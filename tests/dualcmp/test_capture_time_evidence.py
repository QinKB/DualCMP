"""Contracts for capture-time actor/source evidence."""

from dataclasses import replace

import pytest
import torch

from opencood.models.dualcmp.actor_source_tokenizer import (
    ActorAlignedSourceTokenizer,
)
from opencood.models.dualcmp.capture_time_evidence import (
    capture_aligned_actor_boxes,
)
from opencood.models.dualcmp.mtr_actor_adapter import pack_mtr_actor_polylines
from opencood.models.dualcmp.model import DualCmpClipModel
from opencood.models.dualcmp.intention_query_forecast import (
    IntentionQueryForecastHead,
)
from opencood.models.dualcmp.state import ActorState
from opencood.tools.dualcmp.protocol import (
    checkpoint_forecast_contract,
    normalize_train_delays_ms,
    resolve_forecast_ablation,
    validate_evaluation_delay_contract,
    validate_forecast_contract,
    validate_training_delay_contract,
)
from opencood.tools.dualcmp.temporal_protocol import (
    checkpoint_forecast_contract as extended_checkpoint_forecast_contract,
    resolve_forecast_ablation as extended_resolve_forecast_ablation,
)


def _pose(x):
    pose = torch.eye(4)
    pose[0, 3] = float(x)
    return pose


def _state():
    current_pose = _pose(1.0).unsqueeze(0)
    state = ActorState.empty(1, 1, 4, 2, current_pose, torch.tensor([0.1]))
    boxes = torch.tensor([[[
        [5.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0],
        [5.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0],
    ]]])
    poses = torch.stack((_pose(0.0), _pose(1.0))).reshape(1, 1, 2, 4, 4)
    return replace(
        state,
        box=boxes[:, :, -1],
        valid_mask=torch.tensor([[True]]),
        active_mask=torch.tensor([[True]]),
        runtime_id=torch.tensor([[0]]),
        next_runtime_id=torch.tensor([1]),
        last_update_time_s=torch.tensor([[0.1]]),
        forecast_history_box=boxes,
        forecast_history_mask=torch.tensor([[[True, True]]]),
        forecast_history_time_s=torch.tensor([[[0.0, 0.1]]]),
        forecast_history_pose=poses,
    )


def test_capture_time_selects_historical_actor_position_in_current_frame():
    aligned = capture_aligned_actor_boxes(
        _state(), torch.tensor([[0.1, 0.0]]),
        torch.tensor([[True, True]]))
    assert aligned["mask"].tolist() == [[[True, True]]]
    torch.testing.assert_close(aligned["age_s"], torch.tensor([[0.0, 0.1]]))
    torch.testing.assert_close(
        aligned["boxes"][0, 0, 0], _state().box[0, 0], rtol=0, atol=0)
    # Current source sees x=5.  The delayed source captured the actor before
    # the receiver moved +1m, so its historical cells are at current-frame x=4.
    torch.testing.assert_close(
        aligned["boxes"][0, 0, :, 0], torch.tensor([5.0, 4.0]))


def test_capture_time_without_same_identity_history_fails_closed():
    aligned = capture_aligned_actor_boxes(
        _state(), torch.tensor([[0.05]]), torch.tensor([[True]]))
    assert not aligned["mask"].any()
    assert not aligned["boxes"].any()


def test_future_source_capture_is_rejected():
    with pytest.raises(ValueError, match="later than receiver"):
        capture_aligned_actor_boxes(
            _state(), torch.tensor([[0.2]]), torch.tensor([[True]]))


def test_training_distribution_and_evaluation_delay_contracts_are_distinct():
    provenance = {
        "train_delays_ms": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
        "delay_seed": 17,
        "capture_time_evidence": True,
    }
    assert validate_evaluation_delay_contract(
        provenance, 0, True, context="test") == (
            0.0, 100.0, 200.0, 300.0, 400.0, 500.0)
    validate_evaluation_delay_contract(provenance, 500, True, context="test")
    validate_training_delay_contract(
        provenance, [0, 100, 200, 300, 400, 500], 17, True,
        context="test")
    with pytest.raises(RuntimeError, match="training delays"):
        validate_training_delay_contract(
            provenance, [0, 100], 17, True, context="test")
    with pytest.raises(RuntimeError, match="delay seed"):
        validate_training_delay_contract(
            provenance, [0, 100, 200, 300, 400, 500], 18, True,
            context="test")
    with pytest.raises(RuntimeError, match="evidence alignment"):
        validate_evaluation_delay_contract(provenance, 100, False, context="test")
    with pytest.raises(RuntimeError, match="training-delay"):
        validate_evaluation_delay_contract({}, 100, True, context="test")
    with pytest.raises(ValueError, match="unique"):
        normalize_train_delays_ms([0, 0])


def test_legacy_fixed_delay_checkpoint_can_be_evaluated_out_of_distribution():
    provenance = {
        "partner_delay_ms": 100.0,
        "capture_time_evidence": True,
    }
    assert validate_evaluation_delay_contract(
        provenance, 500, True, context="test") == (100.0,)


def test_forecast_ablation_contract_separates_representation_alignment_and_age():
    assert resolve_forecast_ablation("source") == {
        "forecast_representation": "source",
        "capture_alignment": True,
        "age_encoding": True,
        "mean_source_aggregation": False,
    }
    assert extended_resolve_forecast_ablation("source_temporal") == {
        "forecast_representation": "source_temporal",
        "capture_alignment": True,
        "age_encoding": True,
        "mean_source_aggregation": False,
    }
    with pytest.raises(ValueError, match="belongs to the source ablation"):
        extended_resolve_forecast_ablation(
            "source_temporal", mean_source_aggregation=True)
    assert extended_checkpoint_forecast_contract({
        "forecast_representation": "source_temporal",
        "capture_alignment": True,
        "age_encoding": True,
    }) == {
        "forecast_representation": "source_temporal",
        "capture_alignment": True,
        "age_encoding": True,
        "mean_source_aggregation": False,
    }
    assert resolve_forecast_ablation(
        "source", disable_capture_alignment=True) == {
            "forecast_representation": "source",
            "capture_alignment": False,
            "age_encoding": True,
            "mean_source_aggregation": False,
        }
    assert resolve_forecast_ablation(
        "source", disable_age_encoding=True) == {
            "forecast_representation": "source",
            "capture_alignment": True,
            "age_encoding": False,
            "mean_source_aggregation": False,
        }
    assert resolve_forecast_ablation(
        "source", disable_capture_time_evidence=True) == {
            "forecast_representation": "source",
            "capture_alignment": False,
            "age_encoding": False,
            "mean_source_aggregation": False,
        }
    assert resolve_forecast_ablation(
        "source", mean_source_aggregation=True) == {
            "forecast_representation": "source",
            "capture_alignment": True,
            "age_encoding": True,
            "mean_source_aggregation": True,
        }

    legacy = {"capture_time_evidence": False}
    assert checkpoint_forecast_contract(legacy) == {
        "forecast_representation": "source",
        "capture_alignment": False,
        "age_encoding": False,
        "mean_source_aggregation": False,
    }
    assert resolve_forecast_ablation("history") == {
        "forecast_representation": "history",
        "capture_alignment": False,
        "age_encoding": False,
        "mean_source_aggregation": False,
    }
    assert resolve_forecast_ablation("fused") == {
        "forecast_representation": "fused",
        "capture_alignment": False,
        "age_encoding": False,
        "mean_source_aggregation": False,
    }
    provenance = {
        "forecast_representation": "fused",
        "capture_alignment": True,
        "age_encoding": True,
    }
    assert checkpoint_forecast_contract(provenance) == {
        "forecast_representation": "fused",
        "capture_alignment": False,
        "age_encoding": False,
        "mean_source_aggregation": False,
    }
    validate_forecast_contract(provenance, "fused", False, False)
    with pytest.raises(RuntimeError, match="forecast ablation"):
        validate_forecast_contract(provenance, "history", False, False)
    with pytest.raises(ValueError, match="requires source"):
        resolve_forecast_ablation("fused", mean_source_aggregation=True)
    mean_provenance = {
        "forecast_representation": "source",
        "capture_alignment": True,
        "age_encoding": True,
        "mean_source_aggregation": True,
    }
    validate_forecast_contract(
        mean_provenance, "source", True, True,
        mean_source_aggregation=True)
    with pytest.raises(RuntimeError, match="forecast ablation"):
        validate_forecast_contract(mean_provenance, "source", True, True)


def test_mean_source_aggregation_is_mask_aware_and_opt_in():
    head = IntentionQueryForecastHead.__new__(IntentionQueryForecastHead)
    torch.nn.Module.__init__(head)
    head.query_dim = 2
    head.frame_period_s = 0.1
    entries = [(0, 0)]
    tokens = torch.tensor(
        [[[[[1.0, 3.0], [100.0, 100.0]],
           [[3.0, 5.0], [7.0, 9.0]]]]]
    )
    positions = torch.tensor(
        [[[[[1.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
           [[3.0, 0.0, 0.0], [7.0, 0.0, 0.0]]]]]
    )
    mask = torch.tensor([[[[True, False], [True, True]]]])

    direct = head._pack_evidence_memory(
        entries, tokens, positions, mask, None, "source")
    explicit_direct = head._pack_evidence_memory(
        entries, tokens, positions, mask, None, "source",
        aggregate_sources=False)
    averaged = head._pack_evidence_memory(
        entries, tokens, positions, mask, None, "source",
        aggregate_sources=True)

    assert direct[0].shape == (1, 4, 2)
    assert torch.equal(direct[0], tokens.reshape(1, 4, 2))
    assert all(torch.equal(left, right) for left, right in zip(
        direct[:3], explicit_direct[:3]))
    assert averaged[0].shape == (1, 2, 2)
    assert torch.allclose(
        averaged[0], torch.tensor([[[2.0, 4.0], [7.0, 9.0]]]))
    assert torch.allclose(
        averaged[1], torch.tensor([[[2.0, 0.0, 0.0], [7.0, 0.0, 0.0]]]))
    assert torch.equal(averaged[2], torch.tensor([[True, True]]))


def test_forecast_history_uses_observation_time_not_tracker_epsilon():
    state = _state()
    tracker_internal_time = torch.tensor([0.101])
    state = replace(state, receiver_time_s=tracker_internal_time)
    observation_time = torch.tensor([0.1])
    observation_pose = _pose(1.0).unsqueeze(0)
    updated = DualCmpClipModel.apply_learned_update(
        state,
        torch.zeros(1, 1, 4),
        state.box,
        torch.zeros(1, 1, 2),
        torch.ones(1, 1),
        0.2,
        observation_time,
        observation_pose,
    )
    assert updated.receiver_time_s.item() == tracker_internal_time.item()
    assert updated.forecast_history_time_s[0, 0, -1].item() == observation_time.item()
    torch.testing.assert_close(
        updated.forecast_history_pose[0, 0, -1], observation_pose[0])


def test_zero_delay_tokenizer_is_legacy_equivalent():
    torch.manual_seed(7)
    tokenizer = ActorAlignedSourceTokenizer(
        input_dim=4, query_dim=4, max_cav=2,
        lidar_range=[-10, -10, -2, 10, 10, 2], roi_grid_size=3)
    source_bev = torch.randn(1, 2, 4, 8, 8)
    source_mask = torch.tensor([[True, True]])
    boxes = torch.tensor([[[2.0, -1.0, 0.0, 1.5, 2.0, 4.0, 0.2]]])
    actor_mask = torch.tensor([[True]])
    legacy = tokenizer(source_bev, source_mask, boxes, actor_mask)
    specific = tokenizer(
        source_bev, source_mask, boxes[:, :, None].expand(-1, -1, 2, -1),
        actor_mask[:, :, None].expand(-1, -1, 2), reference_boxes=boxes)
    for key in ("tokens", "positions", "token_mask", "sampling_grid"):
        torch.testing.assert_close(legacy[key], specific[key], rtol=0, atol=0)


def test_mtr_actor_timestamp_uses_recorded_history_time():
    boxes = torch.tensor([[[0.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]]])
    history = boxes[:, :, None].expand(-1, -1, 11, -1).clone()
    mask = torch.ones(1, 1, 11, dtype=torch.bool)
    times = torch.tensor(
        [[[-1.0, -0.9, -0.8, -0.7, -0.6, -0.5,
           -0.4, -0.3, -0.2, -0.08, 0.0]]])
    packed = pack_mtr_actor_polylines(
        boxes, torch.tensor([[True]]), history, mask, times,
        torch.tensor([0.0]), min_forecast_history_steps=2,
        min_context_history_steps=2, frame_period_s=0.1,
        timestamp_scale=0.1)
    actual = packed.batch_dict["input_dict"]["obj_trajs"][0, 0, :, 19]
    torch.testing.assert_close(actual, times[0, 0] * 0.1)


def test_regular_mtr_timestamps_match_legacy_zero_delay_contract():
    boxes = torch.tensor([[[0.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]]])
    history = boxes[:, :, None].expand(-1, -1, 11, -1).clone()
    mask = torch.ones(1, 1, 11, dtype=torch.bool)
    receiver_time = torch.tensor([2.0])
    offsets = torch.arange(-10, 1, dtype=torch.float32) * 0.1
    times = receiver_time[:, None, None] + offsets[None, None]
    packed = pack_mtr_actor_polylines(
        boxes, torch.tensor([[True]]), history, mask, times,
        receiver_time, min_forecast_history_steps=2,
        min_context_history_steps=2, frame_period_s=0.1,
        timestamp_scale=0.1)
    actual = packed.batch_dict["input_dict"]["obj_trajs"][0, 0, :, 19]
    legacy = (receiver_time[0] + offsets) * 0.1
    torch.testing.assert_close(actual, legacy, rtol=0, atol=0)
