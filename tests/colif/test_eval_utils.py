from types import SimpleNamespace

import pytest
import torch

from opencood.tools.colif.eval_utils import (
    _history_linear_predictions,
    _history_velocity_estimates,
    _kinematic_lateral_mode_bank,
    _residual_prototype_mode_bank,
    SceneReceiverDistributedSampler,
    _absolute_forecast_error,
    evaluate_colif_strict_online,
)


def test_kinematic_lateral_mode_bank_adds_symmetric_turn_hypotheses():
    times = torch.linspace(0.1, 5.0, 50)
    bank = _kinematic_lateral_mode_bank(
        torch.tensor([2.0, 0.0]),
        heading=torch.tensor(0.0),
        future_times_s=times,
        lateral_endpoint_m=6.0,
    )

    assert bank.shape == (6, 50, 2)
    torch.testing.assert_close(bank[0], torch.zeros_like(bank[0]))
    torch.testing.assert_close(bank[4, -1], torch.tensor([10.0, 6.0]))
    torch.testing.assert_close(bank[5, -1], torch.tensor([10.0, -6.0]))
    # The lateral component grows smoothly from the current state rather than
    # applying an instantaneous endpoint offset.
    assert 0.0 < float(bank[4, 0, 1]) < 6.0


def test_residual_prototype_bank_rotates_local_endpoints_into_receiver_axes():
    times = torch.linspace(0.1, 5.0, 50)
    prototypes = torch.tensor([[2.0, 3.0], [-4.0, -1.0]])
    bank = _residual_prototype_mode_bank(
        torch.tensor([0.0, 2.0]),
        heading=torch.tensor(0.0),
        future_times_s=times,
        prototypes_longitudinal_lateral_m=prototypes,
    )

    assert bank.shape == (2, 50, 2)
    # Motion direction is +y, so its left-hand perpendicular is -x.
    torch.testing.assert_close(bank[0, -1], torch.tensor([-3.0, 12.0]))
    torch.testing.assert_close(bank[1, -1], torch.tensor([1.0, 6.0]))


def test_forecast_error_includes_current_tracking_localization_error():
    prediction = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
    track_current = torch.tensor([2.0, 0.0])
    target = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    gt_current = torch.tensor([0.0, 0.0])
    errors = _absolute_forecast_error(
        prediction, track_current, target, gt_current
    )
    torch.testing.assert_close(errors, torch.tensor([[2.0, 2.0]]))


def test_history_velocity_estimates_recover_linear_motion_with_missing_frames():
    history = torch.zeros(6, 2)
    mask = torch.tensor([True, False, True, True, False, True])
    time = torch.arange(6, dtype=torch.float32) * 0.1
    history[:, 0] = 3.0 + 4.0 * time
    history[:, 1] = -2.0 - 1.5 * time

    ols, median = _history_velocity_estimates(history, mask)

    expected = torch.tensor([4.0, -1.5])
    torch.testing.assert_close(ols, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(median, expected, atol=1e-5, rtol=1e-5)


def test_history_linear_predictions_remove_latest_measurement_noise():
    history = torch.zeros(6, 2)
    time = torch.arange(6, dtype=torch.float32) * 0.1
    history[:, 0] = 3.0 + 4.0 * time
    history[:, 1] = -2.0 - 1.5 * time
    history[-1] += torch.tensor([0.6, -0.4])
    mask = torch.ones(6, dtype=torch.bool)
    future_times = torch.tensor([0.1, 0.2])

    ols, _ = _history_linear_predictions(history, mask, future_times)

    # A fitted line is not forced through the noisy last sample.  Its
    # displacement therefore includes a correction back toward the stable
    # history rather than merely multiplying the latest two-frame velocity.
    assert float(ols[0, 0]) < 0.4
    assert float(ols[0, 1]) > -0.4


def test_scene_receiver_sampler_never_splits_a_sequence() -> None:
    records = [
        SimpleNamespace(scene_id="a", receiver_id="0", start_position=index)
        for index in range(3)
    ] + [
        SimpleNamespace(scene_id="b", receiver_id="0", start_position=index)
        for index in range(2)
    ]
    dataset = SimpleNamespace(clip_records=records)
    rank_zero = set(SceneReceiverDistributedSampler(dataset, 0, 2).indices)
    rank_one = set(SceneReceiverDistributedSampler(dataset, 1, 2).indices)
    assert not rank_zero & rank_one
    assert rank_zero | rank_one == set(range(5))
    assert {0, 1, 2}.issubset(rank_zero) or {0, 1, 2}.issubset(rank_one)


def test_strict_online_metrics_match_perfect_forecast() -> None:
    class PerfectModel(torch.nn.Module):
        def forward(self, runtime, state=None):
            trajectory = torch.zeros(1, 1, 2, 50, 5)
            trajectory[..., 0] = torch.arange(1, 51).view(1, 1, 1, 50) * 0.1
            outputs = {
                "active_mask": torch.tensor([[True]]),
                "forecast_valid_mask": torch.tensor([[True]]),
                "track_boxes": torch.zeros(1, 1, 7),
                "runtime_ids": torch.tensor([[0]]),
                "pred_trajs": trajectory,
            }
            return outputs, object(), {}

    future = torch.zeros(1, 1, 50, 2)
    future[..., 0] = torch.arange(1, 51).view(1, 1, 50) * 0.1
    batch = {
        "scene_receiver_keys": [("scene", 0)],
        "runtime_frames": [{"receiver_frame_idx": torch.tensor([0])}],
        "target_frames": [{
            "current_mask": torch.tensor([[True]]),
            "actor_ids": torch.tensor([[7]]),
            "current_boxes": torch.zeros(1, 1, 7),
            "evidence_source_bits": torch.tensor([[1]]),
            "future_mask": torch.ones(1, 1, 50, dtype=torch.bool),
            "future_xy": future,
        }],
    }
    model = PerfectModel()
    metrics = evaluate_colif_strict_online(model, [batch], torch.device("cpu"))
    assert metrics["minADE_5s"] == 0.0
    assert metrics["minFDE_5s"] == 0.0
    assert metrics["forecast_coverage"] == 1.0
    assert metrics["unmatched_active_rate"] == 0.0
    assert metrics["age_1_actor_count_5s"] == 1
    assert metrics["age_1_minFDE_5s"] == 0.0
    assert metrics["receiver_only_actor_count_5s"] == 1
    assert metrics["receiver_only_minFDE_5s"] == 0.0
    assert metrics["partner_only_actor_count_5s"] == 0


def test_strict_online_reports_partner_only_short_history_forecast() -> None:
    class PartnerOnlyModel(torch.nn.Module):
        def forward(self, runtime, state=None):
            trajectory = torch.zeros(1, 1, 2, 50, 5)
            outputs = {
                "active_mask": torch.tensor([[True]]),
                "forecast_valid_mask": torch.tensor([[True]]),
                "track_boxes": torch.zeros(1, 1, 7),
                "runtime_ids": torch.tensor([[0]]),
                "pred_trajs": trajectory,
                "query_age": torch.tensor([[2]]),
            }
            return outputs, object(), {}

    batch = {
        "scene_receiver_keys": [("scene", 0)],
        "runtime_frames": [{"receiver_frame_idx": torch.tensor([0])}],
        "target_frames": [{
            "current_mask": torch.tensor([[True]]),
            "actor_ids": torch.tensor([[7]]),
            "current_boxes": torch.zeros(1, 1, 7),
            "evidence_source_bits": torch.tensor([[2]]),
            "future_mask": torch.ones(1, 1, 50, dtype=torch.bool),
            "future_xy": torch.zeros(1, 1, 50, 2),
        }],
    }

    metrics = evaluate_colif_strict_online(
        PartnerOnlyModel(), [batch], torch.device("cpu")
    )

    assert metrics["partner_only_actor_count_5s"] == 1
    assert metrics["partner_only_minFDE_5s"] == 0.0
    assert metrics["partner_only_age_2_3_actor_count_5s"] == 1
    assert metrics["partner_only_age_2_3_minFDE_5s"] == 0.0


class _ZeroForecastModel(torch.nn.Module):
    def forward(self, runtime, state=None):
        trajectory = torch.zeros(1, 1, 2, 50, 5)
        outputs = {
            "active_mask": torch.tensor([[True]]),
            "forecast_valid_mask": torch.tensor([[True]]),
            "track_boxes": torch.zeros(1, 1, 7),
            "runtime_ids": torch.tensor([[0]]),
            "pred_trajs": trajectory,
        }
        return outputs, object(), {}


def _single_actor_metric_batch(future_mask: torch.Tensor) -> dict:
    future = torch.zeros(1, 1, 50, 2)
    future[0, 0, :, 0] = torch.arange(1, 51, dtype=torch.float32)
    return {
        "scene_receiver_keys": [("scene", 0)],
        "runtime_frames": [{"receiver_frame_idx": torch.tensor([0])}],
        "target_frames": [{
            "current_mask": torch.tensor([[True]]),
            "actor_ids": torch.tensor([[7]]),
            "current_boxes": torch.zeros(1, 1, 7),
            "evidence_source_bits": torch.tensor([[1]]),
            "future_mask": future_mask.view(1, 1, 50),
            "future_xy": future,
        }],
    }


def test_cmp_fixed_protocol_uses_correct_10hz_endpoints() -> None:
    metrics = evaluate_colif_strict_online(
        _ZeroForecastModel(),
        [_single_actor_metric_batch(torch.ones(50, dtype=torch.bool))],
        torch.device("cpu"),
        forecast_metric_protocol="cmp_fixed",
    )

    assert metrics["minADE_1s"] == pytest.approx(5.5)
    assert metrics["minFDE_1s"] == pytest.approx(10.0)
    assert metrics["minADE_3s"] == pytest.approx(15.5)
    assert metrics["minFDE_3s"] == pytest.approx(30.0)
    assert metrics["minADE_5s"] == pytest.approx(25.5)
    assert metrics["minFDE_5s"] == pytest.approx(50.0)
    assert metrics["actor_count_1s"] == 1
    assert metrics["actor_count_3s"] == 1
    assert metrics["actor_count_5s"] == 1
    assert metrics["forecast_metric_protocol"] == "cmp_fixed"
    assert metrics["forecast_fde_indices"] == {
        "1s": 9,
        "3s": 29,
        "5s": 49,
    }
    assert metrics["forecast_requires_complete_future"] is True


def test_cmp_fixed_excludes_truncated_future_but_online_masked_preserves_it() -> None:
    future_mask = torch.zeros(50, dtype=torch.bool)
    future_mask[:8] = True
    batch = _single_actor_metric_batch(future_mask)

    online_metrics = evaluate_colif_strict_online(
        _ZeroForecastModel(),
        [batch],
        torch.device("cpu"),
        forecast_metric_protocol="online_masked",
    )
    fixed_metrics = evaluate_colif_strict_online(
        _ZeroForecastModel(),
        [batch],
        torch.device("cpu"),
        forecast_metric_protocol="cmp_fixed",
    )

    assert online_metrics["actor_count_1s"] == 1
    assert online_metrics["actor_count_3s"] == 1
    assert online_metrics["actor_count_5s"] == 1
    assert online_metrics["minFDE_1s"] == pytest.approx(8.0)
    assert online_metrics["minFDE_3s"] == pytest.approx(8.0)
    assert online_metrics["minFDE_5s"] == pytest.approx(8.0)
    assert online_metrics["forecast_fde_indices"] is None
    assert (
        online_metrics["forecast_fde_endpoint_rule"]
        == "last_valid_within_horizon"
    )
    assert fixed_metrics["actor_count_1s"] == 0
    assert fixed_metrics["actor_count_3s"] == 0
    assert fixed_metrics["actor_count_5s"] == 0
    # The protocol changes forecast displacement eligibility, not online
    # matching or forecast coverage bookkeeping.
    assert online_metrics["forecast_coverage"] == 1.0
    assert fixed_metrics["forecast_coverage"] == 1.0


def test_cmp_fixed_matches_only_the_complete_future_population() -> None:
    batch = _single_actor_metric_batch(torch.ones(50, dtype=torch.bool))
    targets = batch["target_frames"][0]
    targets["current_mask"] = torch.tensor([[True, True]])
    targets["actor_ids"] = torch.tensor([[7, 8]])
    targets["current_boxes"] = torch.zeros(1, 2, 7)
    targets["current_boxes"][0, 1, 0] = 0.1
    targets["evidence_source_bits"] = torch.tensor([[1, 1]])
    targets["future_mask"] = torch.ones(1, 2, 50, dtype=torch.bool)
    targets["future_mask"][0, 0, 8:] = False
    targets["future_xy"] = torch.zeros(1, 2, 50, 2)
    targets["future_xy"][0, 1, :, 0] = torch.arange(
        1, 51, dtype=torch.float32
    )

    metrics = evaluate_colif_strict_online(
        _ZeroForecastModel(),
        [batch],
        torch.device("cpu"),
        forecast_metric_protocol="cmp_fixed",
    )

    # The closer but truncated target must not steal the only runtime query
    # from the complete-future CMP metric population.
    assert metrics["actor_count_1s"] == 1
    assert metrics["actor_count_3s"] == 1
    assert metrics["actor_count_5s"] == 1
    assert metrics["minFDE_5s"] == pytest.approx(50.1)
    # Coverage deliberately remains defined over all online forecast targets.
    assert metrics["target_actor_frames"] == 2
    assert metrics["matched_actor_frames"] == 1
