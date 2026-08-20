"""Contracts for the legacy-exact DualCMP corridor refiner v2."""

import torch

from opencood.models.colif.dual_source_forecast import (
    DualSourceMotionAnchoredForecastHead,
)
from opencood.models.colif.motion_anchored_forecast import (
    MotionAnchoredForecastHead,
)


_B, _Q, _L, _D = 2, 2, 3, 16


def _args(source_enabled=True):
    return {
        "dual_source_version": 2,
        "source_refinement_enabled": source_enabled,
        "history_frames": 4,
        "future_frames": 4,
        "frame_period_s": 0.1,
        "min_forecast_history_steps": 2,
        "motion_anchor_scales": [0.5, 1.0, 1.5],
        "mode_interaction": "actor_conditioned",
        "motion_context_heads": 4,
        "motion_context_layers": 1,
        "motion_context_ffn": 32,
        "history_encoder_heads": 4,
        "history_encoder_layers": 1,
        "history_encoder_ffn": 32,
        "source_heads": 4,
        "source_bev_dim": _D,
        "lidar_range": [-4.0, -4.0, -3.0, 4.0, 4.0, 3.0],
        "dropout": 0.0,
        "source_dropout": 0.0,
    }


def _head(source_enabled=True):
    return DualSourceMotionAnchoredForecastHead(
        _D, _args(source_enabled=source_enabled)
    )


def _inputs():
    torch.manual_seed(17)
    query = torch.randn(_B, _Q, _D)
    boxes = torch.zeros(_B, _Q, 7)
    boxes[..., 0] = torch.tensor([[0.2, -0.3], [0.4, 0.1]])
    history = torch.zeros(_B, _Q, 4, 7)
    history[..., -3:, 0] = torch.tensor([0.0, 0.1, 0.25])
    history[..., -3:, 1] = torch.tensor([0.0, 0.02, 0.05])
    history_mask = torch.zeros(_B, _Q, 4, dtype=torch.bool)
    history_mask[..., -3:] = True
    valid = torch.ones(_B, _Q, dtype=torch.bool)
    source_bev = torch.randn(_B, _L, _D, 8, 8)
    cav_mask = torch.tensor([[True, True, True], [True, True, False]])
    source_to_receiver = torch.eye(4).view(1, 1, 4, 4).repeat(
        _B, _L, 1, 1
    )
    source_to_receiver[:, 1, 0, 3] = 1.5
    source_to_receiver[:, 2, 1, 3] = -1.0
    return {
        "query": query,
        "boxes": boxes,
        "valid": valid,
        "history": history,
        "history_mask": history_mask,
        "receiver_time": torch.zeros(_B),
        "cooperative": torch.randn(_B, _Q, _D),
        "source_bev": source_bev,
        "cav_mask": cav_mask,
        "source_to_receiver": source_to_receiver,
        "delta_t_s": torch.tensor([[0.0, 0.1, 0.2], [0.0, 0.3, 0.4]]),
    }


def _forward(head, data):
    return head(
        data["query"],
        data["boxes"],
        data["valid"],
        data["history"],
        data["history_mask"],
        data["receiver_time"],
        cooperative_context=data["cooperative"],
        source_bev=data["source_bev"],
        cav_mask=data["cav_mask"],
        source_to_receiver=data["source_to_receiver"],
        delta_t_s=data["delta_t_s"],
    )


def test_motion_base_initialization_matches_legacy_head_exactly():
    torch.manual_seed(23)
    legacy = MotionAnchoredForecastHead(_D, _args())
    torch.manual_seed(23)
    dual = _head()

    legacy_state = legacy.state_dict()
    dual_state = dual.motion_base.state_dict()
    assert legacy_state.keys() == dual_state.keys()
    for key in legacy_state:
        torch.testing.assert_close(legacy_state[key], dual_state[key], atol=0, rtol=0)


def test_initial_dual_base_matches_legacy_forward_and_final_is_identity():
    torch.manual_seed(29)
    legacy = MotionAnchoredForecastHead(_D, _args()).eval()
    torch.manual_seed(29)
    dual = _head().eval()
    data = _inputs()
    torch.manual_seed(41)
    expected = legacy(
        data["query"], data["boxes"], data["valid"], data["history"],
        data["history_mask"], data["receiver_time"],
        cooperative_context=data["cooperative"],
    )
    torch.manual_seed(41)
    actual = _forward(dual, data)

    torch.testing.assert_close(actual["base_pred_trajs"], expected["pred_trajs"], atol=0, rtol=0)
    torch.testing.assert_close(actual["base_pred_scores"], expected["pred_scores"], atol=0, rtol=0)
    torch.testing.assert_close(actual["pred_trajs"], actual["base_pred_trajs"], atol=0, rtol=0)
    torch.testing.assert_close(actual["pred_scores"], actual["base_pred_scores"], atol=0, rtol=0)


def test_source_disabled_reuses_only_legacy_loss_layers():
    output = _forward(_head(source_enabled=False).eval(), _inputs())

    assert output["forecast_loss_layer_names"] == ("proposal", "cross_mode")
    torch.testing.assert_close(
        output["forecast_loss_layer_weights"], torch.tensor([0.5, 0.5])
    )
    assert output["loss_pred_scores_layers"].shape[2] == 2
    assert output["loss_pred_trajs_local_layers"].shape[2] == 2


def test_full_exposes_time_source_router_and_three_loss_layers():
    head = _head().eval()
    output = _forward(head, _inputs())

    assert output["source_gate"].shape == (_B, _Q, head.modes)
    assert output["source_selection_weights"].shape == (
        _B, _Q, head.modes, head.future_frames, _L
    )
    assert output["source_router_weights"].shape == (
        _B, _Q, head.modes, head.future_frames, _L + 1
    )
    assert output["forecast_loss_layer_names"] == (
        "proposal", "cross_mode", "source_final"
    )
    torch.testing.assert_close(
        output["forecast_loss_layer_weights"], torch.tensor([0.5, 0.5, 1.0])
    )


def test_forced_null_preserves_base_after_source_heads_are_nonzero():
    head = _head().eval()
    with torch.no_grad():
        head.source_refiner.velocity_head.weight.fill_(0.2)
        head.source_refiner.log_std_head.weight.fill_(0.2)
        head.source_refiner.mode_score_head.weight.fill_(0.2)
    head.force_null_route = True
    output = _forward(head, _inputs())

    torch.testing.assert_close(output["pred_trajs"], output["base_pred_trajs"], atol=0, rtol=0)
    torch.testing.assert_close(output["pred_scores"], output["base_pred_scores"], atol=0, rtol=0)
    expected = torch.zeros_like(output["source_router_weights"])
    expected[..., 0] = output["forecast_valid_mask"][..., None, None]
    torch.testing.assert_close(output["source_router_weights"], expected)


def test_final_only_gradient_does_not_reach_motion_base():
    head = _head().train()
    output = _forward(head, _inputs())
    final_loss = output["loss_pred_trajs_local_layers"][:, :, 2, ..., :2].square().mean()
    final_loss.backward()

    assert all(
        parameter.grad is None or not torch.count_nonzero(parameter.grad)
        for parameter in head.motion_base.parameters()
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in head.source_refiner.parameters()
    )


def test_base_only_gradient_does_not_reach_source_refiner():
    head = _head().train()
    output = _forward(head, _inputs())
    base_loss = output["loss_pred_trajs_local_layers"][:, :, :2, ..., :2].square().mean()
    base_loss.backward()

    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in head.motion_base.parameters()
    )
    assert all(
        parameter.grad is None or not torch.count_nonzero(parameter.grad)
        for parameter in head.source_refiner.parameters()
    )


def test_invalid_actor_has_zero_router_and_forecast():
    head = _head().eval()
    data = _inputs()
    data["valid"][0, 1] = False
    output = _forward(head, data)

    assert not output["forecast_valid_mask"][0, 1]
    assert torch.count_nonzero(output["pred_scores"][0, 1]) == 0
    assert torch.count_nonzero(output["pred_trajs"][0, 1]) == 0
    assert torch.count_nonzero(output["source_router_weights"][0, 1]) == 0


def test_source_joint_permutation_preserves_final():
    head = _head().eval()
    data = _inputs()
    original = _forward(head, data)
    permutation = torch.tensor([2, 0, 1])
    data["source_bev"] = data["source_bev"][:, permutation]
    data["cav_mask"] = data["cav_mask"][:, permutation]
    data["source_to_receiver"] = data["source_to_receiver"][:, permutation]
    data["delta_t_s"] = data["delta_t_s"][:, permutation]
    permuted = _forward(head, data)

    torch.testing.assert_close(original["pred_trajs"], permuted["pred_trajs"])
    torch.testing.assert_close(original["pred_scores"], permuted["pred_scores"])


def test_source_forward_does_not_advance_global_rng():
    head = _head().train()
    data = _inputs()
    torch.manual_seed(101)
    before = torch.random.get_rng_state().clone()
    _forward(head, data)
    after = torch.random.get_rng_state().clone()

    torch.random.set_rng_state(before)
    head.motion_base(
        data["query"], data["boxes"], data["valid"], data["history"],
        data["history_mask"], data["receiver_time"],
        cooperative_context=data["cooperative"],
    )
    expected_after_base = torch.random.get_rng_state()
    torch.testing.assert_close(after, expected_after_base, atol=0, rtol=0)
