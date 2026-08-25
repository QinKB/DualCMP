import torch

from opencood.models.colif.motion_anchored_forecast import (
    MotionAnchoredForecastHead,
)


def _head(
    mode_interaction="actor_conditioned",
    temporal_history=True,
    proposal_refinement=True,
    trajectory_parameterization="anchored",
):
    return MotionAnchoredForecastHead(
        16,
        {
            "history_frames": 11,
            "future_frames": 50,
            "frame_period_s": 0.1,
            "min_forecast_history_steps": 2,
            "motion_context_heads": 4,
            "motion_context_layers": 1,
            "motion_context_ffn": 32,
            "dropout": 0.0,
            "mode_interaction": mode_interaction,
            "temporal_history": temporal_history,
            "history_encoder_layers": 1,
            "history_encoder_heads": 4,
            "proposal_refinement": proposal_refinement,
            "trajectory_parameterization": trajectory_parameterization,
        },
    )


def test_first_observation_creates_no_long_horizon_forecast():
    head = _head().eval()
    query = torch.zeros(1, 1, 16)
    boxes = torch.zeros(1, 1, 7)
    history = torch.zeros(1, 1, 11, 7)
    mask = torch.zeros(1, 1, 11, dtype=torch.bool)
    mask[..., -1] = True
    output = head(
        query, boxes, torch.ones(1, 1, dtype=torch.bool), history, mask,
        torch.zeros(1),
    )
    assert not output["forecast_valid_mask"].any()
    assert torch.count_nonzero(output["pred_trajs"]) == 0


def test_zero_residual_initialization_equals_multiscale_robust_motion():
    head = _head().eval()
    query = torch.zeros(1, 1, 16)
    boxes = torch.zeros(1, 1, 7)
    history = torch.zeros(1, 1, 11, 7)
    mask = torch.zeros(1, 1, 11, dtype=torch.bool)
    mask[..., -3:] = True
    history[0, 0, -3:, 0] = torch.tensor([0.0, 0.2, 0.4])
    boxes[0, 0, 0] = 0.4
    output = head(
        query, boxes, torch.ones(1, 1, dtype=torch.bool), history, mask,
        torch.zeros(1),
    )
    expected_velocity = torch.tensor([2.0, 0.0])
    final = output["pred_trajs"][0, 0, :, -1, :2]
    expected = (
        expected_velocity[None]
        * head.anchor_scales[:, None]
        * head.future_times_s[-1]
    )
    torch.testing.assert_close(final, expected)
    assert output["forecast_loss_in_receiver_frame"] is True


def test_direct_parameterization_starts_from_zero_displacement() -> None:
    head = _head(trajectory_parameterization="direct").eval()
    query = torch.zeros(1, 1, 16)
    boxes = torch.zeros(1, 1, 7)
    history = torch.zeros(1, 1, 11, 7)
    mask = torch.zeros(1, 1, 11, dtype=torch.bool)
    mask[..., -3:] = True
    history[0, 0, -3:, 0] = torch.tensor([0.0, 0.2, 0.4])

    output = head(
        query,
        boxes,
        torch.ones(1, 1, dtype=torch.bool),
        history,
        mask,
        torch.zeros(1),
    )

    assert output["forecast_valid_mask"].all()
    assert torch.count_nonzero(output["pred_trajs"][..., :2]) == 0


def test_direct_parameterization_is_not_limited_by_anchor_residual_bound() -> None:
    anchored = _head(trajectory_parameterization="anchored").eval()
    direct = _head(trajectory_parameterization="direct").eval()
    direct.load_state_dict(anchored.state_dict(), strict=True)
    with torch.no_grad():
        anchored.proposal_residual_head.bias.fill_(20.0)
        direct.proposal_residual_head.bias.fill_(20.0)
    query = torch.zeros(1, 1, 16)
    boxes = torch.zeros(1, 1, 7)
    history = torch.zeros(1, 1, 11, 7)
    mask = torch.zeros(1, 1, 11, dtype=torch.bool)
    mask[..., -3:] = True
    inputs = (
        query,
        boxes,
        torch.ones(1, 1, dtype=torch.bool),
        history,
        mask,
        torch.zeros(1),
    )

    anchored_xy = anchored(*inputs)["pred_trajs"][..., :2]
    direct_xy = direct(*inputs)["pred_trajs"][..., :2]

    assert float(anchored_xy.abs().max()) < anchored.residual_limit_m
    torch.testing.assert_close(direct_xy, torch.full_like(direct_xy, 20.0))


def test_future_loss_reaches_residual_context_and_mode_heads():
    head = _head().train()
    query = torch.randn(1, 2, 16, requires_grad=True)
    boxes = torch.zeros(1, 2, 7)
    history = torch.zeros(1, 2, 11, 7)
    mask = torch.zeros(1, 2, 11, dtype=torch.bool)
    mask[..., -3:] = True
    history[:, :, -3:, 0] = torch.tensor([0.0, 0.1, 0.2])
    output = head(
        query, boxes, torch.ones(1, 2, dtype=torch.bool), history, mask,
        torch.zeros(1),
    )
    loss = (
        output["loss_pred_trajs_local"][..., :2].square().mean()
        + output["loss_pred_scores"].square().mean()
    )
    loss.backward()
    assert head.residual_head.weight.grad is not None
    assert head.mode_score.weight.grad is not None
    assert head.context_encoder.layers[0].linear1.weight.grad is not None
    assert query.grad is not None


def test_actor_conditioned_interaction_changes_relative_mode_probabilities():
    torch.manual_seed(7)
    additive = _head(
        mode_interaction="additive", proposal_refinement=False
    ).eval()
    conditioned = _head(
        mode_interaction="actor_conditioned", proposal_refinement=False
    ).eval()
    conditioned.load_state_dict(additive.state_dict())
    with torch.no_grad():
        additive.mode_score.weight.normal_(0.0, 0.2)
        conditioned.mode_score.weight.copy_(additive.mode_score.weight)

    query = torch.randn(2, 1, 16)
    boxes = torch.zeros(2, 1, 7)
    history = torch.zeros(2, 1, 11, 7)
    history[:, :, -3:, 0] = torch.tensor([0.0, 0.1, 0.2])
    mask = torch.zeros(2, 1, 11, dtype=torch.bool)
    mask[..., -3:] = True
    valid = torch.ones(2, 1, dtype=torch.bool)
    time = torch.zeros(2)

    additive_scores = additive(
        query, boxes, valid, history, mask, time
    )["pred_scores"][:, 0]
    conditioned_scores = conditioned(
        query, boxes, valid, history, mask, time
    )["pred_scores"][:, 0]

    torch.testing.assert_close(additive_scores[0], additive_scores[1])
    assert not torch.allclose(
        conditioned_scores[0], conditioned_scores[1], atol=1e-5, rtol=1e-5
    )


def test_cooperative_actor_context_conditions_forecast_modes() -> None:
    torch.manual_seed(11)
    head = _head().eval()
    with torch.no_grad():
        head.mode_score.weight.normal_(0.0, 0.2)
    query = torch.randn(1, 1, 16)
    boxes = torch.zeros(1, 1, 7)
    history = torch.zeros(1, 1, 11, 7)
    history[..., -3:, 0] = torch.tensor([0.0, 0.1, 0.2])
    mask = torch.zeros(1, 1, 11, dtype=torch.bool)
    mask[..., -3:] = True
    valid = torch.ones(1, 1, dtype=torch.bool)
    time = torch.zeros(1)

    without_partner = head(
        query,
        boxes,
        valid,
        history,
        mask,
        time,
        cooperative_context=torch.zeros_like(query),
    )["pred_scores"]
    with_partner = head(
        query,
        boxes,
        valid,
        history,
        mask,
        time,
        cooperative_context=torch.randn_like(query),
    )["pred_scores"]

    assert not torch.allclose(without_partner, with_partner)


def test_temporal_history_encoder_preserves_intermediate_motion_structure() -> None:
    torch.manual_seed(13)
    head = _head(temporal_history=True).eval()
    straight = torch.zeros(1, 1, 11, 7)
    curved = straight.clone()
    mask = torch.zeros(1, 1, 11, dtype=torch.bool)
    mask[..., -5:] = True
    straight[0, 0, -5:, 0] = torch.linspace(0.0, 4.0, 5)
    curved[0, 0, -5:, 0] = torch.linspace(0.0, 4.0, 5)
    curved[0, 0, -4:-1, 1] = torch.tensor([0.8, 1.2, 0.8])

    straight_feature = head._encode_temporal_history(straight, mask)
    curved_feature = head._encode_temporal_history(curved, mask)

    assert not torch.allclose(straight_feature, curved_feature)
    assert hasattr(head, "history_encoder")


def test_forecast_loss_reaches_temporal_history_encoder() -> None:
    head = _head(temporal_history=True).train()
    query = torch.randn(1, 1, 16, requires_grad=True)
    boxes = torch.zeros(1, 1, 7)
    history = torch.zeros(1, 1, 11, 7)
    history[..., -4:, 0] = torch.tensor([0.0, 0.1, 0.3, 0.6])
    mask = torch.zeros(1, 1, 11, dtype=torch.bool)
    mask[..., -4:] = True
    output = head(
        query,
        boxes,
        torch.ones(1, 1, dtype=torch.bool),
        history,
        mask,
        torch.zeros(1),
    )
    output["loss_pred_trajs_local"][..., :2].square().mean().backward()

    assert head.history_encoder.layers[0].linear1.weight.grad is not None


def test_physics_proposal_and_refinement_are_both_supervised() -> None:
    head = _head().train()
    query = torch.randn(1, 2, 16, requires_grad=True)
    boxes = torch.zeros(1, 2, 7)
    history = torch.zeros(1, 2, 11, 7)
    history[..., -3:, 0] = torch.tensor([0.0, 0.1, 0.2])
    mask = torch.zeros(1, 2, 11, dtype=torch.bool)
    mask[..., -3:] = True
    output = head(
        query,
        boxes,
        torch.ones(1, 2, dtype=torch.bool),
        history,
        mask,
        torch.zeros(1),
    )

    assert output["loss_pred_trajs_local_layers"].shape[2] == 2
    assert output["loss_pred_scores_layers"].shape[2] == 2
    loss = (
        output["loss_pred_trajs_local_layers"][..., :2].square().mean()
        + output["loss_pred_scores_layers"].square().mean()
    )
    loss.backward()

    assert head.proposal_residual_head.weight.grad is not None
    assert head.mode_refiner.linear1.weight.grad is not None
    assert head.residual_head.weight.grad is not None


def test_direct_ablation_bypasses_all_proposal_parameters() -> None:
    head = _head(proposal_refinement=False).train()
    query = torch.randn(1, 1, 16, requires_grad=True)
    boxes = torch.zeros(1, 1, 7)
    history = torch.zeros(1, 1, 11, 7)
    history[..., -3:, 0] = torch.tensor([0.0, 0.1, 0.2])
    mask = torch.zeros(1, 1, 11, dtype=torch.bool)
    mask[..., -3:] = True
    output = head(
        query,
        boxes,
        torch.ones(1, 1, dtype=torch.bool),
        history,
        mask,
        torch.zeros(1),
    )
    output["loss_pred_trajs_local"][..., :2].square().mean().backward()

    assert head.residual_head.weight.grad is not None
    assert head.proposal_residual_head.weight.grad is None
    assert head.mode_refiner.linear1.weight.grad is None


def test_each_forecast_layer_exposes_its_own_intention_points() -> None:
    head = _head(proposal_refinement=True).eval()
    query = torch.randn(1, 1, 16)
    boxes = torch.zeros(1, 1, 7)
    history = torch.zeros(1, 1, 11, 7)
    history[..., -3:, 0] = torch.tensor([0.0, 0.1, 0.2])
    mask = torch.zeros(1, 1, 11, dtype=torch.bool)
    mask[..., -3:] = True
    with torch.no_grad():
        head.proposal_residual_head.bias.fill_(0.25)
        head.residual_head.bias.fill_(0.5)
    output = head(
        query,
        boxes,
        torch.ones(1, 1, dtype=torch.bool),
        history,
        mask,
        torch.zeros(1),
    )

    intentions = output["loss_intention_points_layers"]
    assert intentions.shape == (1, 1, 2, head.modes, 2)
    torch.testing.assert_close(
        intentions[..., 0, :, :],
        output["loss_pred_trajs_local_layers"][..., 0, :, -1, :2],
    )
    torch.testing.assert_close(
        intentions[..., 1, :, :],
        output["loss_pred_trajs_local_layers"][..., 1, :, -1, :2],
    )
    assert not torch.allclose(intentions[..., 0, :, :], intentions[..., 1, :, :])


def test_ablation_freezes_structurally_inactive_forecast_parameters() -> None:
    head = _head(
        temporal_history=False,
        proposal_refinement=False,
    )
    head.freeze_inactive_parameters(
        actor_cooperative_fusion_enabled=False
    )

    assert not any(
        parameter.requires_grad
        for parameter in head.history_encoder.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in head.proposal_residual_head.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in head.mode_refiner.parameters()
    )
    assert not head.cooperative_scale.requires_grad
    assert all(
        parameter.requires_grad
        for parameter in head.residual_head.parameters()
    )


def test_proposal_inference_ablation_returns_supervised_proposal_layer() -> None:
    head = _head(proposal_refinement=True).eval()
    head.inference_layer = "proposal"
    query = torch.randn(1, 1, 16)
    boxes = torch.zeros(1, 1, 7)
    history = torch.zeros(1, 1, 11, 7)
    history[..., -3:, 0] = torch.tensor([0.0, 0.1, 0.2])
    mask = torch.zeros(1, 1, 11, dtype=torch.bool)
    mask[..., -3:] = True
    output = head(
        query,
        boxes,
        torch.ones(1, 1, dtype=torch.bool),
        history,
        mask,
        torch.zeros(1),
    )

    torch.testing.assert_close(
        output["pred_trajs"],
        output["loss_pred_trajs_local_layers"][:, :, 0],
    )
    torch.testing.assert_close(
        output["pred_scores"],
        torch.softmax(output["loss_pred_scores_layers"][:, :, 0], dim=-1),
    )
