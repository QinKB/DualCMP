import torch

from opencood.models.colif.forecast_motion_prior import ForecastMotionPrior
from opencood.models.colif.measurement_state_update import MeasurementStateUpdate
from opencood.models.colif.state import GlobalForecastState
from opencood.models.point_pillar_colif import PointPillarColif


def _inputs(probabilities=None, valid=True, requires_grad=False):
    cv_box = torch.zeros(1, 1, 7)
    cv_box[..., 0] = 1.0
    cv_velocity = torch.tensor([[[10.0, 0.0]]])
    previous_box = torch.zeros(1, 1, 7)
    means = torch.zeros(1, 1, 2, 5, 2)
    means[0, 0, 0, :, 0] = 1.0
    means[0, 0, 1, :, 0] = 4.0
    means.requires_grad_(requires_grad)
    variances = torch.full((1, 1, 2, 5), 0.25)
    if probabilities is None:
        probabilities = torch.tensor([[[0.5, 0.5]]])
    probabilities = probabilities.clone().requires_grad_(requires_grad)
    pose = torch.eye(4).unsqueeze(0)
    return {
        "cv_box": cv_box,
        "cv_velocity": cv_velocity,
        "previous_box": previous_box,
        "future_mode_probabilities": probabilities,
        "future_relative_means": means,
        "future_mode_variances": variances,
        "future_valid_mask": torch.tensor([[valid]]),
        "previous_pose": pose,
        "current_pose": pose.clone(),
        "previous_time_s": torch.tensor([0.0]),
        "current_time_s": torch.tensor([0.1]),
    }


def _prior(mode):
    return ForecastMotionPrior(
        mode=mode,
        frame_period_s=0.1,
        horizon_steps=5,
        cv_component_weight=0.15,
        cv_std_m=1.0,
        min_std_m=0.25,
        max_std_m=3.0,
        mahalanobis_gate=3.0,
        absolute_gate_m=8.0,
        cv_match_distance_m=4.0,
        proposal_score_weight=0.0,
    )


def test_invalid_forecast_is_exactly_constant_velocity():
    result = _prior("recurrent_bptt")(**_inputs(valid=False))
    torch.testing.assert_close(result.predicted_box[..., :2], torch.tensor([[[1.0, 0.0]]]))
    assert not result.selected_forecast_mask.any()
    assert result.mode_probabilities[..., 0].eq(1.0).all()
    assert result.mode_probabilities[..., 1:].eq(0.0).all()


def test_uniform_forecast_keeps_constant_velocity_map_component():
    result = _prior("recurrent_bptt")(**_inputs())
    torch.testing.assert_close(result.predicted_box[..., :2], torch.tensor([[[1.0, 0.0]]]))
    assert not result.selected_forecast_mask.any()


def test_confident_forecast_mode_becomes_spatial_map_reference():
    probabilities = torch.tensor([[[0.05, 0.95]]])
    result = _prior("recurrent_bptt")(**_inputs(probabilities))
    torch.testing.assert_close(result.predicted_box[..., :2], torch.tensor([[[4.0, 0.0]]]))
    assert result.selected_forecast_mask.all()


def test_forecast_centers_are_transformed_to_current_receiver_frame():
    inputs = _inputs(torch.tensor([[[0.05, 0.95]]]))
    inputs["previous_box"][..., 0] = 10.0
    inputs["current_pose"][:, 0, 3] = 1.0
    result = _prior("recurrent_bptt")(**inputs)
    # Previous-frame absolute forecast center is 10 + 4 = 14 m.  The current
    # receiver moved +1 m in world coordinates, so it is observed at 13 m.
    torch.testing.assert_close(result.mode_centers[0, 0, 2], torch.tensor([13.0, 0.0]))
    torch.testing.assert_close(result.predicted_box[0, 0, :2], torch.tensor([13.0, 0.0]))


def test_cv_candidate_has_priority_over_forecast_mode():
    probabilities = torch.tensor([[[0.05, 0.95]]])
    result = _prior("association")(**_inputs(probabilities))
    proposals = torch.zeros(1, 2, 7)
    proposals[0, 0, 0] = 1.0
    proposals[0, 1, 0] = 4.0
    cost, valid = _prior("association").association_cost(
        result,
        proposals,
        torch.zeros(1, 2),
    )
    assert valid[0, 0].all()
    assert cost[0, 0, 0] < cost[0, 0, 1]
    # Association-only never changes the decoder/state propagation reference.
    torch.testing.assert_close(result.predicted_box[..., :2], torch.tensor([[[1.0, 0.0]]]))


def test_forecast_becomes_fallback_when_cv_has_no_valid_candidate():
    inputs = _inputs(torch.tensor([[[0.05, 0.95]]]))
    inputs["future_relative_means"][0, 0, 1, :, 0] = 8.0
    prior_module = _prior("association")
    result = prior_module(**inputs)
    proposals = torch.zeros(1, 1, 7)
    proposals[0, 0, 0] = 8.0
    cost, valid = prior_module.association_cost(
        result, proposals, torch.zeros(1, 1)
    )
    assert valid[0, 0, 0]
    assert torch.isfinite(cost[0, 0, 0])


def test_propagation_only_keeps_cv_association_when_forecast_is_elsewhere():
    inputs = _inputs(torch.tensor([[[0.05, 0.95]]]))
    inputs["future_relative_means"][0, 0, 1, :, 0] = 8.0
    prior_module = _prior("propagation_detached")
    result = prior_module(**inputs)
    proposals = torch.zeros(1, 2, 7)
    proposals[0, 0, 0] = 1.0
    proposals[0, 1, 0] = 8.0
    cost, valid = prior_module.association_cost(
        result, proposals, torch.zeros(1, 2)
    )
    torch.testing.assert_close(cost[0, 0], torch.tensor([0.0, 7.0]))
    assert valid[0, 0].tolist() == [True, False]
    # The pre-measurement state prior remains forecast-conditioned.  A valid
    # CV measurement overwrites it; with no assignment it survives as the
    # missing-state propagation.
    torch.testing.assert_close(result.predicted_box[0, 0, 0], torch.tensor(8.0))


def test_propagation_modes_control_cross_frame_gradient_without_changing_math():
    detached_inputs = _inputs(
        torch.tensor([[[0.05, 0.95]]]), requires_grad=True
    )
    detached = _prior("propagation_detached")(**detached_inputs)
    assert not detached.predicted_box.requires_grad

    recurrent_inputs = _inputs(
        torch.tensor([[[0.05, 0.95]]]), requires_grad=True
    )
    recurrent = _prior("propagation_bptt")(**recurrent_inputs)
    torch.testing.assert_close(
        recurrent.predicted_box, detached.predicted_box
    )
    recurrent.predicted_box[..., 0].sum().backward()
    assert recurrent_inputs["future_relative_means"].grad is not None
    assert recurrent_inputs["future_relative_means"].grad.abs().sum() > 0


def test_detached_and_recurrent_modes_have_different_cross_frame_gradients():
    detached_inputs = _inputs(
        torch.tensor([[[0.05, 0.95]]]), requires_grad=True
    )
    detached = _prior("detached")(**detached_inputs)
    assert not detached.predicted_box.requires_grad

    recurrent_inputs = _inputs(
        torch.tensor([[[0.05, 0.95]]]), requires_grad=True
    )
    recurrent = _prior("recurrent_bptt")(**recurrent_inputs)
    recurrent.predicted_box[..., 0].sum().backward()
    assert recurrent_inputs["future_relative_means"].grad is not None
    assert recurrent_inputs["future_relative_means"].grad.abs().sum() > 0


def test_elapsed_time_selects_nearest_stored_forecast_step():
    inputs = _inputs(torch.tensor([[[0.05, 0.95]]]))
    inputs["future_relative_means"][0, 0, 1, :, 0] = torch.arange(1.0, 6.0)
    inputs["current_time_s"] = torch.tensor([0.31])
    result = _prior("recurrent_bptt")(**inputs)
    assert result.step_index.tolist() == [2]
    torch.testing.assert_close(result.predicted_box[0, 0, 0], torch.tensor(3.0))


def test_runtime_state_stores_resets_and_detaches_short_forecast_distribution():
    query = torch.randn(2, 3, 8, requires_grad=True)
    boxes = torch.zeros(2, 3, 7)
    pose = torch.eye(4).repeat(2, 1, 1)
    state = GlobalForecastState.empty(
        query_feat=query,
        ref_box=boxes,
        memory_len=4,
        receiver_pose=pose,
        receiver_time_s=torch.zeros(2),
        future_modes=6,
        future_horizon_steps=5,
    )
    assert state.future_mode_probabilities.shape == (2, 3, 6)
    assert state.future_relative_means.shape == (2, 3, 6, 5, 2)
    assert state.future_mode_variances.shape == (2, 3, 6, 5)
    assert state.future_spatial_valid_mask.shape == (2, 3)
    torch.testing.assert_close(state.association_ref_box, boxes)

    state.future_relative_means.requires_grad_(True)
    detached = state.detach_differentiable_tensors()
    assert not detached.future_relative_means.requires_grad

    replacement = GlobalForecastState.empty(
        query_feat=torch.zeros_like(query),
        ref_box=boxes,
        memory_len=4,
        receiver_pose=pose,
        receiver_time_s=torch.zeros(2),
        future_modes=6,
        future_horizon_steps=5,
    )
    state.future_spatial_valid_mask[:] = True
    reset = state.replace_rows(torch.tensor([True, False]), replacement)
    assert not reset.future_spatial_valid_mask[0].any()
    assert reset.future_spatial_valid_mask[1].all()
    reset.validate()


def test_gap_imputation_does_not_overwrite_association_reference():
    association_box = torch.zeros(1, 3, 7)
    association_box[..., 0] = torch.tensor([[1.0, 2.0, 3.0]])
    forecast_box = association_box.clone()
    forecast_box[..., 0] = torch.tensor([[8.0, 9.0, 10.0]])
    active = torch.tensor([[True, True, False]])
    measured = torch.tensor([[False, True, False]])
    selected = torch.tensor([[True, True, True]])

    reported = PointPillarColif._reported_box_after_measurement(
        association_box,
        forecast_box,
        active,
        measured,
        selected,
    )

    # Only the active, genuinely unmeasured slot is imputed.  The tracker-side
    # association reference remains a separate untouched tensor.
    torch.testing.assert_close(
        reported[..., 0], torch.tensor([[8.0, 2.0, 3.0]])
    )
    torch.testing.assert_close(
        association_box[..., 0], torch.tensor([[1.0, 2.0, 3.0]])
    )


def test_private_null_assignment_does_not_steal_the_only_good_proposal():
    propagated = torch.zeros(1, 2, 7)
    active = torch.tensor([[True, True]])
    proposals = torch.zeros(1, 2, 7)
    proposals[0, 0, 0] = 1.0
    proposals[0, 1, 0] = 2.0
    measurements = {
        "proposal_boxes": proposals,
        "proposal_logits": torch.tensor([[4.0, 3.0]]),
        "proposal_mask": torch.tensor([[True, True]]),
    }
    # Query 0 accepts both proposals and prefers proposal 0.  Query 1 accepts
    # neither.  Without private null columns Hungarian assigns query 1 to
    # proposal 0 and wrongly pushes query 0 onto proposal 1 before post-gating.
    cost = torch.tensor([[[1.0, 2.0], [1.1, 100.0]]])
    valid = torch.tensor([[[True, True], [False, False]]])
    reference, births, observed, _ = PointPillarColif._assign_measurements(
        propagated,
        active,
        measurements,
        match_distance=4.0,
        association_cost=cost,
        association_valid=valid,
    )
    assert observed.tolist() == [[True, False]]
    torch.testing.assert_close(reference[0, 0, 0], torch.tensor(1.0))
    assert not births.any()


def test_predictive_cost_changes_runtime_measurement_assignment():
    prior_module = _prior("association")
    inputs = _inputs(torch.tensor([[[0.05, 0.95]]]))
    inputs["future_relative_means"][0, 0, 1, :, 0] = 8.0
    prior = prior_module(**inputs)
    proposals = torch.zeros(1, 1, 7)
    proposals[0, 0, 0] = 8.0
    measurements = {
        "proposal_boxes": proposals,
        "proposal_logits": torch.zeros(1, 1),
        "proposal_mask": torch.tensor([[True]]),
    }
    cost, valid = prior_module.association_cost(
        prior, proposals, measurements["proposal_logits"]
    )
    reference, _, observed, _ = PointPillarColif._assign_measurements(
        prior.predicted_box,
        torch.tensor([[True]]),
        measurements,
        match_distance=4.0,
        association_cost=cost,
        association_valid=valid,
    )
    assert observed.all()
    torch.testing.assert_close(reference[0, 0, 0], torch.tensor(8.0))


def test_measurement_variance_correction_trusts_measurement_more_for_uncertain_prior():
    update = MeasurementStateUpdate(
        query_dim=8,
        min_gain=0.1,
        max_gain=0.95,
        initial_gain=0.75,
    )
    propagated = torch.zeros(1, 2, 7)
    measured = propagated.clone()
    measured[..., 0] = 2.0
    box, _, gain = update(
        torch.zeros(1, 2, 8),
        propagated,
        measured,
        torch.zeros(1, 2, 2),
        torch.zeros(1, 2, 2),
        torch.full((1, 2), 5.0),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 2, dtype=torch.bool),
        prior_variance_xy=torch.tensor([[0.1, 9.0]]),
    )
    assert gain[0, 1] > gain[0, 0]
    assert box[0, 1, 0] > box[0, 0, 0]


def test_forecast_prior_never_reduces_learned_measurement_gain():
    update = MeasurementStateUpdate(
        query_dim=8, min_gain=0.1, max_gain=0.95, initial_gain=0.75
    )
    query = torch.zeros(1, 1, 8)
    propagated = torch.zeros(1, 1, 7)
    measured = propagated.clone()
    measured[..., 0] = 2.0
    arguments = (
        query,
        propagated,
        measured,
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
        torch.full((1, 1), 5.0),
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.bool),
    )
    baseline = update(*arguments)
    corrected = update(
        *arguments,
        prior_variance_xy=torch.full((1, 1), 0.01),
        prior_variance_valid_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    assert corrected[2].item() >= baseline[2].item()
    assert corrected[0][0, 0, 0] >= baseline[0][0, 0, 0]


def test_unselected_future_prior_preserves_learned_measurement_update():
    update = MeasurementStateUpdate(
        query_dim=8, min_gain=0.1, max_gain=0.95, initial_gain=0.75
    )
    query = torch.zeros(1, 1, 8)
    propagated = torch.zeros(1, 1, 7)
    measured = propagated.clone()
    measured[..., 0] = 2.0
    arguments = (
        query,
        propagated,
        measured,
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1),
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.bool),
    )
    baseline = update(*arguments)
    masked = update(
        *arguments,
        prior_variance_xy=torch.full((1, 1), 9.0),
        prior_variance_valid_mask=torch.zeros(1, 1, dtype=torch.bool),
    )
    for actual, expected in zip(masked, baseline):
        torch.testing.assert_close(actual, expected)


def test_missing_measurement_preserves_forecast_predicted_state():
    update = MeasurementStateUpdate(query_dim=8)
    predicted = torch.zeros(1, 1, 7)
    predicted[..., 0] = 4.0
    measured = predicted.clone()
    measured[..., 0] = 100.0
    box, velocity, gain = update(
        torch.zeros(1, 1, 8),
        predicted,
        measured,
        torch.tensor([[[3.0, 0.0]]]),
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1),
        torch.zeros(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.bool),
        prior_variance_xy=torch.ones(1, 1),
    )
    torch.testing.assert_close(box, predicted)
    torch.testing.assert_close(velocity, torch.tensor([[[3.0, 0.0]]]))
    assert gain.item() == 0.0


def test_forecast_output_is_compacted_to_five_step_spatial_state():
    prior = _prior("recurrent_bptt")
    scores = torch.tensor([[[0.25, 0.75]]])
    trajectories = torch.zeros(1, 1, 2, 7, 5)
    trajectories[0, 0, 1, :, 0] = torch.arange(1.0, 8.0)
    trajectories[..., 2] = torch.log(torch.tensor(2.0))
    trajectories[..., 3] = torch.log(torch.tensor(4.0))
    probabilities, means, variances, valid = prior.compact_forecast(
        scores,
        trajectories,
        torch.tensor([[True]]),
    )
    assert probabilities.shape == (1, 1, 2)
    assert means.shape == (1, 1, 2, 5, 2)
    assert variances.shape == (1, 1, 2, 5)
    torch.testing.assert_close(means[0, 0, 1, :, 0], torch.arange(1.0, 6.0))
    # Isotropic variance is the mean of x/y variances: (2^2 + 4^2) / 2.
    torch.testing.assert_close(variances, torch.full_like(variances, 10.0))
    assert valid.all()
