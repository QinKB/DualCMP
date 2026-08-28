import pytest
import torch
from types import SimpleNamespace

from opencood.loss.colif_loss import ColifLoss
from opencood.models.colif.future_adapter import FutureAdapter
from opencood.models.colif.dense_measurement import DenseMeasurementProposalGenerator
from opencood.models.colif.lifecycle import QueryLifecycle
from opencood.models.colif.measurement_state_update import MeasurementStateUpdate
from opencood.models.colif.query_geometry import (
    MetricBoxCoder,
    transform_and_propagate_boxes,
    transform_boxes_between_receiver_frames,
)
from opencood.models.colif.state import GlobalForecastState
from opencood.models.point_pillar_colif import PointPillarColif


def _state(query_dim=8):
    query = torch.randn(1, 4, query_dim)
    boxes = torch.zeros(1, 4, 7)
    return GlobalForecastState.empty(
        query_feat=query,
        ref_box=boxes,
        memory_len=3,
        receiver_pose=torch.eye(4).unsqueeze(0),
        receiver_time_s=torch.zeros(1),
    )


def test_coordinate_propagation_uses_pose_and_current_velocity_only():
    box = torch.zeros(1, 1, 7)
    box[..., 0] = 1.0
    velocity = torch.tensor([[[2.0, 0.0]]])
    previous_pose = torch.eye(4).unsqueeze(0)
    current_pose = torch.eye(4).unsqueeze(0)
    current_pose[:, 0, 3] = 1.0
    propagated, propagated_velocity = transform_and_propagate_boxes(
        box,
        velocity,
        previous_pose,
        current_pose,
        torch.tensor([0.0]),
        torch.tensor([0.1]),
    )
    torch.testing.assert_close(propagated[..., 0], torch.tensor([[0.2]]))
    torch.testing.assert_close(propagated_velocity, velocity)


def test_complete_history_is_reexpressed_in_current_receiver_frame():
    history = torch.zeros(1, 2, 3, 7)
    history[..., 0] = torch.tensor([1.0, 2.0, 3.0])
    previous_pose = torch.eye(4).unsqueeze(0)
    current_pose = torch.eye(4).unsqueeze(0)
    current_pose[:, 0, 3] = 1.0
    transformed = transform_boxes_between_receiver_frames(
        history, previous_pose, current_pose
    )
    torch.testing.assert_close(
        transformed[0, 0, :, 0], torch.tensor([0.0, 1.0, 2.0])
    )
    torch.testing.assert_close(transformed[..., 1:], history[..., 1:])


def test_forecast_history_preserves_capture_time_receiver_coordinates():
    history = torch.zeros(1, 1, 3, 7)
    history[0, 0, -1, 0] = 8.0
    current_native = torch.zeros(1, 1, 7)
    current_native[0, 0, 0] = 7.5
    appended = PointPillarColif._append_native_forecast_history(
        history, current_native, torch.ones(1, 1, dtype=torch.bool)
    )
    # The old state remains 8 m in its own capture frame.  If it had been
    # re-expressed after a 1 m receiver translation it would incorrectly be 7.
    torch.testing.assert_close(appended[0, 0, -2, 0], torch.tensor(8.0))
    torch.testing.assert_close(appended[0, 0, -1, 0], torch.tensor(7.5))


def test_future_adapter_obeys_strict_norm_bound_and_backpropagates():
    query = torch.randn(2, 3, 8, requires_grad=True)
    cue = torch.randn(2, 3, 8, requires_grad=True)
    uncertainty = torch.rand(2, 3, 1)
    valid = torch.ones(2, 3, dtype=torch.bool)
    adapter = FutureAdapter(
        8,
        mode="recurrent_bptt",
        gate_max=0.25,
        clip_norm=2.0,
        dropout_probability=0.0,
    )
    updated, diagnostics = adapter(query, cue, uncertainty, valid)
    assert float(diagnostics["feedback_delta_norm"].max()) <= 0.5 + 1e-6
    updated.square().sum().backward()
    assert cue.grad is not None and cue.grad.abs().sum() > 0
    assert query.grad is not None and torch.isfinite(query.grad).all()


def test_detached_adapter_blocks_gradient_to_future_cue():
    query = torch.randn(1, 2, 8, requires_grad=True)
    cue = torch.randn(1, 2, 8, requires_grad=True)
    adapter = FutureAdapter(
        8, mode="detached", dropout_probability=0.0
    )
    updated, _ = adapter(
        query, cue, torch.zeros(1, 2, 1), torch.ones(1, 2, dtype=torch.bool)
    )
    updated.sum().backward()
    assert cue.grad is None
    assert query.grad is not None


def test_future_adapter_ignores_slots_without_a_valid_previous_forecast():
    """An active first-observation query is not yet a forecast-bearing query."""

    query = torch.randn(1, 3, 8)
    cue = torch.randn_like(query)
    adapter = FutureAdapter(
        8,
        mode="recurrent_bptt",
        dropout_probability=0.0,
    ).eval()
    valid = torch.tensor([[False, True, False]])
    updated, diagnostics = adapter(
        query, cue, torch.zeros(1, 3, 1), valid
    )
    base = adapter.query_norm(query)

    torch.testing.assert_close(updated[:, [0, 2]], base[:, [0, 2]])
    assert not diagnostics["feedback_kept"][0, 0]
    assert diagnostics["feedback_kept"][0, 1]
    assert not diagnostics["feedback_kept"][0, 2]
    assert diagnostics["feedback_gate"][0, 0].item() == 0.0
    assert diagnostics["feedback_gate"][0, 2].item() == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA AMP")
def test_future_adapter_amp_burnin_does_not_detach_linear_weights():
    """Regression for autocast caching weights first seen under no_grad."""

    adapter = FutureAdapter(
        16,
        mode="recurrent_bptt",
        dropout_probability=0.0,
    ).cuda().train()
    query = torch.randn(1, 2, 16, device="cuda")
    cue = torch.randn_like(query)
    uncertainty = torch.rand(1, 2, 1, device="cuda")
    valid = torch.ones(1, 2, dtype=torch.bool, device="cuda")

    with torch.cuda.amp.autocast(enabled=True):
        with torch.no_grad():
            adapter(query, cue, uncertainty, valid)
        updated, _ = adapter(query, cue, uncertainty, valid)
        loss = updated.square().sum()
    loss.backward()

    assert adapter.cue_projection[0].weight.grad is not None
    assert adapter.cue_projection[2].weight.grad is not None
    assert adapter.gate[-1].weight.grad is not None


def test_lifecycle_caps_births_and_preserves_runtime_id():
    lifecycle = QueryLifecycle(
        birth_threshold=0.5,
        keep_threshold=0.25,
        miss_ttl=2,
        birth_suppression_radius=0.1,
        max_active_queries=2,
    )
    scores = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
    boxes = torch.zeros(1, 4, 7)
    boxes[0, :, 0] = torch.arange(4.0)
    first = lifecycle(
        scores,
        boxes,
        torch.zeros(1, 4, dtype=torch.bool),
        torch.zeros(1, 4, dtype=torch.long),
        torch.full((1, 4), -1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
    )
    assert first["active_mask"].sum() == 2
    assert first["runtime_id"][0, :2].tolist() == [0, 1]
    second = lifecycle(
        torch.zeros_like(scores),
        boxes,
        first["active_mask"],
        first["miss_count"],
        first["runtime_id"],
        first["next_runtime_id"],
    )
    assert second["active_mask"].sum() == 2
    assert second["runtime_id"][0, :2].tolist() == [0, 1]


def test_state_contains_compact_cue_not_full_trajectories():
    state = _state()
    assert state.future_cue.shape == state.query_feat.shape
    assert not hasattr(state, "future_trajs")
    assert not hasattr(state, "future_scores")


def test_receiver_future_target_is_rotated_to_actor_heading():
    # Choose radians whose numeric degree value is pi/2, directly exercising
    # the released CMP "degrees consumed as radians" compatibility contract.
    heading = torch.tensor(torch.pi / 2 * torch.pi / 180.0)
    displacement = torch.tensor([[0.0, 2.0]])
    actor_local = ColifLoss._receiver_displacement_to_actor(
        displacement, heading
    )
    torch.testing.assert_close(
        actor_local, torch.tensor([[2.0, 0.0]]), atol=1e-5, rtol=1e-5
    )


def test_future_target_is_centered_on_predicted_track_not_gt_current():
    stored_displacement = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    gt_current = torch.tensor([10.0, 20.0])
    predicted_track = torch.tensor([12.0, 17.0])
    relative = ColifLoss._future_relative_to_track(
        stored_displacement, gt_current, predicted_track
    )
    torch.testing.assert_close(
        relative, torch.tensor([[-1.0, 5.0], [1.0, 7.0]])
    )


def test_query_objectness_reads_differentiable_cmp_dense_evidence():
    geometry = SimpleNamespace(
        lidar_x_min=-2.0,
        lidar_y_min=-2.0,
        lidar_x_max=2.0,
        lidar_y_max=2.0,
    )
    query_logits = torch.zeros(1, 1, requires_grad=True)
    query_boxes = torch.zeros(1, 1, 7, requires_grad=True)
    dense_logits = torch.full((1, 2, 4, 4), 2.0, requires_grad=True)
    fused, sampled = PointPillarColif._fuse_detection_evidence(
        geometry, query_logits, query_boxes, dense_logits
    )
    torch.testing.assert_close(sampled, torch.tensor([[2.0]]))
    torch.testing.assert_close(fused, torch.tensor([[2.0]]))
    fused.sum().backward()
    assert query_logits.grad is not None and query_logits.grad.abs().sum() > 0
    assert dense_logits.grad is not None and dense_logits.grad.abs().sum() > 0


def test_query_objectness_does_not_count_foreground_prior_twice():
    prior_probability = 0.01
    prior_logit = torch.logit(torch.tensor(prior_probability))
    geometry = SimpleNamespace(
        lidar_x_min=-2.0,
        lidar_y_min=-2.0,
        lidar_x_max=2.0,
        lidar_y_max=2.0,
        query_prior_logit=float(prior_logit),
    )
    query_logits = prior_logit.reshape(1, 1)
    query_boxes = torch.zeros(1, 1, 7)
    dense_probability = 0.2
    dense_logit = torch.logit(torch.tensor(dense_probability))
    dense_logits = dense_logit.expand(1, 2, 4, 4).clone()
    fused, _ = PointPillarColif._fuse_detection_evidence(
        geometry, query_logits, query_boxes, dense_logits
    )
    torch.testing.assert_close(fused.sigmoid(), torch.tensor([[dense_probability]]))


def test_zero_box_residual_preserves_full_measurement_reference():
    coder = MetricBoxCoder([-10, -10, -5, 10, 10, 3])
    reference = torch.tensor(
        [[[2.0, -3.0, -0.5, 1.7, 1.8, 4.2, 1.2]]]
    )
    raw = torch.zeros(1, 1, 8)
    raw[..., 7] = 1.0
    decoded = coder.decode_from_reference_box(raw, reference)
    torch.testing.assert_close(decoded, reference)


def test_dense_measurement_seeds_cmp_regressed_box_and_respects_active_space():
    generator = DenseMeasurementProposalGenerator(
        lidar_range=[-4.0, -4.0, -5.0, 4.0, 4.0, 3.0],
        voxel_size=[1.0, 1.0, 8.0],
        anchor_number=2,
        args={
            "max_proposals": 2,
            "pre_topk": 4,
            "score_threshold": 0.1,
            "suppression_radius": 1.0,
            "anchor_dims_hwl": [1.56, 1.6, 3.9],
            "anchor_rotations_deg": [0.0, 90.0],
        },
    )
    logits = torch.full((1, 2, 2, 2), -10.0)
    logits[0, 1, 0, 1] = 5.0
    residuals = torch.zeros(1, 14, 2, 2, requires_grad=True)
    previous_active = torch.zeros(1, 4, dtype=torch.bool)
    previous_boxes = torch.zeros(1, 4, 7)
    proposals = generator(logits, residuals, previous_active, previous_boxes)
    assert proposals["proposal_mask"].sum() == 1
    assert float(proposals["proposal_boxes"][0, 0, 6].detach()) == pytest.approx(
        float(torch.pi / 2)
    )
    proposals["proposal_boxes"].sum().backward()
    assert residuals.grad is not None and residuals.grad.abs().sum() > 0


def test_lifecycle_rejects_unseeded_inactive_births():
    lifecycle = QueryLifecycle(
        birth_threshold=0.5,
        keep_threshold=0.25,
        miss_ttl=2,
        birth_suppression_radius=0.1,
        max_active_queries=4,
    )
    result = lifecycle(
        torch.tensor([[0.9, 0.8]]),
        torch.zeros(1, 2, 7),
        torch.zeros(1, 2, dtype=torch.bool),
        torch.zeros(1, 2, dtype=torch.long),
        torch.full((1, 2), -1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        birth_candidate_mask=torch.tensor([[False, True]]),
    )
    assert result["birth_mask"].tolist() == [[False, True]]


def test_lifecycle_does_not_reuse_one_dense_peak_for_duplicate_tracks():
    lifecycle = QueryLifecycle(
        birth_threshold=0.5,
        keep_threshold=0.25,
        miss_ttl=1,
        birth_suppression_radius=0.1,
        max_active_queries=4,
    )
    result = lifecycle(
        torch.tensor([[0.9, 0.9]]),
        torch.zeros(1, 2, 7),
        torch.tensor([[True, True]]),
        torch.zeros(1, 2, dtype=torch.long),
        torch.tensor([[3, 4]]),
        torch.tensor([5]),
        measurement_observed_mask=torch.tensor([[True, False]]),
    )
    assert result["observed_mask"].tolist() == [[True, False]]
    assert result["miss_count"].tolist() == [[0, 1]]
    second = lifecycle(
        torch.tensor([[0.9, 0.9]]),
        torch.zeros(1, 2, 7),
        result["active_mask"],
        result["miss_count"],
        result["runtime_id"],
        result["next_runtime_id"],
        measurement_observed_mask=torch.tensor([[True, False]]),
    )
    assert second["active_mask"].tolist() == [[True, False]]


def test_measurements_correct_active_reference_before_seeding_newborn():
    propagated = torch.zeros(1, 3, 7)
    propagated[0, 0, 0] = 1.0
    active = torch.tensor([[True, False, False]])
    proposal_boxes = torch.zeros(1, 2, 7)
    proposal_boxes[0, 0, 0] = 1.5
    proposal_boxes[0, 1, 0] = 8.0
    measurements = {
        "proposal_boxes": proposal_boxes,
        "proposal_logits": torch.tensor([[4.0, 3.0]]),
        "proposal_mask": torch.tensor([[True, True]]),
    }
    reference, birth_mask, observed, logits = (
        PointPillarColif._assign_measurements(
            propagated, active, measurements, match_distance=2.0
        )
    )
    assert reference[0, 0, 0] == 1.5
    assert observed[0, 0]
    assert birth_mask[0, 1]
    assert reference[0, 1, 0] == 8.0
    assert logits[0, :2].tolist() == [4.0, 3.0]


def test_measurement_state_update_is_bounded_and_differentiable():
    update = MeasurementStateUpdate(
        query_dim=8, min_gain=0.1, max_gain=0.9, initial_gain=0.7
    )
    query = torch.randn(1, 2, 8, requires_grad=True)
    propagated = torch.zeros(1, 2, 7)
    measured = propagated.clone()
    measured[..., 0] = 2.0
    measured[..., 6] = -torch.pi + 0.1
    propagated[..., 6] = torch.pi - 0.1
    box, velocity, gain = update(
        query,
        propagated,
        measured,
        torch.zeros(1, 2, 2),
        torch.ones(1, 2, 2),
        torch.zeros(1, 2),
        torch.tensor([[True, False]]),
        torch.tensor([[False, False]]),
    )
    assert 0.1 <= float(gain[0, 0]) <= 0.9
    assert gain[0, 1] == 0
    assert box[0, 1, 0] == 0 and velocity[0, 1, 0] == 0
    assert abs(abs(float(box[0, 0, 6])) - torch.pi) < 0.2
    (box.sum() + velocity.sum()).backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()


def test_half_measurements_are_cast_to_float_persistent_state():
    propagated = torch.zeros(1, 2, 7, dtype=torch.float32)
    measurements = {
        "proposal_boxes": torch.ones(1, 1, 7, dtype=torch.float16),
        "proposal_logits": torch.tensor([[2.0]], dtype=torch.float16),
        "proposal_mask": torch.tensor([[True]]),
    }
    reference, _, _, logits = PointPillarColif._assign_measurements(
        propagated,
        torch.zeros(1, 2, dtype=torch.bool),
        measurements,
        match_distance=2.0,
    )
    assert reference.dtype == torch.float32
    assert logits.dtype == torch.float32
    torch.testing.assert_close(reference[0, 0], torch.ones(7))
