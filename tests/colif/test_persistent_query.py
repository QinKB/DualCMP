import pytest
import torch
from types import SimpleNamespace

import numpy as np

from opencood.loss.colif_loss import ColifLoss
from opencood.models.colif.future_adapter import FutureAdapter
from opencood.models.colif.dense_measurement import DenseMeasurementProposalGenerator
from opencood.models.colif.lifecycle import QueryLifecycle
from opencood.models.colif.measurement_state_update import MeasurementStateUpdate
from opencood.models.colif.persistent_query_mtr import PersistentQueryMTRAdapter
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


def test_mtr_local_gmm_is_rotated_to_receiver_axes():
    trajectory = torch.zeros(1, 1, 1, 5)
    trajectory[..., 0] = 2.0
    trajectory[..., 2:4] = torch.log(torch.tensor(3.0))
    rotated = PersistentQueryMTRAdapter._local_gmm_to_receiver(
        trajectory, torch.tensor([torch.pi / 2])
    )
    torch.testing.assert_close(
        rotated[..., :2], torch.tensor([[[[0.0, 2.0]]]]), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(rotated[..., 2:4], trajectory[..., 2:4])
    torch.testing.assert_close(rotated[..., 4], trajectory[..., 4], atol=1e-6, rtol=0)


def test_mtr_history_attributes_match_cmp_v2v4real_contract():
    adapter = PersistentQueryMTRAdapter.__new__(PersistentQueryMTRAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.frame_period_s = 0.1
    adapter.cmp_timestamp_scale = 0.1
    history = torch.zeros(2, 11, 7)
    history[..., 3:6] = torch.tensor([1.5, 1.6, 3.9])
    history[0, -1, :3] = torch.tensor([5.0, 2.0, 0.5])
    history[1, -1, :3] = torch.tensor([8.0, 2.0, 0.5])
    mask = torch.zeros(2, 11, dtype=torch.bool)
    mask[:, -1] = True
    center = history[0, -1].clone()
    features, last_position = adapter._history_features_for_center(
        history, mask, center, 0, torch.tensor(1.0)
    )
    assert features.shape == (2, 11, 22)
    torch.testing.assert_close(features[0, -1, :3], torch.zeros(3))
    torch.testing.assert_close(
        features[0, -1, 3:6], torch.tensor([3.9, 1.6, 1.5])
    )
    torch.testing.assert_close(features[:, -1, 6], torch.ones(2))
    assert features[0, -1, 7] == 1 and features[1, -1, 7] == 0
    torch.testing.assert_close(features[:, -1, 19], torch.tensor([0.1, 0.1]))
    torch.testing.assert_close(last_position[:, 0], torch.tensor([0.0, 3.0]))
    assert torch.count_nonzero(features[:, :-1]) == 0

    later_features, _ = adapter._history_features_for_center(
        history, mask, center, 0, torch.tensor(123.4)
    )
    torch.testing.assert_close(
        later_features[:, -1, 19], torch.tensor([12.34, 12.34])
    )
    torch.testing.assert_close(later_features[..., :19], features[..., :19])
    torch.testing.assert_close(later_features[..., 20:], features[..., 20:])


def test_mtr_history_builder_is_elementwise_equal_to_cmp_dataset_code():
    # Import the bundled CMP implementation itself; this is intentionally an
    # implementation-equivalence test rather than a hand-written expectation.
    from mtr.datasets.v2v4real_multiego_dataset import (
        V2V4RealMultiEgoDataset,
    )

    adapter = PersistentQueryMTRAdapter.__new__(PersistentQueryMTRAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.frame_period_s = 0.1
    adapter.cmp_timestamp_scale = 0.1
    timestamps = 1.2 + torch.arange(11, dtype=torch.float32) * 0.01
    history = torch.zeros(3, 11, 7)
    for actor in range(3):
        frame = torch.arange(11, dtype=torch.float32)
        history[actor, :, 0] = 4.0 * actor + 0.17 * frame
        history[actor, :, 1] = -1.3 * actor + 0.04 * frame
        history[actor, :, 2] = -1.0
        history[actor, :, 3:6] = torch.tensor([1.56, 1.6, 3.9])
        history[actor, :, 6] = 0.2 * actor + 0.01 * frame
    mask = torch.ones(3, 11, dtype=torch.bool)
    center_index = 1
    center = history[center_index, -1]
    actual, actual_last = adapter._history_features_for_center(
        history, mask, center, center_index, torch.tensor(13.0)
    )

    # CMP stores boxes as [x,y,z,l,w,h,yaw_degrees,valid], while CoLIF state
    # stores [x,y,z,h,w,l,yaw_radians].
    cmp_past = torch.cat(
        (
            history[..., :3],
            history[..., [5, 4, 3]],
            torch.rad2deg(history[..., 6:7]),
            mask[..., None].float(),
        ),
        dim=-1,
    ).numpy()
    cmp_center = cmp_past[center_index : center_index + 1, -1]
    dummy_future = np.zeros((3, 1, 8), dtype=np.float32)
    expected, expected_mask, _, _ = (
        V2V4RealMultiEgoDataset.generate_centered_trajs_for_agents(
            center_objects=cmp_center,
            obj_trajs_past=cmp_past,
            obj_types=np.array(["TYPE_VEHICLE"] * 3),
            center_indices=np.array([center_index]),
            sdc_index=0,
            timestamps=timestamps.numpy(),
            obj_trajs_future=dummy_future,
        )
    )
    torch.testing.assert_close(actual, torch.from_numpy(expected[0]))
    torch.testing.assert_close(
        mask, torch.from_numpy(expected_mask[0]).bool()
    )
    expected_last = torch.from_numpy(expected[0, :, :, :3])[:, -1]
    torch.testing.assert_close(actual_last, expected_last)


def test_variable_history_training_is_causal_and_keeps_current_state():
    adapter = PersistentQueryMTRAdapter.__new__(PersistentQueryMTRAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.history_frames = 11
    adapter.history_truncation_probability = 1.0
    adapter.train()
    torch.manual_seed(3)
    original = torch.ones(32, 11, dtype=torch.bool)
    sampled = adapter._sample_variable_history(original)
    assert sampled[:, -1].all()
    assert (sampled.sum(-1) >= 1).all()
    assert (sampled.sum(-1) < 11).any()
    # Every sampled history is a suffix, never a non-causal hole pattern.
    assert ((~sampled[:, :-1]) | sampled[:, 1:]).all()


def test_single_observation_is_state_only_and_not_packed_for_forecast():
    from opencood.hypes_yaml.yaml_utils import load_yaml

    hypes = load_yaml("opencood/hypes_yaml/colif/colif_v2v4real.yaml")
    args = dict(hypes["model"]["args"]["forecast"])
    args["enabled"] = True
    adapter = PersistentQueryMTRAdapter(256, args).eval()
    query = torch.zeros(1, 2, 256)
    boxes = torch.zeros(1, 2, 7)
    boxes[..., 3:6] = torch.tensor([1.56, 1.6, 3.9])
    history = boxes[:, :, None].repeat(1, 1, 11, 1)
    mask = torch.zeros(1, 2, 11, dtype=torch.bool)
    mask[0, 0, -1] = True
    mask[0, 1, -2:] = True
    packed = adapter._pack_context(
        query,
        boxes,
        torch.ones(1, 2, dtype=torch.bool),
        history,
        mask,
        torch.zeros(1),
    )
    entries, batch, _, _, _ = packed
    assert entries == [(0, 1)]
    # The one-frame actor is absent from the context graph as well as from
    # the center set, so it cannot corrupt the mature actor representation.
    assert batch["input_dict"]["obj_trajs"].shape[1] == 1


def test_heading_smoothing_is_causal_and_uses_circular_mean():
    adapter = PersistentQueryMTRAdapter.__new__(PersistentQueryMTRAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.heading_smoothing_steps = 3
    history = torch.zeros(1, 1, 5, 7)
    history[0, 0, :, 6] = torch.tensor(
        [3.12, -3.12, 3.10, 0.2, 1.0]
    )
    mask = torch.ones(1, 1, 5, dtype=torch.bool)
    smoothed = adapter._causal_smooth_headings(history, mask)
    # Circular averaging must remain near pi across the wrap boundary rather
    # than collapsing the first three headings toward zero.
    assert abs(abs(float(smoothed[0, 0, 2, 6])) - torch.pi) < 0.08
    # Changing a future yaw cannot alter an earlier causal smoothed state.
    changed = history.clone()
    changed[0, 0, 4, 6] = -2.0
    changed_smoothed = adapter._causal_smooth_headings(changed, mask)
    torch.testing.assert_close(
        smoothed[0, 0, :4, 6], changed_smoothed[0, 0, :4, 6]
    )


def test_persistent_query_content_is_injected_as_residual():
    adapter = PersistentQueryMTRAdapter.__new__(PersistentQueryMTRAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.query_feature_projection = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        adapter.query_feature_projection.weight.copy_(torch.eye(2))
    center, objects = adapter._inject_query_features(
        torch.zeros(1, 2),
        torch.zeros(1, 2, 2),
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
    )
    torch.testing.assert_close(center, torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(
        objects, torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CMP KNN requires CUDA")
def test_zero_bridge_adapter_is_numerically_equal_to_cmp_mtr_forward():
    from opencood.hypes_yaml.yaml_utils import load_yaml

    hypes = load_yaml(
        "opencood/hypes_yaml/colif/colif_v2v4real.yaml"
    )
    args = dict(hypes["model"]["args"]["forecast"])
    args["enabled"] = True
    args["history_truncation_probability"] = 0.0
    adapter = PersistentQueryMTRAdapter(256, args).cuda().eval()
    report = adapter.load_cmp_checkpoint(
        "MTR/output/v2v4real_multiego_cobevt_c256_no_agg/ckpt/"
        "best_model.pth"
    )
    assert report["loaded_count"] == 880

    boxes = torch.zeros(1, 3, 7, device="cuda")
    boxes[0, :, 0] = torch.tensor([2.0, 8.0, 15.0], device="cuda")
    boxes[0, :, 1] = torch.tensor([-2.0, 1.0, 4.0], device="cuda")
    boxes[0, :, 3:6] = torch.tensor(
        [1.56, 1.6, 3.9], device="cuda"
    )
    boxes[0, :, 6] = torch.tensor([0.1, -0.2, 0.3], device="cuda")
    history = boxes[:, :, None].repeat(1, 1, 11, 1)
    steps = torch.arange(11, device="cuda", dtype=torch.float32) - 10
    history[..., 0] += steps.view(1, 1, -1) * 0.25
    history[..., 1] += steps.view(1, 1, -1) * 0.03
    mask = torch.ones(1, 3, 11, dtype=torch.bool, device="cuda")
    valid = torch.ones(1, 3, dtype=torch.bool, device="cuda")
    query = torch.zeros(1, 3, 256, device="cuda")
    receiver_time = torch.tensor([13.0], device="cuda")

    with torch.no_grad():
        packed = adapter._pack_context(
            query, boxes, valid, history, mask, receiver_time
        )
        entries, cmp_batch, _, _, _ = packed
        cmp_batch["input_dict"]["center_objects_type"] = (
            np.array(["TYPE_VEHICLE"] * len(entries))
        )
        cmp_batch = adapter.context_encoder(cmp_batch)
        cmp_batch = adapter.decoder(cmp_batch)
        actual = adapter(
            query, boxes, valid, history, mask, receiver_time
        )

    expected_scores = cmp_batch["pred_scores"]
    expected_trajs = cmp_batch["pred_trajs"]
    for row, (batch_index, slot) in enumerate(entries):
        torch.testing.assert_close(
            actual["pred_scores"][batch_index, slot],
            expected_scores[row],
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            actual["pred_trajs_local"][batch_index, slot],
            expected_trajs[row],
            atol=1e-5,
            rtol=1e-5,
        )


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
