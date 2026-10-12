"""CPU contracts for DualCMP ActorState and receiver-frame geometry."""

from __future__ import annotations

from dataclasses import fields, replace
import importlib
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from opencood.models.dualcmp.geometry import previous_to_current, propagate_state, transform_filter, wrap_angle
from opencood.models.dualcmp.state import ActorState
from opencood.models.dualcmp.tracker import ActorTracker
from opencood.models.dualcmp.model import DualCmpClipModel
from opencood.models.dualcmp.proposal import decode_proposals
from opencood.models.dualcmp.ab3dmot_assignment import _box3d_from_lidar, associate_car_giou, load_ab3dmot_association
from opencood.models.dualcmp.kalman import matrices, orientation_correction, predict, update


def _pose(x=0.0, y=0.0, yaw=0.0, dtype=torch.float32):
    cosine, sine = torch.cos(torch.tensor(yaw, dtype=dtype)), torch.sin(torch.tensor(yaw, dtype=dtype))
    result = torch.eye(4, dtype=dtype)
    result[0, 0], result[0, 1], result[1, 0], result[1, 1] = cosine, -sine, sine, cosine
    result[0, 3], result[1, 3] = x, y
    return result


def _state(batch_size=1, max_actors=1, history_len=1, requires_grad=False):
    pose = torch.stack([_pose() for _ in range(batch_size)])
    state = ActorState.empty(batch_size, max_actors, 3, history_len, pose, torch.zeros(batch_size))
    box = state.box.clone()
    velocity = state.velocity.clone()
    box[:, 0] = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0])
    velocity[:, 0] = torch.tensor([1.0, 0.0])
    if requires_grad:
        box.requires_grad_()
        velocity.requires_grad_()
    history_box = state.history_box.clone()
    history_box[:, 0, 0] = torch.tensor([2.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0])
    filter_mean = torch.zeros(batch_size, max_actors, 10)
    filter_mean[..., 0], filter_mean[..., 1], filter_mean[..., 2], filter_mean[..., 3] = box[..., 0], box[..., 1], box[..., 2], box[..., 6]
    filter_mean[..., 4], filter_mean[..., 5], filter_mean[..., 6] = box[..., 5], box[..., 4], box[..., 3]
    filter_mean[..., 7], filter_mean[..., 8] = velocity[..., 0], velocity[..., 1]
    filter_cov = torch.eye(10).view(1, 1, 10, 10).repeat(batch_size, max_actors, 1, 1)
    if requires_grad:
        filter_mean = filter_mean.detach().requires_grad_()
    return replace(
        state, box=box, velocity=velocity, valid_mask=torch.ones_like(state.valid_mask),
        active_mask=torch.ones_like(state.active_mask), runtime_id=torch.arange(max_actors).view(1, -1).repeat(batch_size, 1),
        next_runtime_id=torch.full((batch_size,), max_actors, dtype=torch.long),
        last_update_time_s=torch.zeros_like(state.last_update_time_s), history_box=history_box,
        filter_mean=filter_mean, filter_cov=filter_cov,
        history_mask=torch.ones_like(state.history_mask),
    )


def test_known_receiver_transform_yaw_velocity_and_history():
    state = _state()
    current_pose = _pose(10.0, 0.0, torch.pi / 2).unsqueeze(0)
    propagated, valid_rows, dt = propagate_state(state, current_pose, torch.tensor([0.5]), 1.0)
    assert valid_rows.tolist() == [True]
    torch.testing.assert_close(dt, torch.tensor([0.5]))
    torch.testing.assert_close(propagated.box[0, 0, :2], torch.tensor([0.0, 9.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(propagated.velocity[0, 0], torch.tensor([0.0, -1.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(propagated.box[0, 0, 6], torch.tensor(-torch.pi / 2), atol=1e-6, rtol=0)
    torch.testing.assert_close(propagated.history_box[0, 0, 0, :2], torch.tensor([0.0, 8.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(propagated.history_box[0, 0, 0, 6], torch.tensor(-torch.pi / 2), atol=1e-6, rtol=0)


def test_batch_row_reset_has_no_cross_talk():
    state = _state(batch_size=2)
    current_pose = torch.stack([_pose(), _pose(20.0)])
    propagated, valid_rows, _ = propagate_state(state, current_pose, torch.tensor([1.0, 0.0]), 1.0)
    assert valid_rows.tolist() == [True, False]
    assert propagated.valid_mask.tolist() == [[True], [False]]
    torch.testing.assert_close(propagated.box[0, 0, :2], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(propagated.box[1], torch.zeros_like(propagated.box[1]))
    torch.testing.assert_close(propagated.receiver_pose[1], current_pose[1])
    torch.testing.assert_close(propagated.receiver_time_s, torch.tensor([1.0, 0.0]))


@pytest.mark.parametrize("current_time", [0.0, -0.1, 1.1])
def test_invalid_dt_clears_row(current_time):
    state = _state()
    propagated, valid_rows, dt = propagate_state(state, _pose().unsqueeze(0), torch.tensor([current_time]), 1.0)
    assert valid_rows.tolist() == [False]
    torch.testing.assert_close(dt, torch.tensor([current_time]))
    assert not propagated.valid_mask.any() and not propagated.active_mask.any()
    assert propagated.runtime_id.tolist() == [[-1]]
    assert torch.isnan(propagated.last_update_time_s).all()


def test_invalid_dt_clears_slots_but_preserves_sequence_counter():
    state = replace(_state(), runtime_id=torch.tensor([[41]]), next_runtime_id=torch.tensor([42]))
    propagated, valid_rows, _ = propagate_state(state, _pose().unsqueeze(0), torch.tensor([0.0]), 1.0)
    assert valid_rows.tolist() == [False]
    assert propagated.runtime_id.tolist() == [[-1]]
    assert propagated.next_runtime_id.tolist() == [42]


def test_scene_reset_clears_even_valid_dt():
    state = replace(_state(), runtime_id=torch.tensor([[41]]), next_runtime_id=torch.tensor([42]))
    propagated, valid_rows, _ = propagate_state(
        state, _pose(3.0).unsqueeze(0), torch.tensor([0.5]), 1.0,
        scene_reset_mask=torch.tensor([True]),
    )
    assert valid_rows.tolist() == [False]
    assert not propagated.valid_mask.any()
    assert propagated.next_runtime_id.tolist() == [0]
    torch.testing.assert_close(propagated.receiver_pose[0], _pose(3.0))


def test_invalid_and_scene_reset_rows_have_independent_counters():
    state = replace(
        _state(batch_size=2), runtime_id=torch.tensor([[41], [73]]),
        next_runtime_id=torch.tensor([42, 74]),
    )
    propagated, valid_rows, _ = propagate_state(
        state, torch.stack([_pose(), _pose()]), torch.tensor([0.0, 0.5]), 1.0,
        scene_reset_mask=torch.tensor([False, True]),
    )
    assert valid_rows.tolist() == [False, False]
    assert propagated.runtime_id.tolist() == [[-1], [-1]]
    assert propagated.next_runtime_id.tolist() == [42, 0]


def test_counter_reset_mask_fails_closed_outside_reset_rows():
    state = _state()
    with pytest.raises(ValueError, match="reset_mask subset"):
        state.reset_rows(torch.tensor([False]), reset_id_counter_mask=torch.tensor([True]))
    with pytest.raises(ValueError, match="reset_id_counter_mask"):
        state.reset_rows(torch.tensor([True]), reset_id_counter_mask=torch.tensor([1]))


def test_shape_dtype_and_pose_validation_fail_closed():
    state = _state()
    with pytest.raises(ValueError, match="box must have shape"):
        replace(state, box=torch.zeros(1, 1, 6)).validate()
    with pytest.raises(ValueError, match="floating dtype/device"):
        replace(state, velocity=state.velocity.double()).validate()
    singular = _pose().unsqueeze(0)
    singular[0, 0, 0] = 0
    with pytest.raises(ValueError, match="scale or shear"):
        previous_to_current(state.receiver_pose, singular)
    with pytest.raises(ValueError, match="active actors"):
        replace(state, active_mask=torch.ones_like(state.active_mask), valid_mask=torch.zeros_like(state.valid_mask)).validate()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_geometry_state_fails_closed(dtype):
    pose = _pose().unsqueeze(0).to(dtype)
    time_s = torch.zeros(1, dtype=dtype)
    with pytest.raises(TypeError, match="float32 or torch.float64"):
        ActorState.empty(1, 1, 2, 1, pose, time_s)
    with pytest.raises(TypeError, match="dtype must be"):
        ActorState.empty(1, 1, 2, 1, _pose().unsqueeze(0), torch.zeros(1), dtype=dtype)
    state = _state()
    values = {}
    for item in fields(ActorState):
        value = getattr(state, item.name)
        values[item.name] = value.to(dtype) if value.is_floating_point() else value
    with pytest.raises(TypeError, match="float32 or torch.float64"):
        ActorState(**values).validate()


@pytest.mark.parametrize("mutation, message", [
    (lambda pose: pose.__setitem__((0, 0, 0), 2.0), "scale or shear"),
    (lambda pose: pose.__setitem__((0, 0, 1), 0.2), "scale or shear"),
    (lambda pose: pose.__setitem__((0, 0, 0), -1.0), "reflection"),
    (lambda pose: pose.__setitem__((0, 3, 3), 0.9), "homogeneous bottom row"),
])
def test_invalid_pose_matrices_fail_closed(mutation, message):
    state = _state()
    pose = _pose().unsqueeze(0)
    mutation(pose)
    with pytest.raises(ValueError, match=message):
        previous_to_current(state.receiver_pose, pose)


def test_tilt_beyond_near_planar_limit_fails_closed():
    state = _state()
    pose = _pose().unsqueeze(0)
    tilt = 0.11
    pose[0, 1, 1], pose[0, 1, 2] = torch.cos(torch.tensor(tilt)), -torch.sin(torch.tensor(tilt))
    pose[0, 2, 1], pose[0, 2, 2] = torch.sin(torch.tensor(tilt)), torch.cos(torch.tensor(tilt))
    with pytest.raises(ValueError, match="tilt exceeds"):
        previous_to_current(state.receiver_pose, pose)


def test_runtime_id_history_and_counter_contracts():
    state = _state(max_actors=2)
    with pytest.raises(ValueError, match="unique per row"):
        replace(state, runtime_id=torch.tensor([[0, 0]])).validate()
    with pytest.raises(ValueError, match="next_runtime_id"):
        replace(state, next_runtime_id=torch.tensor([1])).validate()
    with pytest.raises(ValueError, match="history_mask"):
        replace(
            state, valid_mask=torch.tensor([[True, False]]), active_mask=torch.tensor([[True, False]]),
            runtime_id=torch.tensor([[0, -1]]), last_update_time_s=torch.tensor([[0.0, float("nan")]]),
        ).validate()
    with pytest.raises(ValueError, match="age and miss_count"):
        replace(state, age=torch.tensor([[-1, 0]])).validate()
    kept_miss = replace(state, active_mask=torch.tensor([[True, False]]))
    kept_miss.validate()


@pytest.mark.parametrize("max_dt", [float("nan"), float("inf"), torch.tensor([1.0, 2.0]), "bad", True])
def test_invalid_max_dt_fails_closed(max_dt):
    with pytest.raises(ValueError, match="max_dt_s"):
        propagate_state(_state(), _pose().unsqueeze(0), torch.tensor([0.5]), max_dt)


def test_propagation_preserves_or_cuts_autograd_as_requested():
    state = _state(requires_grad=True)
    propagated, _, _ = propagate_state(state, _pose().unsqueeze(0), torch.tensor([1.0]), 1.0)
    propagated.filter_mean.sum().backward()
    assert state.filter_mean.grad is not None

    state.filter_mean.grad.zero_()
    detached = state.detached()
    detached.filter_mean.requires_grad_()
    propagated, _, _ = propagate_state(detached, _pose().unsqueeze(0), torch.tensor([1.0]), 1.0)
    propagated.filter_mean.sum().backward()
    assert detached.filter_mean.grad is not None
    assert torch.count_nonzero(state.filter_mean.grad) == 0


def test_angle_wrap_is_principal_interval():
    torch.testing.assert_close(wrap_angle(torch.tensor([3 * torch.pi])), torch.tensor([-torch.pi]), atol=1e-6, rtol=0)


def test_tracker_uses_ab3dmot_giou_hungarian_runtime_id_assignment():
    state = _state(max_actors=2)
    state = replace(state, box=torch.tensor([[[0.0, 0.0, 0.0, 1, 1, 2, 0],
                                               [1.0, 0.0, 0.0, 1, 1, 2, 0]]]),
                    velocity=torch.zeros_like(state.velocity),
                    filter_mean=torch.tensor([[[0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0, 0, 0],
                                               [1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0, 0, 0]]]))
    proposals = {"boxes": torch.tensor([[[0.4, 0.0, 0.0, 1, 1, 2, 0],
                                          [1.4, 0.0, 0.0, 1, 1, 2, 0]]]),
                 "scores": torch.tensor([[0.9, 0.8]]), "mask": torch.tensor([[True, True]])}
    result = ActorTracker(max_miss=2)(state, proposals, _pose().unsqueeze(0), torch.tensor([0.1]), 1.0)
    assert result.miss_count.tolist() == [[0, 0]]
    assert result.active_mask.tolist() == [[True, True]]
    assert result.runtime_id.tolist() == [[0, 1]]


@pytest.mark.parametrize("proposals,tracks", [
    (torch.tensor([[0.2, 0, 0, 1, 1, 2, 0.1], [8, 0, 0, 1, 1, 2, 0]]),
     torch.tensor([[0, 0, 0, 1, 1, 2, 0], [8.2, 0, 0, 1, 1, 2, -0.1]])),
    (torch.tensor([[0.2, 0, 0, 1, 1, 2, 0.3]]), torch.zeros(0, 7)),
    (torch.zeros(0, 7), torch.tensor([[0, 0, 0, 1, 1, 2, 0]])),
])
def test_ab3dmot_adapter_matches_direct_data_association(proposals, tracks):
    _, direct = load_ab3dmot_association()
    matches, _, _, _, _ = direct(_box3d_from_lidar(proposals), _box3d_from_lidar(tracks),
                                  metric="giou_3d", threshold=-0.2, algm="hungar")
    expected = {int(track): int(proposal) for proposal, track in matches.tolist()}
    assert associate_car_giou(proposals, tracks) == expected


def test_tracker_keeps_two_misses_and_deletes_the_third():
    state = _state()
    proposals = {"boxes": torch.zeros(1, 0, 7), "scores": torch.zeros(1, 0),
                 "mask": torch.zeros(1, 0, dtype=torch.bool)}
    tracker = ActorTracker(max_miss=2)
    for time_s, expected_valid in ((0.1, True), (0.2, False)):
        state = tracker(state, proposals, _pose().unsqueeze(0), torch.tensor([time_s]), 1.0)
        assert bool(state.valid_mask[0, 0]) is expected_valid


def test_quality_emission_preserves_max_age_valid_misses():
    state = _state()
    proposals = {"boxes": torch.zeros(1, 0, 7), "scores": torch.zeros(1, 0),
                 "mask": torch.zeros(1, 0, dtype=torch.bool)}
    tracker = ActorTracker(max_miss=2)
    for time_s in (0.1,):
        state = tracker(state, proposals, _pose().unsqueeze(0), torch.tensor([time_s]), 1.0)
        assert DualCmpClipModel.emission_mask(state.valid_mask, torch.tensor([[0.8]]), 0.2).tolist() == [[True]]
    state = tracker(state, proposals, _pose().unsqueeze(0), torch.tensor([0.2]), 1.0)
    assert not DualCmpClipModel.emission_mask(state.valid_mask, torch.tensor([[0.8]]), 0.2).any()


def test_measurement_score_updates_only_on_match_while_quality_updates_on_miss():
    state = _state()
    state = replace(state, score=torch.tensor([[0.3]]))
    proposal = {"boxes": torch.tensor([[[1.1, 0, 0, 1, 1, 2, 0]]]),
                "scores": torch.tensor([[0.9]]), "mask": torch.tensor([[True]])}
    tracker = ActorTracker(max_miss=2)
    state = tracker(state, proposal, _pose().unsqueeze(0), torch.tensor([0.1]), 1.0)
    assert state.score.tolist() == [[pytest.approx(0.9)]]
    no_proposal = {"boxes": torch.zeros(1, 0, 7), "scores": torch.zeros(1, 0),
                   "mask": torch.zeros(1, 0, dtype=torch.bool)}
    state = tracker(state, no_proposal, _pose().unsqueeze(0), torch.tensor([0.2]), 1.0)
    updated = DualCmpClipModel.apply_learned_update(
        state, torch.ones_like(state.feature), state.box + 0.1, state.velocity,
        torch.tensor([[0.8]]), 0.2, state.receiver_time_s,
        state.receiver_pose)
    assert updated.score.tolist() == [[pytest.approx(0.9)]]
    assert updated.quality.tolist() == [[pytest.approx(0.8)]]
    assert updated.active_mask.tolist() == [[True]]


def test_proposal_candidate_mask_is_identical_for_train_and_eval():
    anchors = torch.tensor([[[[0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0]],
                             [[10.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0]]]])
    psm = torch.tensor([[[[3.0, -3.0]]]])
    rm = torch.zeros(1, 7, 1, 2)
    train = decode_proposals(psm, rm, anchors, 2, training=True, score_threshold=0.2, nms_thresh=0.15)
    evaluation = decode_proposals(psm, rm, anchors, 2, training=False, score_threshold=0.2, nms_thresh=0.15)
    assert torch.equal(train["mask"], evaluation["mask"])
    assert train["mask"].tolist() == [[True, False]]


def test_tensor_kalman_matches_local_dmstrack_predict_update_and_orientation():
    if "DMSTRACK_ROOT" not in os.environ:
        pytest.skip("DMSTRACK_ROOT is required for source equivalence")
    root = Path(os.environ["DMSTRACK_ROOT"])
    original_path = list(sys.path)
    try:
        sys.path[:0] = [str(root), str(root / "DMSTrack"), str(root / "AB3DMOT/Xinshuo_PyToolbox")]
        source = importlib.import_module("differentiable_kalman_filter")
    finally:
        sys.path[:] = original_path
    mean = torch.tensor([1.0, 2.0, 0.5, 0.2, 4.0, 1.8, 1.5, 0.3, -0.2, 0.1])
    measurement = torch.tensor([1.6, 1.7, 0.7, -0.1, 4.1, 1.7, 1.6])
    f, h, p, q, r = matrices(torch.float32, torch.device("cpu"))
    reference = source.DifferentiableKalmanFilter(f, h, mean[:, None], p.clone(), q, r,
                                                    False, None, False, None, None, True)
    reference.predict(None); reference.update(measurement, None, None)
    actual_mean, actual_cov = predict(mean, p.clone())
    actual_mean, actual_cov = update(actual_mean, actual_cov, measurement)
    torch.testing.assert_close(actual_mean, reference.x.reshape(-1), atol=1e-6, rtol=0)
    torch.testing.assert_close(actual_cov, reference.P, atol=1e-6, rtol=0)
    # Same orientation source logic, exercised at both correction branches.
    pre, obs = orientation_correction(torch.tensor(2.8), torch.tensor(-2.8))
    assert torch.isfinite(pre) and torch.isfinite(obs)


def test_filter_pose_transform_roundtrip_preserves_mean_and_covariance():
    mean = torch.tensor([[[1.0, -2.0, 0.5, 0.3, 4.0, 1.8, 1.5, 0.2, -0.1, 0.05]]])
    covariance = torch.eye(10).view(1, 1, 10, 10) * 0.7
    previous, current = _pose().unsqueeze(0), _pose(3.0, -1.0, 0.4).unsqueeze(0)
    forward, forward_cov = transform_filter(mean, covariance, previous_to_current(previous, current))
    recovered, recovered_cov = transform_filter(forward, forward_cov, previous_to_current(current, previous))
    torch.testing.assert_close(recovered, mean, atol=1e-5, rtol=0)
    torch.testing.assert_close(recovered_cov, covariance, atol=1e-5, rtol=0)
    torch.testing.assert_close(forward_cov, forward_cov.transpose(-1, -2), atol=1e-6, rtol=0)
    assert torch.isfinite(forward_cov).all()


def test_same_slot_temporal_attention_uses_prior_and_respects_detach_boundary():
    torch.manual_seed(7)
    attention = nn.MultiheadAttention(8, 2, dropout=0.0, batch_first=True)
    attention.eval()
    current = torch.ones(1, 2, 8, requires_grad=True)
    prior = torch.zeros(1, 2, 8, requires_grad=True)
    prior.data[0, 0] = 0.5
    prior.data[0, 1] = 2.0
    valid = torch.tensor([[True, True]])
    fused = DualCmpClipModel.temporal_fuse(attention, current, prior, valid)
    assert not torch.allclose(fused[0, 0], fused[0, 1])
    fused[0, 0].sum().backward()
    assert prior.grad[0, 0].norm() > 0 and prior.grad[0, 1].norm() == 0
    invalid = DualCmpClipModel.temporal_fuse(attention, current.detach(), prior.detach(), torch.tensor([[True, False]]))
    torch.testing.assert_close(invalid[0, 1], current.detach()[0, 1])
    prior.grad.zero_()
    cached = replace(_state(max_actors=2), feature=fused)
    detached = cached.detached()
    next_current = torch.ones(1, 2, 8, requires_grad=True)
    DualCmpClipModel.temporal_fuse(attention, next_current, detached.feature, valid).sum().backward()
    assert torch.count_nonzero(prior.grad) == 0


def test_learned_box_updates_filter_without_overwriting_velocity_measurement_score():
    state = replace(_state(), score=torch.tensor([[0.7]]))
    boxes = (state.box + torch.tensor([[[0.2, 0, 0, 0, 0, 0, 0]]])).detach().requires_grad_()
    result = DualCmpClipModel.apply_learned_update(
        state, torch.ones_like(state.feature), boxes, state.velocity + 3.0,
        torch.tensor([[0.8]]), 0.2, state.receiver_time_s,
        state.receiver_pose)
    assert not torch.allclose(result.filter_mean, state.filter_mean)
    assert result.score.tolist() == [[pytest.approx(0.7)]]
    torch.testing.assert_close(result.velocity, state.velocity + 3.0)
    result.filter_mean.sum().backward()
    assert boxes.grad is not None and boxes.grad.norm() > 0
