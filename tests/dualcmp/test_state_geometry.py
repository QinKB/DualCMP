"""CPU contracts for DualCMP ActorState and receiver-frame geometry."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from opencood.models.dualcmp.geometry import previous_to_current, propagate_state, wrap_angle
from opencood.models.dualcmp.state import ActorState


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
    return replace(
        state, box=box, velocity=velocity, valid_mask=torch.ones_like(state.valid_mask),
        active_mask=torch.ones_like(state.active_mask), runtime_id=torch.arange(max_actors).view(1, -1).repeat(batch_size, 1),
        next_runtime_id=torch.full((batch_size,), max_actors, dtype=torch.long),
        last_update_time_s=torch.zeros_like(state.last_update_time_s), history_box=history_box,
        history_mask=torch.ones_like(state.history_mask),
    )


def test_known_receiver_transform_yaw_velocity_and_history():
    state = _state()
    current_pose = _pose(10.0, 0.0, torch.pi / 2).unsqueeze(0)
    propagated, valid_rows, dt = propagate_state(state, current_pose, torch.tensor([0.5]), 1.0)
    assert valid_rows.tolist() == [True]
    torch.testing.assert_close(dt, torch.tensor([0.5]))
    torch.testing.assert_close(propagated.box[0, 0, :2], torch.tensor([0.0, 8.5]), atol=1e-6, rtol=0)
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
    torch.testing.assert_close(propagated.box[0, 0, :2], torch.tensor([2.0, 0.0]))
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
    propagated.box.sum().backward()
    assert state.box.grad is not None and state.velocity.grad is not None

    state.box.grad.zero_()
    state.velocity.grad.zero_()
    detached = state.detached()
    detached.box.requires_grad_()
    detached.velocity.requires_grad_()
    propagated, _, _ = propagate_state(detached, _pose().unsqueeze(0), torch.tensor([1.0]), 1.0)
    propagated.box.sum().backward()
    assert detached.box.grad is not None and detached.velocity.grad is not None
    assert torch.count_nonzero(state.box.grad) == 0 and torch.count_nonzero(state.velocity.grad) == 0


def test_angle_wrap_is_principal_interval():
    torch.testing.assert_close(wrap_angle(torch.tensor([3 * torch.pi])), torch.tensor([-torch.pi]), atol=1e-6, rtol=0)
