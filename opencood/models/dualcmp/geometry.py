"""Receiver-frame geometry for causal ActorState propagation."""

from __future__ import annotations

import torch
from dataclasses import fields

from .state import ActorState


POSE_ORTHOGONALITY_TOL = 1e-4
POSE_DETERMINANT_TOL = 1e-4
POSE_HOMOGENEOUS_TOL = 1e-5
MAX_PLANAR_TILT_RAD = 0.1


def wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def previous_to_current(previous_pose, current_pose):
    """Return ``inv(P_current) @ P_previous`` for near-planar LiDAR poses.

    Full xyz coordinates are transformed rigidly.  DualCMP's yaw boxes and
    xy velocities use the xy projection, so input poses must be proper,
    near-planar rotations with tilt no greater than ``MAX_PLANAR_TILT_RAD``.
    """
    if previous_pose.shape != current_pose.shape or previous_pose.ndim != 3 or previous_pose.shape[1:] != (4, 4):
        raise ValueError("receiver poses must have matching [B,4,4] shapes")
    if not previous_pose.is_floating_point() or current_pose.dtype != previous_pose.dtype or current_pose.device != previous_pose.device:
        raise ValueError("receiver poses must share floating dtype/device")
    if not torch.isfinite(previous_pose).all() or not torch.isfinite(current_pose).all():
        raise ValueError("receiver poses must be finite")
    _validate_receiver_pose(previous_pose)
    _validate_receiver_pose(current_pose)
    return torch.linalg.inv(current_pose).matmul(previous_pose)


def _validate_receiver_pose(pose):
    rotation = pose[:, :3, :3]
    identity = torch.eye(3, dtype=pose.dtype, device=pose.device).unsqueeze(0)
    orthogonality_error = torch.amax(torch.abs(rotation.transpose(1, 2).matmul(rotation) - identity), dim=(1, 2))
    determinant = torch.linalg.det(rotation)
    bottom = pose[:, 3, :]
    expected_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=pose.dtype, device=pose.device).expand_as(bottom)
    bottom_error = torch.amax(torch.abs(bottom - expected_bottom), dim=1)
    tilt = torch.atan2(torch.linalg.vector_norm(rotation[:, :2, 2], dim=1), rotation[:, 2, 2])
    if torch.any(orthogonality_error > POSE_ORTHOGONALITY_TOL):
        raise ValueError("receiver poses must be rigid rotations without scale or shear")
    if torch.any(torch.abs(determinant - 1.0) > POSE_DETERMINANT_TOL):
        raise ValueError("receiver poses must be proper rotations without reflection")
    if torch.any(bottom_error > POSE_HOMOGENEOUS_TOL):
        raise ValueError("receiver poses must have homogeneous bottom row [0,0,0,1]")
    if torch.any(torch.abs(tilt) > MAX_PLANAR_TILT_RAD):
        raise ValueError("receiver pose tilt exceeds near-planar yaw-box support")


def transform_boxes(boxes, transform):
    """Rigidly transform ``[B,...,7]`` boxes, including receiver-frame yaw."""
    if boxes.ndim < 3 or boxes.shape[-1] != 7 or boxes.shape[0] != transform.shape[0]:
        raise ValueError("boxes must have shape [B,...,7] matching transform batch")
    rotation, translation = transform[:, :3, :3], transform[:, :3, 3]
    result = boxes.clone()
    result[..., :3] = torch.einsum("bij,b...j->b...i", rotation, boxes[..., :3]) + translation.view((translation.shape[0],) + (1,) * (boxes.ndim - 2) + (3,))
    yaw_delta = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
    result[..., 6] = wrap_angle(boxes[..., 6] + yaw_delta.view((yaw_delta.shape[0],) + (1,) * (boxes.ndim - 2)))
    return result


def transform_filter(mean, covariance, transform):
    """Receiver-coordinate adapter for DMSTrack's lidar-order DKF state."""
    rotation, translation = transform[:, :3, :3], transform[:, :3, 3]
    result = mean.clone()
    result[..., :3] = torch.einsum("bij,baj->bai", rotation, mean[..., :3]) + translation[:, None, :]
    result[..., 7:10] = torch.einsum("bij,baj->bai", rotation, mean[..., 7:10])
    yaw_delta = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
    result[..., 3] = wrap_angle(mean[..., 3] + yaw_delta[:, None])
    jacobian = torch.eye(10, dtype=mean.dtype, device=mean.device).unsqueeze(0).repeat(mean.shape[0], 1, 1)
    jacobian[:, :3, :3], jacobian[:, 7:10, 7:10] = rotation, rotation
    covariance = torch.matmul(torch.matmul(jacobian[:, None], covariance), jacobian[:, None].transpose(-1, -2))
    return result, covariance


def propagate_state(state, current_pose, current_time_s, max_dt_s, scene_reset_mask=None):
    """Re-express a state in the current receiver frame and apply CV motion."""
    state.validate()
    try:
        max_dt_tensor = torch.as_tensor(max_dt_s)
        if isinstance(max_dt_s, bool) or max_dt_tensor.dtype == torch.bool or max_dt_tensor.numel() != 1 or max_dt_tensor.is_complex() or not torch.isfinite(max_dt_tensor).all():
            raise ValueError
        max_dt_value = float(max_dt_tensor.item())
    except (TypeError, ValueError, RuntimeError, OverflowError):
        raise ValueError("max_dt_s must be a finite positive scalar")
    if max_dt_value <= 0:
        raise ValueError("max_dt_s must be a finite positive scalar")
    if current_pose.shape != state.receiver_pose.shape or current_time_s.shape != state.receiver_time_s.shape:
        raise ValueError("current pose/time must match state batch shapes")
    if current_pose.dtype != state.feature.dtype or current_time_s.dtype != state.feature.dtype or current_pose.device != state.feature.device or current_time_s.device != state.feature.device:
        raise ValueError("current pose/time must match state dtype/device")
    if scene_reset_mask is None:
        scene_reset_mask = torch.zeros_like(state.receiver_time_s, dtype=torch.bool)
    if scene_reset_mask.shape != state.receiver_time_s.shape or scene_reset_mask.dtype != torch.bool or scene_reset_mask.device != state.feature.device:
        raise ValueError("scene_reset_mask must be bool [B] on the state device")
    transform = previous_to_current(state.receiver_pose, current_pose)
    if not torch.isfinite(current_time_s).all():
        raise ValueError("current_time_s must be finite")
    dt = current_time_s - state.receiver_time_s
    valid_rows = torch.isfinite(dt) & (dt > 0) & (dt <= max_dt_value) & ~scene_reset_mask
    result = state.reset_rows(
        ~valid_rows, current_pose, current_time_s,
        reset_id_counter_mask=scene_reset_mask,
    )
    transformed_box = transform_boxes(state.box, transform)
    transformed_history = transform_boxes(state.history_box, transform)
    rotation_xy = transform[:, :2, :2]
    velocity = torch.einsum("bij,baj->bai", rotation_xy, state.velocity)
    filter_mean, filter_cov = transform_filter(state.filter_mean, state.filter_cov, transform)
    slot_mask = state.valid_mask & valid_rows[:, None]
    history_mask = state.history_mask & valid_rows[:, None, None]
    box = torch.where(slot_mask[:, :, None], transformed_box, result.box)
    velocity = torch.where(slot_mask[:, :, None], velocity, result.velocity)
    history_box = torch.where(history_mask[:, :, :, None], transformed_history, result.history_box)
    filter_mean = torch.where(slot_mask[:, :, None], filter_mean, result.filter_mean)
    filter_cov = torch.where(slot_mask[:, :, None, None], filter_cov, result.filter_cov)
    values = {item.name: getattr(result, item.name) for item in fields(ActorState)}
    values.update({"box": box, "velocity": velocity, "history_box": history_box,
                   "filter_mean": filter_mean, "filter_cov": filter_cov,
                   "receiver_pose": current_pose, "receiver_time_s": current_time_s})
    next_state = ActorState(**values)
    next_state.validate()
    return next_state, valid_rows, dt
