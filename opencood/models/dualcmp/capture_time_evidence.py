"""Align delayed source evidence with causal actor history at capture time."""

from __future__ import annotations

import math

import torch

from .geometry import previous_to_current, transform_boxes


def capture_aligned_actor_boxes(state, capture_time_s, source_mask,
                                time_tolerance_s=1e-4):
    """Return source-specific actor boxes in the current receiver frame.

    Source BEV cells represent geometry at ``capture_time_s`` even though the
    packet is consumed at ``state.receiver_time_s``.  Each source therefore
    samples at the matching runtime actor observation, transformed from that
    observation's receiver pose into the current receiver frame.  Missing
    history fails closed instead of falling back to the actor's current box.
    """
    state.validate()
    if capture_time_s.ndim != 2 or source_mask.shape != capture_time_s.shape:
        raise ValueError("capture_time_s/source_mask must have shape [B,L]")
    if capture_time_s.shape[0] != state.box.shape[0]:
        raise ValueError("capture times must match ActorState batch size")
    if source_mask.dtype != torch.bool or source_mask.device != state.box.device:
        raise ValueError("source_mask must be bool on the ActorState device")
    if (not capture_time_s.is_floating_point()
            or capture_time_s.dtype != state.box.dtype
            or capture_time_s.device != state.box.device):
        raise ValueError("capture_time_s must match ActorState floating dtype/device")
    tolerance = float(time_tolerance_s)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("time_tolerance_s must be finite and non-negative")
    if not torch.isfinite(capture_time_s[source_mask]).all():
        raise ValueError("available source capture times must be finite")

    batch, actors, history = state.forecast_history_mask.shape
    sources = capture_time_s.shape[1]
    safe_capture = torch.where(
        source_mask, capture_time_s, torch.zeros_like(capture_time_s))
    error = torch.abs(
        state.forecast_history_time_s[:, :, :, None]
        - safe_capture[:, None, None, :]
    )
    candidate = (
        state.forecast_history_mask[:, :, :, None]
        & source_mask[:, None, None, :]
    )
    error = torch.where(candidate, error, torch.full_like(error, float("inf")))
    nearest_error, nearest_index = error.min(dim=2)
    age_s = state.receiver_time_s[:, None] - safe_capture
    if torch.any(source_mask & (age_s < -tolerance)):
        raise ValueError("source capture time cannot be later than receiver time")
    causal_source = source_mask & (age_s >= -tolerance)
    valid = (
        state.active_mask[:, :, None]
        & causal_source[:, None, :]
        & torch.isfinite(nearest_error)
        & (nearest_error <= tolerance)
    )

    box_values = state.forecast_history_box[:, :, :, None, :].expand(
        batch, actors, history, sources, 7)
    box_index = nearest_index[:, :, None, :, None].expand(
        batch, actors, 1, sources, 7)
    selected_box = box_values.gather(2, box_index).squeeze(2)

    pose_values = state.forecast_history_pose[:, :, :, None].expand(
        batch, actors, history, sources, 4, 4)
    pose_index = nearest_index[:, :, None, :, None, None].expand(
        batch, actors, 1, sources, 4, 4)
    selected_pose = pose_values.gather(2, pose_index).squeeze(2)
    current_pose = state.receiver_pose[:, None, None].expand(
        batch, actors, sources, 4, 4)
    safe_pose = torch.where(
        valid[:, :, :, None, None], selected_pose, current_pose)
    safe_box = torch.where(valid[:, :, :, None], selected_box,
                           torch.zeros_like(selected_box))
    transform = previous_to_current(
        safe_pose.reshape(-1, 4, 4), current_pose.reshape(-1, 4, 4))
    aligned = transform_boxes(
        safe_box.reshape(-1, 1, 7), transform).reshape(batch, actors, sources, 7)
    # Preserve the synchronous source path exactly.  The selected history box
    # is already in the current receiver frame, so inverse(P) @ P would only
    # introduce avoidable floating-point drift.
    current_source = valid & (age_s[:, None, :].abs() <= tolerance)
    aligned = torch.where(
        current_source[:, :, :, None], selected_box, aligned)
    aligned = torch.where(valid[:, :, :, None], aligned,
                          torch.zeros_like(aligned))
    return {
        "boxes": aligned,
        "mask": valid,
        "age_s": torch.where(causal_source, age_s, torch.zeros_like(age_s)),
        "history_index": nearest_index,
        "time_error_s": torch.where(
            valid, nearest_error, torch.zeros_like(nearest_error)),
    }
