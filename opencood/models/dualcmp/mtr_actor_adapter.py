"""Deterministic adapter from persistent tracks to CMP/MTR actor polylines."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackedMTRActors:
    entries: tuple[tuple[int, int], ...]
    context_slots: torch.Tensor
    center_headings: torch.Tensor
    decoder_headings: torch.Tensor
    batch_dict: dict


def _last_valid_positions(local_xyz: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    time = torch.arange(mask.shape[-1], device=mask.device)
    last = torch.where(
        mask,
        time.view(*([1] * (mask.ndim - 1)), -1),
        torch.full_like(mask, -1, dtype=torch.long),
    ).max(dim=-1).values.clamp_min(0)
    return local_xyz.gather(
        -2, last[..., None, None].expand(*last.shape, 1, 3)
    ).squeeze(-2)


def causal_motion_features(local_xy, mask, timestamps, frame_period_s):
    """MTR velocity/acceleration channels from observed, aligned track boxes.

    V2V4Real has no independent per-observation velocity field here. Use only
    two/three consecutive real past observations and their actual intervals;
    unknown first/gap derivatives stay zero, never copied from future states.
    """
    velocity = torch.zeros_like(local_xy)
    acceleration = torch.zeros_like(local_xy)
    dt = timestamps[:, 1:] - timestamps[:, :-1]
    consecutive = (mask[:, 1:] & mask[:, :-1] & (dt > 0)
                   & (dt <= float(frame_period_s) + 1e-4))
    safe_dt = torch.where(consecutive, dt, torch.ones_like(dt))
    velocity[:, 1:] = torch.where(consecutive[..., None],
        (local_xy[:, 1:] - local_xy[:, :-1]) / safe_dt[..., None],
        torch.zeros_like(local_xy[:, 1:]))
    acceleration_valid = consecutive[:, 1:] & consecutive[:, :-1]
    acceleration[:, 2:] = torch.where(acceleration_valid[..., None],
        (velocity[:, 2:] - velocity[:, 1:-1]) / safe_dt[:, 1:, None],
        torch.zeros_like(local_xy[:, 2:]))
    return velocity, acceleration


def pack_mtr_actor_polylines(
    boxes: torch.Tensor,
    valid_mask: torch.Tensor,
    history_box: torch.Tensor,
    history_mask: torch.Tensor,
    history_time_s: torch.Tensor,
    receiver_time_s: torch.Tensor,
    *,
    min_forecast_history_steps: int,
    min_context_history_steps: int,
    frame_period_s: float,
    timestamp_scale: float = 0.1,
    legacy_heading: bool = True,
) -> PackedMTRActors | None:
    """Build legacy 22-D or aligned, motion-complete 26-D MTR actors.

    With ``legacy_heading=True``, ``history_box`` stores every observation
    in its capture receiver frame, matching the released CMP cache. With
    ``legacy_heading=False``, the caller supplies history re-expressed in
    the prediction-time receiver frame. CoLIF boxes use
    ``[x,y,z,h,w,l,yaw_rad]``; CMP/MTR consumes
    ``[x,y,z,l,w,h,heading_numeric]`` and its released V2V4Real path applies
    trigonometric functions directly to degree-valued headings.  The legacy
    conversion is deliberately confined to this adapter.
    """

    if boxes.ndim != 3 or boxes.shape[-1] != 7:
        raise ValueError("boxes must have shape [B,Q,7]")
    if history_box.shape != (*boxes.shape[:2], history_mask.shape[-1], 7):
        raise ValueError("history_box/history_mask shapes disagree")
    if history_mask.shape[:2] != boxes.shape[:2]:
        raise ValueError("history_mask must have shape [B,Q,T]")
    if history_time_s.shape != history_mask.shape:
        raise ValueError("history_time_s must match history_mask [B,Q,T]")
    if valid_mask.shape != boxes.shape[:2]:
        raise ValueError("valid_mask must have shape [B,Q]")
    if receiver_time_s.shape != (boxes.shape[0],):
        raise ValueError("receiver_time_s must have shape [B]")
    if history_mask.shape[-1] != 11:
        raise ValueError("released V2V4Real MTR requires eleven history frames")
    if frame_period_s <= 0.0 or timestamp_scale <= 0.0:
        raise ValueError("frame period and timestamp scale must be positive")

    steps = history_mask.sum(dim=-1)
    context_valid = valid_mask & (steps >= min_context_history_steps)
    motion_valid = valid_mask & (steps >= min_forecast_history_steps)

    entries: list[tuple[int, int]] = []
    contexts: list[torch.Tensor] = []
    for batch_index in range(boxes.shape[0]):
        slots = torch.where(context_valid[batch_index])[0]
        centers = torch.where(motion_valid[batch_index])[0]
        if slots.numel() == 0:
            continue
        for center_slot in centers.tolist():
            entries.append((batch_index, int(center_slot)))
            contexts.append(slots)
    if not entries:
        return None

    center_count = len(entries)
    max_context = max(int(slots.numel()) for slots in contexts)
    time_count = history_mask.shape[-1]
    feature_count = 22 if legacy_heading else 26
    obj_trajs = history_box.new_zeros(center_count, max_context, time_count, feature_count)
    obj_mask = torch.zeros(
        center_count, max_context, time_count,
        dtype=torch.bool, device=history_mask.device,
    )
    obj_last_pos = history_box.new_zeros(center_count, max_context, 3)
    track_index = torch.zeros(center_count, dtype=torch.long, device=boxes.device)
    context_slots = torch.full(
        (center_count, max_context), -1, dtype=torch.long, device=boxes.device
    )
    center_headings = boxes.new_zeros(center_count)
    decoder_headings = boxes.new_zeros(center_count)

    time_one_hot = torch.eye(time_count, device=boxes.device, dtype=boxes.dtype)
    for row, ((batch_index, center_slot), slots) in enumerate(
        zip(entries, contexts)
    ):
        size = int(slots.numel())
        center_box = boxes[batch_index, center_slot]
        center_heading = center_box[6]
        decoder_heading = torch.rad2deg(center_heading) if legacy_heading else center_heading
        histories = history_box[batch_index, slots]
        mask = history_mask[batch_index, slots].bool()

        delta_xyz = histories[..., :3] - center_box[:3]
        cosine = torch.cos(decoder_heading)
        sine = torch.sin(decoder_heading)
        local_x = cosine * delta_xyz[..., 0] + sine * delta_xyz[..., 1]
        local_y = -sine * delta_xyz[..., 0] + cosine * delta_xyz[..., 1]
        local_xyz = torch.stack((local_x, local_y, delta_xyz[..., 2]), dim=-1)
        relative_heading = (torch.rad2deg(histories[..., 6]) if legacy_heading
                            else histories[..., 6]) - decoder_heading

        features = histories.new_zeros(size, time_count, feature_count)
        features[..., :3] = local_xyz
        features[..., 3] = histories[..., 5]  # length
        features[..., 4] = histories[..., 4]  # width
        features[..., 5] = histories[..., 3]  # height
        features[..., 6] = 1.0  # TYPE_VEHICLE
        center_object_index = int(torch.where(slots == center_slot)[0][0])
        features[center_object_index, :, 7] = 1.0
        features[..., 8:19] = time_one_hot[None]
        timestamps = history_time_s[batch_index, slots] * float(timestamp_scale)
        if not legacy_heading:
            timestamps = (history_time_s[batch_index, slots] - receiver_time_s[batch_index])
        features[..., 19] = timestamps
        features[..., 20] = torch.sin(relative_heading)
        features[..., 21] = torch.cos(relative_heading)
        if not legacy_heading:
            # Original MTR WaymoDataset.generate_centered_trajs_for_agents
            # appends actor-local observed velocity and acceleration. Retain
            # V2V4Real's vehicle/center markers, adapting only these 4 fields.
            velocity, acceleration = causal_motion_features(
                local_xyz[..., :2], mask, history_time_s[batch_index, slots], frame_period_s)
            features[..., 22:24] = velocity
            features[..., 24:26] = acceleration
        features = torch.where(mask[..., None], features, torch.zeros_like(features))

        obj_trajs[row, :size] = features
        obj_mask[row, :size] = mask
        obj_last_pos[row, :size] = _last_valid_positions(local_xyz, mask)
        obj_last_pos[row, :size] = torch.where(
            mask.any(dim=-1, keepdim=True),
            obj_last_pos[row, :size],
            torch.zeros_like(obj_last_pos[row, :size]),
        )
        track_index[row] = center_object_index
        context_slots[row, :size] = slots
        center_headings[row] = center_heading
        decoder_headings[row] = decoder_heading

    input_dict = {
        "obj_trajs": obj_trajs,
        "obj_trajs_mask": obj_mask,
        "obj_trajs_last_pos": obj_last_pos,
        "track_index_to_predict": track_index,
        "center_objects_type": ["TYPE_VEHICLE"] * center_count,
    }
    return PackedMTRActors(
        entries=tuple(entries),
        context_slots=context_slots,
        center_headings=center_headings,
        decoder_headings=decoder_headings,
        batch_dict={
            "input_dict": input_dict,
            "batch_sample_count": [center_count],
        },
    )
