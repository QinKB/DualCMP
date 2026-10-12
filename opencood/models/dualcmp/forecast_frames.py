"""Explicit forecasting coordinate contracts; legacy artifacts stay unchanged."""
import torch

from .geometry import previous_to_current, transform_boxes


ALIGNED_SOURCE = "source_temporal_aligned"
TEMPORAL_SOURCES = ("source_temporal", ALIGNED_SOURCE)


def align_actor_history(history_box, history_mask, history_pose, receiver_pose):
    """Re-express capture-native boxes in the prediction-time receiver frame."""
    if history_pose.shape != (*history_mask.shape, 4, 4):
        raise ValueError("history poses must match actor history")
    current = receiver_pose[:, None, None].expand_as(history_pose)
    safe_pose = torch.where(history_mask[..., None, None], history_pose, current)
    transform = previous_to_current(safe_pose.reshape(-1, 4, 4), current.reshape(-1, 4, 4))
    aligned = transform_boxes(history_box.reshape(-1, 1, 7), transform).reshape_as(history_box)
    return torch.where(history_mask[..., None], aligned, torch.zeros_like(aligned))


def align_source_record(record, final_state):
    """Convert old actor-local ROI positions and source poses to final time.

    Features remain attached to their capture-time actor patch. Only coordinate
    metadata is transformed; no ego motion or future GT is invented.
    """
    positions = record["positions"]
    reference = record["reference_boxes"]
    heading = reference[..., 6, None, None]
    x, y = positions[..., 0], positions[..., 1]
    receiver_points = torch.stack((
        torch.cos(heading) * x - torch.sin(heading) * y,
        torch.sin(heading) * x + torch.cos(heading) * y,
        positions[..., 2],
    ), dim=-1) + reference[..., None, None, :3]
    transform = previous_to_current(record["receiver_pose"], final_state.receiver_pose)
    current_points = torch.einsum("bij,balkj->balki", transform[:, :3, :3], receiver_points)
    current_points = current_points + transform[:, None, None, None, :3, 3]
    delta = current_points - final_state.box[..., None, None, :3]
    final_heading = final_state.box[..., 6, None, None]
    current_local = torch.stack((
        torch.cos(final_heading) * delta[..., 0] + torch.sin(final_heading) * delta[..., 1],
        -torch.sin(final_heading) * delta[..., 0] + torch.cos(final_heading) * delta[..., 1],
        delta[..., 2],
    ), dim=-1)
    source_pose = transform[:, None].matmul(record["source_to_receiver"])
    return current_local, source_pose


def detached_source_history(records):
    """Bounded online state owned by the caller, never by an nn.Module."""
    return tuple({key: value.detach() if isinstance(value, torch.Tensor) else value
                  for key, value in record.items()} for record in records)
