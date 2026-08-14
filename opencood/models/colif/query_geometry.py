"""Coordinate transforms and metric 3D box coding for CoLIF queries."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def inverse_sigmoid(value: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    value = value.clamp(min=eps, max=1.0 - eps)
    return torch.log(value / (1.0 - value))


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def transform_boxes_between_receiver_frames(
    boxes: torch.Tensor,
    previous_pose: torch.Tensor,
    current_pose: torch.Tensor,
) -> torch.Tensor:
    """Express boxes stored in the previous receiver frame in the current one.

    ``boxes`` may have any number of dimensions between batch and the final
    seven box attributes.  This lets the same rigid transform preserve the
    coordinate invariant for both current query references ``[B,Q,7]`` and
    their temporal history ``[B,Q,T,7]``.
    """

    if boxes.ndim < 3 or boxes.shape[-1] != 7:
        raise ValueError("boxes must have shape [B,...,7]")
    if previous_pose.shape != current_pose.shape or previous_pose.shape[-2:] != (4, 4):
        raise ValueError("receiver poses must have matching [B,4,4] shapes")
    if boxes.shape[0] != previous_pose.shape[0]:
        raise ValueError("box batch dimension must match receiver poses")

    previous_to_current = torch.linalg.inv(current_pose) @ previous_pose
    rotation = previous_to_current[:, :3, :3]
    translation = previous_to_current[:, :3, 3]
    transformed = boxes.clone()
    transformed[..., :3] = (
        torch.einsum("bij,b...j->b...i", rotation, boxes[..., :3])
        + translation.view(translation.shape[0], *([1] * (boxes.ndim - 2)), 3)
    )
    yaw_delta = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
    yaw_view = yaw_delta.view(yaw_delta.shape[0], *([1] * (boxes.ndim - 2)))
    transformed[..., 6] = wrap_angle(boxes[..., 6] + yaw_view)
    return transformed


def transform_and_propagate_boxes(
    boxes: torch.Tensor,
    velocity: torch.Tensor,
    previous_pose: torch.Tensor,
    current_pose: torch.Tensor,
    previous_time_s: torch.Tensor,
    current_time_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move prior states into the current receiver frame, then advance them.

    V2V4Real poses are LiDAR-to-world matrices.  Therefore
    ``inv(P_t) @ P_(t-1)`` maps old receiver coordinates into the current
    receiver frame.  Only the current-state velocity advances the spatial
    reference; forecast cues are intentionally absent from this function.
    """

    previous_to_current = torch.linalg.inv(current_pose) @ previous_pose
    rotation = previous_to_current[:, :3, :3]
    propagated = transform_boxes_between_receiver_frames(
        boxes, previous_pose, current_pose
    )
    velocity_current = torch.einsum(
        "bij,bqj->bqi", rotation[:, :2, :2], velocity
    )
    dt = (current_time_s - previous_time_s).clamp(min=0.0, max=1.0)
    propagated[..., :2] = (
        propagated[..., :2] + velocity_current * dt[:, None, None]
    )
    return propagated, velocity_current


class MetricBoxCoder(nn.Module):
    """Decode decoder residuals into V2V4Real ``[x,y,z,h,w,l,yaw]`` boxes."""

    def __init__(self, lidar_range, default_hwl=(1.56, 1.6, 3.9)):
        super().__init__()
        self.register_buffer(
            "lidar_range", torch.as_tensor(lidar_range, dtype=torch.float32)
        )
        self.register_buffer(
            "default_hwl", torch.as_tensor(default_hwl, dtype=torch.float32)
        )

    def metric_xy_to_normalized(self, xy: torch.Tensor) -> torch.Tensor:
        minimum = self.lidar_range[:2]
        extent = self.lidar_range[3:5] - minimum
        return ((xy - minimum) / extent).clamp(1e-5, 1.0 - 1e-5)

    def normalized_xy_to_metric(self, xy: torch.Tensor) -> torch.Tensor:
        minimum = self.lidar_range[:2]
        extent = self.lidar_range[3:5] - minimum
        return xy * extent + minimum

    def decode(
        self, raw_box: torch.Tensor, reference_xy: torch.Tensor
    ) -> torch.Tensor:
        if raw_box.shape[-1] != 8:
            raise ValueError("raw_box must end in 8 values")
        normalized_xy = (
            inverse_sigmoid(reference_xy) + raw_box[..., :2]
        ).sigmoid()
        xy = self.normalized_xy_to_metric(normalized_xy)
        z_min, z_max = self.lidar_range[2], self.lidar_range[5]
        z = raw_box[..., 2].sigmoid() * (z_max - z_min) + z_min
        dimensions = self.default_hwl * torch.exp(
            raw_box[..., 3:6].clamp(min=-2.0, max=2.0)
        )
        yaw = torch.atan2(raw_box[..., 6], raw_box[..., 7])
        return torch.cat(
            [xy, z.unsqueeze(-1), dimensions, yaw.unsqueeze(-1)], dim=-1
        )

    def decode_from_reference_box(
        self, raw_box: torch.Tensor, reference_box: torch.Tensor
    ) -> torch.Tensor:
        """Decode a residual around a complete persistent/measurement box."""

        if raw_box.shape[-1] != 8 or reference_box.shape[-1] != 7:
            raise ValueError("raw/reference boxes must end in 8/7 values")
        reference_xy = self.metric_xy_to_normalized(reference_box[..., :2])
        normalized_xy = (
            inverse_sigmoid(reference_xy) + raw_box[..., :2]
        ).sigmoid()
        xy = self.normalized_xy_to_metric(normalized_xy)
        z = reference_box[..., 2] + raw_box[..., 2] * reference_box[..., 3]
        dimensions = reference_box[..., 3:6].clamp_min(1e-3) * torch.exp(
            raw_box[..., 3:6].clamp(min=-2.0, max=2.0)
        )
        yaw_residual = torch.atan2(raw_box[..., 6], raw_box[..., 7])
        yaw = wrap_angle(reference_box[..., 6] + yaw_residual)
        return torch.cat(
            [xy, z.unsqueeze(-1), dimensions, yaw.unsqueeze(-1)], dim=-1
        )


def normalized_sine_embedding(xy: torch.Tensor, dim: int) -> torch.Tensor:
    """DETR-style sine/cosine embedding for normalized 2D references."""

    if dim % 4:
        raise ValueError("query dimension must be divisible by four")
    features = dim // 4
    scale = 2.0 * math.pi
    index = torch.arange(features, device=xy.device, dtype=xy.dtype)
    temperature = 10000 ** (2 * torch.div(index, 2, rounding_mode="floor") / features)
    encoded = []
    for coordinate in (xy[..., 0], xy[..., 1]):
        value = coordinate[..., None] * scale / temperature
        encoded.append(torch.stack((value.sin(), value.cos()), dim=-1).flatten(-2))
    result = torch.cat(encoded, dim=-1)
    return result[..., :dim]
