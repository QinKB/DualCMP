"""Actor-aligned receiver-frame source BEV tokens (from CoLIF 6b8788e)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorAlignedSourceTokenizer(nn.Module):
    """Sample every source separately; [B,A,L,P,C] is never aggregated here."""
    def __init__(self, input_dim, query_dim, max_cav, lidar_range,
                 roi_grid_size=5, roi_scale=2.0, minimum_half_extent_m=1.0):
        super().__init__()
        self.input_dim, self.query_dim, self.max_cav = int(input_dim), int(query_dim), int(max_cav)
        self.roi_grid_size, self.roi_scale = int(roi_grid_size), float(roi_scale)
        self.minimum_half_extent_m = float(minimum_half_extent_m)
        if self.roi_grid_size < 1 or self.roi_grid_size % 2 == 0:
            raise ValueError("roi_grid_size must be a positive odd integer")
        if self.roi_scale <= 0.0 or self.minimum_half_extent_m <= 0.0:
            raise ValueError("roi_scale and minimum_half_extent_m must be positive")
        if self.input_dim != self.query_dim:
            raise ValueError("input_dim and query_dim must match the MTR actor width")
        self.x_min, self.y_min = float(lidar_range[0]), float(lidar_range[1])
        self.x_max, self.y_max = float(lidar_range[3]), float(lidar_range[4])
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("invalid lidar_range")
        axis = torch.linspace(-1.0, 1.0, self.roi_grid_size)
        offset_y, offset_x = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("unit_roi_offsets", torch.stack((offset_x, offset_y), dim=-1).reshape(-1, 2))

    @property
    def tokens_per_source(self):
        return self.roi_grid_size * self.roi_grid_size

    def _sampling_grid(self, actor_boxes):
        if actor_boxes.ndim != 3 or actor_boxes.shape[-1] != 7:
            raise ValueError("actor_boxes must have shape [B,A,7]")
        half_length = (0.5 * self.roi_scale * actor_boxes[..., 5].clamp_min(1e-3)).clamp_min(self.minimum_half_extent_m)
        half_width = (0.5 * self.roi_scale * actor_boxes[..., 4].clamp_min(1e-3)).clamp_min(self.minimum_half_extent_m)
        unit = self.unit_roi_offsets.to(actor_boxes)
        local_x, local_y = half_length[..., None] * unit[:, 0], half_width[..., None] * unit[:, 1]
        yaw = actor_boxes[..., 6, None]
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        receiver_x = actor_boxes[..., 0, None] + cosine * local_x - sine * local_y
        receiver_y = actor_boxes[..., 1, None] + sine * local_x + cosine * local_y
        grid = torch.stack((2.0 * (receiver_x - self.x_min) / (self.x_max - self.x_min) - 1.0,
                            2.0 * (receiver_y - self.y_min) / (self.y_max - self.y_min) - 1.0), dim=-1)
        spatial_valid = (grid.abs() <= 1.0).all(dim=-1)
        offset_x, offset_y = receiver_x - actor_boxes[..., 0, None], receiver_y - actor_boxes[..., 1, None]
        position = torch.stack((torch.cos(yaw) * offset_x + torch.sin(yaw) * offset_y,
                                -torch.sin(yaw) * offset_x + torch.cos(yaw) * offset_y,
                                torch.zeros_like(offset_x)), dim=-1)
        return grid, spatial_valid, position

    def forward(self, source_bev, source_mask, actor_boxes, actor_valid_mask):
        if source_bev.ndim != 5:
            raise ValueError("source_bev must have shape [B,L,C,H,W]")
        batch, sources, channels, height, width = source_bev.shape
        if sources > self.max_cav or channels != self.input_dim:
            raise ValueError("source_bev disagrees with tokenizer configuration")
        if source_mask.shape != (batch, sources):
            raise ValueError("source_mask must have shape [B,L]")
        if actor_boxes.shape[:2] != actor_valid_mask.shape:
            raise ValueError("actor_valid_mask must have shape [B,A]")
        actors, patches = actor_boxes.shape[1], self.tokens_per_source
        grid, spatial_valid, position = self._sampling_grid(actor_boxes)
        expanded_grid = grid[:, None].expand(batch, sources, actors, patches, 2)
        sampled = F.grid_sample(source_bev.reshape(batch * sources, channels, height, width),
                                expanded_grid.reshape(batch * sources, actors, patches, 2),
                                mode="bilinear", padding_mode="zeros", align_corners=False)
        tokens = sampled.view(batch, sources, channels, actors, patches).permute(0, 3, 1, 4, 2)
        positions = position.to(tokens)[:, :, None].expand(batch, actors, sources, patches, 3)
        token_mask = actor_valid_mask.bool()[..., None, None] & source_mask.bool()[:, None, :, None] & spatial_valid[:, :, None, :]
        return {"tokens": torch.where(token_mask[..., None], tokens, torch.zeros_like(tokens)),
                "positions": torch.where(token_mask[..., None], positions, torch.zeros_like(positions)),
                "token_mask": token_mask, "sampling_grid": grid}
