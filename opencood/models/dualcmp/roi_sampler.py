"""Differentiable actor-local fused-BEV sampling."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class ActorRoiSampler(torch.nn.Module):
    def __init__(self, lidar_range):
        super().__init__()
        self.register_buffer("lidar_range", torch.tensor(lidar_range, dtype=torch.float32))

    def forward(self, fused_bev, boxes):
        if fused_bev.ndim != 4 or boxes.ndim != 3 or boxes.shape[:2] != fused_bev.shape[:1] + boxes.shape[1:2] or boxes.shape[-1] != 7:
            raise ValueError("fused_bev [B,C,H,W] and boxes [B,A,7] are required")
        minimum, maximum = self.lidar_range[:2].to(boxes), self.lidar_range[3:5].to(boxes)
        xy = (boxes[..., :2] - minimum) / (maximum - minimum) * 2.0 - 1.0
        grid = xy.view(boxes.shape[0], boxes.shape[1], 1, 2)
        sampled = F.grid_sample(fused_bev, grid, mode="bilinear", align_corners=True)
        return sampled.squeeze(-1).transpose(1, 2)
