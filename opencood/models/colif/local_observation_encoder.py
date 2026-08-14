"""Per-source anchor observations and differentiable BEV evidence features."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.utils import box_utils_v2v4real as box_utils


def delta_to_boxes3d(deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Decode VoxelPostprocessor-compatible hwl anchor residuals."""

    batch_size = deltas.shape[0]
    deltas = deltas.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 7)
    anchors = anchors.reshape(-1, 7).to(device=deltas.device, dtype=deltas.dtype)
    if anchors.shape[0] != deltas.shape[1]:
        raise ValueError(
            "Anchor/head mismatch: %d anchors for %d residuals"
            % (anchors.shape[0], deltas.shape[1])
        )
    anchors = anchors.unsqueeze(0).expand(batch_size, -1, -1)
    diagonal = torch.sqrt(anchors[..., 4] ** 2 + anchors[..., 5] ** 2)

    boxes = torch.zeros_like(deltas)
    boxes[..., 0] = deltas[..., 0] * diagonal + anchors[..., 0]
    boxes[..., 1] = deltas[..., 1] * diagonal + anchors[..., 1]
    boxes[..., 2] = deltas[..., 2] * anchors[..., 3] + anchors[..., 2]
    boxes[..., 3:6] = torch.exp(deltas[..., 3:6]) * anchors[..., 3:6]
    boxes[..., 6] = deltas[..., 6] + anchors[..., 6]
    return boxes


class LocalObservationEncoder(nn.Module):
    """Decode independent source detections and sample their BEV features."""

    def __init__(self, input_dim: int, anchor_number: int, args):
        super().__init__()
        self.anchor_number = int(anchor_number)
        self.max_observations = int(args.get("max_per_source", 64))
        self.pre_nms_topk = int(args.get("pre_nms_topk", 512))
        self.score_threshold = float(args.get("score_threshold", 0.2))
        self.nms_threshold = float(args.get("nms_threshold", 0.15))
        self.query_dim = int(args.get("query_dim", 256))
        self.prior_probability = float(args.get("prior_probability", 0.01))
        self.lidar_range = tuple(float(value) for value in args["lidar_range"])
        self.voxel_size = tuple(float(value) for value in args["voxel_size"])
        self.anchor_dims_hwl = tuple(float(value) for value in args["anchor_dims_hwl"])
        self.anchor_rotations = tuple(float(value) for value in args["anchor_rotations_deg"])
        if len(self.anchor_rotations) != self.anchor_number:
            raise ValueError("anchor_rotations_deg must match anchor_number")
        if self.max_observations < 1 or self.pre_nms_topk < self.max_observations:
            raise ValueError("pre_nms_topk must be >= max_per_source >= 1")
        if not 0.0 < self.prior_probability < 1.0:
            raise ValueError("prior_probability must be between zero and one")

        self.local_cls_head = nn.Conv2d(input_dim, self.anchor_number, kernel_size=1)
        self.local_reg_head = nn.Conv2d(
            input_dim, 7 * self.anchor_number, kernel_size=1
        )
        self.observation_mlp = nn.Sequential(
            nn.Linear(input_dim + 8, self.query_dim),
            nn.LayerNorm(self.query_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.query_dim, self.query_dim),
        )
        nn.init.normal_(self.local_cls_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(
            self.local_cls_head.bias,
            -math.log((1.0 - self.prior_probability) / self.prior_probability),
        )
        nn.init.zeros_(self.local_reg_head.weight)
        nn.init.zeros_(self.local_reg_head.bias)

    def _anchors(self, height: int, width: int, feature: torch.Tensor) -> torch.Tensor:
        x_min, y_min, _, x_max, y_max, _ = self.lidar_range
        voxel_x, voxel_y, _ = self.voxel_size
        x = torch.linspace(
            x_min + voxel_x, x_max - voxel_x, width,
            device=feature.device, dtype=feature.dtype,
        )
        y = torch.linspace(
            y_min + voxel_y, y_max - voxel_y, height,
            device=feature.device, dtype=feature.dtype,
        )
        cy, cx = torch.meshgrid(y, x, indexing="ij")
        cx = cx.unsqueeze(-1).expand(height, width, self.anchor_number)
        cy = cy.unsqueeze(-1).expand(height, width, self.anchor_number)
        cz = torch.full_like(cx, -1.0)
        h, w, length = self.anchor_dims_hwl
        anchors = torch.stack(
            (
                cx,
                cy,
                cz,
                torch.full_like(cx, h),
                torch.full_like(cx, w),
                torch.full_like(cx, length),
                torch.stack(
                    [torch.full_like(cx[..., 0], math.radians(rotation))
                     for rotation in self.anchor_rotations],
                    dim=-1,
                ),
            ),
            dim=-1,
        )
        return anchors

    def _select(self, boxes: torch.Tensor, scores: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        source_count = boxes.shape[0]
        padded_boxes = boxes.new_zeros((source_count, self.max_observations, 7))
        padded_scores = scores.new_zeros((source_count, self.max_observations))
        padded_mask = torch.zeros(
            (source_count, self.max_observations), dtype=torch.bool, device=boxes.device
        )
        x_min, y_min, z_min, x_max, y_max, z_max = self.lidar_range

        for source_index in range(source_count):
            valid = scores[source_index] >= self.score_threshold
            source_boxes = boxes[source_index][valid]
            source_scores = scores[source_index][valid]
            if source_scores.numel() == 0:
                continue
            in_range = (
                (source_boxes[:, 0] >= x_min) & (source_boxes[:, 0] <= x_max)
                & (source_boxes[:, 1] >= y_min) & (source_boxes[:, 1] <= y_max)
                & (source_boxes[:, 2] >= z_min) & (source_boxes[:, 2] <= z_max)
            )
            source_boxes = source_boxes[in_range]
            source_scores = source_scores[in_range]
            if source_scores.numel() == 0:
                continue
            top_count = min(self.pre_nms_topk, source_scores.numel())
            source_scores, top_indices = torch.topk(source_scores, top_count)
            source_boxes = source_boxes[top_indices]
            corners = box_utils.boxes_to_corners_3d(source_boxes.clone(), order="hwl")
            keep = box_utils.nms_rotated(corners, source_scores, self.nms_threshold)
            keep = torch.as_tensor(
                keep[: self.max_observations], dtype=torch.long, device=boxes.device
            )
            selected_boxes = source_boxes[keep]
            selected_scores = source_scores[keep]
            count = selected_scores.numel()
            padded_boxes[source_index, :count] = selected_boxes
            padded_scores[source_index, :count] = selected_scores
            padded_mask[source_index, :count] = True
        return padded_boxes, padded_scores, padded_mask

    def _sample_features(
        self, local_bev: torch.Tensor, boxes: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        x_min, y_min, _, x_max, y_max, _ = self.lidar_range
        grid_x = 2.0 * (boxes[..., 0] - x_min) / (x_max - x_min) - 1.0
        grid_y = 2.0 * (boxes[..., 1] - y_min) / (y_max - y_min) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(2)
        sampled = F.grid_sample(
            local_bev, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        sampled = sampled.squeeze(-1).transpose(1, 2)
        return sampled * mask.unsqueeze(-1).to(sampled.dtype)

    @staticmethod
    def _scatter_sources(
        local_value: torch.Tensor,
        source_to_batch: torch.Tensor,
        source_slot: torch.Tensor,
        batch_size: int,
        max_sources: int,
    ) -> torch.Tensor:
        flat_shape = (batch_size * max_sources,) + tuple(local_value.shape[1:])
        flat = local_value.new_zeros(flat_shape)
        flat_index = source_to_batch.long() * max_sources + source_slot.long()
        flat = flat.index_copy(0, flat_index, local_value)
        return flat.view((batch_size, max_sources) + tuple(local_value.shape[1:]))

    def forward(self, local_bev: torch.Tensor, runtime_frame: Dict) -> Dict:
        local_psm = self.local_cls_head(local_bev)
        local_rm = self.local_reg_head(local_bev)
        source_count, _, height, width = local_psm.shape
        anchors = self._anchors(height, width, local_bev)
        boxes = delta_to_boxes3d(local_rm, anchors)
        scores = torch.sigmoid(
            local_psm.permute(0, 2, 3, 1).contiguous().view(source_count, -1)
        )
        selected_boxes, selected_scores, selected_mask = self._select(boxes, scores)
        roi_features = self._sample_features(local_bev, selected_boxes, selected_mask)
        observation_input = torch.cat(
            (roi_features, selected_boxes, selected_scores.unsqueeze(-1)), dim=-1
        )
        local_observation_features = self.observation_mlp(observation_input)
        local_observation_features = local_observation_features * selected_mask.unsqueeze(
            -1
        ).to(local_observation_features.dtype)

        batch_size, max_sources = runtime_frame["source_mask"].shape
        source_to_batch = runtime_frame["source_to_batch"]
        source_slot = runtime_frame["source_slot"]
        return {
            "local_bev": local_bev,
            "local_psm": local_psm,
            "local_rm": local_rm,
            "local_anchors": anchors,
            "obs_feat": self._scatter_sources(
                local_observation_features, source_to_batch, source_slot,
                batch_size, max_sources,
            ),
            "obs_box": self._scatter_sources(
                selected_boxes, source_to_batch, source_slot, batch_size, max_sources
            ),
            "obs_score": self._scatter_sources(
                selected_scores, source_to_batch, source_slot, batch_size, max_sources
            ),
            "obs_mask": self._scatter_sources(
                selected_mask, source_to_batch, source_slot, batch_size, max_sources
            ),
        }
