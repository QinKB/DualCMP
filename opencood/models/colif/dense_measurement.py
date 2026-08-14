"""Sparse current-frame measurements from CMP's own fused dense heads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def decode_anchor_residuals(
    residuals: torch.Tensor, anchors: torch.Tensor
) -> torch.Tensor:
    """Decode VoxelPostprocessor-compatible ``hwl`` anchor residuals."""

    batch = residuals.shape[0]
    residuals = residuals.permute(0, 2, 3, 1).contiguous().view(batch, -1, 7)
    anchors = anchors.reshape(-1, 7).to(
        device=residuals.device, dtype=residuals.dtype
    )
    if anchors.shape[0] != residuals.shape[1]:
        raise ValueError(
            "Dense anchor/head mismatch: %d anchors for %d residuals"
            % (anchors.shape[0], residuals.shape[1])
        )
    anchors = anchors.unsqueeze(0).expand(batch, -1, -1)
    diagonal = torch.sqrt(anchors[..., 4].square() + anchors[..., 5].square())
    boxes = torch.zeros_like(residuals)
    boxes[..., 0] = residuals[..., 0] * diagonal + anchors[..., 0]
    boxes[..., 1] = residuals[..., 1] * diagonal + anchors[..., 1]
    boxes[..., 2] = residuals[..., 2] * anchors[..., 3] + anchors[..., 2]
    boxes[..., 3:6] = torch.exp(residuals[..., 3:6]) * anchors[..., 3:6]
    boxes[..., 6] = residuals[..., 6] + anchors[..., 6]
    return boxes


class DenseMeasurementProposalGenerator(nn.Module):
    """Turn the existing CoBEVT ``psm/rm`` into sparse query measurements.

    Proposal indices are intentionally discrete, as in DETR two-stage proposal
    selection.  The selected boxes remain differentiable with respect to the
    regression head.  Spatial local maxima and center-distance suppression keep
    this lightweight and avoid introducing a second detector or tracker.
    """

    def __init__(
        self,
        lidar_range,
        voxel_size,
        anchor_number: int,
        args: dict,
    ):
        super().__init__()
        self.lidar_range = tuple(float(value) for value in lidar_range)
        self.voxel_size = tuple(float(value) for value in voxel_size)
        self.anchor_number = int(anchor_number)
        self.max_proposals = int(args.get("max_proposals", 48))
        self.pre_topk = int(args.get("pre_topk", 256))
        self.score_threshold = float(args.get("score_threshold", 0.05))
        self.suppression_radius = float(args.get("suppression_radius", 2.0))
        self.anchor_dims_hwl = tuple(
            float(value) for value in args.get("anchor_dims_hwl", [1.56, 1.6, 3.9])
        )
        self.anchor_rotations = tuple(
            math.radians(float(value))
            for value in args.get("anchor_rotations_deg", [0.0, 90.0])
        )
        if len(self.anchor_rotations) != self.anchor_number:
            raise ValueError("measurement anchor rotations must match anchor_number")
        if self.max_proposals < 1 or self.pre_topk < self.max_proposals:
            raise ValueError("Require pre_topk >= max_proposals >= 1")
        if not 0.0 <= self.score_threshold < 1.0:
            raise ValueError("measurement score_threshold must lie in [0, 1)")
        if self.suppression_radius <= 0.0:
            raise ValueError("measurement suppression radius must be positive")

    def _anchors(self, height: int, width: int, reference: torch.Tensor):
        x_min, y_min, _, x_max, y_max, _ = self.lidar_range
        voxel_x, voxel_y, _ = self.voxel_size
        x = torch.linspace(
            x_min + voxel_x,
            x_max - voxel_x,
            width,
            device=reference.device,
            dtype=reference.dtype,
        )
        y = torch.linspace(
            y_min + voxel_y,
            y_max - voxel_y,
            height,
            device=reference.device,
            dtype=reference.dtype,
        )
        cy, cx = torch.meshgrid(y, x, indexing="ij")
        cx = cx.unsqueeze(-1).expand(height, width, self.anchor_number)
        cy = cy.unsqueeze(-1).expand(height, width, self.anchor_number)
        cz = torch.full_like(cx, -1.0)
        h, w, length = self.anchor_dims_hwl
        rotations = torch.as_tensor(
            self.anchor_rotations, device=reference.device, dtype=reference.dtype
        ).view(1, 1, self.anchor_number).expand(height, width, -1)
        return torch.stack(
            (
                cx,
                cy,
                cz,
                torch.full_like(cx, h),
                torch.full_like(cx, w),
                torch.full_like(cx, length),
                rotations,
            ),
            dim=-1,
        )

    def forward(
        self,
        dense_logits: torch.Tensor,
        dense_residuals: torch.Tensor,
        previous_active: torch.Tensor,
        propagated_boxes: torch.Tensor,
    ) -> dict:
        batch, anchors_per_cell, height, width = dense_logits.shape
        if anchors_per_cell != self.anchor_number:
            raise ValueError("dense classification head disagrees with anchors")
        if previous_active.shape != propagated_boxes.shape[:2]:
            raise ValueError("previous active mask and propagated boxes disagree")

        anchors = self._anchors(height, width, dense_logits)
        all_boxes = decode_anchor_residuals(dense_residuals, anchors)
        logits_by_cell = dense_logits.permute(0, 2, 3, 1).contiguous()
        probabilities_by_cell = logits_by_cell.sigmoid()
        cell_scores, best_anchor = probabilities_by_cell.max(dim=-1)
        local_peak = cell_scores >= (
            F.max_pool2d(
                cell_scores.unsqueeze(1), kernel_size=3, stride=1, padding=1
            ).squeeze(1)
            - 1e-7
        )
        candidate_scores = cell_scores.masked_fill(~local_peak, -1.0)

        proposal_boxes = all_boxes.new_zeros(
            batch, self.max_proposals, 7
        )
        proposal_logits = dense_logits.new_full(
            (batch, self.max_proposals), -20.0
        )
        proposal_mask = torch.zeros(
            batch,
            self.max_proposals,
            dtype=torch.bool,
            device=dense_logits.device,
        )
        flat_logits = logits_by_cell.view(batch, -1, self.anchor_number)

        # Selection and suppression do not carry gradients; selected tensor
        # values below are gathered from the original differentiable heads.
        with torch.no_grad():
            ordered_cells = []
            for batch_index in range(batch):
                flat_score = candidate_scores[batch_index].reshape(-1)
                valid = torch.where(flat_score >= self.score_threshold)[0]
                if valid.numel():
                    count = min(self.pre_topk, int(valid.numel()))
                    order = valid[torch.topk(flat_score[valid], count).indices]
                else:
                    order = valid
                accepted = []
                for cell_tensor in order:
                    cell = int(cell_tensor)
                    anchor_index = int(best_anchor[batch_index].reshape(-1)[cell])
                    flat_anchor = cell * self.anchor_number + anchor_index
                    center = all_boxes[batch_index, flat_anchor, :2].detach()
                    if accepted:
                        accepted_centers = torch.stack(
                            [all_boxes[batch_index, item, :2].detach() for item in accepted]
                        )
                        if torch.norm(
                            accepted_centers - center[None], dim=-1
                        ).min() < self.suppression_radius:
                            continue
                    accepted.append(flat_anchor)
                    if len(accepted) >= self.max_proposals:
                        break
                ordered_cells.append(accepted)

        for batch_index, accepted in enumerate(ordered_cells):
            if not accepted:
                continue
            indices = torch.as_tensor(
                accepted, dtype=torch.long, device=dense_logits.device
            )
            count = int(indices.numel())
            proposal_boxes[batch_index, :count] = all_boxes[
                batch_index, indices
            ]
            cell_indices = torch.div(
                indices, self.anchor_number, rounding_mode="floor"
            )
            anchor_indices = indices.remainder(self.anchor_number)
            proposal_logits[batch_index, :count] = flat_logits[
                batch_index, cell_indices, anchor_indices
            ]
            proposal_mask[batch_index, :count] = True
        return {
            "proposal_boxes": proposal_boxes,
            "proposal_logits": proposal_logits,
            "proposal_mask": proposal_mask,
            "all_dense_boxes": all_boxes,
        }
