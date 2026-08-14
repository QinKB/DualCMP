"""Actor-conditioned cooperative evidence sampling over aligned source BEVs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorConditionedCooperativeFusion(nn.Module):
    """Select source-local BEV evidence separately for every actor query.

    All source feature maps must already share the receiver coordinate frame.
    The module samples a small metric patch around each actor from every CAV,
    then lets the actor query attend across source and spatial patch tokens.
    """

    def __init__(
        self,
        input_dim: int,
        query_dim: int,
        heads: int,
        max_cav: int,
        lidar_range,
        patch_radius_m: float = 2.0,
        patch_size: int = 3,
        dropout: float = 0.1,
        geometry_conditioning: bool = False,
        source_index_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.query_dim = int(query_dim)
        self.max_cav = int(max_cav)
        self.patch_size = int(patch_size)
        self.patch_radius_m = float(patch_radius_m)
        self.geometry_conditioning = bool(geometry_conditioning)
        self.source_index_embedding = bool(source_index_embedding)
        if self.patch_size < 1 or self.patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        if self.patch_radius_m < 0.0:
            raise ValueError("patch_radius_m must be non-negative")
        if self.query_dim % int(heads):
            raise ValueError("query_dim must be divisible by heads")
        self.x_min, self.y_min = float(lidar_range[0]), float(lidar_range[1])
        self.x_max, self.y_max = float(lidar_range[3]), float(lidar_range[4])
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("invalid lidar_range")

        self.source_projection = nn.Linear(self.input_dim, self.query_dim)
        # Instantiate both alternatives so ablations consume exactly the same
        # RNG and leave every downstream forecast parameter identically
        # initialized.  The inactive branch is frozen explicitly.
        self.source_embedding = nn.Embedding(self.max_cav, self.query_dim)
        self.geometry_projection = nn.Sequential(
            nn.Linear(7, self.query_dim),
            nn.GELU(),
            nn.Linear(self.query_dim, self.query_dim),
        )
        self.patch_projection = nn.Sequential(
            nn.Linear(2, self.query_dim),
            nn.GELU(),
            nn.Linear(self.query_dim, self.query_dim),
        )
        self.query_norm = nn.LayerNorm(self.query_dim)
        self.source_attention = nn.MultiheadAttention(
            self.query_dim,
            int(heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.query_dim)

        axis = torch.linspace(-1.0, 1.0, self.patch_size)
        offset_y, offset_x = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer(
            "unit_patch_offsets",
            torch.stack((offset_x, offset_y), dim=-1).reshape(-1, 2),
        )

    def _sampling_grid(self, boxes: torch.Tensor) -> torch.Tensor:
        center_x = 2.0 * (
            (boxes[..., 0] - self.x_min) / (self.x_max - self.x_min)
        ) - 1.0
        center_y = 2.0 * (
            (boxes[..., 1] - self.y_min) / (self.y_max - self.y_min)
        ) - 1.0
        centers = torch.stack((center_x, center_y), dim=-1)
        metric_scale = boxes.new_tensor(
            [
                2.0 * self.patch_radius_m / (self.x_max - self.x_min),
                2.0 * self.patch_radius_m / (self.y_max - self.y_min),
            ]
        )
        offsets = self.unit_patch_offsets.to(boxes) * metric_scale
        return centers[..., None, :] + offsets.view(1, 1, -1, 2)

    def _source_geometry(
        self,
        actor_boxes: torch.Tensor,
        source_to_receiver: torch.Tensor,
        agents: int,
    ) -> torch.Tensor:
        """Encode actor-relative sensor geometry without fixed source slots."""

        batch, queries = actor_boxes.shape[:2]
        if source_to_receiver.shape != (batch, agents, 4, 4):
            raise ValueError(
                "source_to_receiver must have shape [B,L,4,4]"
            )
        source_origin = source_to_receiver[..., :2, 3]
        relative = (
            actor_boxes[..., None, :2] - source_origin[:, None, :, :]
        )
        diagonal = (
            (self.x_max - self.x_min) ** 2
            + (self.y_max - self.y_min) ** 2
        ) ** 0.5
        distance = torch.linalg.vector_norm(
            relative, dim=-1, keepdim=True
        )
        bearing = torch.atan2(relative[..., 1], relative[..., 0])
        actor_yaw = actor_boxes[..., 6, None]
        source_yaw = torch.atan2(
            source_to_receiver[..., 1, 0],
            source_to_receiver[..., 0, 0],
        )[:, None, :]
        bearing_delta = bearing - actor_yaw
        heading_delta = source_yaw - actor_yaw
        normalizer = actor_boxes.new_tensor(
            [self.x_max - self.x_min, self.y_max - self.y_min]
        )
        geometry = torch.cat(
            (
                relative / normalizer,
                distance / diagonal,
                torch.sin(bearing_delta)[..., None],
                torch.cos(bearing_delta)[..., None],
                torch.sin(heading_delta)[..., None],
                torch.cos(heading_delta)[..., None],
            ),
            dim=-1,
        )
        return self.geometry_projection(geometry)

    def forward(
        self,
        source_bev: torch.Tensor,
        cav_mask: torch.Tensor,
        actor_query: torch.Tensor,
        actor_boxes: torch.Tensor,
        actor_valid_mask: torch.Tensor,
        source_to_receiver: torch.Tensor | None = None,
    ) -> dict:
        if source_bev.ndim != 5:
            raise ValueError("source_bev must have shape [B,L,C,H,W]")
        batch, agents, channels, height, width = source_bev.shape
        if agents > self.max_cav or channels != self.input_dim:
            raise ValueError("source_bev disagrees with fusion configuration")
        if cav_mask.shape != (batch, agents):
            raise ValueError("cav_mask must have shape [B,L]")
        if actor_query.shape[:2] != actor_boxes.shape[:2]:
            raise ValueError("actor query and box counts disagree")
        if actor_valid_mask.shape != actor_query.shape[:2]:
            raise ValueError("actor_valid_mask must have shape [B,Q]")

        queries = actor_query.shape[1]
        patch_tokens = self.patch_size * self.patch_size
        grid = self._sampling_grid(actor_boxes)
        # grid_sample needs one grid per source feature map.  All source maps
        # share the same receiver-aligned metric grid.
        expanded_grid = grid[:, None].expand(
            batch, agents, queries, patch_tokens, 2
        )
        sampled = F.grid_sample(
            source_bev.reshape(batch * agents, channels, height, width),
            expanded_grid.reshape(
                batch * agents, queries, patch_tokens, 2
            ),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.view(
            batch, agents, channels, queries, patch_tokens
        ).permute(0, 3, 1, 4, 2)
        tokens = self.source_projection(sampled)
        if self.source_index_embedding:
            tokens = tokens + self.source_embedding.weight[
                :agents
            ].view(1, 1, agents, 1, self.query_dim)
        if self.geometry_conditioning:
            if source_to_receiver is None:
                raise ValueError(
                    "geometry-conditioned fusion requires source transforms"
                )
            geometry = self._source_geometry(
                actor_boxes, source_to_receiver, agents
            )
            tokens = tokens + geometry[..., None, :]
        tokens = tokens + self.patch_projection(
            self.unit_patch_offsets.to(tokens)
        ).view(1, 1, 1, patch_tokens, self.query_dim)
        tokens = tokens.reshape(
            batch * queries, agents * patch_tokens, self.query_dim
        )

        key_padding = (~cav_mask.bool())[:, None, :, None].expand(
            batch, queries, agents, patch_tokens
        ).reshape(batch * queries, agents * patch_tokens)
        query = self.query_norm(actor_query).reshape(
            batch * queries, 1, self.query_dim
        )
        attended, weights = self.source_attention(
            query,
            tokens,
            tokens,
            key_padding_mask=key_padding,
            need_weights=True,
            average_attn_weights=True,
        )
        attended = self.output_norm(attended[:, 0]).view(
            batch, queries, self.query_dim
        )
        weights = weights[:, 0].view(
            batch, queries, agents, patch_tokens
        )
        valid = actor_valid_mask.bool()
        attended = torch.where(
            valid[..., None], attended, torch.zeros_like(attended)
        )
        weights = torch.where(
            valid[..., None, None], weights, torch.zeros_like(weights)
        )
        return {
            "context": attended,
            "attention_weights": weights,
        }

    def freeze_inactive_parameters(self) -> None:
        if not self.source_index_embedding:
            for parameter in self.source_embedding.parameters():
                parameter.requires_grad = False
        if not self.geometry_conditioning:
            for parameter in self.geometry_projection.parameters():
                parameter.requires_grad = False
