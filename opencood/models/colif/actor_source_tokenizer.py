"""Actor-aligned tokenization of receiver-frame source BEV features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorAlignedSourceTokenizer(nn.Module):
    """Keep per-source RoI evidence until motion-mode decoding.

    Source BEVs already share the receiver coordinate frame.  For every
    tracked actor, a regular grid is defined in the actor's local box frame,
    enlarged by ``roi_scale``, rotated by the actor heading, and sampled from
    every available source.  The returned source and grid dimensions remain
    explicit; this module deliberately performs no source aggregation.
    """

    def __init__(
        self,
        input_dim: int,
        query_dim: int,
        max_cav: int,
        lidar_range,
        roi_grid_size: int = 5,
        roi_scale: float = 2.0,
        minimum_half_extent_m: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.query_dim = int(query_dim)
        self.max_cav = int(max_cav)
        self.roi_grid_size = int(roi_grid_size)
        self.roi_scale = float(roi_scale)
        self.minimum_half_extent_m = float(minimum_half_extent_m)
        if self.roi_grid_size < 1 or self.roi_grid_size % 2 == 0:
            raise ValueError("roi_grid_size must be a positive odd integer")
        if self.roi_scale <= 0.0:
            raise ValueError("roi_scale must be positive")
        if self.minimum_half_extent_m <= 0.0:
            raise ValueError("minimum_half_extent_m must be positive")

        self.x_min, self.y_min = float(lidar_range[0]), float(lidar_range[1])
        self.x_max, self.y_max = float(lidar_range[3]), float(lidar_range[4])
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("invalid lidar_range")

        # Bias-free content projection prevents an empty sampled feature from
        # becoming a learned source observation solely through a linear bias.
        self.content_projection = nn.Linear(
            self.input_dim, self.query_dim, bias=False
        )
        self.offset_projection = nn.Sequential(
            nn.Linear(2, self.query_dim),
            nn.GELU(),
            nn.Linear(self.query_dim, self.query_dim),
        )

        axis = torch.linspace(-1.0, 1.0, self.roi_grid_size)
        offset_y, offset_x = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer(
            "unit_roi_offsets",
            torch.stack((offset_x, offset_y), dim=-1).reshape(-1, 2),
        )

    @property
    def tokens_per_source(self) -> int:
        return self.roi_grid_size * self.roi_grid_size

    def _sampling_grid(
        self, actor_boxes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if actor_boxes.ndim != 3 or actor_boxes.shape[-1] != 7:
            raise ValueError("actor_boxes must have shape [B,Q,7]")

        # Boxes use xyzhwlr order.  Local x follows box length and local y
        # follows box width, matching proposal-aligned RoI pooling.
        half_length = (
            0.5 * self.roi_scale * actor_boxes[..., 5].clamp_min(1e-3)
        ).clamp_min(self.minimum_half_extent_m)
        half_width = (
            0.5 * self.roi_scale * actor_boxes[..., 4].clamp_min(1e-3)
        ).clamp_min(self.minimum_half_extent_m)
        unit = self.unit_roi_offsets.to(actor_boxes)
        local_x = half_length[..., None] * unit[:, 0]
        local_y = half_width[..., None] * unit[:, 1]

        yaw = actor_boxes[..., 6, None]
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        receiver_x = actor_boxes[..., 0, None] + cosine * local_x - sine * local_y
        receiver_y = actor_boxes[..., 1, None] + sine * local_x + cosine * local_y

        grid_x = 2.0 * (receiver_x - self.x_min) / (
            self.x_max - self.x_min
        ) - 1.0
        grid_y = 2.0 * (receiver_y - self.y_min) / (
            self.y_max - self.y_min
        ) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)
        spatial_valid = (grid.abs() <= 1.0).all(dim=-1)
        # MTR's released V2V4Real intentions were clustered in its legacy local
        # frame, where numeric degree headings were consumed as radians.  The
        # physical sampling grid remains correct; only the positional value
        # supplied to intention attention is expressed in that same frame.
        receiver_offset_x = receiver_x - actor_boxes[..., 0, None]
        receiver_offset_y = receiver_y - actor_boxes[..., 1, None]
        decoder_heading = torch.rad2deg(yaw)
        decoder_cosine = torch.cos(decoder_heading)
        decoder_sine = torch.sin(decoder_heading)
        decoder_local_x = (
            decoder_cosine * receiver_offset_x
            + decoder_sine * receiver_offset_y
        )
        decoder_local_y = (
            -decoder_sine * receiver_offset_x
            + decoder_cosine * receiver_offset_y
        )
        half_x_range = 0.5 * (self.x_max - self.x_min)
        half_y_range = 0.5 * (self.y_max - self.y_min)
        position_input = torch.stack(
            (
                decoder_local_x / half_x_range,
                decoder_local_y / half_y_range,
            ),
            dim=-1,
        )
        return grid, spatial_valid, position_input

    def forward(
        self,
        source_bev: torch.Tensor,
        cav_mask: torch.Tensor,
        actor_boxes: torch.Tensor,
        actor_valid_mask: torch.Tensor,
    ) -> dict:
        if source_bev.ndim != 5:
            raise ValueError("source_bev must have shape [B,L,C,H,W]")
        batch, sources, channels, height, width = source_bev.shape
        if sources > self.max_cav or channels != self.input_dim:
            raise ValueError("source_bev disagrees with tokenizer configuration")
        if cav_mask.shape != (batch, sources):
            raise ValueError("cav_mask must have shape [B,L]")
        if actor_boxes.shape[:2] != actor_valid_mask.shape:
            raise ValueError("actor_valid_mask must have shape [B,Q]")

        queries = actor_boxes.shape[1]
        patch_tokens = self.tokens_per_source
        grid, spatial_valid, position_input = self._sampling_grid(actor_boxes)
        expanded_grid = grid[:, None].expand(
            batch, sources, queries, patch_tokens, 2
        )
        sampled = F.grid_sample(
            source_bev.reshape(batch * sources, channels, height, width),
            expanded_grid.reshape(batch * sources, queries, patch_tokens, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.view(
            batch, sources, channels, queries, patch_tokens
        ).permute(0, 3, 1, 4, 2)

        tokens = self.content_projection(sampled)
        position = self.offset_projection(
            position_input.to(tokens)
        )[:, :, None].expand(
            batch, queries, sources, patch_tokens, self.query_dim
        )
        token_mask = (
            actor_valid_mask.bool()[..., None, None]
            & cav_mask.bool()[:, None, :, None]
            & spatial_valid[:, :, None, :]
        )
        tokens = torch.where(
            token_mask[..., None], tokens, torch.zeros_like(tokens)
        )
        position = torch.where(
            token_mask[..., None], position, torch.zeros_like(position)
        )
        return {
            "tokens": tokens,
            "positions": position,
            "token_mask": token_mask,
            "sampling_grid": grid,
        }
