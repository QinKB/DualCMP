"""Trajectory-aligned source-BEV sampling and conservative refinement."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CorridorSampleBatch:
    samples: torch.Tensor
    valid: torch.Tensor
    grid: torch.Tensor


@dataclass(frozen=True)
class SourceRefinerOutput:
    final_logits: torch.Tensor
    final_trajs: torch.Tensor
    source_gate: torch.Tensor
    selection_weights: torch.Tensor
    router_weights: torch.Tensor
    velocity_delta: torch.Tensor
    mean_delta: torch.Tensor


class TrajectoryCorridorSampler(nn.Module):
    """Sample fixed metric offsets along receiver-frame trajectories."""

    def __init__(
        self,
        lidar_range,
        lateral_offset_m: float = 1.5,
        longitudinal_offset_m: float = 1.5,
    ) -> None:
        super().__init__()
        if lidar_range is None or len(lidar_range) != 6:
            raise ValueError("lidar_range must contain six metric bounds")
        self.x_min = float(lidar_range[0])
        self.y_min = float(lidar_range[1])
        self.x_max = float(lidar_range[3])
        self.y_max = float(lidar_range[4])
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("invalid lidar_range")
        self.lateral_offset_m = float(lateral_offset_m)
        self.longitudinal_offset_m = float(longitudinal_offset_m)
        if self.lateral_offset_m < 0.0 or self.longitudinal_offset_m < 0.0:
            raise ValueError("corridor offsets must be non-negative")

    def _metric_grid(
        self, actor_boxes: torch.Tensor, proposal_mean: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        absolute = actor_boxes[..., None, None, :2] + proposal_mean
        displacement = torch.empty_like(proposal_mean)
        displacement[..., 0, :] = proposal_mean[..., 0, :]
        displacement[..., 1:, :] = (
            proposal_mean[..., 1:, :] - proposal_mean[..., :-1, :]
        )
        norm = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
        yaw = actor_boxes[..., 6]
        fallback = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
        fallback = fallback[..., None, None, :].expand_as(displacement)
        tangent = torch.where(
            norm > 1e-4,
            displacement / norm.clamp_min(1e-6),
            fallback,
        )
        normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)
        offsets = torch.stack(
            (
                torch.zeros_like(tangent),
                self.lateral_offset_m * normal,
                -self.lateral_offset_m * normal,
                self.longitudinal_offset_m * tangent,
                -self.longitudinal_offset_m * tangent,
            ),
            dim=-2,
        )
        positions = absolute[..., None, :] + offsets
        grid_x = 2.0 * (positions[..., 0] - self.x_min) / (
            self.x_max - self.x_min
        ) - 1.0
        grid_y = 2.0 * (positions[..., 1] - self.y_min) / (
            self.y_max - self.y_min
        ) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)
        spatial_valid = (grid.abs() <= 1.0).all(dim=-1)
        return grid, spatial_valid

    def forward(
        self,
        source_bev: torch.Tensor,
        actor_boxes: torch.Tensor,
        proposal_mean: torch.Tensor,
        cav_mask: torch.Tensor,
    ) -> CorridorSampleBatch:
        if source_bev.ndim != 5:
            raise ValueError("source_bev must have shape [B,L,C,H,W]")
        batch, sources, channels, height, width = source_bev.shape
        if actor_boxes.ndim != 3 or actor_boxes.shape[0] != batch:
            raise ValueError("actor_boxes must have shape [B,Q,7]")
        if actor_boxes.shape[-1] != 7:
            raise ValueError("actor_boxes must end in seven box fields")
        if proposal_mean.ndim != 5 or proposal_mean.shape[:2] != actor_boxes.shape[:2]:
            raise ValueError("proposal_mean must have shape [B,Q,K,T,2]")
        if proposal_mean.shape[-1] != 2:
            raise ValueError("proposal_mean must end in xy")
        if cav_mask.shape != (batch, sources):
            raise ValueError("cav_mask must have shape [B,L]")

        grid, spatial_valid = self._metric_grid(actor_boxes, proposal_mean)
        queries, modes, steps, offsets = grid.shape[1:5]
        if sources == 0:
            return CorridorSampleBatch(
                samples=source_bev.new_zeros(
                    batch, queries, modes, steps, 0, channels
                ),
                valid=torch.zeros(
                    batch,
                    queries,
                    modes,
                    steps,
                    0,
                    dtype=torch.bool,
                    device=source_bev.device,
                ),
                grid=grid,
            )
        source_flat = source_bev.reshape(
            batch * sources, channels, height, width
        )
        pooled = None
        valid_count = source_bev.new_zeros(
            batch, queries, modes, steps, sources, 1
        )
        valid_any = torch.zeros(
            batch,
            queries,
            modes,
            steps,
            sources,
            dtype=torch.bool,
            device=source_bev.device,
        )
        for offset_index in range(offsets):
            offset_grid = grid[..., offset_index, :]
            source_grid = offset_grid[:, None].expand(
                batch, sources, queries, modes, steps, 2
            )
            sampled_offset = F.grid_sample(
                source_flat,
                source_grid.reshape(
                    batch * sources, queries * modes, steps, 2
                ),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled_offset = sampled_offset.view(
                batch, sources, channels, queries, modes, steps
            ).permute(0, 3, 4, 5, 1, 2)
            offset_valid = spatial_valid[
                ..., offset_index, None
            ].expand(
                batch, queries, modes, steps, sources
            ) & cav_mask[:, None, None, None, :].bool()
            contribution = sampled_offset * offset_valid[..., None].to(
                sampled_offset.dtype
            )
            pooled = contribution if pooled is None else pooled + contribution
            valid_count = valid_count + offset_valid[..., None].to(
                valid_count.dtype
            )
            valid_any = valid_any | offset_valid
        pooled = pooled / valid_count.clamp_min(1.0)
        pooled = torch.where(
            valid_any[..., None], pooled, torch.zeros_like(pooled)
        )
        return CorridorSampleBatch(samples=pooled, valid=valid_any, grid=grid)


class _ZeroPreservingTemporalBlock(nn.Module):
    """Bias-free pre-norm attention for which f(0) is exactly zero."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, bias=False, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        safe_valid = valid.clone()
        empty = ~safe_valid.any(dim=-1)
        safe_valid[empty, -1] = True
        normalized = self.norm1(tokens)
        attended = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )[0]
        result = tokens + self.dropout(attended)
        result = result + self.dropout(self.ffn(self.norm2(result)))
        result = torch.where(valid[..., None], result, torch.zeros_like(result))
        return result


class SourceCorridorRefiner(nn.Module):
    """Select source evidence along time and apply a gated motion residual."""

    def __init__(
        self,
        query_dim: int,
        source_bev_dim: int,
        modes: int,
        future_frames: int,
        args: dict,
    ) -> None:
        super().__init__()
        self.query_dim = int(query_dim)
        self.source_bev_dim = int(source_bev_dim)
        self.modes = int(modes)
        self.future_frames = int(future_frames)
        self.frame_period_s = float(args.get("frame_period_s", 0.1))
        self.velocity_limit_mps = float(
            args.get("source_velocity_limit_mps", 3.0)
        )
        self.position_limit_m = float(args.get("source_residual_limit_m", 4.0))
        self.gate_prior = float(args.get("source_gate_prior", 0.1))
        dropout = float(args.get("dropout", 0.1))
        heads = int(args.get("source_heads", 8))
        if not 0.0 < self.gate_prior < 1.0:
            raise ValueError("source_gate_prior must lie strictly between zero and one")
        if self.query_dim % heads:
            raise ValueError("source_heads must divide query_dim")
        if self.frame_period_s <= 0.0:
            raise ValueError("frame_period_s must be positive")
        if self.velocity_limit_mps <= 0.0 or self.position_limit_m <= 0.0:
            raise ValueError("source motion limits must be positive")

        self.value_projection = nn.Linear(
            self.source_bev_dim, self.query_dim, bias=False
        )
        self.time_film = nn.Parameter(
            torch.randn(self.future_frames, self.query_dim) * 0.02
        )
        self.base_state_projection = nn.Linear(5, self.query_dim)
        self.relevance = nn.Sequential(
            nn.LayerNorm(self.query_dim),
            nn.Linear(self.query_dim, self.query_dim),
            nn.GELU(),
            nn.Linear(self.query_dim, 1),
        )
        self.availability_beta_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(0.1)), dtype=torch.float32)
        )
        self.temporal = _ZeroPreservingTemporalBlock(
            self.query_dim, heads=heads, dropout=dropout
        )
        self.uncertainty_projection = nn.Linear(2, self.query_dim)
        self.gate_hidden = nn.Sequential(
            nn.LayerNorm(self.query_dim),
            nn.Linear(self.query_dim, self.query_dim),
            nn.GELU(),
        )
        self.gate_head = nn.Linear(self.query_dim, 1)
        self.velocity_head = nn.Linear(self.query_dim, 2, bias=False)
        self.log_std_head = nn.Linear(self.query_dim, 2, bias=False)
        self.raw_rho_head = nn.Linear(self.query_dim, 1, bias=False)
        self.mode_score_head = nn.Linear(self.query_dim, 1, bias=False)

        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(
            self.gate_head.bias, math.log(self.gate_prior / (1.0 - self.gate_prior))
        )
        for head in (
            self.velocity_head,
            self.log_std_head,
            self.raw_rho_head,
            self.mode_score_head,
        ):
            nn.init.zeros_(head.weight)

    @staticmethod
    def bound_vector_norm(vector: torch.Tensor, limit: float) -> torch.Tensor:
        norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        magnitude = float(limit) * torch.tanh(norm / float(limit))
        scale = torch.where(
            norm > 1e-6,
            magnitude / norm.clamp_min(1e-8),
            torch.ones_like(norm),
        )
        return vector * scale

    @staticmethod
    def integrate_velocity(
        velocity: torch.Tensor,
        frame_period_s: float,
        position_limit_m: float,
    ) -> torch.Tensor:
        position = torch.cumsum(velocity * float(frame_period_s), dim=-2)
        norm = torch.linalg.vector_norm(position, dim=-1, keepdim=True)
        scale = torch.clamp(float(position_limit_m) / norm.clamp_min(1e-8), max=1.0)
        return position * scale

    @property
    def availability_beta(self) -> torch.Tensor:
        return F.softplus(self.availability_beta_raw)

    def forward(
        self,
        samples: torch.Tensor,
        sample_valid: torch.Tensor,
        base_trajs: torch.Tensor,
        base_logits: torch.Tensor,
        geometry: torch.Tensor,
        age: torch.Tensor,
        availability_logits: torch.Tensor,
        actor_valid: torch.Tensor,
        force_null: bool = False,
        availability_bias_enabled: bool = True,
    ) -> SourceRefinerOutput:
        if samples.ndim != 6:
            raise ValueError("samples must have shape [B,Q,K,T,L,C]")
        batch, queries, modes, steps, sources, channels = samples.shape
        if (modes, steps, channels) != (
            self.modes,
            self.future_frames,
            self.source_bev_dim,
        ):
            raise ValueError("samples disagree with refiner dimensions")
        if sample_valid.shape != samples.shape[:-1]:
            raise ValueError("sample_valid shape mismatch")
        if base_trajs.shape != (batch, queries, modes, steps, 5):
            raise ValueError("base_trajs shape mismatch")
        if base_logits.shape != (batch, queries, modes):
            raise ValueError("base_logits shape mismatch")
        if geometry.shape != (batch, queries, sources, self.query_dim):
            raise ValueError("geometry shape mismatch")
        if age.shape != geometry.shape:
            raise ValueError("age shape mismatch")
        if availability_logits.shape != (batch, queries, sources):
            raise ValueError("availability_logits shape mismatch")
        if actor_valid.shape != (batch, queries):
            raise ValueError("actor_valid shape mismatch")

        projected = self.value_projection(samples)
        source_valid = sample_valid.bool()
        source_tokens = torch.where(
            source_valid[..., None], projected, torch.zeros_like(projected)
        )

        state = self.base_state_projection(base_trajs.detach())
        score_feature = torch.tanh(
            source_tokens
            + state[..., None, :]
            + geometry[:, :, None, None, :, :]
            + age[:, :, None, None, :, :]
        )
        relevance = self.relevance(score_feature).squeeze(-1).float()
        availability_bias = F.logsigmoid(
            availability_logits[:, :, None, None, :].float()
        ).clamp_min(-20.0)
        beta = (
            self.availability_beta.float()
            if availability_bias_enabled
            else self.availability_beta_raw.float() * 0.0
        )
        relevance = relevance + beta * availability_bias
        has_source = source_valid.any(dim=-1)
        safe_logits = relevance.masked_fill(
            ~source_valid, torch.finfo(relevance.dtype).min
        )
        safe_logits = torch.where(
            has_source[..., None], safe_logits, torch.zeros_like(safe_logits)
        )
        selection = torch.softmax(safe_logits, dim=-1).to(projected.dtype)
        selection = torch.where(source_valid, selection, torch.zeros_like(selection))
        selection = selection / selection.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        source_context = (selection[..., None] * source_tokens).sum(dim=-2)
        source_context = source_context * (
            1.0
            + torch.tanh(self.time_film).view(1, 1, 1, steps, self.query_dim)
        )
        time_valid = has_source & actor_valid[:, :, None, None]
        flat_context = source_context.reshape(-1, steps, self.query_dim)
        flat_valid = time_valid.reshape(-1, steps)
        encoded = self.temporal(flat_context, flat_valid).view_as(source_context)

        valid_float_time = time_valid[..., None].to(encoded.dtype)
        pooled = (encoded * valid_float_time).sum(dim=-2) / valid_float_time.sum(
            dim=-2
        ).clamp_min(1.0)
        uncertainty = base_trajs.detach()[..., 2:4].mean(dim=-2)
        gate_feature = self.gate_hidden(
            pooled + self.uncertainty_projection(uncertainty)
        )
        gate = torch.sigmoid(self.gate_head(gate_feature).squeeze(-1))
        gate_valid = actor_valid[:, :, None] & has_source.any(dim=-1)
        if force_null:
            gate_valid = torch.zeros_like(gate_valid)
        gate = torch.where(gate_valid, gate, torch.zeros_like(gate))

        raw_velocity = self.velocity_head(encoded)
        velocity_delta = self.bound_vector_norm(
            raw_velocity, self.velocity_limit_mps
        )
        mean_delta = self.integrate_velocity(
            velocity_delta,
            self.frame_period_s,
            self.position_limit_m,
        )
        gate_state = gate[..., None, None]
        mean_delta = gate_state * mean_delta
        log_std_delta = gate_state * 0.5 * torch.tanh(self.log_std_head(encoded))
        raw_rho_delta = gate_state * 0.5 * torch.tanh(self.raw_rho_head(encoded))
        mode_delta = gate * self.mode_score_head(pooled).squeeze(-1).clamp(-5.0, 5.0)

        base_mean = base_trajs.detach()[..., :2]
        base_log_std = base_trajs.detach()[..., 2:4]
        base_rho = base_trajs.detach()[..., 4:5]
        base_raw_rho = torch.atanh((2.0 * base_rho).clamp(-0.999, 0.999))
        final_mean = base_mean + mean_delta
        final_log_std = (base_log_std + log_std_delta).clamp(-1.609, 5.0)
        final_rho = 0.5 * torch.tanh(base_raw_rho + raw_rho_delta)
        final_trajs = torch.cat((final_mean, final_log_std, final_rho), dim=-1)
        final_logits = base_logits.detach() + mode_delta

        gate_time = gate[..., None]
        null = (1.0 - gate_time)[..., None]
        routed_sources = gate_time[..., None] * selection
        router = torch.cat((null.expand(batch, queries, modes, steps, 1), routed_sources), dim=-1)
        return SourceRefinerOutput(
            final_logits=final_logits,
            final_trajs=final_trajs,
            source_gate=gate,
            selection_weights=selection,
            router_weights=router,
            velocity_delta=velocity_delta,
            mean_delta=mean_delta,
        )
