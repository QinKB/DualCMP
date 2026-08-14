"""Differentiable predict--correct update for persistent actor state."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MeasurementStateUpdate(nn.Module):
    """Fuse propagated state and current fused-BEV measurement."""

    def __init__(self, query_dim, min_gain=0.1, max_gain=0.95,
                 initial_gain=0.75, innovation_scale_m=4.0):
        super().__init__()
        if not 0.0 <= min_gain < initial_gain < max_gain <= 1.0:
            raise ValueError(
                "Require 0 <= min_gain < initial_gain < max_gain <= 1"
            )
        if innovation_scale_m <= 0.0:
            raise ValueError("innovation_scale_m must be positive")
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)
        self.innovation_scale_m = float(innovation_scale_m)
        hidden = max(32, query_dim // 2)
        self.query_norm = nn.LayerNorm(query_dim)
        self.gate = nn.Sequential(
            nn.Linear(query_dim + 2, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        normalized = (initial_gain - min_gain) / (max_gain - min_gain)
        initial_logit = math.log(normalized / (1.0 - normalized))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, initial_logit)

    def forward(self, query_feat, propagated_box, measured_box,
                propagated_velocity, measured_velocity, measurement_logit,
                observed_mask, birth_mask, prior_variance_xy=None,
                prior_variance_valid_mask=None):
        innovation = measured_box[..., :2] - propagated_box[..., :2]
        innovation_norm = torch.linalg.vector_norm(
            innovation, dim=-1, keepdim=True
        ) / self.innovation_scale_m
        feature = torch.cat(
            (
                self.query_norm(query_feat),
                measurement_logit.sigmoid().unsqueeze(-1),
                innovation_norm.clamp(max=4.0),
            ),
            dim=-1,
        )
        learned_gain = self.min_gain + (self.max_gain - self.min_gain) * self.gate(
            feature
        ).sigmoid()
        learned_gain = torch.where(
            birth_mask[..., None], torch.ones_like(learned_gain), learned_gain
        )
        learned_gain = learned_gain * observed_mask[..., None].to(learned_gain.dtype)
        if prior_variance_xy is None:
            spatial_gain = learned_gain
        else:
            if prior_variance_xy.shape != observed_mask.shape:
                raise ValueError("prior_variance_xy must have shape [B,Q]")
            prior_variance = prior_variance_xy.clamp_min(1e-4)
            measurement_probability = measurement_logit.sigmoid().clamp_min(0.05)
            measurement_variance = measurement_probability.reciprocal()
            kalman_gain = prior_variance / (prior_variance + measurement_variance)
            kalman_gain = kalman_gain.clamp(self.min_gain, self.max_gain)[..., None]
            # A forecast prior may increase measurement trust when it is
            # uncertain, but it may never weaken the pretrained tracker's
            # learned correction of a current fused-BEV measurement.
            kalman_gain = torch.maximum(kalman_gain, learned_gain)
            if prior_variance_valid_mask is not None:
                if prior_variance_valid_mask.shape != observed_mask.shape:
                    raise ValueError(
                        "prior_variance_valid_mask must have shape [B,Q]"
                    )
                kalman_gain = torch.where(
                    prior_variance_valid_mask[..., None],
                    kalman_gain,
                    learned_gain,
                )
            spatial_gain = torch.where(
                birth_mask[..., None], torch.ones_like(kalman_gain), kalman_gain
            )
            spatial_gain = spatial_gain * observed_mask[..., None].to(
                spatial_gain.dtype
            )
        state_box = propagated_box + learned_gain * (measured_box - propagated_box)
        state_box = state_box.clone()
        state_box[..., :2] = propagated_box[..., :2] + spatial_gain * (
            measured_box[..., :2] - propagated_box[..., :2]
        )
        yaw_delta = torch.atan2(
            torch.sin(measured_box[..., 6] - propagated_box[..., 6]),
            torch.cos(measured_box[..., 6] - propagated_box[..., 6]),
        )
        state_box[..., 6] = propagated_box[..., 6] + learned_gain[..., 0] * yaw_delta
        state_velocity = propagated_velocity + spatial_gain * (
            measured_velocity - propagated_velocity
        )
        return state_box, state_velocity, spatial_gain[..., 0]
