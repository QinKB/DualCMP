"""Fixed-slot multimodal GMM forecast head."""

import math

import torch
import torch.nn as nn


class MTRStyleGMMForecastHead(nn.Module):
    def __init__(self, query_dim=256, modes=6, future_frames=50, token_dim=256):
        super().__init__()
        self.modes = modes
        self.future_frames = future_frames
        self.mode_embedding = nn.Parameter(torch.randn(modes, query_dim) * 0.02)
        self.score_head = nn.Linear(query_dim, 1)
        self.traj_head = nn.Linear(query_dim, future_frames * 5)
        self.token_head = nn.Linear(query_dim, token_dim)
        # A random log-std near the lower clamp (-5) turns ordinary 30--50 m
        # early-training errors into million-scale NLL.  Start from a broad,
        # unbiased Gaussian and let every parameter remain learnable.
        nn.init.zeros_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)
        nn.init.zeros_(self.traj_head.weight)
        initial_trajectory = torch.zeros(future_frames, 5)
        initial_trajectory[:, 2:4] = math.log(5.0)
        with torch.no_grad():
            self.traj_head.bias.copy_(initial_trajectory.reshape(-1))

    def forward(self, context, active_mask):
        mode_feature = context.unsqueeze(2) + self.mode_embedding.view(1, 1, self.modes, -1)
        scores = self.score_head(mode_feature).squeeze(-1)
        raw = self.traj_head(mode_feature).view(
            context.shape[0], context.shape[1], self.modes, self.future_frames, 5
        )
        trajectories = torch.cat(
            (
                raw[..., :2], raw[..., 2:4].clamp(-5.0, 3.0),
                torch.tanh(raw[..., 4:5]) * 0.95,
            ), dim=-1,
        )
        tokens = self.token_head(mode_feature)
        scores = scores.masked_fill(~active_mask.unsqueeze(-1), -1e4)
        trajectories = trajectories * active_mask[..., None, None, None].to(
            trajectories.dtype
        )
        tokens = tokens * active_mask[..., None, None].to(tokens.dtype)
        return scores, trajectories, tokens
