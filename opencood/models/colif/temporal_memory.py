"""Short, variable-length temporal memory for persistent actor queries."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalQueryMemory(nn.Module):
    def __init__(self, query_dim: int):
        super().__init__()
        self.gru = nn.GRU(query_dim + 7, query_dim, batch_first=True)
        self.output_norm = nn.LayerNorm(query_dim)

    def forward(
        self,
        history_feat: torch.Tensor,
        history_box: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, query_count, memory_len, query_dim = history_feat.shape
        inputs = torch.cat([history_feat, history_box], dim=-1)
        inputs = inputs * history_mask[..., None].to(inputs.dtype)
        outputs, _ = self.gru(inputs.reshape(-1, memory_len, query_dim + 7))
        memory = outputs[:, -1].reshape(batch_size, query_count, query_dim)
        valid = history_mask.any(dim=-1)
        memory = torch.where(valid[..., None], memory, torch.zeros_like(memory))
        return self.output_norm(memory)

    @staticmethod
    def append(
        history_feat: torch.Tensor,
        history_box: torch.Tensor,
        history_mask: torch.Tensor,
        query_feat: torch.Tensor,
        box: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        next_feat = torch.cat([history_feat[..., 1:, :], query_feat[..., None, :]], dim=-2)
        next_box = torch.cat([history_box[..., 1:, :], box[..., None, :]], dim=-2)
        next_mask = torch.cat([history_mask[..., 1:], valid_mask[..., None]], dim=-1)
        return next_feat, next_box, next_mask
