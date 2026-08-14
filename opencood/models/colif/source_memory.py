"""Per-query, per-source temporal evidence memory."""

from __future__ import annotations

import torch
import torch.nn as nn


class SourceMemoryUpdater(nn.Module):
    def __init__(self, query_dim, memory_dim, max_sources, memory_len):
        super().__init__()
        self.max_sources = max_sources
        self.memory_len = memory_len
        self.write = nn.Linear(query_dim + 9, memory_dim)
        self.query = nn.Linear(query_dim, memory_dim)
        self.readout = nn.Linear(memory_dim, query_dim)

    def forward(self, state, assignment, aligned):
        previous_memory = state.source_memory
        previous_mask = state.source_memory_mask
        previous_time = state.source_memory_time
        observation_value = torch.cat(
            (
                aligned["feat"], aligned["box"], aligned["score"].unsqueeze(-1),
                aligned["delta_t_s"].unsqueeze(-1),
            ),
            dim=-1,
        )
        write_tokens = self.write(observation_value)
        updated_memories = []
        updated_masks = []
        updated_times = []
        for source in range(self.max_sources):
            source_mask = (aligned["source"] == source) & aligned["mask"]
            weights = assignment * source_mask.unsqueeze(1).to(assignment.dtype)
            weight_sum = weights.sum(-1, keepdim=True)
            token = torch.einsum("bqo,bod->bqd", weights, write_tokens)
            token = token / weight_sum.clamp_min(1e-6)
            has_write = weight_sum.squeeze(-1) > 1e-4
            capture_time = aligned.get(
                "capture_time_s", torch.zeros_like(aligned["score"])
            )
            write_time = torch.einsum(
                "bqo,bo->bq", weights, capture_time
            ) / weight_sum.squeeze(-1).clamp_min(1e-6)
            shifted = torch.cat(
                (token.unsqueeze(2), previous_memory[:, :, source, :-1]), dim=2
            )
            shifted_mask = torch.cat(
                (has_write.unsqueeze(2), previous_mask[:, :, source, :-1]), dim=2
            )
            updated_memories.append(torch.where(
                has_write.unsqueeze(-1).unsqueeze(-1), shifted,
                previous_memory[:, :, source],
            ))
            updated_masks.append(torch.where(
                has_write.unsqueeze(-1), shifted_mask,
                previous_mask[:, :, source],
            ))
            shifted_time = torch.cat(
                (write_time.unsqueeze(2), previous_time[:, :, source, :-1]), dim=2
            )
            updated_times.append(torch.where(
                has_write.unsqueeze(-1), shifted_time,
                previous_time[:, :, source],
            ))

        memory = torch.stack(updated_memories, dim=2)
        memory_mask = torch.stack(updated_masks, dim=2)
        memory_time = torch.stack(updated_times, dim=2)

        flat_memory = memory.flatten(2, 3)
        flat_mask = memory_mask.flatten(2, 3)
        query = self.query(state.query_feat)
        logits = torch.einsum("bqd,bqld->bql", query, flat_memory)
        logits = logits / (query.shape[-1] ** 0.5)
        masked_logits = logits.masked_fill(~flat_mask, -1e4)
        maximum = masked_logits.max(-1, keepdim=True).values
        attention = torch.exp(masked_logits - maximum) * flat_mask.to(logits.dtype)
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-6)
        read = torch.einsum("bql,bqld->bqd", attention, flat_memory)
        return memory, memory_mask, memory_time, self.readout(read)
