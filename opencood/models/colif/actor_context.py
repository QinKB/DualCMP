"""Interaction context over active persistent actor queries."""

import torch
import torch.nn as nn


class ActorContextEncoder(nn.Module):
    def __init__(self, query_dim=256, layers=2, heads=8):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=query_dim, nhead=heads, dim_feedforward=query_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.position = nn.Linear(7, query_dim)

    def forward(self, query, boxes, active_mask):
        result = query.clone()
        encoded_input = query + self.position(boxes)
        for batch_index in range(query.shape[0]):
            indices = torch.nonzero(active_mask[batch_index], as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            active = encoded_input[batch_index, indices].unsqueeze(0)
            result[batch_index, indices] = self.encoder(active).squeeze(0)
        return result
