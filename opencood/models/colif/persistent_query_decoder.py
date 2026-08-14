"""MOTR-derived 3D persistent query decoder over CMP fused BEV."""

from __future__ import annotations

import torch
import torch.nn as nn

from opencood.models.colif.motr_vendor import DeformableTransformer
from opencood.models.colif.query_geometry import (
    MetricBoxCoder,
    inverse_sigmoid,
    normalized_sine_embedding,
)


class SinePositionEmbedding2D(nn.Module):
    def __init__(self, dim: int, temperature: float = 10000.0):
        super().__init__()
        if dim % 4:
            raise ValueError("position dimension must be divisible by four")
        self.dim = dim
        self.temperature = temperature

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = feature.shape
        y = torch.linspace(0.0, 1.0, height, device=feature.device, dtype=feature.dtype)
        x = torch.linspace(0.0, 1.0, width, device=feature.device, dtype=feature.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        xy = torch.stack((grid_x, grid_y), dim=-1)
        pos = normalized_sine_embedding(xy, self.dim).permute(2, 0, 1)
        return pos.unsqueeze(0).expand(batch, -1, -1, -1)


class QueryInteractionUpdate(nn.Module):
    """Batch-safe content update retaining MOTR QIM's self-attention/FFN form."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, prior: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        query = prior + observed
        attended = self.attention(query, query, observed, need_weights=False)[0]
        value = self.norm1(observed + attended)
        return self.norm2(prior + value + self.ffn(value))


class PersistentQueryDecoder(nn.Module):
    def __init__(self, args, lidar_range):
        super().__init__()
        dim = int(args.get("hidden_dim", 256))
        self.dim = dim
        self.num_queries = int(args.get("num_queries", 128))
        self.box_coder = MetricBoxCoder(
            lidar_range, args.get("default_hwl", [1.56, 1.6, 3.9])
        )
        self.input_projection = nn.Conv2d(
            int(args.get("input_dim", 256)), dim, kernel_size=1
        )
        self.position_embedding = SinePositionEmbedding2D(dim)
        self.transformer = DeformableTransformer(
            d_model=dim,
            nhead=int(args.get("nheads", 8)),
            num_encoder_layers=int(args.get("enc_layers", 2)),
            num_decoder_layers=int(args.get("dec_layers", 3)),
            dim_feedforward=int(args.get("dim_feedforward", 1024)),
            dropout=float(args.get("dropout", 0.1)),
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=1,
            dec_n_points=int(args.get("dec_n_points", 4)),
            enc_n_points=int(args.get("enc_n_points", 4)),
            two_stage=False,
            decoder_self_cross=True,
        )
        self.newborn_content = nn.Embedding(self.num_queries, dim)
        self.slot_position = nn.Embedding(self.num_queries, dim)
        self.newborn_reference = nn.Embedding(self.num_queries, 2)
        self.reference_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim)
        )
        self.class_head = nn.Linear(dim, 1)
        self.box_head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, 8)
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, 2)
        )
        self.query_update = QueryInteractionUpdate(
            dim, int(args.get("nheads", 8)), float(args.get("dropout", 0.1))
        )
        prior_probability = float(args.get("prior_probability", 0.01))
        bias = -torch.log(torch.tensor((1.0 - prior_probability) / prior_probability))
        nn.init.constant_(self.class_head.bias, float(bias))
        nn.init.zeros_(self.box_head[-1].weight)
        nn.init.zeros_(self.box_head[-1].bias)
        # A zero sine/cosine pair makes atan2's derivative undefined.
        # Initialize every query at yaw=0 with (sin, cos)=(0, 1).
        with torch.no_grad():
            self.box_head[-1].bias[7] = 1.0
        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)

    def initial_query_content(self, batch_size: int) -> torch.Tensor:
        return self.newborn_content.weight.unsqueeze(0).expand(batch_size, -1, -1)

    def initial_reference_xy(self, batch_size: int) -> torch.Tensor:
        return self.newborn_reference.weight.sigmoid().unsqueeze(0).expand(
            batch_size, -1, -1
        )

    def initial_metric_boxes(self, batch_size: int) -> torch.Tensor:
        ref = self.initial_reference_xy(batch_size)
        raw = torch.zeros(
            batch_size,
            self.num_queries,
            8,
            device=ref.device,
            dtype=ref.dtype,
        )
        raw[..., 6] = 0.0
        raw[..., 7] = 1.0
        return self.box_coder.decode(raw, ref)

    def forward(
        self,
        fused_bev: torch.Tensor,
        query_content: torch.Tensor,
        reference_box: torch.Tensor,
    ) -> dict:
        # The pinned MOTR CUDA operator has a float32 kernel only.  Keep this
        # localized island outside AMP; the cast still propagates gradients
        # back into the mixed-precision CoBEVT feature.
        with torch.cuda.amp.autocast(enabled=False):
            return self._forward_float(
                fused_bev.float(), query_content.float(), reference_box.float()
            )

    def _forward_float(
        self,
        fused_bev: torch.Tensor,
        query_content: torch.Tensor,
        reference_box: torch.Tensor,
    ) -> dict:
        source = self.input_projection(fused_bev)
        batch, _, height, width = source.shape
        if query_content.shape != (batch, self.num_queries, self.dim):
            raise ValueError("query_content shape disagrees with decoder config")
        reference_xy = self.box_coder.metric_xy_to_normalized(reference_box[..., :2])
        query_position = self.reference_mlp(
            normalized_sine_embedding(reference_xy, self.dim)
        ) + self.slot_position.weight.unsqueeze(0)
        query_embed = torch.cat([query_position, query_content], dim=-1)
        mask = torch.zeros(batch, height, width, dtype=torch.bool, device=source.device)
        position = self.position_embedding(source)
        hs, initial_ref, intermediate_ref, _, _ = self.transformer(
            [source],
            [mask],
            [position],
            query_embed=query_embed,
            ref_pts=inverse_sigmoid(reference_xy),
        )
        del initial_ref, intermediate_ref
        layer_logits = self.class_head(hs).squeeze(-1)
        layer_raw_boxes = self.box_head(hs)
        layer_boxes = torch.stack(
            [
                self.box_coder.decode_from_reference_box(raw, reference_box)
                for raw in layer_raw_boxes
            ],
            dim=0,
        )
        layer_velocity = self.velocity_head(hs)
        query_feat = self.query_update(query_content, hs[-1])
        return {
            "exist_logits": layer_logits[-1],
            "boxes": layer_boxes[-1],
            "velocity": layer_velocity[-1],
            "query_feat": query_feat,
            "decoder_embedding": hs[-1],
            "aux_exist_logits": layer_logits[:-1],
            "aux_boxes": layer_boxes[:-1],
            "aux_velocity": layer_velocity[:-1],
        }
