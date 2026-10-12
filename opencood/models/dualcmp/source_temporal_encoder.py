"""Compact causal per-source evidence history for motion forecasting."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .source_evidence_policy import FEATURE_ENCODERS


class SourceTemporalEncoder(nn.Module):
    """Pool actor-local source grids, then encode each source through time.

    Inputs keep source identity until the intention decoder boundary:
    ``[B, A, L, T, K, C]``.  Temporal attention is independent for every
    actor/source/spatial token, so no cross-source averaging happens here.
    """

    def __init__(self, feature_dim, roi_grid_size=5, pooled_grid_size=2,
                 heads=8, layers=1, dropout=0.1, frame_period_s=0.1,
                 pose_scale_m=100.0, standard_position_encoding=False,
                 feature_encoder="legacy_additive_v1"):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.roi_grid_size = int(roi_grid_size)
        self.pooled_grid_size = int(pooled_grid_size)
        self.frame_period_s = float(frame_period_s)
        self.pose_scale_m = float(pose_scale_m)
        self.standard_position_encoding = bool(standard_position_encoding)
        self.feature_encoder = str(feature_encoder)
        if self.feature_encoder not in FEATURE_ENCODERS:
            raise ValueError("unknown Source feature encoder")
        if self.feature_encoder == "motion_correlation_v1":
            raise ValueError("motion correlation must use SourceMotionEncoder, not legacy temporal attention")
        if self.feature_encoder != "legacy_additive_v1" and not self.standard_position_encoding:
            raise ValueError("new feature encoders require the explicit aligned protocol")
        self.qk_position_only = self.feature_encoder in ("temporal_qk_v1", "factorized_qk_v1")
        if self.roi_grid_size < 1 or self.pooled_grid_size < 1:
            raise ValueError("source temporal grid sizes must be positive")
        if self.pooled_grid_size > self.roi_grid_size:
            raise ValueError("pooled source grid cannot exceed the ROI grid")
        if self.feature_dim % int(heads):
            raise ValueError("source temporal feature_dim must divide heads")
        if (int(layers) < 1 or not math.isfinite(self.frame_period_s)
                or self.frame_period_s <= 0 or not math.isfinite(self.pose_scale_m)
                or self.pose_scale_m <= 0):
            raise ValueError("invalid source temporal encoder configuration")
        self.metadata_projection = nn.Sequential(
            nn.Linear(4, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=int(heads),
            dim_feedforward=2 * self.feature_dim,
            dropout=float(dropout),
            batch_first=True,
        )
        self.spatial_encoder = None
        if self.qk_position_only:
            self.temporal_encoder = nn.ModuleList([
                self._position_layer(int(heads), float(dropout)) for _ in range(int(layers))])
            if self.feature_encoder == "factorized_qk_v1":
                self.spatial_encoder = nn.ModuleList([self._position_layer(int(heads), float(dropout))])
        else:
            self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=int(layers))
            if self.feature_encoder == "factorized_additive_v1":
                spatial_layer = nn.TransformerEncoderLayer(
                    d_model=self.feature_dim, nhead=int(heads),
                    dim_feedforward=2 * self.feature_dim, dropout=float(dropout), batch_first=True)
                self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=1)

    def _position_layer(self, heads, dropout):
        # Reuse MTR's actual post-norm Transformer and its Q/K-only positions.
        # The bundled global MHA lacks the local-attention keyword arguments
        # unconditionally sent by that layer. Adapt only those unused None
        # arguments; do not modify the vendored implementation or its math.
        from . import mtr_vendor  # Establish the pinned local MTR import root.
        from mtr.models_v2v4real.utils.transformer.transformer_encoder_layer import TransformerEncoderLayer
        from mtr.models_v2v4real.utils.transformer.multi_head_attention import MultiheadAttention

        class DenseAttention(MultiheadAttention):
            def forward(self, query, key, value, **kwargs):
                for name in ("index_pair", "query_batch_cnt", "key_batch_cnt", "index_pair_batch"):
                    if kwargs.pop(name, None) is not None:
                        raise ValueError("Source dense attention cannot consume local KNN indices")
                return super().forward(query, key, value, **kwargs)

        layer = TransformerEncoderLayer(self.feature_dim, heads,
            dim_feedforward=2 * self.feature_dim, dropout=dropout,
            normalize_before=False, use_local_attn=False)
        layer.self_attn = DenseAttention(self.feature_dim, heads, dropout=dropout)
        return layer

    def _attention(self, sequence, valid, position, layers):
        nonempty = valid.any(dim=-1)
        output = sequence * 0.0
        if bool(nonempty.any()):
            selected = sequence[nonempty]
            if self.qk_position_only:
                selected = selected.transpose(0, 1)
                selected_pos = position[nonempty].transpose(0, 1)
                for layer in layers:
                    selected = layer(selected, src_key_padding_mask=~valid[nonempty], pos=selected_pos)
                selected = selected.transpose(0, 1)
            else:
                selected = layers(selected, src_key_padding_mask=~valid[nonempty])
            output[nonempty] = selected
        return torch.where(valid[..., None], output, torch.zeros_like(output))

    @property
    def pooled_tokens(self):
        return self.pooled_grid_size * self.pooled_grid_size

    def pool_frame(self, tokens, positions, token_mask):
        """Mask-aware adaptive pooling from a square ROI to ``K`` tokens."""
        if tokens.ndim != 5 or tokens.shape[-1] != self.feature_dim:
            raise ValueError("tokens must have shape [B,A,L,P,C]")
        if positions.shape != (*tokens.shape[:-1], 3):
            raise ValueError("positions must have shape [B,A,L,P,3]")
        if token_mask.shape != tokens.shape[:-1] or token_mask.dtype != torch.bool:
            raise ValueError("token_mask must be bool [B,A,L,P]")
        expected = self.roi_grid_size * self.roi_grid_size
        if tokens.shape[-2] != expected:
            raise ValueError("source ROI token count disagrees with roi_grid_size")

        leading = tokens.shape[:3]
        count = leading[0] * leading[1] * leading[2]
        grid = self.roi_grid_size
        pooled = self.pooled_grid_size
        weights = token_mask.reshape(count, 1, grid, grid).to(tokens.dtype)

        def masked_pool(values):
            channels = values.shape[-1]
            image = values.reshape(count, grid, grid, channels).permute(0, 3, 1, 2)
            numerator = F.adaptive_avg_pool2d(
                image * weights, (pooled, pooled))
            denominator = F.adaptive_avg_pool2d(
                weights, (pooled, pooled))
            output = numerator / denominator.clamp_min(1e-6)
            output = output.permute(0, 2, 3, 1).reshape(
                *leading, self.pooled_tokens, channels)
            return output, denominator

        pooled_tokens, denominator = masked_pool(tokens)
        pooled_positions, _ = masked_pool(positions.to(tokens))
        pooled_mask = (denominator > 0).reshape(
            *leading, self.pooled_tokens)
        pooled_tokens = torch.where(
            pooled_mask[..., None], pooled_tokens,
            torch.zeros_like(pooled_tokens))
        pooled_positions = torch.where(
            pooled_mask[..., None], pooled_positions,
            torch.zeros_like(pooled_positions))
        return pooled_tokens, pooled_positions, pooled_mask

    @staticmethod
    def same_runtime_mask(runtime_ids, final_runtime_ids):
        """Keep history only when the slot still owns the same runtime actor."""
        if runtime_ids.ndim != 3 or final_runtime_ids.shape != runtime_ids.shape[:2]:
            raise ValueError("runtime ids must have [B,A,T] and [B,A] shapes")
        return (
            (runtime_ids == final_runtime_ids[:, :, None])
            & (final_runtime_ids[:, :, None] >= 0)
        )

    def forward(self, tokens, positions, token_mask, age_s,
                source_to_receiver_xyyaw):
        """Encode compact source histories without mixing CAV identities."""
        if tokens.ndim != 6 or tokens.shape[-1] != self.feature_dim:
            raise ValueError("tokens must have shape [B,A,L,T,K,C]")
        if positions.shape != (*tokens.shape[:-1], 3):
            raise ValueError("positions must match tokens with xyz coordinates")
        if token_mask.shape != tokens.shape[:-1] or token_mask.dtype != torch.bool:
            raise ValueError("token_mask must be bool [B,A,L,T,K]")
        if age_s.shape != tokens.shape[:4]:
            raise ValueError("age_s must have shape [B,A,L,T]")
        shared_pose_shape = (tokens.shape[0], tokens.shape[2], tokens.shape[3], 3)
        actor_pose_shape = (*tokens.shape[:4], 3)
        if source_to_receiver_xyyaw.shape not in (shared_pose_shape, actor_pose_shape):
            raise ValueError("source pose metadata must have shape [B,L,T,3] or [B,A,L,T,3]")
        valid_time = token_mask.any(dim=-1)
        if not torch.isfinite(age_s[valid_time]).all() or torch.any(
                age_s[valid_time] < -1e-6):
            raise ValueError("valid source ages must be finite and non-negative")
        if not torch.isfinite(source_to_receiver_xyyaw).all():
            raise ValueError("source pose metadata must be finite")

        batch, actors, sources, times, patches, channels = tokens.shape
        pose = source_to_receiver_xyyaw
        if pose.ndim == 4:
            pose = pose[:, None].expand(-1, actors, -1, -1, -1)
        safe_age = torch.where(valid_time, age_s, torch.zeros_like(age_s))
        metadata = torch.cat(
            (
                (safe_age / self.frame_period_s)[..., None],
                pose[..., :2] / self.pose_scale_m,
                pose[..., 2:3] / math.pi,
            ),
            dim=-1,
        )
        metadata = self.metadata_projection(metadata)[..., None, :]
        encoded = tokens + metadata
        attention_position = metadata.expand_as(tokens)
        if self.standard_position_encoding:
            # Reuse MTR/DAB-DETR sine positions, not a learned MLP substitute
            # for temporal attention. Spatial positions now encode real motion
            # in the final actor frame, including historical displacement.
            from .mtr_vendor import position_encoding_utils
            temporal_position = torch.stack((safe_age / self.frame_period_s,
                                             torch.zeros_like(safe_age)), dim=-1)
            time_encoding = position_encoding_utils.gen_sineembed_for_position(
                temporal_position.reshape(-1, 1, 2), hidden_dim=channels
            ).reshape(batch, actors, sources, times, channels)
            spatial_encoding = position_encoding_utils.gen_sineembed_for_position(
                positions[..., :2].reshape(-1, 1, 2), hidden_dim=channels
            ).reshape_as(tokens)
            encoded = encoded + time_encoding[..., None, :] + spatial_encoding
            attention_position = attention_position + time_encoding[..., None, :] + spatial_encoding

        if self.qk_position_only:
            encoded = tokens
        if self.spatial_encoder is not None:
            spatial_shape = (batch * actors * sources * times, patches, channels)
            encoded = self._attention(encoded.reshape(spatial_shape),
                token_mask.reshape(spatial_shape[:2]),
                attention_position.reshape(spatial_shape), self.spatial_encoder).reshape_as(tokens)

        sequence = encoded.permute(0, 1, 2, 4, 3, 5).reshape(
            batch * actors * sources * patches, times, channels)
        valid = token_mask.permute(0, 1, 2, 4, 3).reshape(
            batch * actors * sources * patches, times)
        sequence = torch.where(
            valid[..., None], sequence, torch.zeros_like(sequence))
        if self.standard_position_encoding:
            # MTR polyline encoders process only non-empty rows. Do the same
            # for aligned Source: masked unused slots must not consume the
            # Transformer activation budget or be fake valid observations.
            position_sequence = attention_position.permute(0, 1, 2, 4, 3, 5).reshape_as(sequence)
            output = self._attention(sequence, valid, position_sequence, self.temporal_encoder)
        else:
            safe_valid = valid.clone()
            empty = ~safe_valid.any(dim=-1)
            safe_valid[empty, 0] = True
            output = self.temporal_encoder(sequence, src_key_padding_mask=~safe_valid)
        output = torch.where(valid[..., None], output, torch.zeros_like(output))
        output = output.reshape(
            batch, actors, sources, patches, times, channels
        ).permute(0, 1, 2, 4, 3, 5)
        return {
            "tokens": output,
            "positions": positions,
            "token_mask": token_mask,
        }
