"""Same-observer spatial correspondence, followed by causal BEV time encoding.

The scaled dot-product is adapted from MIT-licensed GenFlow3D's RAFT
``layers/corr.py::CorrBlock.correlation``; correlation-conv/feature-concat
reuses ``layers/update.py::SmallMotionEncoder``; masked ConvGRU adapts
``layers/update.py::ConvGRU``. No flow refinement, flow
supervision, detector or independent tracker is introduced.
"""
import math

import torch
import torch.nn as nn

from .source_temporal_encoder import SourceTemporalEncoder


class ConvGRU(nn.Module):
    """SLIM/RAFT ConvGRU equations (Copyright 2022 MBition GmbH, MIT)."""

    def __init__(self, channels):
        super().__init__()
        self.convz = nn.Conv2d(2 * channels, channels, 3, padding=1)
        self.convr = nn.Conv2d(2 * channels, channels, 3, padding=1)
        self.convq = nn.Conv2d(2 * channels, channels, 3, padding=1)

    def forward(self, hidden, observation):
        joined = torch.cat((hidden, observation), dim=1)
        update = torch.sigmoid(self.convz(joined))
        reset = torch.sigmoid(self.convr(joined))
        candidate = torch.tanh(self.convq(torch.cat((reset * hidden, observation), dim=1)))
        return (1 - update) * hidden + update * candidate


class SourceActorReadout(nn.Module):
    """History-conditioned full-memory readout, before MTR actor interaction."""

    def __init__(self, actor_dim, memory_dim, heads=8, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(actor_dim, memory_dim)
        self.attention = nn.MultiheadAttention(memory_dim, heads, dropout=dropout)
        self.output_projection = nn.Linear(memory_dim, actor_dim)
        # Fairseq scale_attn: normalize the new attention message before adding
        # it to history, not the full Source bank or MTR's existing BatchNorms.
        self.output_normalization = nn.LayerNorm(actor_dim)
        self.last_counts = {"contexts": 0, "neighbors": 0, "memory_rows": 0}

    def forward(self, features, entries, context_slots, tokens, positions, token_mask):
        from .mtr_vendor import position_encoding_utils

        if (tokens.ndim != 5 or positions.shape != (*tokens.shape[:-1], 3)
                or token_mask.shape != tokens.shape[:-1] or token_mask.dtype != torch.bool
                or tokens.shape[-1] != self.attention.embed_dim
                or features.shape[:2] != context_slots.shape
                or len(entries) != features.shape[0]):
            raise ValueError("actor Source readout requires packed contexts and full [B,A,L,P,D] memory")
        batches = context_slots.new_tensor([entry[0] for entry in entries])
        centers = context_slots.new_tensor([entry[1] for entry in entries])
        row, col = torch.where(context_slots >= 0)
        owner = context_slots[row, col]
        observed = token_mask[batches[row], owner].flatten(1).any(-1)
        row, col, owner = row[observed], col[observed], owner[observed]
        self.last_counts = {"contexts": int(row.numel()),
                            "neighbors": int((owner != centers[row]).sum()), "memory_rows": 0}
        if not row.numel():
            return features
        identities = torch.stack((batches[row], owner), dim=1)
        unique, inverse = torch.unique(identities, dim=0, return_inverse=True)
        memory = tokens[unique[:, 0], unique[:, 1]].flatten(1, 2).transpose(0, 1)
        local_xy = positions[unique[:, 0], unique[:, 1], ..., :2].flatten(1, 2).transpose(0, 1)
        mask = token_mask[unique[:, 0], unique[:, 1]].flatten(1)
        memory = torch.where(mask.T[..., None], memory, torch.zeros_like(memory))
        local_xy = torch.where(mask.T[..., None], local_xy, torch.zeros_like(local_xy))
        if not torch.isfinite(memory).all() or not torch.isfinite(local_xy).all():
            raise ValueError("observed Source memory/positions must be finite")
        # Keys and query anchors share the memory owner's current actor frame.
        # Context-conditioned history content keeps MTR's relative heading.
        key_pos = position_encoding_utils.gen_sineembed_for_position(local_xy, hidden_dim=memory.shape[-1])
        query = features.new_zeros(features.shape[0], unique.shape[0], memory.shape[-1])
        query[row, inverse] = self.input_projection(features[row, col])
        query_pos = position_encoding_utils.gen_sineembed_for_position(
            query.new_zeros(*query.shape[:-1], 2), hidden_dim=query.shape[-1])
        readout, _ = self.attention(query + query_pos, memory + key_pos, memory,
                                    key_padding_mask=~mask, need_weights=False)
        output = features.clone()
        message = self.output_normalization(self.output_projection(readout[row, inverse]))
        output[row, col] = features[row, col] + message
        self.last_counts["memory_rows"] = int(unique.shape[0])
        return output


class SourceMotionEncoder(SourceTemporalEncoder):
    """Keep [B,A,L,T,K,C] identities until within-source motion is encoded."""

    def __init__(self, feature_dim, roi_grid_size=5, pooled_grid_size=5,
                 heads=8, layers=1, dropout=0.1, frame_period_s=0.1,
                 pose_scale_m=100.0, standard_position_encoding=True,
                 feature_encoder="motion_correlation_v1"):
        nn.Module.__init__(self)
        self.feature_dim = int(feature_dim)
        self.roi_grid_size = int(roi_grid_size)
        self.pooled_grid_size = int(pooled_grid_size)
        self.frame_period_s = float(frame_period_s)
        self.pose_scale_m = float(pose_scale_m)
        self.feature_encoder = str(feature_encoder)
        if (not standard_position_encoding or self.feature_encoder != "motion_correlation_v1"
                or self.pooled_grid_size != self.roi_grid_size
                or self.feature_dim < 8 or self.feature_dim % 8
                or int(layers) < 1 or not math.isfinite(self.frame_period_s)
                or self.frame_period_s <= 0 or not math.isfinite(self.pose_scale_m)
                or self.pose_scale_m <= 0):
            raise ValueError("motion correlation requires aligned time and an unpooled spatial grid")
        self.feature_projection = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, self.feature_dim), nn.ReLU())
        # Each candidate keeps its score AND physical dx/dy; following an ROI
        # must not silently turn an actor's displacement into zero motion.
        self.correlation_encoder = nn.Sequential(
            nn.Conv2d(4 * self.pooled_tokens + 4, self.feature_dim, 1), nn.ReLU())
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, 3, padding=1), nn.ReLU())
        self.temporal_encoder = nn.ModuleList([
            ConvGRU(self.feature_dim)
            for _ in range(int(layers))])

    @staticmethod
    def correspondence(features, positions, mask, age_s, frame_period_s):
        """All spatial pairs within ONE observer; duplicate/gap frames excluded.

        [N,T,K,D] -> [N,T,K,K]. Physical offsets are metres in the final
        actor frame, not patch indices. Scores are features, not calibrated
        probabilities or velocity estimates.
        """
        count, times, patches, width = features.shape
        corr = features.new_zeros(count, times, patches, patches)
        offsets = features.new_zeros(count, times, patches, patches, 2)
        pair_mask = torch.zeros_like(corr, dtype=torch.bool)
        dt = age_s[:, :-1] - age_s[:, 1:]
        consecutive = (dt > 1e-6) & (dt <= float(frame_period_s) + 1e-4)
        valid = mask.clone()
        # Only an observed predecessor can make this capture a duplicate.
        # Invalid predecessors have sanitized timestamps, not real captures.
        valid[:, 1:] &= (~mask[:, :-1].any(-1) | (dt > 1e-6))[..., None]
        if times > 1:
            pair_mask[:, 1:] = (valid[:, 1:, :, None] & mask[:, :-1, None, :]
                                & consecutive[..., None, None])
            corr[:, 1:] = torch.matmul(features[:, 1:], features[:, :-1].transpose(-1, -2)) / math.sqrt(width)
            offsets[:, 1:] = positions[:, 1:, :, None, :2] - positions[:, :-1, None, :, :2]
        corr = torch.where(pair_mask, corr, torch.zeros_like(corr))
        offsets = torch.where(pair_mask[..., None], offsets, torch.zeros_like(offsets))
        return corr, offsets, pair_mask, valid

    def forward(self, tokens, positions, token_mask, age_s, source_to_receiver_xyyaw, encode_age=True):
        if (tokens.ndim != 6 or tokens.shape[-2:] != (self.pooled_tokens, self.feature_dim)
                or positions.shape != (*tokens.shape[:-1], 3)
                or token_mask.shape != tokens.shape[:-1] or token_mask.dtype != torch.bool
                or age_s.shape != tokens.shape[:4]):
            raise ValueError("motion input must be tokens [B,A,L,T,K,C], xyz, bool mask and ages")
        batch, actors, sources, times, patches, channels = tokens.shape
        pose = source_to_receiver_xyyaw
        if pose.shape == (batch, sources, times, 3):
            pose = pose[:, None].expand(-1, actors, -1, -1, -1)
        if pose.shape != (batch, actors, sources, times, 3):
            raise ValueError("motion observer pose must be [B,A,L,T,3] or [B,L,T,3]")
        valid_time = token_mask.any(-1)
        if (not torch.isfinite(age_s[valid_time]).all() or (age_s[valid_time] < -1e-6).any()
                or not torch.isfinite(tokens[token_mask]).all()
                or not torch.isfinite(positions[token_mask]).all()
                or not torch.isfinite(pose).all()):
            raise ValueError("motion evidence/geometry/time must be finite and causal")
        grid = self.pooled_grid_size
        count = batch * actors * sources
        mask = token_mask.reshape(count, times, patches)
        # Skip unused actor/source slots before materializing cost volumes.
        nonempty = mask.flatten(1).any(-1)
        result = tokens.new_zeros(count, times, patches, channels)
        result_mask = torch.zeros_like(mask)
        if bool(nonempty.any()):
            selected = torch.where(token_mask[..., None], tokens, torch.zeros_like(tokens))
            selected = selected.reshape(count, times, patches, channels)[nonempty]
            selected_mask = mask[nonempty]
            size = selected.shape[0]
            image = selected.reshape(size * times, grid, grid, channels).permute(0, 3, 1, 2)
            features = self.feature_projection(image).permute(0, 2, 3, 1).reshape(size, times, patches, channels)
            features = torch.where(selected_mask[..., None], features, torch.zeros_like(features))
            ages = torch.where(valid_time, age_s, torch.zeros_like(age_s)).reshape(count, times)[nonempty]
            xyz = positions.reshape(count, times, patches, 3)[nonempty]
            corr, offsets, pairs, valid = self.correspondence(features, xyz, selected_mask, ages, self.frame_period_s)
            geometry = pose.reshape(count, times, 3)[nonempty]
            metadata_age = ages if encode_age else torch.zeros_like(ages)
            metadata = torch.cat(((metadata_age / self.frame_period_s)[..., None],
                                   geometry[..., :2] / self.pose_scale_m,
                                   geometry[..., 2:] / math.pi), -1)
            candidates = torch.cat((corr, offsets[..., 0], offsets[..., 1],
                                    pairs.to(tokens),
                                    metadata[:, :, None].expand(-1, -1, patches, -1)), -1)
            cost_image = candidates.reshape(size * times, grid, grid, -1).permute(0, 3, 1, 2)
            cost_feature = self.correlation_encoder(cost_image)
            appearance = features.reshape(size * times, grid, grid, channels).permute(0, 3, 1, 2)
            encoded = self.spatial_encoder(appearance + cost_feature)
            volume_mask = valid.reshape(size, times, grid, grid)[:, :, None]
            encoded = encoded.reshape(size, times, channels, grid, grid)
            encoded = torch.where(volume_mask, encoded, torch.zeros_like(encoded))
            for layer in self.temporal_encoder:
                hidden = encoded.new_zeros(size, channels, grid, grid)
                outputs = []
                for time in range(times):
                    # Missing/repeated captures do not invent a state update.
                    hidden = torch.where(volume_mask[:, time], layer(hidden, encoded[:, time]), hidden)
                    outputs.append(torch.where(volume_mask[:, time], hidden, torch.zeros_like(hidden)))
                encoded = torch.stack(outputs, dim=1)
            result[nonempty] = encoded.permute(0, 1, 3, 4, 2).reshape(size, times, patches, channels)
            result_mask[nonempty] = valid
        result = result.reshape_as(tokens)
        result_mask = result_mask.reshape_as(token_mask)
        # Keep observations verbatim; learned motion supplements them rather
        # than replacing them. Source/time/patch axes survive until readout.
        observations = torch.where(result_mask[..., None], tokens, torch.zeros_like(tokens))
        return {"tokens": torch.cat((observations, result), -1),
                "positions": positions, "token_mask": result_mask}
