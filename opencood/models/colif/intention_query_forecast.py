"""MTR intention-query forecasting with source-preserving actor evidence.

The decoder keeps MTR's fixed, data-clustered endpoint intentions as its only
motion-mode prior.  CoLIF contributes causal persistent-query history and
per-CAV actor-aligned evidence; it does not manufacture velocity-scaled
trajectory anchors.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from easydict import EasyDict

from opencood.models.colif.mtr_vendor import (
    MTRDecoder,
    batch_nms,
    position_encoding_utils,
)


class IntentionQueryForecastHead(nn.Module):
    """Forecast persistent actors through MTR's 64 fixed endpoint queries.

    The public output retains six NMS-selected trajectories.  Training keeps
    all 64 queries at every decoder layer and chooses the positive query from
    the fixed V2V4Real endpoint intentions, exactly as in MTR.
    """

    def __init__(self, query_dim: int, args: dict):
        super().__init__()
        self.query_dim = int(query_dim)
        self.history_frames = int(args.get("history_frames", 11))
        self.future_frames = int(args.get("future_frames", 50))
        self.frame_period_s = float(args.get("frame_period_s", 0.1))
        self.min_forecast_history_steps = int(
            args.get("min_forecast_history_steps", 2)
        )
        if self.frame_period_s <= 0.0:
            raise ValueError("frame_period_s must be positive")
        if not 2 <= self.min_forecast_history_steps <= self.history_frames:
            raise ValueError("min_forecast_history_steps is outside history range")

        project_root = Path(__file__).resolve().parents[3]
        decoder_config = EasyDict(
            OBJECT_TYPE=["TYPE_VEHICLE"],
            NUM_FUTURE_FRAMES=self.future_frames,
            NUM_MOTION_MODES=int(args.get("modes", 6)),
            USE_PLACE_HOLDER=False,
            # Formal runs must use the clustered V2V4Real endpoint prior.
            INTENTION_RANDOM_INIT=False,
            INTENTION_POINTS_FILE=str(
                Path(
                    args.get(
                        "intention_points_file",
                        project_root
                        / "preprocessed_data/v2v4real/"
                        "v2v4real_cluster_64_center_dict.pkl",
                    )
                ).resolve()
            ),
            D_MODEL=int(args.get("d_model", 512)),
            NUM_DECODER_LAYERS=int(args.get("decoder_layers", 6)),
            NUM_ATTN_HEAD=int(args.get("attention_heads", 8)),
            MAP_D_MODEL=int(args.get("map_d_model", 256)),
            DROPOUT_OF_ATTN=float(args.get("dropout", 0.1)),
            LOSS_WEIGHTS=EasyDict(cls=1.0, reg=1.0, vel=0.5),
            NMS_DIST_THRESH=float(args.get("nms_distance", 2.5)),
        )
        self.decoder = MTRDecoder(in_channels=self.query_dim, config=decoder_config)
        self.modes = int(decoder_config.NUM_MOTION_MODES)
        self.decoder_layers = int(decoder_config.NUM_DECODER_LAYERS)
        self.decoder_dim = int(decoder_config.D_MODEL)
        self.raw_modes = int(
            getattr(
                self.decoder,
                self.decoder.intention_points["TYPE_VEHICLE"],
            ).shape[0]
        )
        if self.raw_modes != 64:
            raise ValueError("V2V4Real MTR decoder must expose 64 intention points")
        if self.modes > self.raw_modes:
            raise ValueError("output modes cannot exceed intention query count")

        # Causal online history is represented by a standard learned temporal
        # encoder over normalized observed boxes; no hand-built velocity,
        # dispersion or scale hypothesis enters the decoder.
        self.temporal_history_enabled = bool(args.get("temporal_history", True))
        lidar_range = torch.as_tensor(
            args.get("lidar_range", [-102.4, -38.4, -3.0, 102.4, 38.4, 1.0]),
            dtype=torch.float32,
        )
        if lidar_range.shape != (6,):
            raise ValueError("lidar_range must contain six values")
        self.register_buffer(
            "history_position_scale",
            (lidar_range[3:6] - lidar_range[:3]).mul(0.5).clamp_min(1e-3),
        )
        history_size_scale = torch.as_tensor(
            args.get("history_size_scale", [3.0, 3.0, 8.0]),
            dtype=torch.float32,
        )
        if history_size_scale.shape != (3,) or not bool(
            (history_size_scale > 0).all()
        ):
            raise ValueError("history_size_scale must contain three positive values")
        self.register_buffer("history_size_scale", history_size_scale)
        self.query_norm = nn.LayerNorm(self.query_dim)
        self.history_input_projection = nn.Linear(9, self.query_dim)
        self.history_position = nn.Parameter(
            torch.randn(self.history_frames, self.query_dim) * 0.02
        )
        dropout = float(args.get("dropout", 0.1))
        history_layer = nn.TransformerEncoderLayer(
            d_model=self.query_dim,
            nhead=int(args.get("history_encoder_heads", 8)),
            dim_feedforward=int(args.get("history_encoder_ffn", self.query_dim * 4)),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(
            history_layer,
            num_layers=int(args.get("history_encoder_layers", 2)),
        )
        self.history_norm = nn.LayerNorm(self.query_dim)

        # MTR's object decoder supplies standard actor interaction.  DualCMP
        # then reads two *independent* actor-local memories: a fused consensus
        # RoI and source-preserved RoIs.  Both memories share this one linear
        # query_dim -> MTR-d_model projection: it changes attention feature
        # width only, and does not mix the two representations.
        source_heads = int(args.get("source_attention_heads", 8))
        if self.decoder_dim % source_heads:
            raise ValueError("source_attention_heads must divide MTR d_model")
        source_ffn = int(args.get("source_attention_ffn", self.decoder_dim * 4))
        self.source_token_norm = nn.LayerNorm(self.query_dim, elementwise_affine=False)
        self.attention_memory_projection = nn.Linear(self.query_dim, self.decoder_dim)
        # The two reads are deliberately parameter-shared.  Thus fused-only,
        # source-only and dual runs have identical trainable parameter counts;
        # only their available memory differs.
        self.attention_query_norms = nn.ModuleList(
            [nn.LayerNorm(self.decoder_dim) for _ in range(self.decoder_layers)]
        )
        self.shared_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.decoder_dim, source_heads, dropout=dropout,
                    bias=False, batch_first=True,
                )
                for _ in range(self.decoder_layers)
            ]
        )
        self.attention_dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(self.decoder_layers)]
        )
        self.attention_output_norms = nn.ModuleList(
            [nn.LayerNorm(self.decoder_dim) for _ in range(self.decoder_layers)]
        )
        self.dual_ffn_norms = nn.ModuleList(
            [nn.LayerNorm(self.decoder_dim) for _ in range(self.decoder_layers)]
        )
        self.dual_ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.decoder_dim, source_ffn),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(source_ffn, self.decoder_dim),
                )
                for _ in range(self.decoder_layers)
            ]
        )
        self.dual_ffn_dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(self.decoder_layers)]
        )
        self.latent_projection = nn.Linear(self.decoder_dim, self.query_dim)
        self.enabled = bool(args.get("enabled", True))

        unused = [
            self.decoder.in_proj_map,
            self.decoder.map_decoder_layers,
        ]
        if self.decoder.map_query_content_mlps is not None:
            unused.extend(
                [self.decoder.map_query_content_mlps, self.decoder.map_query_embed_mlps]
            )
        self._unused_map_modules = unused
        self.freeze_unused_map_parameters()

    def freeze_unused_map_parameters(self) -> None:
        """Freeze MTR map branches that V2V4Real's map-free path never reads."""

        for module in self._unused_map_modules:
            for parameter in module.parameters():
                parameter.requires_grad = False

    def freeze_inactive_parameters(
        self, actor_cooperative_fusion_enabled: bool = True
    ) -> None:
        """Keep DDP traversal aligned with the selected controlled ablation."""

        if not self.temporal_history_enabled:
            for module in (self.history_input_projection, self.history_encoder, self.history_norm):
                for parameter in module.parameters():
                    parameter.requires_grad = False
            self.history_position.requires_grad = False
        if not actor_cooperative_fusion_enabled:
            for module in (
                self.source_token_norm,
                self.attention_memory_projection,
                self.attention_query_norms,
                self.shared_attentions,
                self.attention_output_norms,
                self.dual_ffn_norms,
                self.dual_ffns,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def load_cmp_checkpoint(self, checkpoint_path: str) -> dict:
        """Strictly load only the published MTR motion decoder parameters."""

        payload = torch.load(checkpoint_path, map_location="cpu")
        source = payload.get("model_state", payload)
        prefix = "motion_transformer.motion_decoder."
        decoder_state = {
            key[len(prefix) :]: value
            for key, value in source.items()
            if key.startswith(prefix)
        }
        if not decoder_state:
            raise ValueError(f"checkpoint has no {prefix} parameters")
        result = self.decoder.load_state_dict(decoder_state, strict=True)
        self.freeze_unused_map_parameters()
        return {
            "loaded_count": len(decoder_state),
            "decoder_loaded_count": len(decoder_state),
            "newly_initialized": (
                "online_history_encoder, shared_dual_attention, "
                "attention_memory_projection, latent_projection"
            ),
            "missing": list(result.missing_keys),
            "unexpected": list(result.unexpected_keys),
        }

    def _right_align_history(self, history_box, history_mask):
        if history_box.shape[:-1] != history_mask.shape or history_box.shape[-1] != 7:
            raise ValueError("history_box/history_mask shapes disagree")
        length = history_box.shape[-2]
        if length > self.history_frames:
            return history_box[..., -self.history_frames :, :], history_mask[..., -self.history_frames :]
        if length == self.history_frames:
            return history_box, history_mask
        pad = self.history_frames - length
        return (
            torch.cat((history_box.new_zeros(*history_box.shape[:-2], pad, 7), history_box), dim=-2),
            torch.cat(
                (
                    torch.zeros(*history_mask.shape[:-1], pad, dtype=torch.bool, device=history_mask.device),
                    history_mask,
                ),
                dim=-1,
            ),
        )

    def _encode_temporal_history(self, history_box, history_mask):
        if not self.temporal_history_enabled:
            return history_box.new_zeros(*history_mask.shape[:-1], self.query_dim)
        xyz = history_box[..., :3]
        index = torch.arange(self.history_frames, device=history_mask.device)
        latest_index = torch.where(
            history_mask,
            index.view(*([1] * (history_mask.ndim - 1)), -1),
            torch.full_like(history_mask, -1, dtype=torch.long),
        ).max(dim=-1).values.clamp_min(0)
        latest_xyz = xyz.gather(
            -2, latest_index[..., None, None].expand(*latest_index.shape, 1, 3)
        ).squeeze(-2)
        relative_xyz = (
            xyz - latest_xyz[..., None, :]
        ) / self.history_position_scale.to(history_box)
        time = torch.linspace(-1.0, 0.0, self.history_frames, device=xyz.device, dtype=xyz.dtype)
        token_input = torch.cat(
            (
                relative_xyz,
                history_box[..., 3:6] / self.history_size_scale.to(history_box),
                torch.sin(history_box[..., 6:7]),
                torch.cos(history_box[..., 6:7]),
                time.view(*([1] * (history_box.ndim - 2)), self.history_frames, 1).expand(*history_box.shape[:-2], self.history_frames, 1),
            ),
            dim=-1,
        )
        tokens = self.history_input_projection(token_input) + self.history_position.view(
            *([1] * (token_input.ndim - 2)), self.history_frames, self.query_dim
        )
        flat_tokens = tokens.reshape(-1, self.history_frames, self.query_dim)
        flat_mask = history_mask.reshape(-1, self.history_frames).bool()
        safe_mask = flat_mask.clone()
        safe_mask[~safe_mask.any(dim=-1), -1] = True
        encoded = self.history_encoder(flat_tokens, src_key_padding_mask=~safe_mask)
        pooled = encoded[
            torch.arange(encoded.shape[0], device=encoded.device), latest_index.reshape(-1)
        ].view(*history_mask.shape[:-1], self.query_dim)
        pooled = self.history_norm(pooled)
        return torch.where(history_mask.any(dim=-1)[..., None], pooled, torch.zeros_like(pooled))

    def _pack_actor_context(self, feature, boxes, valid_mask, motion_valid):
        """Pack variable actor sets into MTR's center/object memory tensors."""

        entries = []
        contexts = []
        for batch_index in range(feature.shape[0]):
            context_slots = torch.where(valid_mask[batch_index])[0]
            centers = torch.where(motion_valid[batch_index])[0]
            contexts.append(context_slots)
            entries.extend((batch_index, int(slot)) for slot in centers)
        if not entries:
            return None
        max_context = max(int(item.numel()) for item in contexts)
        count = len(entries)
        center_feature = feature.new_zeros(count, self.query_dim)
        object_feature = feature.new_zeros(count, max_context, self.query_dim)
        object_mask = torch.zeros(count, max_context, dtype=torch.bool, device=feature.device)
        object_pos = feature.new_zeros(count, max_context, 3)
        headings = feature.new_zeros(count)
        decoder_headings = feature.new_zeros(count)
        context_slots = torch.full((count, max_context), -1, dtype=torch.long, device=feature.device)
        for row, (batch_index, center_slot) in enumerate(entries):
            slots = contexts[batch_index]
            actor_box = boxes[batch_index, center_slot]
            heading = actor_box[6]
            # The released V2V4Real MTR intentions/checkpoint were built from
            # degree-valued headings consumed as trigonometric angles.  Keep
            # this compatibility conversion local to the MTR coordinate frame.
            decoder_heading = torch.rad2deg(heading)
            delta = boxes[batch_index, slots, :3] - actor_box[:3]
            cosine, sine = torch.cos(decoder_heading), torch.sin(decoder_heading)
            local_x = cosine * delta[:, 0] + sine * delta[:, 1]
            local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
            size = int(slots.numel())
            center_feature[row] = feature[batch_index, center_slot]
            object_feature[row, :size] = feature[batch_index, slots]
            object_mask[row, :size] = True
            object_pos[row, :size] = torch.stack((local_x, local_y, delta[:, 2]), dim=-1)
            headings[row] = heading
            decoder_headings[row] = decoder_heading
            context_slots[row, :size] = slots
        return (
            entries, center_feature, object_feature, object_mask, object_pos,
            headings, decoder_headings, context_slots,
        )

    def _pack_evidence_memory(self, entries, tokens, positions, token_mask, name):
        """Pack only each target actor's local RoI tokens for one evidence path.

        ``consensus`` accepts [B,Q,P,D] or [B,Q,1,P,D]; ``source`` uses the
        explicit [B,Q,L,P,D] source axis.  Neither path contains neighbour
        source tokens: actor interaction belongs exclusively to MTR's standard
        object decoder.
        """

        if tokens is None:
            return None
        if tokens.ndim == 4:
            tokens = tokens.unsqueeze(2)
            positions = positions.unsqueeze(2) if positions is not None else None
            token_mask = token_mask.unsqueeze(2) if token_mask is not None else None
        if tokens.ndim != 5 or tokens.shape[-1] != self.query_dim:
            raise ValueError(f"{name}_tokens must have shape [B,Q,L,P,D]")
        if positions is None or positions.shape != tokens.shape:
            raise ValueError(f"{name}_positions must have shape [B,Q,L,P,D]")
        if token_mask is None or token_mask.shape != tokens.shape[:-1]:
            raise ValueError(f"{name}_token_mask must have shape [B,Q,L,P]")
        count = len(entries)
        _, _, sources, patches, dim = tokens.shape
        content = self.source_token_norm(tokens) + positions
        token_count = sources * patches
        memory = tokens.new_zeros(count, token_count, self.decoder_dim)
        memory_mask = torch.zeros(count, token_count, dtype=torch.bool, device=tokens.device)
        for row, (batch_index, target_slot) in enumerate(entries):
            memory[row] = self.attention_memory_projection(
                content[batch_index, target_slot].reshape(token_count, dim)
            )
            memory_mask[row] = token_mask[batch_index, target_slot].reshape(token_count).bool()
        return memory, memory_mask, sources, patches

    def _read_evidence(self, base_query, memory_bundle, layer_index, intention_query, dynamic_center):
        """Read one memory independently; None/masked memory has exact zero contribution."""

        actor_count, raw_modes, _ = base_query.shape
        if memory_bundle is None:
            return (
                base_query.new_zeros(actor_count, raw_modes, self.decoder_dim),
                base_query.new_zeros(actor_count, raw_modes, 0, 0),
                torch.zeros(actor_count, dtype=torch.bool, device=base_query.device),
            )
        memory, valid, sources, patches = memory_bundle
        has_memory = valid.any(dim=-1)
        safe_valid = valid.clone()
        safe_valid[~has_memory, 0] = True
        dynamic_sine = position_encoding_utils.gen_sineembed_for_position(
            dynamic_center, hidden_dim=self.decoder_dim
        ).permute(1, 0, 2)
        # Both representations are queried by the same static intention and
        # current dynamic endpoint encoding.  Shared weights make their
        # ablations parameter-identical while the memories stay independent.
        attention_query = (
            self.attention_query_norms[layer_index](base_query)
            + intention_query.permute(1, 0, 2)
            + dynamic_sine
        )
        context, weights = self.shared_attentions[layer_index](
            attention_query, memory, memory,
            key_padding_mask=~safe_valid,
            need_weights=True, average_attn_weights=True,
        )
        context = torch.where(has_memory[:, None, None], context, torch.zeros_like(context))
        weights = torch.where(has_memory[:, None, None], weights, torch.zeros_like(weights))
        return (
            context,
            weights.reshape(actor_count, raw_modes, sources, patches),
            has_memory,
        )

    @staticmethod
    def _local_gmm_to_receiver(trajectories, heading):
        cosine, sine = torch.cos(heading)[:, None, None], torch.sin(heading)[:, None, None]
        local_x, local_y = trajectories[..., 0], trajectories[..., 1]
        receiver_x = cosine * local_x - sine * local_y
        receiver_y = sine * local_x + cosine * local_y
        std_x = trajectories[..., 2].clamp(-1.609, 5.0).exp()
        std_y = trajectories[..., 3].clamp(-1.609, 5.0).exp()
        rho = trajectories[..., 4].clamp(-0.5, 0.5)
        var_x, var_y = std_x.square(), std_y.square()
        covariance = rho * std_x * std_y
        receiver_var_x = (cosine.square() * var_x + sine.square() * var_y - 2 * cosine * sine * covariance).clamp_min(1e-8)
        receiver_var_y = (sine.square() * var_x + cosine.square() * var_y + 2 * cosine * sine * covariance).clamp_min(1e-8)
        receiver_covariance = cosine * sine * (var_x - var_y) + (cosine.square() - sine.square()) * covariance
        receiver_std_x, receiver_std_y = receiver_var_x.sqrt(), receiver_var_y.sqrt()
        receiver_rho = (receiver_covariance / (receiver_std_x * receiver_std_y)).clamp(-0.999, 0.999)
        return torch.stack((receiver_x, receiver_y, receiver_std_x.log(), receiver_std_y.log(), receiver_rho), dim=-1)

    def empty_outputs(self, query_feat, valid_mask):
        batch, queries, _ = query_feat.shape
        trajectories = query_feat.new_zeros(batch, queries, self.modes, self.future_frames, 5)
        return {
            "pred_scores": query_feat.new_full((batch, queries, self.modes), -20.0),
            "pred_trajs": trajectories,
            "pred_trajs_local": trajectories.clone(),
            "loss_pred_scores": query_feat.new_full((batch, queries, self.raw_modes), -20.0),
            "loss_pred_trajs_local": query_feat.new_zeros(batch, queries, self.raw_modes, self.future_frames, 5),
            "loss_pred_scores_layers": query_feat.new_full((batch, queries, self.decoder_layers, self.raw_modes), -20.0),
            "loss_pred_trajs_local_layers": query_feat.new_zeros(batch, queries, self.decoder_layers, self.raw_modes, self.future_frames, 5),
            "loss_intention_points": query_feat.new_zeros(batch, queries, self.raw_modes, 2),
            "loss_intention_points_layers": query_feat.new_zeros(batch, queries, self.decoder_layers, self.raw_modes, 2),
            "future_cue": query_feat.new_zeros(batch, queries, self.query_dim),
            "future_uncertainty": query_feat.new_zeros(batch, queries, 1),
            "cooperative_context": query_feat.new_zeros(batch, queries, self.query_dim),
            "consensus_context": query_feat.new_zeros(batch, queries, self.query_dim),
            "source_context": query_feat.new_zeros(batch, queries, self.query_dim),
            "cooperative_attention_weights": query_feat.new_zeros(batch, queries, 0, 0),
            "consensus_attention_weights": query_feat.new_zeros(batch, queries, 0),
            "source_attention_weights": query_feat.new_zeros(batch, queries, 0, 0),
            "mode_consensus_attention_weights": query_feat.new_zeros(batch, queries, self.modes, 0),
            "mode_source_attention_weights": query_feat.new_zeros(batch, queries, self.modes, 0, 0),
            "forecast_valid_mask": torch.zeros_like(valid_mask),
            "forecast_history_steps": torch.zeros_like(valid_mask, dtype=torch.long),
            "forecast_frame_heading": query_feat.new_zeros(batch, queries),
            "forecast_loss_in_receiver_frame": False,
            "packed_entries": (),
            "packed_context_slots": torch.empty(0, 0, dtype=torch.long, device=query_feat.device),
            "packed_dense_trajs_local": query_feat.new_zeros(0, 0, self.future_frames, 5),
            "packed_center_heading": query_feat.new_zeros(0),
        }

    def _decode_mtr(
        self, center_feature, object_feature, object_mask, object_pos,
        consensus_memory, source_memory,
    ):
        decoder = self.decoder
        center = decoder.in_proj_center_obj(center_feature)
        projected_object = object_feature.new_zeros(*object_feature.shape[:-1], self.decoder_dim)
        projected_object[object_mask] = decoder.in_proj_obj(object_feature[object_mask])
        projected_object, dense_trajs = decoder.apply_dense_future_prediction(projected_object, object_mask, object_pos)
        types = ["TYPE_VEHICLE"] * center.shape[0]
        intention_query, intention_points = decoder.get_motion_query(types)
        query_content = torch.zeros_like(intention_query)
        center_repeated = center[None].expand(self.raw_modes, -1, -1)
        dynamic_center = intention_points
        predictions = []
        last_consensus_attention = last_source_attention = None
        last_consensus_context = last_source_context = None
        for layer_index in range(self.decoder_layers):
            object_query = decoder.apply_cross_attention(
                kv_feature=projected_object,
                kv_mask=object_mask,
                kv_pos=object_pos,
                query_content=query_content,
                query_embed=intention_query,
                attention_layer=decoder.obj_decoder_layers[layer_index],
                dynamic_query_center=dynamic_center,
                layer_idx=layer_index,
            )
            base_query = decoder.query_feature_fusion_layers[layer_index](
                torch.cat((center_repeated, object_query), dim=-1).flatten(0, 1)
            ).view(self.raw_modes, center.shape[0], self.decoder_dim)
            base_query = base_query.permute(1, 0, 2)
            (
                consensus_context,
                consensus_attention,
                consensus_active,
            ) = self._read_evidence(
                base_query, consensus_memory, layer_index, intention_query, dynamic_center
            )
            source_context, source_attention, source_active = self._read_evidence(
                base_query, source_memory, layer_index, intention_query, dynamic_center
            )
            # Independent consensus/source reads are summed from the same MTR
            # base query.  Variance-preserving path scaling and shared post-norm
            # prevent dual from winning merely through a larger residual norm.
            active_paths = (
                consensus_active.to(base_query.dtype)
                + source_active.to(base_query.dtype)
            ).clamp_min(1.0)
            evidence_context = (
                consensus_context + source_context
            ) / active_paths.sqrt()[:, None, None]
            query_batch = self.attention_output_norms[layer_index](
                base_query
                + self.attention_dropouts[layer_index](evidence_context)
            )
            query_batch = self.dual_ffn_norms[layer_index](
                query_batch
                + self.dual_ffn_dropouts[layer_index](
                    self.dual_ffns[layer_index](query_batch)
                )
            )
            query_content = query_batch.permute(1, 0, 2)
            flattened = query_content.permute(1, 0, 2).reshape(-1, self.decoder_dim)
            score = decoder.motion_cls_heads[layer_index](flattened).view(center.shape[0], self.raw_modes)
            traj = decoder.motion_reg_heads[layer_index](flattened).view(
                center.shape[0], self.raw_modes, self.future_frames, 5
            )
            predictions.append((score, traj))
            dynamic_center = traj[:, :, -1, :2].permute(1, 0, 2).contiguous()
            last_consensus_attention = consensus_attention
            last_source_attention = source_attention
            last_consensus_context = consensus_context
            last_source_context = source_context
        fixed_intentions = intention_points.permute(1, 0, 2).contiguous()
        return (
            predictions, fixed_intentions, dense_trajs, query_content.permute(1, 0, 2),
            last_consensus_attention, last_source_attention,
            last_consensus_context, last_source_context,
        )

    def forward(
        self,
        query_feat,
        boxes,
        valid_mask,
        history_box,
        history_mask,
        receiver_time_s,
        consensus_tokens=None,
        consensus_positions=None,
        consensus_token_mask=None,
        source_tokens=None,
        source_positions=None,
        source_token_mask=None,
    ):
        del receiver_time_s
        with torch.cuda.amp.autocast(enabled=False):
            query_feat = query_feat.float()
            boxes = boxes.float()
            history_box = history_box.float()
            output = self.empty_outputs(query_feat, valid_mask)
            if not self.enabled:
                return output
            history_box, history_mask = self._right_align_history(history_box, history_mask)
            steps = history_mask.sum(dim=-1)
            motion_valid = valid_mask & (steps >= self.min_forecast_history_steps)
            output["forecast_history_steps"] = steps
            output["forecast_frame_heading"] = boxes[..., 6]
            if not motion_valid.any():
                return output
            feature = self.query_norm(query_feat) + self._encode_temporal_history(
                history_box, history_mask
            )
            packed = self._pack_actor_context(feature, boxes, valid_mask, motion_valid)
            if packed is None:
                return output
            (
                entries, center, objects, object_mask, object_pos, headings,
                decoder_headings, context_slots,
            ) = packed
            if consensus_tokens is not None:
                consensus_tokens = consensus_tokens.float()
                consensus_positions = (
                    consensus_positions.float() if consensus_positions is not None else None
                )
            if source_tokens is not None:
                source_tokens = source_tokens.float()
                source_positions = source_positions.float() if source_positions is not None else None
            consensus_memory = self._pack_evidence_memory(
                entries, consensus_tokens, consensus_positions,
                consensus_token_mask, "consensus",
            )
            source_memory = self._pack_evidence_memory(
                entries, source_tokens, source_positions, source_token_mask, "source",
            )
            (
                predictions, intentions, dense, latent, consensus_attn, source_attn,
                consensus_context, source_context,
            ) = self._decode_mtr(
                center, objects, object_mask, object_pos, consensus_memory, source_memory
            )
            layer_scores = torch.stack([item[0] for item in predictions], dim=1)
            layer_trajs = torch.stack([item[1] for item in predictions], dim=1)
            raw_scores, raw_trajs = predictions[-1]
            probabilities = torch.softmax(raw_scores, dim=-1)
            selected_local, selected_scores, selected_indices = batch_nms(
                raw_trajs, probabilities, self.decoder.model_cfg.NMS_DIST_THRESH, self.modes
            )
            selected_logits = raw_scores.gather(1, selected_indices)
            selected_probability = torch.softmax(selected_logits, dim=-1)
            selected_receiver = self._local_gmm_to_receiver(
                selected_local, decoder_headings,
            )
            selected_latent = latent.gather(
                1, selected_indices[..., None].expand(-1, -1, self.decoder_dim)
            )
            entropy = -(
                probabilities * probabilities.clamp_min(1e-8).log()
            ).sum(dim=-1, keepdim=True) / math.log(self.raw_modes)

            consensus_sources = consensus_memory[2] if consensus_memory is not None else 0
            consensus_patches = consensus_memory[3] if consensus_memory is not None else 0
            source_sources = source_memory[2] if source_memory is not None else 0
            source_patches = source_memory[3] if source_memory is not None else 0
            selected_consensus_attn = consensus_attn.gather(
                1, selected_indices[..., None, None].expand(
                    -1, -1, consensus_sources, consensus_patches
                )
            ) if consensus_memory is not None else raw_trajs.new_zeros(len(entries), self.modes, 0, 0)
            selected_source_attn = source_attn.gather(
                1, selected_indices[..., None, None].expand(
                    -1, -1, source_sources, source_patches
                )
            ) if source_memory is not None else raw_trajs.new_zeros(len(entries), self.modes, 0, 0)
            selected_consensus_context = consensus_context.gather(
                1, selected_indices[..., None].expand(-1, -1, self.decoder_dim)
            )
            selected_source_context = source_context.gather(
                1, selected_indices[..., None].expand(-1, -1, self.decoder_dim)
            )

            output["packed_entries"] = tuple(entries)
            output["packed_context_slots"] = context_slots
            output["packed_dense_trajs_local"] = dense
            output["packed_center_heading"] = headings
            if consensus_memory is not None:
                output["mode_consensus_attention_weights"] = raw_trajs.new_zeros(
                    query_feat.shape[0], query_feat.shape[1], self.modes,
                    consensus_sources * consensus_patches,
                )
                output["consensus_attention_weights"] = raw_trajs.new_zeros(
                    query_feat.shape[0], query_feat.shape[1], consensus_sources * consensus_patches,
                )
            if source_memory is not None:
                output["mode_source_attention_weights"] = raw_trajs.new_zeros(
                    query_feat.shape[0], query_feat.shape[1], self.modes,
                    source_sources, source_patches,
                )
                output["source_attention_weights"] = raw_trajs.new_zeros(
                    query_feat.shape[0], query_feat.shape[1], source_sources, source_patches,
                )
                output["cooperative_attention_weights"] = raw_trajs.new_zeros(
                    query_feat.shape[0], query_feat.shape[1], source_sources, source_patches,
                )
            for row, (batch_index, slot) in enumerate(entries):
                output["pred_scores"][batch_index, slot] = selected_scores[row]
                output["pred_trajs"][batch_index, slot] = selected_receiver[row]
                output["pred_trajs_local"][batch_index, slot] = selected_local[row]
                output["loss_pred_scores"][batch_index, slot] = raw_scores[row]
                output["loss_pred_trajs_local"][batch_index, slot] = raw_trajs[row]
                output["loss_pred_scores_layers"][batch_index, slot] = layer_scores[row]
                output["loss_pred_trajs_local_layers"][batch_index, slot] = layer_trajs[row]
                output["loss_intention_points"][batch_index, slot] = intentions[row]
                output["loss_intention_points_layers"][batch_index, slot] = intentions[row].unsqueeze(0).expand(self.decoder_layers, -1, -1)
                output["future_cue"][batch_index, slot] = self.latent_projection(selected_latent[row]).mul(selected_probability[row, :, None]).sum(dim=0)
                output["future_uncertainty"][batch_index, slot] = entropy[row]
                consensus_summary = self.latent_projection(selected_consensus_context[row]).mul(selected_probability[row, :, None]).sum(dim=0)
                source_summary = self.latent_projection(selected_source_context[row]).mul(selected_probability[row, :, None]).sum(dim=0)
                output["consensus_context"][batch_index, slot] = consensus_summary
                output["source_context"][batch_index, slot] = source_summary
                output["cooperative_context"][batch_index, slot] = consensus_summary + source_summary
                if consensus_memory is not None:
                    mode_consensus = selected_consensus_attn[row].flatten(1)
                    output["mode_consensus_attention_weights"][batch_index, slot] = mode_consensus
                    output["consensus_attention_weights"][batch_index, slot] = torch.einsum(
                        "k,kp->p", selected_probability[row], mode_consensus
                    )
                if source_memory is not None:
                    output["mode_source_attention_weights"][batch_index, slot] = selected_source_attn[row]
                    output["source_attention_weights"][batch_index, slot] = torch.einsum(
                        "k,klp->lp", selected_probability[row], selected_source_attn[row]
                    )
                    output["cooperative_attention_weights"][batch_index, slot] = output[
                        "source_attention_weights"
                    ][batch_index, slot]
                output["forecast_valid_mask"][batch_index, slot] = True
            return output
