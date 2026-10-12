"""CMP/MTR forecasting with a source-preserving Dual-BEV extension."""

from __future__ import annotations

from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

from opencood.models.dualcmp.mtr_actor_adapter import pack_mtr_actor_polylines
from opencood.models.dualcmp.forecast_frames import ALIGNED_SOURCE
from opencood.models.dualcmp.forecast_normalization import configure_forecast_normalization
from opencood.models.dualcmp.source_motion_encoder import SourceActorReadout
from opencood.models.dualcmp.mtr_vendor import (
    MTREncoder, MTRDecoder, batch_nms, position_encoding_utils,
)

GMM_VARIANCES = ("log_std_clamp_v1", "softplus_std_v1")


def parameterize_gmm_variance(trajectory, contract):
    """Keep the public [xy,log_std_xy,rho] interface and old graph unchanged.

    Standard positive-scale parameterization prevents exponential cold-start
    overshoot; native MTR's likelihood and its scale limits remain unchanged.
    """
    if contract == "log_std_clamp_v1":
        return trajectory
    if contract != "softplus_std_v1" or trajectory.shape[-1] != 5:
        raise ValueError("unsupported bivariate GMM variance parameterization")
    log_std = (F.softplus(trajectory[..., 2:4]) + math.exp(-1.609)).log()
    return torch.cat((trajectory[..., :2], log_std, trajectory[..., 4:]), dim=-1)


class IntentionQueryForecastHead(nn.Module):
    """Run the released V2V4Real MTR actor encoder and motion decoder.

    Intention queries read actor-aligned BEV memory after standard MTR updates.
    The motion-correlation family also supplies target/neighbor history with
    same-observer motion evidence before scene interaction. With no evidence,
    both hooks are absent and the released MTR path is unchanged.
    """

    def __init__(self, query_dim: int, args: dict):
        super().__init__()
        self.query_dim = int(query_dim)
        self.evidence_dim = self.query_dim * (2 if args.get("source_feature_encoder") == "motion_correlation_v1" else 1)
        self.forecast_coordinate_frame = ("current_receiver" if args.get("representation")
                                          == ALIGNED_SOURCE else "legacy_capture")
        self.history_frames = int(args.get("history_frames", 11))
        self.future_frames = int(args.get("future_frames", 50))
        self.frame_period_s = float(args.get("frame_period_s", 0.1))
        self.timestamp_scale = float(args.get("mtr_timestamp_scale", 0.1))
        self.min_forecast_history_steps = int(
            args.get("min_forecast_history_steps", 2)
        )
        self.min_context_history_steps = int(
            args.get("min_context_history_steps", 2)
        )
        if self.history_frames != 11:
            raise ValueError("released V2V4Real MTR requires eleven history frames")
        if self.frame_period_s <= 0.0:
            raise ValueError("frame_period_s must be positive")
        if not 2 <= self.min_forecast_history_steps <= self.history_frames:
            raise ValueError("min_forecast_history_steps is outside history range")
        if not 1 <= self.min_context_history_steps <= self.min_forecast_history_steps:
            raise ValueError(
                "min_context_history_steps must lie in "
                "[1, min_forecast_history_steps]"
            )

        context_config = EasyDict(
            NUM_OF_ATTN_NEIGHBORS=int(args.get("context_neighbors", 16)),
            NUM_INPUT_ATTR_AGENT=26 if self.forecast_coordinate_frame == "current_receiver" else 22,
            NUM_CHANNEL_IN_MLP_AGENT=int(
                args.get("context_encoder_hidden", self.query_dim)
            ),
            NUM_LAYER_IN_MLP_AGENT=int(args.get("context_encoder_mlp_layers", 3)),
            D_MODEL=self.query_dim,
            NUM_ATTN_LAYERS=int(args.get("context_encoder_layers", 6)),
            NUM_ATTN_HEAD=int(args.get("context_encoder_heads", 8)),
            DROPOUT_OF_ATTN=float(args.get("dropout", 0.1)),
            USE_LOCAL_ATTN=bool(args.get("context_encoder_local_attention", True)),
        )
        self.context_encoder = MTREncoder(context_config)

        project_root = Path(__file__).resolve().parents[3]
        intention_file = Path(
            args.get(
                "intention_points_file",
                "preprocessed_data/v2v4real/"
                "v2v4real_cluster_64_center_dict.pkl",
            )
        )
        if not intention_file.is_absolute():
            intention_file = project_root / intention_file
        decoder_config = EasyDict(
            OBJECT_TYPE=["TYPE_VEHICLE"],
            NUM_FUTURE_FRAMES=self.future_frames,
            NUM_MOTION_MODES=int(args.get("modes", 6)),
            USE_PLACE_HOLDER=False,
            INTENTION_RANDOM_INIT=False,
            INTENTION_POINTS_FILE=str(intention_file.resolve()),
            D_MODEL=int(args.get("d_model", 512)),
            NUM_DECODER_LAYERS=int(args.get("decoder_layers", 6)),
            NUM_ATTN_HEAD=int(args.get("attention_heads", 8)),
            MAP_D_MODEL=int(args.get("map_d_model", 256)),
            DROPOUT_OF_ATTN=float(args.get("dropout", 0.1)),
            LOSS_WEIGHTS=EasyDict(cls=1.0, reg=1.0, vel=0.5),
            NMS_DIST_THRESH=float(args.get("nms_distance", 2.5)),
            PHYSICAL_DENSE_MOTION=self.forecast_coordinate_frame == "current_receiver",
            FRAME_PERIOD_S=self.frame_period_s,
        )
        self.decoder = MTRDecoder(
            in_channels=self.context_encoder.num_out_channels,
            config=decoder_config,
        )
        self.modes = int(decoder_config.NUM_MOTION_MODES)
        self.decoder_layers = int(decoder_config.NUM_DECODER_LAYERS)
        self.decoder_dim = int(decoder_config.D_MODEL)
        self.raw_modes = int(self.decoder.intention_points["TYPE_VEHICLE"].shape[0])
        if self.raw_modes != 64:
            raise ValueError("V2V4Real MTR decoder must expose 64 intention points")
        if self.modes > self.raw_modes:
            raise ValueError("output modes cannot exceed intention query count")

        (
            self.evidence_input_projection,
            self.evidence_decoder_layers,
        ) = self.decoder.build_transformer_decoder(
            in_channels=self.evidence_dim,
            d_model=self.decoder_dim,
            nhead=int(args.get("attention_heads", 8)),
            dropout=float(args.get("dropout", 0.1)),
            num_decoder_layers=self.decoder_layers,
            use_local_attn=False,
        )
        if self.forecast_coordinate_frame == "current_receiver":
            # DAB-DETR/MTR's existing option keeps the static intention identity
            # in every cross-attention layer. Without a map branch, globally
            # read object/BEV memories can collapse dynamic endpoint queries;
            # self-attention positional Q/K alone cannot restore identical V.
            for layer in list(self.decoder.obj_decoder_layers) + list(self.evidence_decoder_layers):
                layer.keep_query_pos = True
        for layer in self.evidence_decoder_layers:
            layer.capture_attention = True
            def capture_attention(module, inputs, outputs, owner=layer):
                owner.last_cross_attention_weights = outputs[1].detach() if outputs[1] is not None else None
            layer.cross_attn.register_forward_hook(capture_attention)

        self.enabled = bool(args.get("enabled", True))
        self.source_evidence_enabled = bool(args.get("source_evidence", True))
        self.temporal_source_enabled = (
            str(args.get("representation", "source")) == "source_temporal")
        self.temporal_source_fusion_gates = None
        if self.temporal_source_enabled:
            self.temporal_source_fusion_gates = nn.Parameter(
                torch.full((self.decoder_layers,), 0.1)
            )
        self._unused_map_modules = [
            self.context_encoder.map_encoders,
            self.decoder.in_proj_map,
            self.decoder.map_decoder_layers,
        ]
        if self.decoder.map_query_content_mlps is not None:
            self._unused_map_modules.extend(
                [
                    self.decoder.map_query_content_mlps,
                    self.decoder.map_query_embed_mlps,
                ]
            )
        self._unused_query_position_modules = [
            *(layer.ca_qpos_proj for layer in self.decoder.obj_decoder_layers[1:] if not layer.keep_query_pos),
            *(layer.ca_qpos_proj for layer in self.evidence_decoder_layers[1:] if not layer.keep_query_pos),
        ]
        self.freeze_unused_parameters()

        self.source_actor_readout = None
        if self.evidence_dim == 2 * self.query_dim and self.forecast_coordinate_frame == "current_receiver":
            self.source_actor_readout = SourceActorReadout(
                self.query_dim, self.evidence_dim,
                int(args.get("context_encoder_heads", 8)), float(args.get("dropout", 0.1)))
        self.forecast_normalization = str(args.get("forecast_normalization", "batch_norm_v1"))
        self.forecast_variance = str(args.get("forecast_variance", "log_std_clamp_v1"))
        if self.forecast_variance not in GMM_VARIANCES or (self.forecast_coordinate_frame != "current_receiver" and self.forecast_variance != "log_std_clamp_v1"):
            raise ValueError("new variance parameterization is aligned-only; legacy ablations stay unchanged")
        if self.forecast_coordinate_frame != "current_receiver" and self.forecast_normalization != "batch_norm_v1":
            raise ValueError("new normalization is aligned-only; legacy ablations stay unchanged")
        configure_forecast_normalization(self, self.forecast_normalization)
        # Replacing inactive norm parameters must not unfreeze unused branches.
        self.freeze_unused_parameters()

    def freeze_unused_parameters(self) -> None:
        for module in (
            *self._unused_map_modules,
            *self._unused_query_position_modules,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def freeze_unused_map_parameters(self) -> None:
        self.freeze_unused_parameters()

    def freeze_inactive_parameters(
        self, actor_cooperative_fusion_enabled: bool = True
    ) -> None:
        if not actor_cooperative_fusion_enabled:
            for module in (
                self.evidence_input_projection,
                self.evidence_decoder_layers,
                *([self.source_actor_readout] if self.source_actor_readout is not None else []),
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = False

    @staticmethod
    def _right_align_history(history_box, history_mask, history_time_s,
                             history_frames):
        if history_box.shape[:-1] != history_mask.shape or history_box.shape[-1] != 7:
            raise ValueError("history_box/history_mask shapes disagree")
        if history_time_s.shape != history_mask.shape:
            raise ValueError("history_time_s must match history_mask")
        length = history_box.shape[-2]
        if length > history_frames:
            return (history_box[..., -history_frames:, :],
                    history_mask[..., -history_frames:],
                    history_time_s[..., -history_frames:])
        if length == history_frames:
            return history_box, history_mask, history_time_s
        pad = history_frames - length
        return (
            torch.cat(
                (history_box.new_zeros(*history_box.shape[:-2], pad, 7), history_box),
                dim=-2,
            ),
            torch.cat(
                (
                    torch.zeros(
                        *history_mask.shape[:-1], pad,
                        dtype=torch.bool, device=history_mask.device,
                    ),
                    history_mask,
                ),
                dim=-1,
            ),
            torch.cat(
                (history_time_s.new_zeros(*history_time_s.shape[:-1], pad),
                 history_time_s),
                dim=-1,
            ),
        )

    def _pack_evidence_memory(
        self, entries, tokens, positions, token_mask, age_s, name,
        aggregate_sources=False,
    ):
        if tokens is None:
            return None
        if tokens.ndim == 4:
            tokens = tokens.unsqueeze(2)
            positions = positions.unsqueeze(2) if positions is not None else None
            token_mask = token_mask.unsqueeze(2) if token_mask is not None else None
            age_s = age_s.unsqueeze(2) if age_s is not None else None
        if tokens.ndim != 5 or tokens.shape[-1] != self.evidence_dim:
            raise ValueError(f"{name}_tokens must have shape [B,Q,L,P,D]")
        if positions is None or positions.shape != (*tokens.shape[:-1], 3):
            raise ValueError(f"{name}_positions must have shape [B,Q,L,P,3]")
        if token_mask is None or token_mask.shape != tokens.shape[:-1]:
            raise ValueError(f"{name}_token_mask must have shape [B,Q,L,P]")
        if age_s is not None:
            if age_s.shape != tokens.shape[:3]:
                raise ValueError(f"{name}_age_s must have shape [B,Q,L]")
            valid_age = token_mask.any(dim=-1)
            if not torch.isfinite(age_s[valid_age]).all() or torch.any(
                age_s[valid_age] < -1e-6
            ):
                raise ValueError(f"{name}_age_s must be finite and non-negative")
            safe_age = torch.where(valid_age, age_s, torch.zeros_like(age_s))
            age_steps = safe_age / self.frame_period_s
            age_position = torch.stack(
                (age_steps.reshape(-1), torch.zeros_like(age_steps).reshape(-1)),
                dim=-1,
            ).unsqueeze(1)
            zero_position = torch.zeros_like(age_position)
            age_encoding = (
                position_encoding_utils.gen_sineembed_for_position(
                    age_position, hidden_dim=self.evidence_dim)
                - position_encoding_utils.gen_sineembed_for_position(
                    zero_position, hidden_dim=self.evidence_dim)
            ).squeeze(1).reshape(*safe_age.shape, self.evidence_dim)
            tokens = tokens + age_encoding[:, :, :, None, :]
        if aggregate_sources:
            weights = token_mask[..., None].to(dtype=tokens.dtype)
            denominator = weights.sum(dim=2, keepdim=True).clamp_min(1.0)
            tokens = (tokens * weights).sum(dim=2, keepdim=True) / denominator
            positions = (
                positions * weights.to(dtype=positions.dtype)
            ).sum(dim=2, keepdim=True) / denominator.to(dtype=positions.dtype)
            token_mask = token_mask.any(dim=2, keepdim=True)
        count = len(entries)
        _, _, sources, patches, dim = tokens.shape
        token_count = sources * patches
        memory = tokens.new_zeros(count, token_count, dim)
        memory_pos = tokens.new_zeros(count, token_count, 3)
        memory_mask = torch.zeros(
            count, token_count, dtype=torch.bool, device=tokens.device
        )
        for row, (batch_index, target_slot) in enumerate(entries):
            memory[row] = tokens[batch_index, target_slot].reshape(token_count, dim)
            memory_pos[row] = positions[batch_index, target_slot].reshape(
                token_count, 3
            )
            memory_mask[row] = token_mask[batch_index, target_slot].reshape(
                token_count
            ).bool()
        return memory, memory_pos, memory_mask, sources, patches

    @staticmethod
    def _combine_evidence(consensus, source):
        bundles = [item for item in (consensus, source) if item is not None]
        if not bundles:
            return None
        return (
            torch.cat([item[0] for item in bundles], dim=1),
            torch.cat([item[1] for item in bundles], dim=1),
            torch.cat([item[2] for item in bundles], dim=1),
        )

    @staticmethod
    def _fuse_parallel_source(query_content, source_query,
                              layer_input_query, gate):
        """Keep the MTR query primary and add only source-caused change."""
        if not (query_content.shape == source_query.shape
                == layer_input_query.shape):
            raise ValueError("parallel source query shapes must match")
        return query_content + gate * (source_query - layer_input_query)

    def _evidence_hook(self, memory_bundle, attention_by_layer,
                       parallel_source=False):
        if memory_bundle is None:
            return None
        raw_memory, memory_pos, memory_mask = memory_bundle
        has_memory = memory_mask.any(dim=-1)
        if not bool(has_memory.any()):
            return None
        safe_mask = memory_mask.clone()
        safe_mask[~has_memory, 0] = True
        projected = raw_memory.new_zeros(
            *raw_memory.shape[:-1], self.decoder_dim
        )
        projected[safe_mask] = self.evidence_input_projection(raw_memory[safe_mask])

        def hook(*, query_content, intention_query, dynamic_query_center,
                 layer_idx, layer_input_query=None):
            attention_query = (
                layer_input_query
                if parallel_source and layer_input_query is not None
                else query_content
            )
            updated = self.decoder.apply_cross_attention(
                kv_feature=projected,
                kv_mask=safe_mask,
                kv_pos=memory_pos,
                query_content=attention_query,
                query_embed=intention_query,
                attention_layer=self.evidence_decoder_layers[layer_idx],
                dynamic_query_center=dynamic_query_center,
                layer_idx=layer_idx,
            )
            weights = getattr(
                self.evidence_decoder_layers[layer_idx],
                "last_cross_attention_weights",
                None,
            )
            if weights is not None:
                attention_by_layer[layer_idx] = torch.where(
                    has_memory[:, None, None],
                    weights,
                    torch.zeros_like(weights),
                )
            if parallel_source:
                if (layer_input_query is None
                        or self.temporal_source_fusion_gates is None):
                    raise RuntimeError(
                        "temporal source fusion requires parallel MTR features")
                updated = self._fuse_parallel_source(
                    query_content, updated, layer_input_query,
                    self.temporal_source_fusion_gates[layer_idx],
                )
            return torch.where(
                has_memory[None, :, None], updated, query_content
            )

        return hook

    @staticmethod
    def _local_gmm_to_receiver(trajectories, heading):
        cosine = torch.cos(heading)[:, None, None]
        sine = torch.sin(heading)[:, None, None]
        local_x, local_y = trajectories[..., 0], trajectories[..., 1]
        receiver_x = cosine * local_x - sine * local_y
        receiver_y = sine * local_x + cosine * local_y
        std_x = trajectories[..., 2].clamp(-1.609, 5.0).exp()
        std_y = trajectories[..., 3].clamp(-1.609, 5.0).exp()
        rho = trajectories[..., 4].clamp(-0.5, 0.5)
        var_x, var_y = std_x.square(), std_y.square()
        covariance = rho * std_x * std_y
        receiver_var_x = (
            cosine.square() * var_x + sine.square() * var_y
            - 2 * cosine * sine * covariance
        ).clamp_min(1e-8)
        receiver_var_y = (
            sine.square() * var_x + cosine.square() * var_y
            + 2 * cosine * sine * covariance
        ).clamp_min(1e-8)
        receiver_covariance = (
            cosine * sine * (var_x - var_y)
            + (cosine.square() - sine.square()) * covariance
        )
        receiver_std_x, receiver_std_y = (
            receiver_var_x.sqrt(), receiver_var_y.sqrt()
        )
        receiver_rho = (
            receiver_covariance / (receiver_std_x * receiver_std_y)
        ).clamp(-0.999, 0.999)
        return torch.stack(
            (
                receiver_x,
                receiver_y,
                receiver_std_x.log(),
                receiver_std_y.log(),
                receiver_rho,
            ),
            dim=-1,
        )

    def empty_outputs(self, query_feat, valid_mask):
        batch, queries, _ = query_feat.shape
        trajectories = query_feat.new_zeros(
            batch, queries, self.modes, self.future_frames, 5
        )
        return {
            "pred_scores": query_feat.new_zeros(batch, queries, self.modes),
            "selected_mode_indices": torch.zeros(
                batch, queries, self.modes,
                dtype=torch.long, device=query_feat.device,
            ),
            "pred_trajs": trajectories,
            "pred_trajs_local": trajectories.clone(),
            "loss_pred_scores": query_feat.new_full(
                (batch, queries, self.raw_modes), -20.0
            ),
            "loss_pred_trajs_local": query_feat.new_zeros(
                batch, queries, self.raw_modes, self.future_frames, 5
            ),
            "loss_pred_scores_layers": query_feat.new_full(
                (batch, queries, self.decoder_layers, self.raw_modes), -20.0
            ),
            "loss_pred_trajs_local_layers": query_feat.new_zeros(
                batch, queries, self.decoder_layers, self.raw_modes,
                self.future_frames, 5,
            ),
            "loss_intention_points": query_feat.new_zeros(
                batch, queries, self.raw_modes, 2
            ),
            "loss_intention_points_layers": query_feat.new_zeros(
                batch, queries, self.decoder_layers, self.raw_modes, 2
            ),
            "future_cue": query_feat.new_zeros(batch, queries, self.query_dim),
            "future_uncertainty": query_feat.new_zeros(batch, queries, 1),
            "cooperative_context": query_feat.new_zeros(
                batch, queries, self.query_dim
            ),
            "consensus_context": query_feat.new_zeros(
                batch, queries, self.query_dim
            ),
            "source_context": query_feat.new_zeros(batch, queries, self.query_dim),
            "cooperative_attention_weights": query_feat.new_zeros(
                batch, queries, 0, 0
            ),
            "consensus_attention_weights": query_feat.new_zeros(
                batch, queries, 0
            ),
            "source_attention_weights": query_feat.new_zeros(
                batch, queries, 0, 0
            ),
            "mode_consensus_attention_weights": query_feat.new_zeros(
                batch, queries, self.modes, 0
            ),
            "mode_source_attention_weights": query_feat.new_zeros(
                batch, queries, self.modes, 0, 0
            ),
            "forecast_valid_mask": torch.zeros_like(valid_mask),
            "forecast_history_steps": torch.zeros_like(
                valid_mask, dtype=torch.long
            ),
            "forecast_frame_heading": query_feat.new_zeros(batch, queries),
            "forecast_loss_in_receiver_frame": False,
            "forecast_coordinate_frame": self.forecast_coordinate_frame,
            "packed_entries": (),
            "packed_context_slots": torch.empty(
                0, 0, dtype=torch.long, device=query_feat.device
            ),
            "packed_dense_trajs_local": query_feat.new_zeros(
                0, 0, self.future_frames, 5
            ),
            "packed_center_heading": query_feat.new_zeros(0),
        }

    def forward(
        self,
        query_feat,
        boxes,
        valid_mask,
        history_box,
        history_mask,
        history_time_s,
        receiver_time_s,
        consensus_tokens=None,
        consensus_positions=None,
        consensus_token_mask=None,
        source_tokens=None,
        source_positions=None,
        source_token_mask=None,
        source_age_s=None,
        mean_source_aggregation=False,
    ):
        with torch.cuda.amp.autocast(enabled=False):
            query_feat = query_feat.float()
            boxes = boxes.float()
            history_box = history_box.float()
            history_time_s = history_time_s.float()
            receiver_time_s = receiver_time_s.float()
            output = self.empty_outputs(query_feat, valid_mask)
            if not self.enabled:
                return output
            history_box, history_mask, history_time_s = self._right_align_history(
                history_box, history_mask, history_time_s, self.history_frames
            )
            output["forecast_history_steps"] = history_mask.sum(dim=-1)
            output["forecast_frame_heading"] = boxes[..., 6]
            packed = pack_mtr_actor_polylines(
                boxes,
                valid_mask,
                history_box,
                history_mask,
                history_time_s,
                receiver_time_s,
                min_forecast_history_steps=self.min_forecast_history_steps,
                min_context_history_steps=self.min_context_history_steps,
                frame_period_s=self.frame_period_s,
                timestamp_scale=self.timestamp_scale,
                legacy_heading=(self.forecast_coordinate_frame == "legacy_capture"),
            )
            if packed is None:
                return output

            consensus = self._pack_evidence_memory(
                packed.entries,
                consensus_tokens.float() if consensus_tokens is not None else None,
                consensus_positions.float()
                if consensus_positions is not None else None,
                consensus_token_mask,
                None,
                "consensus",
            )
            source = (
                self._pack_evidence_memory(
                    packed.entries,
                    source_tokens.float() if source_tokens is not None else None,
                    source_positions.float()
                    if source_positions is not None
                    else None,
                    source_token_mask,
                    source_age_s.float() if source_age_s is not None else None,
                    "source",
                    aggregate_sources=bool(mean_source_aggregation),
                )
                if self.source_evidence_enabled
                else None
            )
            combined = self._combine_evidence(consensus, source)
            attention_by_layer = {}
            query_hook = self._evidence_hook(
                combined, attention_by_layer,
                parallel_source=self.temporal_source_enabled)
            # No zero edges on normal supervised graphs. The trainer handles
            # a missing trainable evidence branch after loss-local matching.
            output["needs_forecast_ddp_anchor"] = (
                query_hook is None and any(parameter.requires_grad
                    for parameter in self.evidence_input_projection.parameters()))

            actor_hook = None
            if self.source_actor_readout is not None and source is not None:
                actor_hook = lambda features: self.source_actor_readout(
                    features, packed.entries, packed.context_slots,
                    source_tokens.float(), source_positions.float(), source_token_mask)
            encoded = self.context_encoder(packed.batch_dict, actor_feature_hook=actor_hook)
            decoded = self.decoder(
                encoded,
                query_feature_hook=query_hook,
                store_training_targets=False,
            )
            predictions = self.decoder.forward_ret_dict["pred_list"]
            predictions = [(scores, parameterize_gmm_variance(trajs, self.forecast_variance))
                           for scores, trajs in predictions]
            layer_scores = torch.stack([item[0] for item in predictions], dim=1)
            layer_trajs = torch.stack([item[1] for item in predictions], dim=1)
            raw_scores, raw_trajs = predictions[-1]
            probabilities = torch.softmax(raw_scores, dim=-1)
            selected_local, selected_scores, selected_indices = batch_nms(
                raw_trajs,
                probabilities,
                self.decoder.model_cfg.NMS_DIST_THRESH,
                self.modes,
            )
            selected_receiver = self._local_gmm_to_receiver(
                selected_local, packed.decoder_headings
            )
            intentions = self.decoder.forward_ret_dict["intention_points"]
            dense = parameterize_gmm_variance(self.decoder.forward_ret_dict["pred_dense_trajs"], self.forecast_variance)

            consensus_sources = consensus[3] if consensus is not None else 0
            consensus_patches = consensus[4] if consensus is not None else 0
            source_sources = source[3] if source is not None else 0
            source_patches = source[4] if source is not None else 0
            last_attention = attention_by_layer.get(
                self.decoder_layers - 1,
                raw_trajs.new_zeros(
                    len(packed.entries), self.raw_modes,
                    consensus_sources * consensus_patches
                    + source_sources * source_patches,
                ),
            )
            selected_attention = last_attention.gather(
                1,
                selected_indices[..., None].expand(
                    -1, -1, last_attention.shape[-1]
                ),
            )
            consensus_count = consensus_sources * consensus_patches
            selected_consensus_attention = selected_attention[..., :consensus_count]
            selected_source_attention = selected_attention[..., consensus_count:]
            if source is not None:
                selected_source_attention = selected_source_attention.view(
                    len(packed.entries), self.modes,
                    source_sources, source_patches,
                )

            output["packed_entries"] = packed.entries
            output["packed_context_slots"] = packed.context_slots
            output["packed_dense_trajs_local"] = dense
            output["packed_center_heading"] = packed.center_headings
            if consensus is not None:
                output["mode_consensus_attention_weights"] = raw_trajs.new_zeros(
                    query_feat.shape[0], query_feat.shape[1], self.modes,
                    consensus_count,
                )
                output["consensus_attention_weights"] = raw_trajs.new_zeros(
                    query_feat.shape[0], query_feat.shape[1], consensus_count
                )
            if source is not None:
                output["mode_source_attention_weights"] = raw_trajs.new_zeros(
                    query_feat.shape[0], query_feat.shape[1], self.modes,
                    source_sources, source_patches,
                )
                output["source_attention_weights"] = raw_trajs.new_zeros(
                    query_feat.shape[0], query_feat.shape[1],
                    source_sources, source_patches,
                )
                output["cooperative_attention_weights"] = raw_trajs.new_zeros(
                    query_feat.shape[0], query_feat.shape[1],
                    source_sources, source_patches,
                )

            for row, (batch_index, slot) in enumerate(packed.entries):
                output["pred_scores"][batch_index, slot] = selected_scores[row]
                output["selected_mode_indices"][batch_index, slot] = (
                    selected_indices[row]
                )
                output["pred_trajs"][batch_index, slot] = selected_receiver[row]
                output["pred_trajs_local"][batch_index, slot] = selected_local[row]
                output["loss_pred_scores"][batch_index, slot] = raw_scores[row]
                output["loss_pred_trajs_local"][batch_index, slot] = raw_trajs[row]
                output["loss_pred_scores_layers"][batch_index, slot] = layer_scores[row]
                output["loss_pred_trajs_local_layers"][batch_index, slot] = layer_trajs[row]
                output["loss_intention_points"][batch_index, slot] = intentions[row]
                output["loss_intention_points_layers"][batch_index, slot] = (
                    intentions[row].unsqueeze(0).expand(
                        self.decoder_layers, -1, -1
                    )
                )
                if consensus is not None:
                    mode_consensus = selected_consensus_attention[row]
                    output["mode_consensus_attention_weights"][batch_index, slot] = mode_consensus
                    output["consensus_attention_weights"][batch_index, slot] = torch.einsum(
                        "k,kp->p", selected_scores[row], mode_consensus
                    )
                if source is not None:
                    mode_source = selected_source_attention[row]
                    output["mode_source_attention_weights"][batch_index, slot] = mode_source
                    output["source_attention_weights"][batch_index, slot] = torch.einsum(
                        "k,klp->lp", selected_scores[row], mode_source
                    )
                    output["cooperative_attention_weights"][batch_index, slot] = output[
                        "source_attention_weights"
                    ][batch_index, slot]
                output["forecast_valid_mask"][batch_index, slot] = True
            return output
