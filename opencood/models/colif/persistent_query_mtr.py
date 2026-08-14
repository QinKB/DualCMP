"""Variable-history persistent queries adapted to CMP's complete MTR stack."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from easydict import EasyDict

from opencood.models.colif.motr_vendor import MTREncoder, MTRDecoder, batch_nms


class PersistentQueryMTRAdapter(nn.Module):
    """Run CMP's context encoder and decoder on causal persistent-query history.

    CMP trained its MTR decoder on features produced by ``MTREncoder`` from a
    fixed 11-slot tensor whose validity mask permits missing observations.
    CoLIF preserves that pretrained feature contract while changing eligibility:
    every active query is forecast immediately, even if only its current slot is
    valid.  No GT identity or trajectory enters this runtime adapter.
    """

    def __init__(self, query_dim: int, args: dict):
        super().__init__()
        project_root = Path(__file__).resolve().parents[3]
        self.query_dim = int(query_dim)
        self.history_frames = int(args.get("history_frames", 11))
        if self.history_frames != 11:
            raise ValueError(
                "The released V2V4Real MTR checkpoint requires 11 history slots"
            )
        self.frame_period_s = float(args.get("frame_period_s", 0.1))
        # CMP's V2V4Real dataset encodes timestamps as 0.1 * (0.1 * index).
        # Reproduce that released-checkpoint contract instead of silently
        # feeding physical seconds to weights trained with the upstream scale.
        self.cmp_timestamp_scale = float(args.get("cmp_timestamp_scale", 0.1))
        if self.frame_period_s <= 0.0 or self.cmp_timestamp_scale <= 0.0:
            raise ValueError("MTR history time scales must be positive")

        context_config = EasyDict(
            NUM_OF_ATTN_NEIGHBORS=int(args.get("context_neighbors", 16)),
            NUM_INPUT_ATTR_AGENT=22,
            NUM_CHANNEL_IN_MLP_AGENT=int(args.get("context_mlp_channels", 256)),
            NUM_LAYER_IN_MLP_AGENT=int(args.get("context_mlp_layers", 3)),
            D_MODEL=int(args.get("context_dim", 256)),
            NUM_ATTN_LAYERS=int(args.get("context_layers", 6)),
            NUM_ATTN_HEAD=int(args.get("context_heads", 8)),
            DROPOUT_OF_ATTN=float(args.get("dropout", 0.1)),
            USE_LOCAL_ATTN=bool(args.get("context_local_attention", True)),
        )
        self.context_encoder = MTREncoder(context_config)

        decoder_config = EasyDict(
            OBJECT_TYPE=["TYPE_VEHICLE"],
            NUM_FUTURE_FRAMES=int(args.get("future_frames", 50)),
            NUM_MOTION_MODES=int(args.get("modes", 6)),
            USE_PLACE_HOLDER=False,
            INTENTION_RANDOM_INIT=bool(args.get("intention_random_init", False)),
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
        self.decoder = MTRDecoder(
            in_channels=self.context_encoder.num_out_channels,
            config=decoder_config,
        )
        self.modes = int(decoder_config.NUM_MOTION_MODES)
        intention_buffer = getattr(
            self.decoder,
            self.decoder.intention_points["TYPE_VEHICLE"],
        )
        self.raw_modes = int(intention_buffer.shape[0])
        if self.raw_modes < self.modes:
            raise ValueError("MTR intention count is smaller than output modes")
        self.future_frames = int(decoder_config.NUM_FUTURE_FRAMES)
        self.decoder_layers = int(decoder_config.NUM_DECODER_LAYERS)
        self.history_truncation_probability = float(
            args.get("history_truncation_probability", 0.5)
        )
        if not 0.0 <= self.history_truncation_probability <= 1.0:
            raise ValueError(
                "forecast.history_truncation_probability must lie in [0, 1]"
            )
        self.min_forecast_history_steps = int(
            args.get("min_forecast_history_steps", 1)
        )
        self.min_context_history_steps = int(
            args.get("min_context_history_steps", 1)
        )
        self.heading_smoothing_steps = int(args.get("heading_smoothing_steps", 1))
        if not 1 <= self.min_forecast_history_steps <= self.history_frames:
            raise ValueError("min_forecast_history_steps is outside history range")
        if not 1 <= self.min_context_history_steps <= self.history_frames:
            raise ValueError("min_context_history_steps is outside history range")
        if self.min_context_history_steps > self.min_forecast_history_steps:
            raise ValueError(
                "context history requirement cannot exceed center requirement"
            )
        if not 1 <= self.heading_smoothing_steps <= self.history_frames:
            raise ValueError("heading_smoothing_steps is outside history range")
        # The trajectory encoder remains exactly CMP-compatible, while this
        # zero-initialized residual is the explicit CoLIF bridge from the
        # persistent perception/query state into both the center actor and
        # its interacting object tokens.  At initialization the official MTR
        # output is bit-for-bit unchanged; Phase 2 learns only useful visual
        # state corrections instead of replacing the pretrained trajectory
        # representation.
        self.query_feature_projection = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, context_config.D_MODEL),
            nn.GELU(),
            nn.Linear(context_config.D_MODEL, context_config.D_MODEL),
        )
        nn.init.zeros_(self.query_feature_projection[-1].weight)
        nn.init.zeros_(self.query_feature_projection[-1].bias)
        self.latent_projection = nn.Linear(decoder_config.D_MODEL, query_dim)
        self.enabled = bool(args.get("enabled", True))

        # These map branches are constructed by upstream CMP/MTR but are not
        # consumed by the V2V4Real map-free forward used here.
        unused = [
            self.context_encoder.map_encoders,
            self.decoder.in_proj_map,
            self.decoder.map_decoder_layers,
        ]
        if self.decoder.map_query_content_mlps is not None:
            unused.extend(
                [
                    self.decoder.map_query_content_mlps,
                    self.decoder.map_query_embed_mlps,
                ]
            )
        self._unused_map_modules = unused
        self.freeze_unused_map_parameters()

    def freeze_unused_map_parameters(self):
        for module in self._unused_map_modules:
            for parameter in module.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _load_prefixed(module, source, prefix):
        module_state = {
            key[len(prefix) :]: value
            for key, value in source.items()
            if key.startswith(prefix)
        }
        result = module.load_state_dict(module_state, strict=True)
        return len(module_state), list(result.missing_keys), list(result.unexpected_keys)

    def load_cmp_checkpoint(self, checkpoint_path: str) -> dict:
        payload = torch.load(checkpoint_path, map_location="cpu")
        source = payload.get("model_state", payload)
        context_count, context_missing, context_unexpected = self._load_prefixed(
            self.context_encoder,
            source,
            "motion_transformer.context_encoder.",
        )
        decoder_count, decoder_missing, decoder_unexpected = self._load_prefixed(
            self.decoder,
            source,
            "motion_transformer.motion_decoder.",
        )
        self.freeze_unused_map_parameters()
        return {
            "loaded_count": context_count + decoder_count,
            "context_loaded_count": context_count,
            "decoder_loaded_count": decoder_count,
            "missing_nonpersistent": context_missing + decoder_missing,
            "unexpected": context_unexpected + decoder_unexpected,
        }

    def _right_align_history(self, history_box, history_mask):
        if history_box.shape[:-1] != history_mask.shape or history_box.shape[-1] != 7:
            raise ValueError("history_box/history_mask shapes disagree")
        length = history_box.shape[-2]
        if length == self.history_frames:
            return history_box, history_mask
        if length > self.history_frames:
            return (
                history_box[..., -self.history_frames :, :],
                history_mask[..., -self.history_frames :],
            )
        pad = self.history_frames - length
        box_padding = history_box.new_zeros(*history_box.shape[:-2], pad, 7)
        mask_padding = torch.zeros(
            *history_mask.shape[:-1],
            pad,
            dtype=torch.bool,
            device=history_mask.device,
        )
        return (
            torch.cat((box_padding, history_box), dim=-2),
            torch.cat((mask_padding, history_mask), dim=-1),
        )

    def _sample_variable_history(self, history_mask: torch.Tensor) -> torch.Tensor:
        """Causally truncate mature tracks during training.

        The runtime model must forecast at actor age one, but a seven-frame
        causal burn-in otherwise exposes Phase 2 almost exclusively to mature
        histories.  Random suffix truncation supplies the missing training
        distribution without adding observations, GT state, or future data.
        """

        if not self.training or self.history_truncation_probability <= 0.0:
            return history_mask
        truncate = torch.rand(
            history_mask.shape[:-1], device=history_mask.device
        ) < self.history_truncation_probability
        lengths = torch.randint(
            1,
            self.history_frames + 1,
            history_mask.shape[:-1],
            device=history_mask.device,
        )
        lengths = torch.where(
            truncate,
            lengths,
            torch.full_like(lengths, self.history_frames),
        )
        time_index = torch.arange(
            self.history_frames, device=history_mask.device
        )
        suffix = time_index.view(*([1] * (history_mask.ndim - 1)), -1) >= (
            self.history_frames - lengths[..., None]
        )
        return history_mask & suffix

    def _inject_query_features(
        self,
        center_feature: torch.Tensor,
        object_feature: torch.Tensor,
        center_query: torch.Tensor,
        context_query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            center_feature + self.query_feature_projection(center_query),
            object_feature + self.query_feature_projection(context_query),
        )

    def _causal_smooth_headings(
        self, history_box: torch.Tensor, history_mask: torch.Tensor
    ) -> torch.Tensor:
        """Circularly smooth query yaw without using a future observation.

        CMP's AB3DMOT history contains Kalman-filtered headings.  Persistent
        query boxes are direct neural measurements and are appreciably noisier.
        The released V2V4Real MTR then amplifies any radian error by converting
        the yaw to degrees and consuming that number as a trigonometric angle.
        This causal circular window restores the missing tracker-side heading
        filtering while retaining the checkpoint's exact downstream contract.
        """

        if self.heading_smoothing_steps == 1:
            return history_box
        smoothed = history_box.clone()
        yaw = history_box[..., 6]
        for time_index in range(history_box.shape[-2]):
            start = max(0, time_index + 1 - self.heading_smoothing_steps)
            window_mask = history_mask[..., start : time_index + 1]
            sine = (
                torch.sin(yaw[..., start : time_index + 1])
                * window_mask.to(yaw.dtype)
            ).sum(dim=-1)
            cosine = (
                torch.cos(yaw[..., start : time_index + 1])
                * window_mask.to(yaw.dtype)
            ).sum(dim=-1)
            valid = window_mask.any(dim=-1)
            mean_yaw = torch.atan2(sine, cosine)
            smoothed[..., time_index, 6] = torch.where(
                valid, mean_yaw, yaw[..., time_index]
            )
        return smoothed

    def _history_features_for_center(
        self,
        history_box,
        history_mask,
        center_box,
        center_context_index,
        receiver_time_s,
    ):
        """Reproduce CMP's 22-D V2V4Real agent-polyline attributes."""

        context_count, timestamp_count, _ = history_box.shape
        relative_xyz = history_box[..., :3] - center_box[None, None, :3]
        # The released V2V4Real tracking cache stores degrees, while its MTR
        # dataset leaves degrees-to-radians conversion commented out before
        # these trigonometric operations.  Preserve canonical radians in the
        # persistent tracker and reproduce that legacy pretrained contract
        # only inside this adapter boundary.
        cmp_center_heading = torch.rad2deg(center_box[6])
        cosine = torch.cos(cmp_center_heading)
        sine = torch.sin(cmp_center_heading)
        relative_x = cosine * relative_xyz[..., 0] + sine * relative_xyz[..., 1]
        relative_y = -sine * relative_xyz[..., 0] + cosine * relative_xyz[..., 1]
        relative_xyz = torch.stack(
            (relative_x, relative_y, relative_xyz[..., 2]), dim=-1
        )

        # CoLIF boxes use [h,w,l]; CMP/MTR tracking trajectories use [l,w,h].
        dimensions_lwh = history_box[..., [5, 4, 3]]
        object_flags = history_box.new_zeros(context_count, timestamp_count, 2)
        object_flags[..., 0] = 1.0
        object_flags[center_context_index, :, 1] = 1.0

        time_one_hot = torch.eye(
            timestamp_count,
            dtype=history_box.dtype,
            device=history_box.device,
        ).unsqueeze(0).expand(context_count, -1, -1)
        # CMP constructs ``0.1 * (0.1 * frame_index)`` for the whole receiver
        # timeline and then slices the eleven history frames.  Therefore the
        # temporal scalar is scene-relative (for example frame 120 produces
        # 1.10, ..., 1.20), not a [0.00, ..., 0.10] clock restarted for every
        # sample.  ``receiver_time_s`` is the physical 10 Hz scene time; the
        # second scale below reproduces CMP's historical 0.01-per-frame input.
        relative_time_s = (
            torch.arange(
            timestamp_count,
            dtype=history_box.dtype,
            device=history_box.device,
            )
            - (timestamp_count - 1)
        ) * self.frame_period_s
        timestamps = (
            receiver_time_s.to(history_box.dtype) + relative_time_s
        ) * self.cmp_timestamp_scale
        timestamps = timestamps.view(1, timestamp_count, 1).expand(
            context_count, -1, -1
        )

        relative_heading = (
            torch.rad2deg(history_box[..., 6]) - cmp_center_heading
        )
        heading_embedding = torch.stack(
            (torch.sin(relative_heading), torch.cos(relative_heading)), dim=-1
        )
        features = torch.cat(
            (
                relative_xyz,
                dimensions_lwh,
                object_flags,
                time_one_hot,
                timestamps,
                heading_embedding,
            ),
            dim=-1,
        )
        if features.shape[-1] != 22:
            raise AssertionError("CMP V2V4Real history attributes must have 22 channels")
        features = features * history_mask[..., None].to(features.dtype)

        last_position = relative_xyz.new_zeros(context_count, 3)
        for time_index in range(timestamp_count):
            valid = history_mask[:, time_index]
            last_position[valid] = relative_xyz[valid, time_index]
        return features, last_position

    def _pack_context(
        self,
        query_feat,
        boxes,
        valid_mask,
        history_box,
        history_mask,
        receiver_time_s,
    ):
        history_box, history_mask = self._right_align_history(
            history_box, history_mask
        )
        history_mask = self._sample_variable_history(history_mask)
        history_box = self._causal_smooth_headings(history_box, history_mask)
        if boxes.shape[:2] != valid_mask.shape:
            raise ValueError("boxes and valid_mask shapes disagree")
        if history_box.shape[:2] != valid_mask.shape:
            raise ValueError("history batch/query dimensions disagree")
        if receiver_time_s.shape != (query_feat.shape[0],):
            raise ValueError("receiver_time_s must have shape [batch]")

        history_steps = history_mask.sum(dim=-1)
        center_valid_mask = valid_mask & (
            history_steps >= self.min_forecast_history_steps
        )
        context_valid_mask = valid_mask & (
            history_steps >= self.min_context_history_steps
        )
        entries = []
        active_by_batch = []
        for batch_index in range(query_feat.shape[0]):
            active = torch.where(context_valid_mask[batch_index])[0]
            active_by_batch.append(active)
            centers = torch.where(center_valid_mask[batch_index])[0]
            entries.extend((batch_index, int(slot)) for slot in centers)
        if not entries:
            return None

        max_context = max(int(indices.numel()) for indices in active_by_batch)
        actor_count = len(entries)
        trajectories = query_feat.new_zeros(
            actor_count, max_context, self.history_frames, 22
        )
        trajectory_mask = torch.zeros(
            actor_count,
            max_context,
            self.history_frames,
            dtype=torch.bool,
            device=query_feat.device,
        )
        last_position = query_feat.new_zeros(actor_count, max_context, 3)
        center_index = torch.zeros(actor_count, dtype=torch.long, device=query_feat.device)
        context_query = query_feat.new_zeros(
            actor_count, max_context, self.query_dim
        )
        center_query = query_feat.new_zeros(actor_count, self.query_dim)
        center_heading = query_feat.new_zeros(actor_count)
        context_slot_indices = torch.full(
            (actor_count, max_context),
            -1,
            dtype=torch.long,
            device=query_feat.device,
        )

        for row, (batch_index, center_slot) in enumerate(entries):
            context_slots = active_by_batch[batch_index]
            count = int(context_slots.numel())
            matching_center = torch.where(context_slots == center_slot)[0]
            if matching_center.numel() != 1:
                raise AssertionError("active center query must occur exactly once")
            center_context_index = int(matching_center[0])
            current_box = boxes[batch_index, center_slot].clone()
            current_valid = torch.where(
                history_mask[batch_index, center_slot]
            )[0]
            if current_valid.numel():
                current_box[6] = history_box[
                    batch_index, center_slot, int(current_valid[-1]), 6
                ]
            features, positions = self._history_features_for_center(
                history_box[batch_index, context_slots],
                history_mask[batch_index, context_slots],
                current_box,
                center_context_index,
                receiver_time_s[batch_index],
            )
            trajectories[row, :count] = features
            trajectory_mask[row, :count] = history_mask[batch_index, context_slots]
            last_position[row, :count] = positions
            center_index[row] = center_context_index
            context_query[row, :count] = query_feat[batch_index, context_slots]
            context_slot_indices[row, :count] = context_slots
            center_query[row] = query_feat[batch_index, center_slot]
            center_heading[row] = current_box[6]

        return entries, {
            "input_dict": {
                "obj_trajs": trajectories,
                "obj_trajs_mask": trajectory_mask,
                "obj_trajs_last_pos": last_position,
                "track_index_to_predict": center_index,
            },
            "batch_sample_count": [actor_count],
            "colif_context_slots": context_slot_indices,
        }, center_query, context_query, center_heading

    @staticmethod
    def _local_gmm_to_receiver(trajectories, heading):
        """Rotate actor-local bivariate GMM parameters to receiver axes."""

        if trajectories.shape[-1] != 5:
            raise ValueError("MTR GMM trajectories must have five parameters")
        if trajectories.shape[0] != heading.shape[0]:
            raise ValueError("heading count must match packed center actors")
        cosine = torch.cos(heading)[:, None, None]
        sine = torch.sin(heading)[:, None, None]
        local_x, local_y = trajectories[..., 0], trajectories[..., 1]
        receiver_x = cosine * local_x - sine * local_y
        receiver_y = sine * local_x + cosine * local_y

        std_x = trajectories[..., 2].clamp(-1.609, 5.0).exp()
        std_y = trajectories[..., 3].clamp(-1.609, 5.0).exp()
        rho = trajectories[..., 4].clamp(-0.5, 0.5)
        variance_x = std_x.square()
        variance_y = std_y.square()
        covariance_xy = rho * std_x * std_y
        receiver_variance_x = (
            cosine.square() * variance_x
            + sine.square() * variance_y
            - 2.0 * cosine * sine * covariance_xy
        ).clamp_min(1e-8)
        receiver_variance_y = (
            sine.square() * variance_x
            + cosine.square() * variance_y
            + 2.0 * cosine * sine * covariance_xy
        ).clamp_min(1e-8)
        receiver_covariance_xy = (
            cosine * sine * (variance_x - variance_y)
            + (cosine.square() - sine.square()) * covariance_xy
        )
        receiver_std_x = receiver_variance_x.sqrt()
        receiver_std_y = receiver_variance_y.sqrt()
        receiver_rho = (
            receiver_covariance_xy / (receiver_std_x * receiver_std_y)
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
        return {
            "pred_scores": query_feat.new_full((batch, queries, self.modes), -20.0),
            "pred_trajs": query_feat.new_zeros(
                batch, queries, self.modes, self.future_frames, 5
            ),
            "pred_trajs_local": query_feat.new_zeros(
                batch, queries, self.modes, self.future_frames, 5
            ),
            # Training follows CMP/MTR on all 64 intention modes.  NMS to six
            # modes is an inference/output operation and must not delete the
            # other modes before best-mode supervision is computed.
            "loss_pred_scores": query_feat.new_full(
                (batch, queries, self.raw_modes), -20.0
            ),
            "loss_pred_trajs_local": query_feat.new_zeros(
                batch, queries, self.raw_modes, self.future_frames, 5
            ),
            # CMP/MTR applies the same trajectory and mode supervision to all
            # decoder layers and averages their losses.  Keep the final-layer
            # aliases above for public/backward-compatible outputs, while the
            # criterion consumes these complete layer stacks.
            "loss_pred_scores_layers": query_feat.new_full(
                (batch, queries, self.decoder_layers, self.raw_modes), -20.0
            ),
            "loss_pred_trajs_local_layers": query_feat.new_zeros(
                batch,
                queries,
                self.decoder_layers,
                self.raw_modes,
                self.future_frames,
                5,
            ),
            "loss_intention_points": query_feat.new_zeros(
                batch, queries, self.raw_modes, 2
            ),
            "future_cue": query_feat.new_zeros(batch, queries, self.query_dim),
            "future_uncertainty": query_feat.new_zeros(batch, queries, 1),
            "forecast_valid_mask": torch.zeros_like(valid_mask),
            "forecast_history_steps": torch.zeros_like(valid_mask, dtype=torch.long),
            "forecast_frame_heading": query_feat.new_zeros(batch, queries),
            "forecast_loss_in_receiver_frame": False,
            # Criterion-only packed tensors for CMP's dense future auxiliary
            # objective.  They contain no GT identity or future target and are
            # never copied into recurrent runtime state.
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
        receiver_time_s,
        cooperative_context=None,
    ):
        # CMP's KNN, BatchNorm and long-horizon GMM path is kept in float32.
        with torch.cuda.amp.autocast(enabled=False):
            if cooperative_context is not None:
                if cooperative_context.shape != query_feat.shape:
                    raise ValueError("cooperative_context must match query_feat")
                query_feat = query_feat + cooperative_context
            return self._forward_float(
                query_feat.float(),
                boxes.float(),
                valid_mask,
                history_box.float(),
                history_mask,
                receiver_time_s.float(),
            )

    def _forward_float(
        self,
        query_feat,
        boxes,
        valid_mask,
        history_box,
        history_mask,
        receiver_time_s,
    ):
        output = self.empty_outputs(query_feat, valid_mask)
        if not self.enabled:
            return output
        packed = self._pack_context(
            query_feat,
            boxes,
            valid_mask,
            history_box,
            history_mask,
            receiver_time_s,
        )
        if packed is None:
            return output
        entries, context_batch, center_query, context_query, packed_heading = packed
        context_batch = self.context_encoder(context_batch)
        center = context_batch["center_objects_feature"]
        obj = context_batch["obj_feature"]
        obj_mask = context_batch["obj_mask"]
        obj_pos = context_batch["obj_pos"]
        center, obj = self._inject_query_features(
            center, obj, center_query, context_query
        )

        decoder = self.decoder
        center = decoder.in_proj_center_obj(center)
        valid_obj = decoder.in_proj_obj(obj[obj_mask])
        projected_obj = obj.new_zeros(*obj.shape[:-1], valid_obj.shape[-1])
        projected_obj[obj_mask] = valid_obj
        projected_obj, dense_trajs = decoder.apply_dense_future_prediction(
            projected_obj, obj_mask, obj_pos
        )
        types = ["TYPE_VEHICLE"] * len(entries)
        pred_list = decoder.apply_transformer_decoder(
            center, types, projected_obj, obj_mask, obj_pos
        )
        if len(pred_list) != self.decoder_layers:
            raise AssertionError("MTR decoder layer count changed at runtime")
        layer_scores = torch.stack([item[0] for item in pred_list], dim=1)
        layer_trajs = torch.stack([item[1] for item in pred_list], dim=1)
        raw_scores, raw_trajs = pred_list[-1]
        raw_intention_points = decoder.forward_ret_dict["intention_points"]
        probabilities = torch.softmax(raw_scores, dim=-1)
        selected_trajs, selected_probabilities, selected_indices = batch_nms(
            pred_trajs=raw_trajs,
            pred_scores=probabilities,
            dist_thresh=decoder.model_cfg.NMS_DIST_THRESH,
            num_ret_modes=self.modes,
        )
        selected_scores = raw_scores.gather(1, selected_indices)
        packed_heading_legacy = torch.rad2deg(packed_heading)
        selected_trajs_receiver = self._local_gmm_to_receiver(
            selected_trajs, packed_heading_legacy
        )
        latent = decoder.forward_ret_dict["query_content_list"].permute(1, 0, 2)
        selected_latent = latent.gather(
            1,
            selected_indices[..., None].expand(-1, -1, latent.shape[-1]),
        )
        selected_latent = self.latent_projection(selected_latent)
        selected_probability = torch.softmax(selected_scores, dim=-1)
        cue = torch.einsum("mk,mkd->md", selected_probability, selected_latent)
        entropy = -(
            selected_probability
            * torch.log(selected_probability.clamp_min(1e-8))
        ).sum(dim=-1, keepdim=True) / math.log(self.modes)

        output["packed_entries"] = tuple(entries)
        output["packed_context_slots"] = context_batch[
            "colif_context_slots"
        ]
        output["packed_dense_trajs_local"] = dense_trajs
        output["packed_center_heading"] = packed_heading

        for row, (batch_index, slot) in enumerate(entries):
            # CMP's public decoder contract returns probabilities after
            # softmax/NMS.  Keep raw logits only in the explicit loss tensors.
            output["pred_scores"][batch_index, slot] = (
                selected_probabilities[row]
            )
            output["pred_trajs"][batch_index, slot] = selected_trajs_receiver[row]
            output["pred_trajs_local"][batch_index, slot] = selected_trajs[row]
            output["loss_pred_scores"][batch_index, slot] = raw_scores[row]
            output["loss_pred_trajs_local"][batch_index, slot] = raw_trajs[row]
            output["loss_pred_scores_layers"][batch_index, slot] = (
                layer_scores[row]
            )
            output["loss_pred_trajs_local_layers"][batch_index, slot] = (
                layer_trajs[row]
            )
            output["loss_intention_points"][batch_index, slot] = (
                raw_intention_points[row]
            )
            output["future_cue"][batch_index, slot] = cue[row]
            output["future_uncertainty"][batch_index, slot] = entropy[row]
            output["forecast_valid_mask"][batch_index, slot] = True
            output["forecast_history_steps"][batch_index, slot] = history_mask[
                batch_index, slot
            ].sum()
            output["forecast_frame_heading"][batch_index, slot] = packed_heading[row]
        return output
