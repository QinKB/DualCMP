"""History-conditioned multimodal forecasting for persistent queries."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionAnchoredForecastHead(nn.Module):
    """Predict learned residuals around history-derived motion anchors.

    A single observation creates a persistent actor state but cannot identify
    velocity.  Once two observations exist, pairwise history velocities form a
    robust motion estimate.  Six generic speed hypotheses provide a stable
    multimodal backbone; query/context features learn interaction-conditioned
    residual trajectories, uncertainty, and mode probability.  The module uses
    only observations available at the current receiver time and is actor-ID
    free.
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
        scales = args.get(
            "motion_anchor_scales", [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]
        )
        self.modes = len(scales)
        self.mode_interaction = str(
            args.get("mode_interaction", "additive")
        )
        self.trajectory_parameterization = str(
            args.get("trajectory_parameterization", "anchored")
        )
        if self.mode_interaction not in ("additive", "actor_conditioned"):
            raise ValueError("Unsupported motion-anchor mode interaction")
        if self.trajectory_parameterization not in ("anchored", "direct"):
            raise ValueError(
                "trajectory_parameterization must be anchored or direct"
            )
        if self.modes < 2:
            raise ValueError("motion-anchored forecast requires multiple modes")
        if self.frame_period_s <= 0.0:
            raise ValueError("frame_period_s must be positive")
        if not 2 <= self.min_forecast_history_steps <= self.history_frames:
            raise ValueError(
                "motion anchors require between 2 and history_frames observations"
            )
        self.register_buffer(
            "anchor_scales", torch.as_tensor(scales, dtype=torch.float32)
        )
        self.register_buffer(
            "future_times_s",
            torch.arange(1, self.future_frames + 1, dtype=torch.float32)
            * self.frame_period_s,
        )

        descriptor_dim = 11
        self.query_norm = nn.LayerNorm(self.query_dim)
        self.cooperative_norm = nn.LayerNorm(self.query_dim)
        self.cooperative_scale = nn.Parameter(torch.tensor(0.1))
        self.temporal_history_enabled = bool(
            args.get("temporal_history", True)
        )
        self.history_projection = nn.Sequential(
            nn.Linear(14, self.query_dim),
            nn.GELU(),
            nn.Linear(self.query_dim, self.query_dim),
        )
        self.history_position = nn.Parameter(
            torch.randn(self.history_frames, self.query_dim) * 0.02
        )
        history_layer = nn.TransformerEncoderLayer(
            d_model=self.query_dim,
            nhead=int(args.get("history_encoder_heads", 8)),
            dim_feedforward=int(
                args.get("history_encoder_ffn", self.query_dim * 4)
            ),
            dropout=float(args.get("dropout", 0.1)),
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(
            history_layer,
            num_layers=int(args.get("history_encoder_layers", 2)),
        )
        self.history_norm = nn.LayerNorm(self.query_dim)
        self.history_scale = nn.Parameter(torch.tensor(0.1))
        self.motion_projection = nn.Sequential(
            nn.Linear(descriptor_dim, self.query_dim),
            nn.GELU(),
            nn.Linear(self.query_dim, self.query_dim),
        )
        self.box_projection = nn.Sequential(
            nn.Linear(8, self.query_dim), nn.GELU(), nn.Linear(self.query_dim, self.query_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.query_dim,
            nhead=int(args.get("motion_context_heads", 8)),
            dim_feedforward=int(args.get("motion_context_ffn", self.query_dim * 4)),
            dropout=float(args.get("dropout", 0.1)),
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(args.get("motion_context_layers", 2)),
        )
        self.mode_embedding = nn.Parameter(
            torch.randn(self.modes, self.query_dim) * 0.02
        )
        self.proposal_refinement = bool(
            args.get("proposal_refinement", True)
        )
        self.inference_layer = str(
            args.get("inference_layer", "proposal")
        )
        if self.inference_layer not in ("proposal", "refined"):
            raise ValueError("inference_layer must be proposal or refined")
        self.loss_layers = 2 if self.proposal_refinement else 1
        self.mode_refiner = nn.TransformerEncoderLayer(
            d_model=self.query_dim,
            nhead=int(args.get("motion_context_heads", 8)),
            dim_feedforward=int(
                args.get("motion_context_ffn", self.query_dim * 4)
            ),
            dropout=float(args.get("dropout", 0.1)),
            batch_first=True,
            norm_first=True,
        )
        self.proposal_mode_score = nn.Linear(self.query_dim, 1)
        self.proposal_residual_head = nn.Linear(
            self.query_dim, self.future_frames * 2
        )
        self.proposal_log_std_head = nn.Linear(
            self.query_dim, self.future_frames * 2
        )
        self.proposal_rho_head = nn.Linear(
            self.query_dim, self.future_frames
        )
        self.mode_score = nn.Linear(self.query_dim, 1)
        self.residual_head = nn.Linear(
            self.query_dim, self.future_frames * 2
        )
        self.log_std_head = nn.Linear(
            self.query_dim, self.future_frames * 2
        )
        self.rho_head = nn.Linear(self.query_dim, self.future_frames)
        self.future_token = nn.Linear(self.query_dim, self.query_dim)
        self.residual_limit_m = float(args.get("motion_residual_limit_m", 12.0))
        if self.residual_limit_m <= 0.0:
            raise ValueError("motion_residual_limit_m must be positive")

        # The anchored model starts from the history-derived physics model.  The
        # direct ablation uses the same heads but starts from zero displacement.
        for score_head in (self.proposal_mode_score, self.mode_score):
            if self.mode_interaction == "actor_conditioned":
                nn.init.normal_(score_head.weight, mean=0.0, std=0.02)
            else:
                nn.init.zeros_(score_head.weight)
            nn.init.zeros_(score_head.bias)
        for residual_head in (
            self.proposal_residual_head,
            self.residual_head,
        ):
            nn.init.zeros_(residual_head.weight)
            nn.init.zeros_(residual_head.bias)
        for std_head in (
            self.proposal_log_std_head,
            self.log_std_head,
        ):
            nn.init.zeros_(std_head.weight)
            nn.init.constant_(std_head.bias, math.log(5.0))
        for rho_head in (self.proposal_rho_head, self.rho_head):
            nn.init.zeros_(rho_head.weight)
            nn.init.zeros_(rho_head.bias)
        self.enabled = bool(args.get("enabled", True))

        pair_i, pair_j = torch.triu_indices(
            self.history_frames, self.history_frames, offset=1
        )
        self.register_buffer("pair_i", pair_i)
        self.register_buffer("pair_j", pair_j)

    def freeze_unused_map_parameters(self):
        """Interface parity with the optional CMP/MTR adapter."""

    def freeze_inactive_parameters(
        self, actor_cooperative_fusion_enabled: bool = True
    ) -> None:
        """Remove ablated branches from optimization and DDP traversal."""

        if not self.temporal_history_enabled:
            for module in (
                self.history_projection,
                self.history_encoder,
                self.history_norm,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = False
            self.history_position.requires_grad = False
            self.history_scale.requires_grad = False
        if not self.proposal_refinement:
            for module in (
                self.mode_refiner,
                self.proposal_mode_score,
                self.proposal_residual_head,
                self.proposal_log_std_head,
                self.proposal_rho_head,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = False
        if not actor_cooperative_fusion_enabled:
            for parameter in self.cooperative_norm.parameters():
                parameter.requires_grad = False
            self.cooperative_scale.requires_grad = False

    def _right_align_history(self, history_box, history_mask):
        length = history_box.shape[-2]
        if length > self.history_frames:
            return (
                history_box[..., -self.history_frames :, :],
                history_mask[..., -self.history_frames :],
            )
        if length == self.history_frames:
            return history_box, history_mask
        pad = self.history_frames - length
        box_pad = history_box.new_zeros(*history_box.shape[:-2], pad, 7)
        mask_pad = torch.zeros(
            *history_mask.shape[:-1],
            pad,
            dtype=torch.bool,
            device=history_mask.device,
        )
        return (
            torch.cat((box_pad, history_box), dim=-2),
            torch.cat((mask_pad, history_mask), dim=-1),
        )

    def _robust_motion(self, history_box, history_mask):
        """Return pairwise-median velocity and dispersion from observed history."""

        xy = history_box[..., :2]
        pair_valid = (
            history_mask[..., self.pair_i] & history_mask[..., self.pair_j]
        )
        elapsed = (
            (self.pair_j - self.pair_i).to(xy.dtype) * self.frame_period_s
        )
        pair_velocity = (
            xy[..., self.pair_j, :] - xy[..., self.pair_i, :]
        ) / elapsed.view(*([1] * (xy.ndim - 2)), -1, 1)
        count = pair_valid.sum(dim=-1)
        median_index = ((count - 1).clamp_min(0) // 2)[..., None, None]
        masked = torch.where(
            pair_valid[..., None],
            pair_velocity,
            torch.full_like(pair_velocity, float("inf")),
        )
        ordered = masked.sort(dim=-2).values
        median_velocity = ordered.gather(
            -2, median_index.expand(*count.shape, 1, 2)
        ).squeeze(-2)
        median_velocity = torch.where(
            (count > 0)[..., None],
            median_velocity,
            torch.zeros_like(median_velocity),
        )
        deviation = torch.linalg.vector_norm(
            pair_velocity - median_velocity[..., None, :], dim=-1
        )
        dispersion = (
            deviation * pair_valid.to(deviation.dtype)
        ).sum(dim=-1) / count.clamp_min(1).to(deviation.dtype)
        return median_velocity, dispersion[..., None]

    def _motion_descriptor(
        self, boxes, history_box, history_mask, velocity, dispersion
    ):
        speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
        steps = history_mask.sum(dim=-1, keepdim=True).to(boxes.dtype)
        yaw = boxes[..., 6:7]
        last_step = history_box[..., -1, :2]
        current_offset = last_step - boxes[..., :2]
        return torch.cat(
            (
                velocity / 20.0,
                speed / 20.0,
                dispersion / 10.0,
                steps / float(self.history_frames),
                torch.sin(yaw),
                torch.cos(yaw),
                current_offset / 5.0,
                boxes[..., 3:5] / 5.0,
            ),
            dim=-1,
        )

    def _encode_temporal_history(self, history_box, history_mask):
        """Encode the observed history prefix with bidirectional self-attention."""

        history_box, history_mask = self._right_align_history(
            history_box, history_mask
        )
        if not self.temporal_history_enabled:
            return history_box.new_zeros(
                *history_mask.shape[:-1], self.query_dim
            )
        xy = history_box[..., :2]
        latest_index = torch.where(
            history_mask,
            torch.arange(
                self.history_frames,
                device=history_mask.device,
            ).view(*([1] * (history_mask.ndim - 1)), -1),
            torch.full_like(history_mask, -1, dtype=torch.long),
        ).max(dim=-1).values.clamp_min(0)
        gather_xy = latest_index[..., None, None].expand(
            *latest_index.shape, 1, 2
        )
        latest_xy = xy.gather(-2, gather_xy).squeeze(-2)
        relative_xy = (xy - latest_xy[..., None, :]) / 20.0

        displacement = torch.zeros_like(xy)
        transition = history_mask[..., 1:] & history_mask[..., :-1]
        displacement[..., 1:, :] = (
            xy[..., 1:, :] - xy[..., :-1, :]
        ) * transition[..., None].to(xy.dtype)
        normalized_displacement = displacement / 2.0
        acceleration = torch.zeros_like(xy)
        acceleration[..., 2:, :] = (
            displacement[..., 2:, :] - displacement[..., 1:-1, :]
        ) * (
            transition[..., 1:] & transition[..., :-1]
        )[..., None].to(xy.dtype)
        normalized_acceleration = acceleration / 2.0
        speed = torch.linalg.vector_norm(
            normalized_displacement, dim=-1, keepdim=True
        )
        yaw = history_box[..., 6:7]
        time = torch.linspace(
            -1.0,
            0.0,
            self.history_frames,
            device=history_box.device,
            dtype=history_box.dtype,
        ).view(*([1] * (history_box.ndim - 2)), self.history_frames, 1)
        token_input = torch.cat(
            (
                relative_xy,
                normalized_displacement,
                normalized_acceleration,
                history_box[..., 2:3] / 5.0,
                history_box[..., 3:6] / 8.0,
                torch.sin(yaw),
                torch.cos(yaw),
                time.expand(*history_box.shape[:-2], self.history_frames, 1),
                speed,
            ),
            dim=-1,
        )
        tokens = self.history_projection(token_input)
        tokens = tokens + self.history_position.view(
            *([1] * (tokens.ndim - 2)), self.history_frames, self.query_dim
        )
        flat_tokens = tokens.reshape(-1, self.history_frames, self.query_dim)
        flat_mask = history_mask.reshape(-1, self.history_frames).bool()
        has_history = flat_mask.any(dim=-1)
        safe_mask = flat_mask.clone()
        safe_mask[~has_history, -1] = True
        encoded = self.history_encoder(
            flat_tokens,
            src_key_padding_mask=~safe_mask,
        )
        flat_latest = latest_index.reshape(-1)
        row = torch.arange(encoded.shape[0], device=encoded.device)
        pooled = encoded[row, flat_latest]
        pooled = self.history_norm(pooled).view(
            *history_mask.shape[:-1], self.query_dim
        )
        pooled = torch.where(
            history_mask.any(dim=-1)[..., None],
            pooled,
            torch.zeros_like(pooled),
        )
        return pooled

    @staticmethod
    def _normalized_box_features(boxes):
        return torch.cat(
            (
                boxes[..., :2] / boxes.new_tensor([102.4, 38.4]),
                boxes[..., 2:3] / 5.0,
                boxes[..., 3:6] / boxes.new_tensor([3.0, 3.0, 8.0]),
                torch.sin(boxes[..., 6:7]),
                torch.cos(boxes[..., 6:7]),
            ),
            dim=-1,
        )

    def _interact(self, feature, valid_mask):
        result = feature.clone()
        for batch_index in range(feature.shape[0]):
            slots = torch.where(valid_mask[batch_index])[0]
            if slots.numel():
                result[batch_index, slots] = self.context_encoder(
                    feature[batch_index, slots].unsqueeze(0)
                ).squeeze(0)
        return result

    def _condition_modes(self, context):
        mode = self.mode_embedding.view(
            1, 1, self.modes, self.query_dim
        )
        actor = context.unsqueeze(2)
        if self.mode_interaction == "additive":
            return actor + mode
        # The additive form makes a shared linear mode head actor-invariant
        # after softmax: w^T actor is the same offset for every mode.  FiLM-like
        # multiplication retains the physics-initialized mode tokens while
        # allowing each actor feature to affect modes differently.
        return actor * (1.0 + torch.tanh(mode)) + mode

    def empty_outputs(self, query_feat, valid_mask):
        batch, queries, _ = query_feat.shape
        trajectories = query_feat.new_zeros(
            batch, queries, self.modes, self.future_frames, 5
        )
        scores = query_feat.new_zeros(batch, queries, self.modes)
        return {
            "pred_scores": scores,
            "pred_trajs": trajectories,
            "pred_trajs_local": trajectories,
            "loss_pred_scores": scores,
            "loss_pred_trajs_local": trajectories,
            "loss_pred_scores_layers": scores.unsqueeze(2).expand(
                batch, queries, self.loss_layers, self.modes
            ),
            "loss_pred_trajs_local_layers": trajectories.unsqueeze(2).expand(
                batch,
                queries,
                self.loss_layers,
                self.modes,
                self.future_frames,
                5,
            ),
            "loss_intention_points": query_feat.new_zeros(
                batch, queries, self.modes, 2
            ),
            "loss_intention_points_layers": query_feat.new_zeros(
                batch, queries, self.loss_layers, self.modes, 2
            ),
            "future_cue": query_feat.new_zeros(batch, queries, self.query_dim),
            "future_uncertainty": query_feat.new_zeros(batch, queries, 1),
            "forecast_valid_mask": torch.zeros_like(valid_mask),
            "forecast_history_steps": torch.zeros_like(valid_mask, dtype=torch.long),
            "forecast_frame_heading": query_feat.new_zeros(batch, queries),
            "forecast_loss_in_receiver_frame": True,
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
        del receiver_time_s
        output = self.empty_outputs(query_feat, valid_mask)
        if not self.enabled:
            return output
        history_box, history_mask = self._right_align_history(
            history_box, history_mask
        )
        steps = history_mask.sum(dim=-1)
        motion_valid = valid_mask & (steps >= self.min_forecast_history_steps)
        velocity, dispersion = self._robust_motion(history_box, history_mask)
        descriptor = self._motion_descriptor(
            boxes, history_box, history_mask, velocity, dispersion
        )
        temporal_history = self._encode_temporal_history(
            history_box, history_mask
        )
        if cooperative_context is None:
            cooperative_context = torch.zeros_like(query_feat)
        if cooperative_context.shape != query_feat.shape:
            raise ValueError("cooperative_context must match query_feat")
        feature = (
            self.query_norm(query_feat)
            + self.cooperative_scale.tanh()
            * self.cooperative_norm(cooperative_context)
            + self.history_scale.tanh() * temporal_history
            + self.motion_projection(descriptor)
            + self.box_projection(self._normalized_box_features(boxes))
        )
        context = self._interact(feature, motion_valid)
        mode_feature = self._condition_modes(context)
        if self.trajectory_parameterization == "anchored":
            anchors = (
                velocity[..., None, None, :]
                * self.anchor_scales.view(1, 1, self.modes, 1, 1)
                * self.future_times_s.view(1, 1, 1, self.future_frames, 1)
            )
        else:
            anchors = velocity.new_zeros(
                *velocity.shape[:-1], self.modes, self.future_frames, 2
            )
        if self.proposal_refinement:
            proposal_logits = self.proposal_mode_score(mode_feature).squeeze(-1)
            proposal_raw = self.proposal_residual_head(mode_feature).view(
                *mode_feature.shape[:-1], self.future_frames, 2
            )
            proposal_residual = (
                proposal_raw
                if self.trajectory_parameterization == "direct"
                else self.residual_limit_m * torch.tanh(
                    proposal_raw / self.residual_limit_m
                )
            )
            proposal_mean = anchors + proposal_residual
            proposal_log_std = self.proposal_log_std_head(mode_feature).view(
                *mode_feature.shape[:-1], self.future_frames, 2
            ).clamp(-1.609, 5.0)
            proposal_rho = 0.5 * torch.tanh(
                self.proposal_rho_head(mode_feature).view(
                    *mode_feature.shape[:-1], self.future_frames, 1
                )
            )
            proposal_trajectories = torch.cat(
                (proposal_mean, proposal_log_std, proposal_rho), dim=-1
            )
            refined_feature = self.mode_refiner(
                mode_feature.reshape(-1, self.modes, self.query_dim)
            ).view_as(mode_feature)
            residual_limit = 0.5 * self.residual_limit_m
            base_mean = proposal_mean
        else:
            refined_feature = mode_feature
            residual_limit = self.residual_limit_m
            base_mean = anchors
        logits = self.mode_score(refined_feature).squeeze(-1)
        refinement_raw = self.residual_head(refined_feature).view(
            *refined_feature.shape[:-1], self.future_frames, 2
        )
        refinement = (
            refinement_raw
            if (
                self.trajectory_parameterization == "direct"
                and not self.proposal_refinement
            )
            else residual_limit * torch.tanh(
                refinement_raw / self.residual_limit_m
            )
        )
        mean = base_mean + refinement
        log_std = self.log_std_head(refined_feature).view(
            *refined_feature.shape[:-1], self.future_frames, 2
        ).clamp(-1.609, 5.0)
        rho = 0.5 * torch.tanh(
            self.rho_head(refined_feature).view(
                *refined_feature.shape[:-1], self.future_frames, 1
            )
        )
        trajectories = torch.cat((mean, log_std, rho), dim=-1)
        use_proposal = (
            self.proposal_refinement and self.inference_layer == "proposal"
        )
        public_logits = proposal_logits if use_proposal else logits
        public_trajectories = (
            proposal_trajectories if use_proposal else trajectories
        )
        public_mode_feature = mode_feature if use_proposal else refined_feature
        probabilities = torch.softmax(public_logits, dim=-1)
        mode_tokens = self.future_token(public_mode_feature)
        cue = torch.einsum("bqk,bqkd->bqd", probabilities, mode_tokens)
        entropy = -(
            probabilities * probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1, keepdim=True) / math.log(self.modes)

        float_mask = motion_valid[..., None, None, None].to(trajectories.dtype)
        trajectories = trajectories * float_mask
        public_trajectories = public_trajectories * float_mask
        probabilities = probabilities * motion_valid[..., None].to(probabilities.dtype)
        logits = logits.masked_fill(~motion_valid[..., None], -20.0)
        cue = cue * motion_valid[..., None].to(cue.dtype)
        entropy = entropy * motion_valid[..., None].to(entropy.dtype)
        if self.proposal_refinement:
            proposal_logits = proposal_logits.masked_fill(
                ~motion_valid[..., None], -20.0
            )
            loss_scores_layers = torch.stack(
                (proposal_logits, logits), dim=2
            )
            loss_trajs_layers = torch.stack(
                (proposal_trajectories, trajectories), dim=2
            )
            loss_intentions_layers = torch.stack(
                (proposal_mean[..., -1, :], mean[..., -1, :]), dim=2
            )
        else:
            loss_scores_layers = logits.unsqueeze(2)
            loss_trajs_layers = trajectories.unsqueeze(2)
            loss_intentions_layers = mean[..., -1, :].unsqueeze(2)
        layer_float_mask = motion_valid[..., None, None, None, None].to(
            loss_trajs_layers.dtype
        )
        loss_trajs_layers = loss_trajs_layers * layer_float_mask
        output.update(
            {
                "pred_scores": probabilities,
                "pred_trajs": public_trajectories,
                "pred_trajs_local": public_trajectories,
                "loss_pred_scores": logits,
                "loss_pred_trajs_local": trajectories,
                "loss_pred_scores_layers": loss_scores_layers,
                "loss_pred_trajs_local_layers": loss_trajs_layers,
                "loss_intention_points": mean[..., -1, :],
                "loss_intention_points_layers": loss_intentions_layers,
                "future_cue": cue,
                "future_uncertainty": entropy,
                "forecast_valid_mask": motion_valid,
                "forecast_history_steps": steps,
                "forecast_frame_heading": boxes[..., 6],
            }
        )
        return output
