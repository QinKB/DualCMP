"""CoLIF: CMP fused BEV with persistent 3D forecast queries."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from opencood.models.point_pillar_cobevt import PointPillarCoBEVT
from opencood.models.colif import (
    ActorConditionedCooperativeFusion,
    DenseMeasurementProposalGenerator,
    ForecastMotionPrior,
    FutureAdapter,
    GlobalForecastState,
    MeasurementStateUpdate,
    MotionAnchoredForecastHead,
    PersistentQueryDecoder,
    PersistentQueryMTRAdapter,
    TemporalQueryMemory,
)
from opencood.models.colif.lifecycle import QueryLifecycle
from opencood.models.colif.query_geometry import (
    transform_and_propagate_boxes,
    transform_boxes_between_receiver_frames,
)


FORECAST_BEV_REPRESENTATIONS = ("per_cav", "fused")


def select_forecast_bev_sources(
    representation: str,
    encoded: dict,
    source_to_receiver: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Select spatial evidence only for the forecast reader.

    Tracking always consumes ``encoded['fused_bev']`` before this selection.
    The fused ablation therefore changes forecast evidence without changing
    measurements, query identities, or histories.
    """

    representation = str(representation)
    if representation == "per_cav":
        return (
            encoded["regrouped_source_bev"],
            encoded["cav_mask"].bool(),
            source_to_receiver,
        )
    if representation == "fused":
        fused_bev = encoded["fused_bev"]
        batch = fused_bev.shape[0]
        cav_mask = torch.ones(
            batch, 1, dtype=torch.bool, device=fused_bev.device
        )
        identity = torch.eye(
            4, dtype=fused_bev.dtype, device=fused_bev.device
        ).view(1, 1, 4, 4).expand(batch, 1, 4, 4)
        return fused_bev.unsqueeze(1), cav_mask, identity
    raise ValueError(
        "actor_cooperative_fusion.representation must be one of %s"
        % (FORECAST_BEV_REPRESENTATIONS,)
    )


class PointPillarColif(nn.Module):
    """Online receiver-centric model whose forward contract is supervision-free."""

    def __init__(self, args):
        super().__init__()
        self.cobevt = PointPillarCoBEVT(args)
        decoder_args = args.get("persistent_query", {})
        self.query_decoder = PersistentQueryDecoder(
            decoder_args, lidar_range=args["lidar_range"]
        )
        self.max_queries = self.query_decoder.num_queries
        self.query_dim = self.query_decoder.dim
        prior_probability = float(decoder_args.get("prior_probability", 0.01))
        self.query_prior_logit = float(
            torch.log(torch.tensor(prior_probability / (1.0 - prior_probability)))
        )
        lidar_range = tuple(float(value) for value in args["lidar_range"])
        self.lidar_x_min, self.lidar_y_min = lidar_range[0], lidar_range[1]
        self.lidar_x_max, self.lidar_y_max = lidar_range[3], lidar_range[4]
        state_args = args.get("state", {})
        self.memory_len = int(state_args.get("memory_len", 6))
        self.tracker_memory_len = int(
            state_args.get("tracker_memory_len", min(self.memory_len, 4))
        )
        if not 1 <= self.tracker_memory_len <= self.memory_len:
            raise ValueError(
                "state.tracker_memory_len must lie in [1, state.memory_len]"
            )
        self.temporal_memory = TemporalQueryMemory(self.query_dim)
        cooperative_args = args.get("actor_cooperative_fusion", {})
        self.actor_cooperative_fusion_enabled = bool(
            cooperative_args.get("enabled", True)
        )
        self.forecast_bev_representation = str(
            cooperative_args.get("representation", "per_cav")
        )
        if self.forecast_bev_representation not in FORECAST_BEV_REPRESENTATIONS:
            raise ValueError(
                "actor_cooperative_fusion.representation must be one of %s"
                % (FORECAST_BEV_REPRESENTATIONS,)
            )
        self.actor_cooperative_fusion = ActorConditionedCooperativeFusion(
            input_dim=int(cooperative_args.get("input_dim", 256)),
            query_dim=self.query_dim,
            heads=int(cooperative_args.get("heads", 8)),
            max_cav=int(args["max_cav"]),
            lidar_range=args["lidar_range"],
            patch_radius_m=float(
                cooperative_args.get("patch_radius_m", 4.0)
            ),
            patch_size=int(cooperative_args.get("patch_size", 3)),
            dropout=float(cooperative_args.get("dropout", 0.1)),
            geometry_conditioning=bool(
                cooperative_args.get("geometry_conditioning", False)
            ),
            source_index_embedding=bool(
                cooperative_args.get("source_index_embedding", True)
            ),
        )
        self.measurement_proposals = DenseMeasurementProposalGenerator(
            lidar_range=args["lidar_range"],
            voxel_size=args["voxel_size"],
            anchor_number=int(args["anchor_number"]),
            args=args.get("measurement_proposals", {}),
        )
        self.measurement_match_distance = float(
            args.get("measurement_proposals", {}).get("match_distance", 4.0)
        )
        if self.measurement_match_distance <= 0.0:
            raise ValueError("measurement proposal match distance must be positive")
        self.memory_scale = nn.Parameter(torch.tensor(0.1))
        measurement_update_args = args.get("measurement_state_update", {})
        self.measurement_state_update = MeasurementStateUpdate(
            query_dim=self.query_dim,
            min_gain=float(measurement_update_args.get("min_gain", 0.1)),
            max_gain=float(measurement_update_args.get("max_gain", 0.95)),
            initial_gain=float(
                measurement_update_args.get("initial_gain", 0.75)
            ),
            innovation_scale_m=float(
                measurement_update_args.get("innovation_scale_m", 4.0)
            ),
        )
        future_args = args.get("future_adapter", {})
        # Kept only so historical checkpoints retain an identical parameter
        # schema.  The falsified latent-content recurrence is no longer used
        # by the main model path.
        self.phase2_feedback_mode = "none"
        self.future_adapter = FutureAdapter(
            query_dim=self.query_dim,
            mode="none",
            gate_max=float(future_args.get("gate_max", 0.25)),
            clip_norm=float(future_args.get("clip_norm", 4.0)),
            dropout_probability=float(
                future_args.get("dropout_probability", 0.3)
            ),
        )
        self.lifecycle = QueryLifecycle(
            birth_threshold=float(state_args.get("birth_threshold", 0.45)),
            keep_threshold=float(state_args.get("keep_threshold", 0.25)),
            miss_ttl=int(state_args.get("miss_ttl", 4)),
            birth_suppression_radius=float(
                state_args.get("birth_suppression_radius", 2.0)
            ),
            max_active_queries=int(state_args.get("max_active_queries", 32)),
        )
        forecast_args = dict(args.get("forecast", {}))
        forecast_args.setdefault("lidar_range", args["lidar_range"])
        forecast_architecture = str(
            forecast_args.get("architecture", "cmp_mtr")
        )
        if forecast_architecture == "cmp_mtr":
            self.forecast_head = PersistentQueryMTRAdapter(
                self.query_dim, forecast_args
            )
        elif forecast_architecture == "motion_anchored":
            self.forecast_head = MotionAnchoredForecastHead(
                self.query_dim, forecast_args
            )
        else:
            raise ValueError(
                "Unknown forecast.architecture: %s" % forecast_architecture
            )
        prior_args = args.get("forecast_motion_prior", {})
        self.phase2_future_prior_mode = str(
            prior_args.get("phase2_mode", prior_args.get("mode", "none"))
        )
        self.forecast_motion_prior = ForecastMotionPrior(
            mode=str(prior_args.get("mode", "none")),
            frame_period_s=float(
                prior_args.get(
                    "frame_period_s", forecast_args.get("frame_period_s", 0.1)
                )
            ),
            horizon_steps=int(prior_args.get("horizon_steps", 5)),
            cv_component_weight=float(
                prior_args.get("cv_component_weight", 0.15)
            ),
            cv_std_m=float(prior_args.get("cv_std_m", 1.0)),
            min_std_m=float(prior_args.get("min_std_m", 0.25)),
            max_std_m=float(prior_args.get("max_std_m", 3.0)),
            mahalanobis_gate=float(prior_args.get("mahalanobis_gate", 3.0)),
            absolute_gate_m=float(prior_args.get("absolute_gate_m", 8.0)),
            cv_match_distance_m=self.measurement_match_distance,
            proposal_score_weight=float(
                prior_args.get("proposal_score_weight", 0.25)
            ),
        )
        if self.phase2_future_prior_mode not in ForecastMotionPrior.MODES:
            raise ValueError("Invalid forecast_motion_prior.phase2_mode")
        self.future_prior_modes = int(self.forecast_head.modes)
        self.future_prior_steps = self.forecast_motion_prior.horizon_steps

    def _fuse_detection_evidence(self, query_logits, query_boxes, dense_logits):
        """Read CMP's fused detection heatmap at every decoded query center.

        This is a continuous measurement update, not a decoded detection or
        tracker interface: no score threshold, NMS, GT or discrete assignment
        is used.  A 3x3 local maximum tolerates sub-cell center residuals while
        preserving gradients to both the query and CoBEVT branches.
        """

        heatmap = F.max_pool2d(dense_logits, kernel_size=3, stride=1, padding=1)
        heatmap = heatmap.amax(dim=1, keepdim=True)
        normalized_x = 2.0 * (
            (query_boxes[..., 0] - self.lidar_x_min)
            / (self.lidar_x_max - self.lidar_x_min)
        ) - 1.0
        normalized_y = 2.0 * (
            (query_boxes[..., 1] - self.lidar_y_min)
            / (self.lidar_y_max - self.lidar_y_min)
        ) - 1.0
        sampling_grid = torch.stack((normalized_x, normalized_y), dim=-1)
        sampled = F.grid_sample(
            heatmap,
            sampling_grid[:, None],
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[:, 0, 0]
        # The query classifier and dense head are both trained with the same
        # foreground prior.  Summing their logits directly counts that prior
        # twice and suppresses newborn actors (0.01 + a 0.2 dense score became
        # roughly 0.002).  Subtracting one prior logit is the calibrated
        # product-of-experts form; at initialization the fused score exactly
        # equals CMP's dense evidence.
        prior_logit = float(getattr(self, "query_prior_logit", 0.0))
        return query_logits + sampled - prior_logit, sampled

    @staticmethod
    def _assign_measurements(
        propagated_box: torch.Tensor,
        previous_active: torch.Tensor,
        measurements: dict,
        match_distance: float,
        association_cost: torch.Tensor | None = None,
        association_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign active tracks with private nulls, then seed newborn slots."""

        reference_box = propagated_box.clone()
        birth_candidate_mask = torch.zeros_like(previous_active)
        measurement_observed_mask = torch.zeros_like(previous_active)
        assigned_logit = propagated_box.new_full(previous_active.shape, -20.0)
        proposal_boxes = measurements["proposal_boxes"]
        expected_cost_shape = (
            propagated_box.shape[0],
            propagated_box.shape[1],
            proposal_boxes.shape[1],
        )
        if association_cost is not None and association_cost.shape != expected_cost_shape:
            raise ValueError("association cost must have shape [B,Q,P]")
        if association_valid is not None and association_valid.shape != expected_cost_shape:
            raise ValueError("association validity must have shape [B,Q,P]")
        if (association_cost is None) != (association_valid is None):
            raise ValueError("association cost and validity must be provided together")
        for batch_index in range(propagated_box.shape[0]):
            proposals = torch.where(measurements["proposal_mask"][batch_index])[0]
            if not proposals.numel():
                continue
            used_proposals = torch.zeros(
                proposals.numel(),
                dtype=torch.bool,
                device=propagated_box.device,
            )
            active = torch.where(previous_active[batch_index])[0]
            if active.numel():
                with torch.no_grad():
                    if association_cost is None:
                        pair_cost = torch.cdist(
                            propagated_box[batch_index, active, :2].float(),
                            proposal_boxes[batch_index, proposals, :2].float(),
                        )
                        pair_valid = pair_cost <= match_distance
                    else:
                        pair_cost = association_cost[
                            batch_index, active[:, None], proposals[None, :]
                        ].float()
                        pair_valid = association_valid[
                            batch_index, active[:, None], proposals[None, :]
                        ]
                    # Every active query owns one private null column.  Invalid
                    # proposal pairs are more expensive than that null, so a
                    # bad row cannot steal a good proposal from another query.
                    active_count = int(active.numel())
                    proposal_count = int(proposals.numel())
                    invalid_cost = pair_cost.new_tensor(1e6)
                    null_cost = pair_cost.new_tensor(1e5)
                    augmented = pair_cost.new_full(
                        (active_count, proposal_count + active_count),
                        float(invalid_cost),
                    )
                    augmented[:, :proposal_count] = torch.where(
                        pair_valid,
                        pair_cost.clamp(max=float(null_cost) - 1.0),
                        invalid_cost,
                    )
                    diagonal = torch.arange(active_count, device=augmented.device)
                    augmented[diagonal, proposal_count + diagonal] = null_cost
                    rows, columns = linear_sum_assignment(
                        augmented.detach().cpu().numpy()
                    )
                for row, column in zip(rows.tolist(), columns.tolist()):
                    if column >= int(proposals.numel()):
                        continue
                    slot = active[row]
                    proposal = proposals[column]
                    reference_box[batch_index, slot] = measurements[
                        "proposal_boxes"
                    ][batch_index, proposal].to(reference_box.dtype)
                    assigned_logit[batch_index, slot] = measurements[
                        "proposal_logits"
                    ][batch_index, proposal].to(assigned_logit.dtype)
                    measurement_observed_mask[batch_index, slot] = True
                    used_proposals[column] = True

            inactive = torch.where(~previous_active[batch_index])[0]
            remaining = proposals[~used_proposals]
            count = min(int(inactive.numel()), int(remaining.numel()))
            if count == 0:
                continue
            slots = inactive[:count]
            proposal_indices = remaining[:count]
            reference_box[batch_index, slots] = measurements["proposal_boxes"][
                batch_index, proposal_indices
            ].to(reference_box.dtype)
            assigned_logit[batch_index, slots] = measurements["proposal_logits"][
                batch_index, proposal_indices
            ].to(assigned_logit.dtype)
            birth_candidate_mask[batch_index, slots] = True
            measurement_observed_mask[batch_index, slots] = True
        return (
            reference_box,
            birth_candidate_mask,
            measurement_observed_mask,
            assigned_logit,
        )

    def initialize_state(
        self,
        batch_size: int,
        device: torch.device,
        receiver_pose: torch.Tensor | None = None,
        receiver_time_s: torch.Tensor | None = None,
    ) -> GlobalForecastState:
        del device  # parameters define exact device/dtype under AMP/DDP
        content = self.query_decoder.initial_query_content(batch_size)
        boxes = self.query_decoder.initial_metric_boxes(batch_size)
        if receiver_pose is None:
            receiver_pose = torch.eye(
                4, device=content.device, dtype=content.dtype
            ).repeat(batch_size, 1, 1)
        if receiver_time_s is None:
            receiver_time_s = torch.zeros(
                batch_size, device=content.device, dtype=content.dtype
            )
        return GlobalForecastState.empty(
            query_feat=content,
            ref_box=boxes,
            memory_len=self.memory_len,
            receiver_pose=receiver_pose,
            receiver_time_s=receiver_time_s,
            future_modes=self.future_prior_modes,
            future_horizon_steps=self.future_prior_steps,
        )

    def _prepare_prior(
        self, state: GlobalForecastState, runtime_frame: dict
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        cv_box, cv_velocity = transform_and_propagate_boxes(
            state.association_ref_box,
            state.velocity,
            state.receiver_pose,
            runtime_frame["receiver_pose"],
            state.receiver_time_s,
            runtime_frame["receiver_time_s"],
        )
        spatial_prior = self.forecast_motion_prior(
            cv_box=cv_box,
            cv_velocity=cv_velocity,
            previous_box=state.ref_box,
            future_mode_probabilities=state.future_mode_probabilities,
            future_relative_means=state.future_relative_means,
            future_mode_variances=state.future_mode_variances,
            future_valid_mask=state.future_spatial_valid_mask,
            previous_pose=state.receiver_pose,
            current_pose=runtime_frame["receiver_pose"],
            previous_time_s=state.receiver_time_s,
            current_time_s=runtime_frame["receiver_time_s"],
        )
        # The legacy adapter remains in the checkpoint schema only.  Running it
        # in mode=none preserves the Phase-1 query normalization without
        # injecting a forecast latent into query content.
        query_prior, feedback_aux = self.future_adapter(
            state.query_feat,
            torch.zeros_like(state.future_cue),
            torch.zeros_like(state.future_uncertainty),
            torch.zeros_like(state.future_valid_mask),
        )
        history_box_current = transform_boxes_between_receiver_frames(
            state.history_box,
            state.receiver_pose,
            runtime_frame["receiver_pose"],
        )
        temporal = self.temporal_memory(
            state.history_feat[..., -self.tracker_memory_len :, :],
            history_box_current[..., -self.tracker_memory_len :, :],
            state.history_mask[..., -self.tracker_memory_len :],
        )
        query_prior = query_prior + self.memory_scale.tanh() * temporal
        return (
            cv_box,
            spatial_prior,
            query_prior,
            history_box_current,
            feedback_aux,
        )

    @staticmethod
    def _reported_box_after_measurement(
        association_box: torch.Tensor,
        forecast_box: torch.Tensor,
        active_mask: torch.Tensor,
        measurement_observed_mask: torch.Tensor,
        selected_forecast_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Impute only the public gap state without mutating filter state."""

        if association_box.shape != forecast_box.shape:
            raise ValueError("association and forecast boxes must have equal shape")
        expected = association_box.shape[:2]
        for mask in (
            active_mask, measurement_observed_mask, selected_forecast_mask
        ):
            if mask.shape != expected:
                raise ValueError("gap-imputation masks must have shape [B,Q]")
        use_forecast = (
            active_mask & ~measurement_observed_mask & selected_forecast_mask
        )
        return torch.where(
            use_forecast[..., None], forecast_box, association_box
        )

    def _advance_state(
        self,
        state: GlobalForecastState,
        decoded: dict,
        propagated_box: torch.Tensor,
        reported_propagated_box: torch.Tensor,
        propagated_velocity: torch.Tensor,
        prior_variance_xy: torch.Tensor,
        prior_variance_valid_mask: torch.Tensor,
        reported_forecast_mask: torch.Tensor,
        history_box_current: torch.Tensor,
        forecast_history_box: torch.Tensor,
        runtime_frame: dict,
    ) -> tuple[GlobalForecastState, dict]:
        scores = decoded["exist_logits"].sigmoid()
        lifecycle = self.lifecycle(
            scores,
            decoded["boxes"],
            state.active_mask,
            state.miss_count,
            state.runtime_id,
            state.next_runtime_id,
            birth_candidate_mask=decoded.get("birth_candidate_mask"),
            measurement_observed_mask=decoded.get(
                "measurement_observed_mask"
            ),
        )
        active = lifecycle["active_mask"]
        measured = lifecycle["observed_mask"] | lifecycle["birth_mask"]
        state_box, state_velocity, measurement_gain = (
            self.measurement_state_update(
                decoded["query_feat"],
                propagated_box,
                decoded["boxes"],
                propagated_velocity,
                decoded["velocity"],
                decoded.get(
                    "measurement_logits",
                    torch.full_like(decoded["exist_logits"], -20.0),
                ),
                measured,
                lifecycle["birth_mask"],
                prior_variance_xy=prior_variance_xy,
                prior_variance_valid_mask=prior_variance_valid_mask,
            )
        )

        reported_box = self._reported_box_after_measurement(
            state_box,
            reported_propagated_box,
            active,
            measured,
            reported_forecast_mask,
        )

        newborn_content = self.query_decoder.initial_query_content(
            scores.shape[0]
        )
        newborn_box = self.query_decoder.initial_metric_boxes(scores.shape[0])
        query_feat = torch.where(
            active[..., None], decoded["query_feat"], newborn_content
        )
        state_box = torch.where(active[..., None], state_box, newborn_box)
        reported_box = torch.where(
            active[..., None], reported_box, newborn_box
        )
        state_velocity = torch.where(
            active[..., None], state_velocity, torch.zeros_like(state_velocity)
        )
        history_feat, history_box, history_mask = self.temporal_memory.append(
            state.history_feat,
            history_box_current,
            state.history_mask,
            decoded["query_feat"],
            state_box,
            active,
        )
        forecast_history_box = self._append_native_forecast_history(
            forecast_history_box, reported_box, active
        )
        history_feat = torch.where(
            active[..., None, None], history_feat, torch.zeros_like(history_feat)
        )
        history_box = torch.where(
            active[..., None, None], history_box, torch.zeros_like(history_box)
        )
        history_mask = history_mask & active[..., None]
        age = torch.where(
            lifecycle["birth_mask"],
            torch.ones_like(state.age),
            torch.where(active, state.age + 1, torch.zeros_like(state.age)),
        )
        generation = state.generation + (
            state.active_mask & ~active
        ).to(state.generation.dtype)
        next_state = GlobalForecastState(
            query_feat=query_feat,
            ref_box=reported_box,
            association_ref_box=state_box,
            velocity=state_velocity,
            exist_logit=decoded["exist_logits"],
            active_mask=active,
            age=age,
            miss_count=lifecycle["miss_count"],
            runtime_id=lifecycle["runtime_id"],
            generation=generation,
            next_runtime_id=lifecycle["next_runtime_id"],
            history_feat=history_feat,
            history_box=history_box,
            forecast_history_box=forecast_history_box,
            history_mask=history_mask,
            future_cue=torch.zeros_like(query_feat),
            future_uncertainty=torch.zeros_like(state.future_uncertainty),
            future_valid_mask=torch.zeros_like(state.future_valid_mask),
            future_mode_probabilities=torch.zeros_like(
                state.future_mode_probabilities
            ),
            future_relative_means=torch.zeros_like(state.future_relative_means),
            future_mode_variances=torch.zeros_like(state.future_mode_variances),
            future_spatial_valid_mask=torch.zeros_like(
                state.future_spatial_valid_mask
            ),
            receiver_pose=runtime_frame["receiver_pose"],
            receiver_time_s=runtime_frame["receiver_time_s"],
        )
        next_state.validate()
        return next_state, {**lifecycle, "measurement_gain": measurement_gain}

    @staticmethod
    def _append_native_forecast_history(
        forecast_history_box: torch.Tensor,
        current_native_box: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Append a capture-time receiver-local state for CMP/MTR.

        Unlike ``history_box`` used by the tracker, this memory must not be
        transformed when the receiver moves: the released CMP trajectory
        caches store every observation in the receiver frame of its own
        capture.  Keeping this operation explicit prevents a future refactor
        from silently coupling the two coordinate contracts again.
        """

        if forecast_history_box.shape[:-2] != current_native_box.shape[:-1]:
            raise ValueError("forecast history and current boxes disagree")
        appended = torch.cat(
            (
                forecast_history_box[..., 1:, :],
                current_native_box[..., None, :],
            ),
            dim=-2,
        )
        return torch.where(
            active_mask[..., None, None], appended, torch.zeros_like(appended)
        )

    def _forward_single(
        self, runtime_frame: dict, prev_state=None, run_forecast: bool = True
    ):
        encoded = self.cobevt.encode_fused_bev(runtime_frame)
        fused_bev = encoded["fused_bev"]
        (
            forecast_source_bev,
            forecast_cav_mask,
            forecast_source_to_receiver,
        ) = select_forecast_bev_sources(
            self.forecast_bev_representation,
            encoded,
            runtime_frame.get("source_to_receiver"),
        )
        auxiliary_psm = self.cobevt.cls_head(fused_bev)
        auxiliary_rm = self.cobevt.reg_head(fused_bev)
        batch_size = fused_bev.shape[0]
        if prev_state is None:
            prev_state = self.initialize_state(
                batch_size,
                fused_bev.device,
                runtime_frame["receiver_pose"],
                runtime_frame["receiver_time_s"],
            )
        prev_state.validate()
        (
            cv_propagated_box,
            spatial_prior,
            query_prior,
            history_box_current,
            feedback_aux,
        ) = self._prepare_prior(prev_state, runtime_frame)
        measurements = self.measurement_proposals(
            auxiliary_psm,
            auxiliary_rm,
            prev_state.active_mask,
            cv_propagated_box,
        )
        association_cost, association_valid = (
            self.forecast_motion_prior.association_cost(
                spatial_prior,
                measurements["proposal_boxes"],
                measurements["proposal_logits"],
            )
        )
        (
            decoder_reference,
            birth_candidate_mask,
            measurement_observed_mask,
            proposal_seed_logits,
        ) = self._assign_measurements(
                (
                    cv_propagated_box
                    if self.forecast_motion_prior.mode in (
                        "propagation_detached", "propagation_bptt"
                    )
                    else spatial_prior.predicted_box
                ),
                prev_state.active_mask,
                measurements,
                self.measurement_match_distance,
                association_cost=association_cost,
                association_valid=association_valid,
            )
        decoded = self.query_decoder(
            fused_bev, query_prior, decoder_reference
        )
        decoded["birth_candidate_mask"] = birth_candidate_mask
        raw_query_exist_logits = decoded["exist_logits"]
        fused_exist_logits, dense_exist_logits = self._fuse_detection_evidence(
            raw_query_exist_logits, decoded["boxes"], auxiliary_psm
        )
        decoded = {**decoded, "exist_logits": fused_exist_logits}
        decoded["measurement_observed_mask"] = measurement_observed_mask
        decoded["measurement_logits"] = proposal_seed_logits
        separate_gap_imputation = self.forecast_motion_prior.mode in (
            "propagation_detached", "propagation_bptt"
        )
        tracking_propagated_box = (
            cv_propagated_box
            if separate_gap_imputation else spatial_prior.predicted_box
        )
        tracking_prior_variance_valid = (
            spatial_prior.selected_forecast_mask
            & ~torch.full_like(
                spatial_prior.selected_forecast_mask,
                separate_gap_imputation,
            )
        )
        next_state, lifecycle = self._advance_state(
            prev_state,
            decoded,
            tracking_propagated_box,
            spatial_prior.predicted_box,
            spatial_prior.predicted_velocity,
            spatial_prior.predicted_variance_xy,
            tracking_prior_variance_valid,
            spatial_prior.selected_forecast_mask,
            history_box_current,
            prev_state.forecast_history_box,
            runtime_frame,
        )
        if run_forecast and self.forecast_head.enabled:
            if self.actor_cooperative_fusion_enabled:
                cooperative = self.actor_cooperative_fusion(
                    forecast_source_bev,
                    forecast_cav_mask,
                    next_state.query_feat,
                    next_state.ref_box,
                    next_state.active_mask,
                    source_to_receiver=forecast_source_to_receiver,
                )
            else:
                cooperative = {
                    "context": torch.zeros_like(next_state.query_feat),
                    "attention_weights": next_state.query_feat.new_zeros(
                        next_state.query_feat.shape[0],
                        next_state.query_feat.shape[1],
                        forecast_cav_mask.shape[1],
                        self.actor_cooperative_fusion.patch_size ** 2,
                    ),
                }
            forecast = self.forecast_head(
                next_state.query_feat,
                next_state.ref_box,
                next_state.active_mask,
                next_state.forecast_history_box,
                next_state.history_mask,
                next_state.receiver_time_s,
                cooperative_context=cooperative["context"],
            )
        else:
            cooperative = {
                "context": torch.zeros_like(next_state.query_feat),
                "attention_weights": next_state.query_feat.new_zeros(
                    next_state.query_feat.shape[0],
                    next_state.query_feat.shape[1],
                    forecast_cav_mask.shape[1],
                    self.actor_cooperative_fusion.patch_size ** 2,
                ),
            }
            forecast = self.forecast_head.empty_outputs(
                next_state.query_feat,
                next_state.active_mask,
            )
        (
            state_mode_probabilities,
            state_relative_means,
            state_mode_variances,
            state_spatial_valid,
        ) = self.forecast_motion_prior.compact_forecast(
            forecast["pred_scores"],
            forecast["pred_trajs"],
            forecast["forecast_valid_mask"] & next_state.active_mask,
        )
        next_state = replace(
            next_state,
            future_cue=torch.zeros_like(next_state.future_cue),
            future_uncertainty=torch.zeros_like(next_state.future_uncertainty),
            future_valid_mask=torch.zeros_like(next_state.future_valid_mask),
            future_mode_probabilities=state_mode_probabilities,
            future_relative_means=state_relative_means,
            future_mode_variances=state_mode_variances,
            future_spatial_valid_mask=state_spatial_valid,
        )
        training_anchor = fused_bev.new_zeros(())
        if self.training and torch.is_grad_enabled():
            for parameter in self.parameters():
                if parameter.requires_grad:
                    training_anchor = parameter.reshape(-1)[0] * 0.0
                    break
            # The final frame's compact spatial forecast has no next-frame
            # consumer inside this clip.  Anchor it at zero so DDP observes the
            # legitimate terminal recurrent branch without changing its loss.
            training_anchor = training_anchor + 0.0 * (
                state_mode_probabilities.sum()
                + state_relative_means.sum()
                + state_mode_variances.sum()
            )
        outputs = {
            "boxes": decoded["boxes"],
            "track_boxes": next_state.ref_box,
            "velocity": decoded["velocity"],
            "track_velocity": next_state.velocity,
            "exist_logits": decoded["exist_logits"],
            "exist_scores": decoded["exist_logits"].sigmoid(),
            "raw_query_exist_logits": raw_query_exist_logits,
            "dense_exist_logits": dense_exist_logits,
            "active_mask": next_state.active_mask,
            "runtime_ids": next_state.runtime_id,
            "query_age": next_state.age,
            "miss_count": next_state.miss_count,
            "forecast_valid_mask": forecast["forecast_valid_mask"],
            "forecast_history_steps": forecast["forecast_history_steps"],
            "forecast_frame_heading": forecast["forecast_frame_heading"],
            "forecast_loss_in_receiver_frame": forecast[
                "forecast_loss_in_receiver_frame"
            ],
            "packed_entries": forecast["packed_entries"],
            "packed_context_slots": forecast["packed_context_slots"],
            "packed_dense_trajs_local": forecast[
                "packed_dense_trajs_local"
            ],
            "packed_center_heading": forecast["packed_center_heading"],
            "pred_scores": forecast["pred_scores"],
            "pred_trajs": forecast["pred_trajs"],
            "pred_trajs_local": forecast["pred_trajs_local"],
            "loss_pred_scores": forecast["loss_pred_scores"],
            "loss_pred_trajs_local": forecast["loss_pred_trajs_local"],
            "loss_pred_scores_layers": forecast[
                "loss_pred_scores_layers"
            ],
            "loss_pred_trajs_local_layers": forecast[
                "loss_pred_trajs_local_layers"
            ],
            "loss_intention_points": forecast["loss_intention_points"],
            "aux_exist_logits": decoded["aux_exist_logits"],
            "aux_boxes": decoded["aux_boxes"],
            "aux_velocity": decoded["aux_velocity"],
            "birth_candidate_mask": birth_candidate_mask,
            "proposal_seed_logits": proposal_seed_logits,
            "measurement_observed_mask": measurement_observed_mask,
            "measurement_gain": lifecycle["measurement_gain"],
            "future_prior_valid_mask": spatial_prior.future_valid_mask,
            "future_prior_selected_forecast_mask": (
                spatial_prior.selected_forecast_mask
            ),
            "future_prior_variance_xy": spatial_prior.predicted_variance_xy,
            "future_prior_step_index": spatial_prior.step_index,
            "cooperative_attention_weights": cooperative[
                "attention_weights"
            ],
            "auxiliary_psm": auxiliary_psm,
            "auxiliary_rm": auxiliary_rm,
            # Gives actor-empty ranks a root and completes terminal recurrent
            # branches; DDP still discovers truly absent conditional branches.
            "training_anchor": training_anchor,
        }
        aux = {
            "fused_bev": fused_bev,
            "source_bev": encoded["source_bev"],
            "regrouped_source_bev": encoded["regrouped_source_bev"],
            "actor_cooperative_context": cooperative["context"],
            "actor_cooperative_attention_weights": cooperative[
                "attention_weights"
            ],
            "cav_mask": encoded["cav_mask"],
            "forecast_cav_mask": forecast_cav_mask,
            "cv_propagated_box": cv_propagated_box,
            "propagated_box": spatial_prior.predicted_box,
            "propagated_velocity": spatial_prior.predicted_velocity,
            "future_prior_mode_centers": spatial_prior.mode_centers,
            "future_prior_mode_variances": spatial_prior.mode_variances,
            "future_prior_mode_probabilities": spatial_prior.mode_probabilities,
            "future_prior_valid_mask": spatial_prior.future_valid_mask,
            "future_prior_selected_forecast_mask": (
                spatial_prior.selected_forecast_mask
            ),
            "decoder_reference_box": decoder_reference,
            "measurement_proposal_boxes": measurements["proposal_boxes"],
            "measurement_proposal_logits": measurements["proposal_logits"],
            "measurement_proposal_mask": measurements["proposal_mask"],
            "measurement_observed_mask": measurement_observed_mask,
            "decoder_embedding": decoded["decoder_embedding"],
            **feedback_aux,
            **lifecycle,
        }
        return outputs, next_state, aux

    def forward(
        self,
        runtime_frame,
        prev_state=None,
        detach_interval=None,
        loss_start_index: int = 0,
    ):
        """Run one frame or a chronologically ordered clip.

        A clip is deliberately exposed through the ordinary ``forward``
        entrypoint so DDP observes one forward graph followed by one backward.
        Calling a DDP wrapper four times and only then back-propagating once
        makes its reducer retain only the final conditional MTR graph and can
        deadlock when ranks contain different numbers of active actors.
        """
        if isinstance(runtime_frame, (list, tuple)):
            return self.forward_clip(
                runtime_frame,
                initial_state=prev_state,
                detach_interval=detach_interval,
                loss_start_index=loss_start_index,
            )
        return self._forward_single(runtime_frame, prev_state)

    def forward_frame(self, runtime_frame, prev_state=None):
        return self._forward_single(runtime_frame, prev_state)

    def forward_clip(
        self,
        runtime_frames,
        initial_state=None,
        detach_interval=None,
        loss_start_index: int = 0,
    ):
        if not 0 <= loss_start_index <= len(runtime_frames):
            raise ValueError("loss_start_index is outside the runtime clip")
        state = initial_state
        outputs_by_time = []
        aux_by_time = []
        for time_index, runtime_frame in enumerate(runtime_frames):
            if time_index < loss_start_index:
                with torch.no_grad():
                    outputs, state, aux = self._forward_single(
                        runtime_frame,
                        state,
                        run_forecast=self.forecast_motion_prior.mode != "none",
                    )
            else:
                outputs, state, aux = self._forward_single(runtime_frame, state)
            outputs_by_time.append(outputs)
            aux_by_time.append(aux)
            if (
                detach_interval is not None
                and detach_interval > 0
                and time_index >= loss_start_index
                and (time_index - loss_start_index + 1) % detach_interval == 0
                and time_index + 1 < len(runtime_frames)
            ):
                state = state.detach_differentiable_tensors()
        return outputs_by_time, state, aux_by_time
