"""Causal DualCMP clip model: detector features, proposals, and actor state."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from dataclasses import replace

from opencood.models.point_pillar_fax import PointPillarFax

from .proposal import decode_proposals
from .roi_sampler import ActorRoiSampler
from .state import ActorState
from .tracker import ActorTracker
from .kalman import box_to_measurement, orientation_correction, update as kalman_update
from .box_refinement import SparseBox3DEncoder, SparseBox3DRefinementModule, decode_box, encode_box, linear_relu_ln
from .actor_source_tokenizer import ActorAlignedSourceTokenizer
from .source_temporal_encoder import SourceTemporalEncoder
from .source_motion_encoder import SourceMotionEncoder
from .source_evidence_policy import EVIDENCE_POLICIES, FEATURE_ENCODERS, apply_evidence_policy
from .intention_query_forecast import IntentionQueryForecastHead
from .capture_time_evidence import capture_aligned_actor_boxes
from .forecast_frames import (ALIGNED_SOURCE, TEMPORAL_SOURCES, align_actor_history,
                              align_source_record, detached_source_history)


class DualCmpClipModel(nn.Module):
    def __init__(self, detector_args, anchors, lidar_range, max_actors=128,
                 proposal_top_k=128, max_miss=2, proposal_score_threshold=0.20,
                 proposal_nms_thresh=0.15, emission_threshold=None,
                 forecast_args=None):
        super().__init__()
        self.detector = PointPillarFax(detector_args)
        self.register_buffer("anchors", torch.as_tensor(anchors, dtype=torch.float32))
        self.max_actors = int(max_actors)
        self.proposal_top_k = int(proposal_top_k)
        self.proposal_score_threshold = float(proposal_score_threshold)
        self.proposal_nms_thresh = float(proposal_nms_thresh)
        self.emission_threshold = float(proposal_score_threshold if emission_threshold is None else emission_threshold)
        self.roi = ActorRoiSampler(lidar_range)
        self.tracker = ActorTracker(max_miss=max_miss)
        feature_dim = int(detector_args["fax_fusion"]["input_dim"])
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=8, dropout=0.1, batch_first=True)
        # Sparse4D blocks.linear_relu_ln, preserving the head's norm -> refine
        # boundary. Cache the normalized feature, not the unbounded recurrent
        # Linear/ReLU activation. The two LN parameters belong to all A/B/C
        # checkpoints; pre-normalization checkpoints must not strict-resume.
        self.track_update = nn.Sequential(*linear_relu_ln(feature_dim, 1, 1))
        self.box_encoder = SparseBox3DEncoder()
        self.box_refinement = SparseBox3DRefinementModule()
        self.forecast_head = None
        self.actor_source_tokenizer = None
        self.source_temporal_encoder = None
        self.capture_time_evidence_enabled = False
        self.capture_alignment_enabled = False
        self.age_encoding_enabled = False
        self.mean_source_aggregation_enabled = False
        self.forecast_representation = "history"
        self.aligned_evidence_policy = "source_temporal"
        self.source_feature_encoder = "legacy_additive_v1"
        # Kept opt-in so Phase-A tracking checkpoints strict-load unchanged.
        if forecast_args is not None:
            forecast_args = dict(forecast_args)
            legacy_capture = bool(forecast_args.get("capture_time_evidence", True))
            self.capture_alignment_enabled = bool(
                forecast_args.get("capture_alignment", legacy_capture))
            self.age_encoding_enabled = bool(
                forecast_args.get("age_encoding", legacy_capture))
            self.capture_time_evidence_enabled = (
                self.capture_alignment_enabled and self.age_encoding_enabled)
            self.forecast_representation = str(
                forecast_args.get("representation", "source"))
            self.aligned_evidence_policy = str(forecast_args.get("aligned_evidence_policy", "source_temporal"))
            self.source_feature_encoder = str(forecast_args.get("source_feature_encoder", "legacy_additive_v1"))
            if self.aligned_evidence_policy not in EVIDENCE_POLICIES or self.source_feature_encoder not in FEATURE_ENCODERS:
                raise ValueError("unknown aligned evidence/feature contract")
            if self.forecast_representation != ALIGNED_SOURCE and (
                    self.aligned_evidence_policy != "source_temporal" or self.source_feature_encoder != "legacy_additive_v1"):
                raise ValueError("new evidence/feature choices are aligned-only; legacy ablations stay unchanged")
            if self.forecast_representation == ALIGNED_SOURCE and self.aligned_evidence_policy == "history":
                forecast_args["source_evidence"] = False
            self.mean_source_aggregation_enabled = bool(
                forecast_args.get("mean_source_aggregation", False))
            source_representations = ("source", *TEMPORAL_SOURCES)
            if self.forecast_representation not in (
                    "history", "fused", *source_representations):
                raise ValueError(
                    "forecast representation must be history, fused, source, "
                    "or a temporal source representation")
            if self.forecast_representation not in source_representations:
                if self.mean_source_aggregation_enabled:
                    raise ValueError(
                        "mean source aggregation requires source representation")
                self.capture_alignment_enabled = False
                self.age_encoding_enabled = False
                self.capture_time_evidence_enabled = False
            self.actor_source_tokenizer = ActorAlignedSourceTokenizer(
                input_dim=feature_dim, query_dim=feature_dim,
                max_cav=int(detector_args["max_cav"]), lidar_range=lidar_range,
                roi_grid_size=int(forecast_args.get("roi_grid_size", 5)),
                roi_scale=float(forecast_args.get("roi_scale", 2.0)),
                minimum_half_extent_m=float(forecast_args.get("minimum_half_extent_m", 1.0)),
                legacy_heading=(self.forecast_representation != ALIGNED_SOURCE))
            if self.forecast_representation in TEMPORAL_SOURCES:
                if self.mean_source_aggregation_enabled:
                    raise ValueError(
                        "mean source aggregation belongs to the source ablation")
                encoder_type = SourceMotionEncoder if self.source_feature_encoder == "motion_correlation_v1" else SourceTemporalEncoder
                self.source_temporal_encoder = encoder_type(
                    feature_dim=feature_dim,
                    roi_grid_size=int(forecast_args.get("roi_grid_size", 5)),
                    pooled_grid_size=int(
                        forecast_args.get("source_temporal_pooled_grid_size",
                                          5 if self.forecast_representation == ALIGNED_SOURCE else 2)),
                    heads=int(forecast_args.get("source_temporal_heads", 8)),
                    layers=int(forecast_args.get("source_temporal_layers", 1)),
                    dropout=float(forecast_args.get("dropout", 0.1)),
                    frame_period_s=float(
                        forecast_args.get("frame_period_s", 0.1)),
                    pose_scale_m=float(
                        forecast_args.get("source_pose_scale_m", 100.0)),
                    standard_position_encoding=(self.forecast_representation == ALIGNED_SOURCE),
                    feature_encoder=self.source_feature_encoder,
                )
            self.forecast_head = IntentionQueryForecastHead(feature_dim, forecast_args)
            if self.forecast_representation == "history":
                for parameter in self.actor_source_tokenizer.parameters():
                    parameter.requires_grad = False
                self.forecast_head.freeze_inactive_parameters(False)
            if self.forecast_representation == ALIGNED_SOURCE and self.aligned_evidence_policy == "history":
                self.forecast_head.freeze_inactive_parameters(False)
                for parameter in self.source_temporal_encoder.parameters():
                    parameter.requires_grad = False

    @staticmethod
    def emission_mask(valid_mask, quality, threshold):
        """AB3DMOT-style max-age validity plus learned quality threshold."""
        return valid_mask & (quality >= float(threshold))

    @staticmethod
    def temporal_fuse(attention, current_feature, previous_feature, valid_mask):
        """Sparse4D temp_gnn adapted to one query/key/value per runtime slot."""
        batch_size, actor_count, feature_dim = current_feature.shape
        attended, _ = attention(
            current_feature.reshape(batch_size * actor_count, 1, feature_dim),
            previous_feature.reshape(batch_size * actor_count, 1, feature_dim),
            previous_feature.reshape(batch_size * actor_count, 1, feature_dim),
            need_weights=False)
        fused = current_feature + attended.reshape(batch_size, actor_count, feature_dim)
        return torch.where(valid_mask[..., None], fused, current_feature)

    def _source_capture(self, state, frame, source_mask):
        source_count = source_mask.shape[1]
        if self.capture_alignment_enabled:
            return capture_aligned_actor_boxes(
                state, frame["capture_time_s"].float(), source_mask)
        age_s = state.receiver_time_s[:, None] - frame["capture_time_s"].float()
        return {
            "age_s": torch.where(
                source_mask, age_s, torch.zeros_like(age_s)),
            "mask": state.active_mask[:, :, None] & source_mask[:, None, :],
            "boxes": state.box[:, :, None].expand(
                -1, -1, source_count, -1),
            "time_error_s": torch.zeros_like(
                state.active_mask[:, :, None].expand(
                    -1, -1, source_count), dtype=state.box.dtype),
        }

    @staticmethod
    def _source_pose_xyyaw(source_to_receiver):
        if source_to_receiver.ndim != 4 or source_to_receiver.shape[-2:] != (4, 4):
            raise ValueError("source_to_receiver must have shape [B,L,4,4]")
        return torch.stack(
            (
                source_to_receiver[..., 0, 3],
                source_to_receiver[..., 1, 3],
                torch.atan2(source_to_receiver[..., 1, 0],
                            source_to_receiver[..., 0, 0]),
            ),
            dim=-1,
        )

    def _temporal_source_memory(self, records, final_state,
                                shuffle_source_time=False):
        if self.source_temporal_encoder is None or not records:
            raise RuntimeError("source temporal evidence requires frame records")
        history = self.forecast_head.history_frames
        records = records[-history:]
        tokens = torch.stack([item["tokens"] for item in records], dim=3)
        if self.forecast_representation == ALIGNED_SOURCE:
            aligned = [align_source_record(item, final_state) for item in records]
            records = [dict(item, positions=value[0], source_pose=self._source_pose_xyyaw(value[1]))
                       for item, value in zip(records, aligned)]
        positions = torch.stack(
            [item["positions"] for item in records], dim=3)
        token_mask = torch.stack(
            [item["token_mask"] for item in records], dim=3)
        runtime_ids = torch.stack(
            [item["runtime_id"] for item in records], dim=2)
        if shuffle_source_time:
            tokens = tokens.flip(3)
            positions = positions.flip(3)
            token_mask = token_mask.flip(3)
            runtime_ids = runtime_ids.flip(2)
        same_actor = self.source_temporal_encoder.same_runtime_mask(
            runtime_ids, final_state.runtime_id)
        token_mask = (
            token_mask
            & same_actor[:, :, None, :, None]
            & final_state.active_mask[:, :, None, None, None]
        )

        capture_time = torch.stack(
            [item["capture_time_s"] for item in records], dim=2)
        age = final_state.receiver_time_s[:, None, None] - capture_time
        age = age[:, None].expand(-1, final_state.box.shape[1], -1, -1)
        if not self.age_encoding_enabled and self.source_feature_encoder != "motion_correlation_v1":
            age = torch.zeros_like(age)
        source_pose = torch.stack(
            [item["source_pose"] for item in records], dim=2)
        if self.forecast_representation == ALIGNED_SOURCE:
            # Observer-to-target geometry is per actor, not one shared CAV
            # translation for all targets. Preserve CAV identity and express
            # distance/view direction in the same frame as the source patches.
            delta = source_pose[:, None, ..., :2] - final_state.box[:, :, None, None, :2]
            yaw = final_state.box[:, :, None, None, 6]
            relative_yaw = source_pose[:, None, ..., 2] - yaw
            source_pose = torch.stack((
                torch.cos(yaw) * delta[..., 0] + torch.sin(yaw) * delta[..., 1],
                -torch.sin(yaw) * delta[..., 0] + torch.cos(yaw) * delta[..., 1],
                torch.atan2(torch.sin(relative_yaw), torch.cos(relative_yaw)),
            ), dim=-1)
        inputs = (tokens, positions, token_mask, age, source_pose)
        if self.forecast_representation == ALIGNED_SOURCE:
            inputs = apply_evidence_policy(inputs, self.aligned_evidence_policy)
        if self.source_feature_encoder == "motion_correlation_v1":
            # Capture time is protocol validity, even when its learned feature
            # is ablated. Zeroing timestamps would discard all but frame one.
            encoded = self.source_temporal_encoder(*inputs, encode_age=self.age_encoding_enabled)
        else:
            encoded = self.source_temporal_encoder(*inputs)
        batch, actors, sources, times, patches, channels = (
            encoded["tokens"].shape)
        memory = {
            "tokens": encoded["tokens"].reshape(
                batch, actors, sources * times, patches, channels),
            "positions": encoded["positions"].reshape(
                batch, actors, sources * times, patches, 3),
            "token_mask": encoded["token_mask"].reshape(
                batch, actors, sources * times, patches),
            "age_s": age,
            "runtime_ids": runtime_ids,
        }
        return memory

    @classmethod
    def apply_learned_update(cls, state, update, boxes, velocities, quality,
                             threshold, observation_time_s,
                             observation_pose):
        """Update learned state without replacing the tracker measurement score."""
        if observation_time_s.shape != state.receiver_time_s.shape:
            raise ValueError("observation_time_s must have shape [B]")
        if observation_pose.shape != state.receiver_pose.shape:
            raise ValueError("observation_pose must have shape [B,4,4]")
        if (observation_time_s.dtype != state.box.dtype
                or observation_pose.dtype != state.box.dtype
                or observation_time_s.device != state.box.device
                or observation_pose.device != state.box.device):
            raise ValueError(
                "observation time/pose must match ActorState dtype/device")
        valid = state.valid_mask
        measurement = box_to_measurement(boxes)
        corrected_yaw, observed_yaw = orientation_correction(
            state.filter_mean[..., 3], measurement[..., 3])
        mean = torch.cat((state.filter_mean[..., :3], corrected_yaw[..., None],
                          state.filter_mean[..., 4:]), dim=-1)
        measurement = torch.cat((measurement[..., :3], observed_yaw[..., None], measurement[..., 4:]), dim=-1)
        candidate_mean, candidate_cov = kalman_update(mean, state.filter_cov, measurement)
        return replace(
            state,
            feature=torch.where(valid[..., None], update, state.feature),
            box=torch.where(valid[..., None], boxes, state.box),
            velocity=torch.where(valid[..., None], velocities, state.velocity),
            quality=torch.where(valid, quality, state.quality),
            score=state.score,
            filter_mean=torch.where(valid[..., None], candidate_mean, state.filter_mean),
            filter_cov=torch.where(valid[..., None, None], candidate_cov, state.filter_cov),
            active_mask=cls.emission_mask(valid, quality, threshold),
            history_box=torch.where(
                valid[..., None, None],
                torch.cat((boxes.unsqueeze(2), state.history_box[:, :, :-1]), dim=2),
                state.history_box),
            history_mask=torch.where(valid[..., None], torch.ones_like(state.history_mask), state.history_mask),
            # CoLIF uniad_tracker::Tracker._append_forecast_history keeps
            # capture-frame boxes separate from tracking memory re-expressed
            # by geometry.propagate_state.  Only emitted observations append.
            forecast_history_box=torch.where(
                cls.emission_mask(valid, quality, threshold)[..., None, None],
                torch.cat((state.forecast_history_box[:, :, 1:], boxes.unsqueeze(2)), dim=2),
                state.forecast_history_box),
            forecast_history_mask=torch.where(
                cls.emission_mask(valid, quality, threshold)[..., None],
                torch.cat((state.forecast_history_mask[:, :, 1:],
                           cls.emission_mask(valid, quality, threshold).unsqueeze(2)), dim=2),
                state.forecast_history_mask),
            forecast_history_time_s=torch.where(
                cls.emission_mask(valid, quality, threshold)[..., None],
                torch.cat((state.forecast_history_time_s[:, :, 1:],
                           observation_time_s[:, None, None].expand(
                               -1, state.box.shape[1], 1)), dim=2),
                state.forecast_history_time_s),
            forecast_history_pose=torch.where(
                cls.emission_mask(valid, quality, threshold)[..., None, None, None],
                torch.cat((state.forecast_history_pose[:, :, 1:],
                           observation_pose[:, None, None].expand(
                               -1, state.box.shape[1], 1, -1, -1)), dim=2),
                state.forecast_history_pose),
        )

    def forward(self, runtime_frames, state=None, max_dt_s=0.2,
                detector_autocast=False, run_forecast=False,
                current_frame_only_grad=False,
                shuffle_source_time=False, source_history=None):
        """Advance only from runtime frames; actor state remains FP32.

        ``detector_autocast`` deliberately encloses the dense detector only.
        Geometry, lifecycle and persistent state run outside AMP so their
        dtype invariant is independent of the trainer's precision policy.
        """
        outputs = []
        temporal_source_records = list(source_history or ())
        if temporal_source_records and self.forecast_representation not in TEMPORAL_SOURCES:
            raise ValueError("source_history requires a temporal Source model")
        if run_forecast and self.forecast_head is None:
            raise RuntimeError("run_forecast requires an opt-in forecast head")
        if shuffle_source_time and self.forecast_representation not in TEMPORAL_SOURCES:
            raise ValueError(
                "shuffle_source_time requires source_temporal representation")
        for frame_index, frame in enumerate(runtime_frames):
            # Scene/receiver transitions and non-contiguous frames clear memory
            # row-wise. Tracker runtime IDs alone cannot identify a new scene.
            owner = tuple(zip(frame["scene_id"], frame["receiver_id"].tolist()))
            if "sequence_segment" in frame:
                owner = tuple((scene, receiver, segment) for (scene, receiver), segment
                              in zip(owner, frame["sequence_segment"].tolist()))
                if state is not None and bool(frame["sequence_segment_start"].any()):
                    state = state.reset_rows(frame["sequence_segment_start"],
                                             receiver_pose=frame["receiver_pose"].float(),
                                             receiver_time_s=frame["receiver_time_s"].float())
            if temporal_source_records:
                previous = temporal_source_records[-1]
                delta = frame["receiver_time_s"].float() - previous["receiver_time_s"]
                same_owner = torch.tensor([left == right for left, right in zip(
                    previous["owner"], owner)], device=delta.device, dtype=torch.bool)
                continuous = same_owner & (delta > 0) & (delta <= max_dt_s)
                temporal_source_records = [dict(item, token_mask=item["token_mask"]
                                                & continuous[:, None, None, None])
                                           for item in temporal_source_records]
            warmup_no_grad = bool(
                current_frame_only_grad
                and frame_index < len(runtime_frames) - 1
            )
            with torch.set_grad_enabled(torch.is_grad_enabled() and not warmup_no_grad):
                with autocast(enabled=bool(detector_autocast)):
                    detector = self.detector(frame)
            # AB3DMOT proposal decoding and the repository detection loss
            # allocate FP32 anchors/targets; leave the AMP region before both.
            detector["psm"] = detector["psm"].float()
            detector["rm"] = detector["rm"].float()
            detector["fused_bev"] = detector["fused_bev"].float()
            pose, time_s = frame["receiver_pose"].float(), frame["receiver_time_s"].float()
            interval = time_s.new_full(time_s.shape, 0.1) if state is None else time_s - state.receiver_time_s
            interval = torch.where((interval > 0) & (interval <= max_dt_s), interval,
                                   time_s.new_full(time_s.shape, 0.1))
            if state is None:
                state = ActorState.empty(pose.shape[0], self.max_actors,
                                         detector["fused_bev"].shape[1], 11,
                                         pose, time_s)
                proposals = decode_proposals(detector["psm"], detector["rm"], self.anchors,
                                             self.proposal_top_k, training=self.training,
                                             score_threshold=self.proposal_score_threshold,
                                             nms_thresh=self.proposal_nms_thresh)
                state = self.tracker(
                    state, proposals, pose, time_s + 1e-3, max_dt_s)
            else:
                proposals = decode_proposals(detector["psm"], detector["rm"], self.anchors,
                                             self.proposal_top_k, training=self.training,
                                             score_threshold=self.proposal_score_threshold,
                                             nms_thresh=self.proposal_nms_thresh)
                state = self.tracker(state, proposals, pose, time_s, max_dt_s)
            detector["proposals"] = proposals
            detector["state"] = state
            with torch.set_grad_enabled(torch.is_grad_enabled() and not warmup_no_grad):
                detector["roi_feature"] = self.roi(detector["fused_bev"], state.box)
                detector["temporal_feature"] = self.temporal_fuse(
                    self.temporal_attention, detector["roi_feature"], state.feature,
                    state.valid_mask)
                update = self.track_update(detector["temporal_feature"])
                anchor_box = torch.where(state.valid_mask[..., None], state.box,
                                         torch.cat((state.box[..., :3], torch.ones_like(state.box[..., 3:6]), state.box[..., 6:]), -1))
                anchor_velocity = torch.where(state.valid_mask[..., None], state.velocity / 0.1,
                                              torch.zeros_like(state.velocity))
                anchor = encode_box(anchor_box, anchor_velocity)
                detector["track_box_code"], detector["track_exist_logits"] = self.box_refinement(
                    update, anchor, self.box_encoder(anchor), interval)
                decoded_box, decoded_velocity = decode_box(detector["track_box_code"])
                detector["track_box"] = torch.where(state.valid_mask[..., None], decoded_box, state.box)
                detector["track_velocity"] = torch.where(state.valid_mask[..., None], decoded_velocity, state.velocity)
            quality = detector["track_exist_logits"].sigmoid()
            state = self.apply_learned_update(
                state, update, detector["track_box"], detector["track_velocity"],
                quality, self.emission_threshold, time_s, pose)
            state.validate()
            detector["state"] = state
            temporal_capture = None
            if self.forecast_representation in TEMPORAL_SOURCES:
                with torch.set_grad_enabled(
                        torch.is_grad_enabled() and not warmup_no_grad):
                    temporal_capture = self._source_capture(
                        state, frame, detector["source_mask"])
                    if self.forecast_representation == ALIGNED_SOURCE and self.aligned_evidence_policy == "fused_temporal":
                        memory_bev = detector["fused_bev"][:, None].float()
                        memory_source_mask = torch.ones(memory_bev.shape[:2], dtype=torch.bool, device=memory_bev.device)
                        memory_capture_time = state.receiver_time_s[:, None]
                        memory_pose = torch.eye(4, device=memory_bev.device)[None, None].expand(memory_bev.shape[0], 1, -1, -1)
                        frame_evidence = self.actor_source_tokenizer(memory_bev, memory_source_mask, state.box, state.active_mask)
                    else:
                        memory_capture_time = frame["capture_time_s"].float()
                        memory_pose = frame["source_to_receiver"].float()
                        frame_evidence = self.actor_source_tokenizer(
                            detector["source_bev"].float(), detector["source_mask"],
                            temporal_capture["boxes"], temporal_capture["mask"], reference_boxes=state.box)
                    pooled = self.source_temporal_encoder.pool_frame(
                        frame_evidence["tokens"],
                        frame_evidence["positions"],
                        frame_evidence["token_mask"])
                temporal_source_records.append({
                    "tokens": pooled[0],
                    "positions": pooled[1],
                    "token_mask": pooled[2],
                    "runtime_id": state.runtime_id,
                    "capture_time_s": memory_capture_time,
                    "source_pose": self._source_pose_xyyaw(memory_pose),
                    "source_to_receiver": memory_pose,
                    "reference_boxes": state.box,
                    "receiver_pose": state.receiver_pose,
                    "receiver_time_s": state.receiver_time_s,
                    "owner": owner,
                })
                temporal_source_records = temporal_source_records[-self.forecast_head.history_frames:]
            if run_forecast and frame_index == len(runtime_frames) - 1:
                source_evidence = consensus_evidence = None
                source_age_s = None
                source_count = detector["source_mask"].shape[1]
                if self.forecast_representation in (
                        "source", *TEMPORAL_SOURCES):
                    evidence_bev = detector["source_bev"].view_as(
                        detector["source_bev"])
                elif self.forecast_representation == "fused":
                    evidence_bev = detector["fused_bev"].view_as(
                        detector["fused_bev"])
                else:
                    evidence_bev = None
                if self.forecast_representation == ALIGNED_SOURCE:
                    if self.aligned_evidence_policy == "fused_temporal":
                        evidence_bev = detector["fused_bev"][:, None]
                    elif self.aligned_evidence_policy == "history":
                        evidence_bev = None

                if self.forecast_representation == "source":
                    capture = self._source_capture(
                        state, frame, detector["source_mask"])
                elif self.forecast_representation in TEMPORAL_SOURCES:
                    capture = temporal_capture
                else:
                    capture = {
                        "age_s": state.box.new_zeros(
                            state.box.shape[0], source_count),
                        "mask": torch.zeros(
                            state.box.shape[0], self.max_actors, source_count,
                            dtype=torch.bool, device=state.box.device),
                        "time_error_s": state.box.new_zeros(
                            state.box.shape[0], self.max_actors, source_count),
                    }

                if self.forecast_representation == "source":
                    source_evidence = self.actor_source_tokenizer(
                        evidence_bev, detector["source_mask"],
                        capture["boxes"], capture["mask"],
                        reference_boxes=state.box)
                    if self.age_encoding_enabled:
                        source_age_s = capture["age_s"][:, None].expand(
                            -1, self.max_actors, -1)
                elif self.forecast_representation in TEMPORAL_SOURCES:
                    source_evidence = self._temporal_source_memory(
                        temporal_source_records, state,
                        shuffle_source_time=shuffle_source_time)
                elif self.forecast_representation == "fused":
                    fused_mask = torch.ones(
                        evidence_bev.shape[0], 1, dtype=torch.bool,
                        device=evidence_bev.device)
                    consensus_evidence = self.actor_source_tokenizer(
                        evidence_bev[:, None], fused_mask,
                        state.box, state.active_mask)
                history_box = state.forecast_history_box
                if self.forecast_representation == ALIGNED_SOURCE:
                    history_box = align_actor_history(history_box, state.forecast_history_mask,
                                                      state.forecast_history_pose, state.receiver_pose)
                forecast = self.forecast_head(
                    # Tracker latent must not be forecast input; source tokens
                    # and current boxes/history form the MTR input template.
                    detector["fused_bev"].mean(dim=(-1, -2)).unsqueeze(1).expand(
                        -1, self.max_actors, -1),
                    state.box, state.active_mask, history_box,
                    state.forecast_history_mask,
                    state.forecast_history_time_s, state.receiver_time_s,
                    consensus_tokens=(consensus_evidence["tokens"]
                                      if consensus_evidence is not None else None),
                    consensus_positions=(consensus_evidence["positions"]
                                         if consensus_evidence is not None else None),
                    consensus_token_mask=(consensus_evidence["token_mask"]
                                          if consensus_evidence is not None else None),
                    source_tokens=(source_evidence["tokens"]
                                   if source_evidence is not None else None),
                    source_positions=(source_evidence["positions"]
                                      if source_evidence is not None else None),
                    source_token_mask=(source_evidence["token_mask"]
                                       if source_evidence is not None else None),
                    source_age_s=source_age_s,
                    mean_source_aggregation=(
                        self.mean_source_aggregation_enabled))
                forecast["source_capture_age_s"] = capture["age_s"]
                forecast["source_capture_box_mask"] = capture["mask"]
                forecast["source_capture_time_error_s"] = capture["time_error_s"]
                if self.forecast_representation in TEMPORAL_SOURCES:
                    forecast["source_temporal_age_s"] = source_evidence["age_s"]
                    forecast["source_temporal_runtime_ids"] = (
                        source_evidence["runtime_ids"])
                    forecast["source_temporal_token_mask"] = (
                        source_evidence["token_mask"])
                detector["forecast"] = forecast
                if self.forecast_representation == ALIGNED_SOURCE:
                    forecast["aligned_evidence_policy"] = self.aligned_evidence_policy
                    forecast["source_feature_encoder"] = self.source_feature_encoder
                detector["forecast_evidence_bev"] = evidence_bev
                if self.forecast_representation in (
                        "source", *TEMPORAL_SOURCES) and not (
                        self.forecast_representation == ALIGNED_SOURCE and self.aligned_evidence_policy in ("fused_temporal", "history")):
                    detector["forecast_source_bev"] = evidence_bev
            outputs.append(detector)
            # Phase-B uses ten causal tracking warmup frames only to build
            # capture-native history.  It never performs eleven-frame
            # perception BPTT; the final frame retains the forecast graph.
            if run_forecast and frame_index < len(runtime_frames) - 1:
                state = state.detached()
        if outputs and self.forecast_representation in TEMPORAL_SOURCES:
            outputs[-1]["source_history"] = detached_source_history(temporal_source_records)
        return outputs, state
