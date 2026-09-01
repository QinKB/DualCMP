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
from .kalman import box_to_measurement, update as kalman_update
from .actor_source_tokenizer import ActorAlignedSourceTokenizer
from .intention_query_forecast import IntentionQueryForecastHead


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
        self.track_update = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU())
        self.track_exist = nn.Linear(feature_dim, 1)
        self.track_box = nn.Linear(feature_dim, 7)
        self.track_velocity = nn.Linear(feature_dim, 2)
        self.forecast_head = None
        self.actor_source_tokenizer = None
        # Kept opt-in so Phase-A tracking checkpoints strict-load unchanged.
        if forecast_args is not None:
            forecast_args = dict(forecast_args)
            self.actor_source_tokenizer = ActorAlignedSourceTokenizer(
                input_dim=feature_dim, query_dim=feature_dim,
                max_cav=int(detector_args["max_cav"]), lidar_range=lidar_range,
                roi_grid_size=int(forecast_args.get("roi_grid_size", 5)),
                roi_scale=float(forecast_args.get("roi_scale", 2.0)),
                minimum_half_extent_m=float(forecast_args.get("minimum_half_extent_m", 1.0)))
            self.forecast_head = IntentionQueryForecastHead(feature_dim, forecast_args)

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

    @classmethod
    def apply_learned_update(cls, state, update, boxes, velocities, quality, threshold):
        """Update learned state without replacing the tracker measurement score."""
        valid = state.valid_mask
        candidate_mean, candidate_cov = kalman_update(
            state.filter_mean, state.filter_cov, box_to_measurement(boxes))
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
        )

    def forward(self, runtime_frames, state=None, max_dt_s=0.2,
                detector_autocast=False, run_forecast=False,
                current_frame_only_grad=False):
        """Advance only from runtime frames; actor state remains FP32.

        ``detector_autocast`` deliberately encloses the dense detector only.
        Geometry, lifecycle and persistent state run outside AMP so their
        dtype invariant is independent of the trainer's precision policy.
        """
        outputs = []
        if run_forecast and self.forecast_head is None:
            raise RuntimeError("run_forecast requires an opt-in forecast head")
        for frame_index, frame in enumerate(runtime_frames):
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
            if state is None:
                state = ActorState.empty(pose.shape[0], self.max_actors,
                                         detector["fused_bev"].shape[1], 11, pose, time_s)
                proposals = decode_proposals(detector["psm"], detector["rm"], self.anchors,
                                             self.proposal_top_k, training=self.training,
                                             score_threshold=self.proposal_score_threshold,
                                             nms_thresh=self.proposal_nms_thresh)
                state = self.tracker(state, proposals, pose, time_s + 1e-3, max_dt_s)
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
                detector["track_exist_logits"] = self.track_exist(update).squeeze(-1)
                detector["track_box"] = state.box + self.track_box(update)
                detector["track_velocity"] = state.velocity + self.track_velocity(update)
            quality = detector["track_exist_logits"].sigmoid()
            state = self.apply_learned_update(
                state, update, detector["track_box"], detector["track_velocity"],
                quality, self.emission_threshold)
            state.validate()
            detector["state"] = state
            if run_forecast and frame_index == len(runtime_frames) - 1:
                # A view keeps the forecast branch observable without copying
                # the dense tensor; its retained gradient excludes the fused
                # detection/tracking branch that also consumes source_bev.
                forecast_source_bev = detector["source_bev"].view_as(
                    detector["source_bev"]
                )
                evidence = self.actor_source_tokenizer(
                    forecast_source_bev, detector["source_mask"], state.box,
                    state.active_mask)
                forecast = self.forecast_head(
                    # Tracker latent must not be forecast input; source tokens
                    # and current boxes/history form the MTR input template.
                    detector["fused_bev"].mean(dim=(-1, -2)).unsqueeze(1).expand(
                        -1, self.max_actors, -1),
                    state.box, state.active_mask, state.forecast_history_box,
                    state.forecast_history_mask, state.receiver_time_s,
                    source_tokens=evidence["tokens"], source_positions=evidence["positions"],
                    source_token_mask=evidence["token_mask"])
                detector["forecast"] = forecast
                detector["forecast_source_bev"] = forecast_source_bev
            outputs.append(detector)
            # Phase-B uses ten causal tracking warmup frames only to build
            # capture-native history.  It never performs eleven-frame
            # perception BPTT; the final frame retains the forecast graph.
            if run_forecast and frame_index < len(runtime_frames) - 1:
                state = state.detached()
        return outputs, state
