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


class DualCmpClipModel(nn.Module):
    def __init__(self, detector_args, anchors, lidar_range, max_actors=128,
                 proposal_top_k=128, max_miss=2, proposal_score_threshold=0.20,
                 proposal_nms_thresh=0.15, emission_threshold=None):
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
        self.track_update = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU())
        self.track_exist = nn.Linear(feature_dim, 1)
        self.track_box = nn.Linear(feature_dim, 7)
        self.track_velocity = nn.Linear(feature_dim, 2)

    @staticmethod
    def emission_mask(valid_mask, quality, threshold):
        """AB3DMOT-style max-age validity plus learned quality threshold."""
        return valid_mask & (quality >= float(threshold))

    def forward(self, runtime_frames, state=None, max_dt_s=0.2,
                detector_autocast=False):
        """Advance only from runtime frames; actor state remains FP32.

        ``detector_autocast`` deliberately encloses the dense detector only.
        Geometry, lifecycle and persistent state run outside AMP so their
        dtype invariant is independent of the trainer's precision policy.
        """
        outputs = []
        for frame in runtime_frames:
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
                                         detector["fused_bev"].shape[1], 1, pose, time_s)
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
            detector["roi_feature"] = self.roi(detector["fused_bev"], state.box)
            update = self.track_update(detector["roi_feature"])
            detector["track_exist_logits"] = self.track_exist(update).squeeze(-1)
            detector["track_box"] = state.box + self.track_box(update)
            detector["track_velocity"] = state.velocity + self.track_velocity(update)
            quality = detector["track_exist_logits"].sigmoid()
            valid = state.valid_mask
            state = replace(
                state,
                feature=torch.where(valid[..., None], update, state.feature),
                box=torch.where(valid[..., None], detector["track_box"], state.box),
                velocity=torch.where(valid[..., None], detector["track_velocity"], state.velocity),
                quality=torch.where(valid, quality, state.quality),
                score=torch.where(valid, quality, state.score),
                # AB3DMOT output keeps valid tracks through max_age; an
                # unmatched frame must not suppress a quality-qualified actor.
                active_mask=self.emission_mask(valid, quality, self.emission_threshold),
                history_box=torch.where(
                    valid[..., None, None],
                    torch.cat((detector["track_box"].unsqueeze(2), state.history_box[:, :, :-1]), dim=2),
                    state.history_box),
                history_mask=torch.where(valid[..., None], torch.ones_like(state.history_mask), state.history_mask),
            )
            state.validate()
            detector["state"] = state
            outputs.append(detector)
        return outputs, state
