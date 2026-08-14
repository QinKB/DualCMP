"""Transform source-native observations into the current receiver frame."""

from __future__ import annotations

import torch
import torch.nn as nn


class PoseTimeAligner(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.metadata = nn.Sequential(
            nn.Linear(3, feature_dim), nn.ReLU(inplace=True), nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, observations, runtime_frame):
        feature = observations["obs_feat"]
        boxes = observations["obs_box"]
        scores = observations["obs_score"]
        mask = observations["obs_mask"] & runtime_frame["source_mask"].unsqueeze(-1)
        transform = runtime_frame["source_to_receiver"]
        rotation = transform[..., :3, :3]
        translation = transform[..., :3, 3]
        centers = torch.einsum("bsij,bsmj->bsmi", rotation, boxes[..., :3])
        centers = centers + translation.unsqueeze(2)
        yaw_delta = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
        aligned_boxes = boxes.clone()
        aligned_boxes[..., :3] = centers
        aligned_boxes[..., 6] = torch.atan2(
            torch.sin(boxes[..., 6] + yaw_delta.unsqueeze(-1)),
            torch.cos(boxes[..., 6] + yaw_delta.unsqueeze(-1)),
        )
        dt = runtime_frame["delta_t_s"].nan_to_num(0.0)
        receiver_time = runtime_frame.get(
            "receiver_time_s", torch.zeros(feature.shape[0], device=feature.device)
        ).to(device=feature.device, dtype=dt.dtype)
        capture_time = runtime_frame.get(
            "capture_time_s", receiver_time.unsqueeze(-1).expand_as(dt)
        ).to(device=feature.device, dtype=dt.dtype)
        capture_time = torch.where(
            torch.isfinite(capture_time), capture_time, receiver_time.unsqueeze(-1)
        )
        source_score = scores
        source_slot = torch.arange(
            feature.shape[1], device=feature.device, dtype=feature.dtype
        ).view(1, -1).expand(feature.shape[0], -1)
        source_slot = source_slot / max(feature.shape[1] - 1, 1)
        meta = torch.stack((dt.to(feature.dtype), source_score.mean(-1), source_slot), dim=-1)
        feature = feature + self.metadata(meta).unsqueeze(2)
        batch, sources, proposals = mask.shape
        source_index = torch.arange(sources, device=mask.device).view(1, sources, 1)
        source_index = source_index.expand(batch, sources, proposals)
        return {
            "feat": feature.reshape(batch, sources * proposals, -1),
            "box": aligned_boxes.reshape(batch, sources * proposals, 7),
            "score": scores.reshape(batch, sources * proposals),
            "mask": mask.reshape(batch, sources * proposals),
            "source": source_index.reshape(batch, sources * proposals),
            "delta_t_s": dt.unsqueeze(-1).expand(batch, sources, proposals).reshape(
                batch, sources * proposals
            ),
            "capture_time_s": capture_time.unsqueeze(-1).expand(
                batch, sources, proposals
            ).reshape(batch, sources * proposals),
        }
