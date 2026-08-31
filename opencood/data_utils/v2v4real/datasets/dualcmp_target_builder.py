"""Target-only actor and causal velocity construction for DualCMP."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from opencood.utils import box_utils_v2v4real


def source_to_receiver_transform(source_pose: np.ndarray, receiver_pose: np.ndarray) -> np.ndarray:
    source_pose, receiver_pose = np.asarray(source_pose, dtype=np.float64), np.asarray(receiver_pose, dtype=np.float64)
    if source_pose.shape != (4, 4) or receiver_pose.shape != (4, 4):
        raise ValueError("V2V4Real lidar_pose must be a homogeneous 4x4 matrix")
    return (np.linalg.inv(receiver_pose) @ source_pose).astype(np.float32)


class CausalTargetBuilder:
    """Build receiver-centric targets without ever consulting a later frame."""

    def __init__(self, lidar_range: Sequence[float], max_actors: int, order: str = "hwl") -> None:
        self.lidar_range, self.max_actors, self.order = tuple(map(float, lidar_range)), int(max_actors), order
        if self.max_actors < 1 or self.order != "hwl":
            raise ValueError("max_actors must be positive and V2V4Real order must be hwl")

    def actors_in_receiver(self, yaml_params: Mapping, receiver_pose: np.ndarray) -> dict:
        source_pose = np.asarray(yaml_params["lidar_pose"], dtype=np.float32)
        projected = {}
        box_utils_v2v4real.project_world_objects(yaml_params.get("vehicles", {}), projected,
                                                  source_to_receiver_transform(source_pose, receiver_pose),
                                                  list(self.lidar_range), self.order)
        actors = {}
        for object_key, content in projected.items():
            try:
                actor_id = int(object_key)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "DualCMP receiver-local vehicle object key must be a non-negative integer: %r"
                    % (object_key,)
                ) from exc
            if actor_id < 0:
                raise ValueError(
                    "DualCMP receiver-local vehicle object key must be non-negative: %r"
                    % (object_key,)
                )
            if actor_id in actors:
                raise ValueError(
                    "DualCMP receiver-local vehicle object keys map to duplicate actor ID %d"
                    % actor_id
                )
            actors[actor_id] = np.asarray(content["coord"][0], dtype=np.float32)
        return actors

    def build_current(self, current_actors: Mapping[int, np.ndarray], previous_actors: Optional[Mapping[int, np.ndarray]],
                      frame_period_s: float) -> dict:
        if previous_actors is not None and float(frame_period_s) <= 0:
            raise ValueError("Causal velocity requires a positive backward time interval")
        boxes = np.zeros((self.max_actors, 7), dtype=np.float32)
        mask = np.zeros(self.max_actors, dtype=np.bool_)
        actor_ids = np.full(self.max_actors, -1, dtype=np.int64)
        velocity = np.zeros((self.max_actors, 2), dtype=np.float32)
        velocity_mask = np.zeros(self.max_actors, dtype=np.bool_)
        for index, actor_id in enumerate(sorted(current_actors)[: self.max_actors]):
            boxes[index], mask[index], actor_ids[index] = current_actors[actor_id], True, actor_id
            if previous_actors is not None and actor_id in previous_actors:
                # Both boxes are already expressed in the current receiver
                # frame: previous actors are reprojected before this difference.
                velocity[index] = (boxes[index, :2] - previous_actors[actor_id][:2]) / float(frame_period_s)
                velocity_mask[index] = True
        return {"actor_ids": actor_ids, "current_boxes": boxes, "current_mask": mask,
                "current_velocity": velocity, "velocity_mask": velocity_mask}
