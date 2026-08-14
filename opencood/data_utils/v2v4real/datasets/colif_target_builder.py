"""Supervision-only source evidence targets for CoLIF."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from opencood.utils import box_utils_v2v4real


class CanonicalTargetBuilder:
    """Build local positives only when the source point cloud supports them."""

    def __init__(
        self,
        lidar_range: Sequence[float],
        max_objects: int = 100,
        min_points: int = 3,
        order: str = "hwl",
    ) -> None:
        self.lidar_range = tuple(float(value) for value in lidar_range)
        self.max_objects = int(max_objects)
        self.min_points = int(min_points)
        self.order = order
        if self.max_objects < 1 or self.min_points < 1:
            raise ValueError("max_objects and min_points must be positive")
        if self.order != "hwl":
            raise ValueError("The current VoxelPostprocessor target path requires hwl")

    def build_source_target(self, yaml_params: Mapping, lidar_points: np.ndarray) -> dict:
        vehicles = yaml_params.get("vehicles", {})
        projected = {}
        box_utils_v2v4real.project_world_objects(
            vehicles,
            projected,
            # V2V4Real object locations are source-LiDAR local.  This helper's
            # matrix argument is source->target (despite the legacy name
            # ``project_world_objects``), so identity yields native boxes for
            # point-support counting.
            np.eye(4, dtype=np.float64),
            list(self.lidar_range),
            self.order,
        )

        boxes = np.zeros((self.max_objects, 7), dtype=np.float32)
        mask = np.zeros(self.max_objects, dtype=np.float32)
        actor_ids = np.full(self.max_objects, -1, dtype=np.int64)
        association_ignore = np.ones(self.max_objects, dtype=np.bool_)
        point_counts = np.zeros(self.max_objects, dtype=np.int32)
        supported_object_keys = []

        target_index = 0
        for object_key, content in projected.items():
            box = np.asarray(content["coord"], dtype=np.float32)
            corners = box_utils_v2v4real.boxes_to_corners_3d(
                box.copy(), order=self.order
            )[0]
            points = box_utils_v2v4real.get_points_in_rotated_box_3d(
                lidar_points[:, :3], corners
            )
            if points.shape[0] < self.min_points:
                continue
            if target_index >= self.max_objects:
                break
            association_id = content.get("ass_id", -1)
            try:
                association_id = int(association_id)
            except (TypeError, ValueError):
                association_id = -1

            boxes[target_index] = box[0]
            mask[target_index] = 1.0
            actor_ids[target_index] = association_id
            association_ignore[target_index] = association_id < 0
            point_counts[target_index] = int(points.shape[0])
            supported_object_keys.append(str(object_key))
            target_index += 1

        return {
            "source_boxes": boxes,
            "source_box_mask": mask,
            "source_actor_ids": actor_ids,
            "association_ignore_mask": association_ignore,
            "source_point_counts": point_counts,
            "supported_object_keys": tuple(supported_object_keys),
        }
