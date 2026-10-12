"""Target-only actor and causal velocity construction for DualCMP."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from opencood.utils import box_utils_v2v4real


def source_to_receiver_transform(source_pose: np.ndarray, receiver_pose: np.ndarray) -> np.ndarray:
    source_pose, receiver_pose = np.asarray(source_pose, dtype=np.float64), np.asarray(receiver_pose, dtype=np.float64)
    if source_pose.shape != (4, 4) or receiver_pose.shape != (4, 4):
        raise ValueError("V2V4Real lidar_pose must be a homogeneous 4x4 matrix")
    return (np.linalg.inv(receiver_pose) @ source_pose).astype(np.float32)


def transform_tracking_boxes(boxes: np.ndarray, source_pose: np.ndarray,
                             destination_pose: np.ndarray) -> np.ndarray:
    """Rigidly project ``[x,y,z,h,w,l,yaw]`` boxes between LiDAR frames."""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 7 or not np.isfinite(boxes).all():
        raise ValueError("tracking boxes must be finite [N,7]")
    if boxes.shape[0] == 0:
        return boxes.copy()
    corners = box_utils_v2v4real.boxes_to_corners_3d(boxes.copy(), order="hwl")
    projected = box_utils_v2v4real.project_box3d(
        corners, source_to_receiver_transform(source_pose, destination_pose))
    return box_utils_v2v4real.corner_to_center(projected, order="hwl").astype(np.float32)


@lru_cache(maxsize=64)
def load_tracking_label_file(path: str) -> Dict[int, Dict[int, np.ndarray]]:
    """Read evaluator-native KITTI labels into CAV0 reference boxes by frame."""
    result = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) < 17 or fields[2].lower() != "car":
            raise ValueError("invalid tracking label line %d in %s" % (line_number, path))
        frame, track_id = int(fields[0]), int(fields[1])
        # The evaluator's V2V4Real writer stores LiDAR [x,y,z] as KITTI
        # camera [x,z,y]; undo only that documented interface conversion.
        box = np.asarray([float(fields[13]), float(fields[15]), float(fields[14]),
                          float(fields[10]), float(fields[11]), float(fields[12]),
                          float(fields[16])], dtype=np.float32)
        if frame < 0 or track_id < 0 or not np.isfinite(box).all() or np.any(box[3:6] <= 0):
            raise ValueError("invalid tracking label values in %s:%d" % (path, line_number))
        if track_id in result.setdefault(frame, {}):
            raise ValueError("duplicate tracking ID %d in frame %d of %s" % (track_id, frame, path))
        result[frame][track_id] = box
    if not result:
        raise ValueError("tracking label contains no Car states: %s" % path)
    return result


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

    def actors_from_tracking_labels(self, label_root: str, scene_id: str,
                                    receiver_id: str, frame_index: int,
                                    reference_pose: Optional[np.ndarray] = None,
                                    receiver_pose: Optional[np.ndarray] = None) -> dict:
        """Return stable label track IDs in CAV0 reference or receiver coordinates."""
        path = Path(label_root) / (str(scene_id) + "-" + str(receiver_id) + ".txt")
        return self.actors_from_tracking_label_file(
            path, frame_index, reference_pose, receiver_pose)

    def actors_from_tracking_label_file(self, path: Path, frame_index: int,
                                        reference_pose: Optional[np.ndarray] = None,
                                        receiver_pose: Optional[np.ndarray] = None) -> dict:
        """Read one evaluator-native CAV0 label file at its local frame ordinal."""
        if not path.is_file():
            raise FileNotFoundError("tracking label missing: %s" % path)
        labels = load_tracking_label_file(str(path))
        if int(frame_index) not in labels:
            raise ValueError("tracking label missing frame %d in %s" % (frame_index, path))
        actors = labels[int(frame_index)]
        if reference_pose is None and receiver_pose is None:
            return {track_id: box.copy() for track_id, box in actors.items()}
        if reference_pose is None or receiver_pose is None:
            raise ValueError("both reference_pose and receiver_pose are required for projection")
        ids = sorted(actors)
        transformed = transform_tracking_boxes(
            np.stack([actors[track_id] for track_id in ids]), reference_pose, receiver_pose)
        return {track_id: transformed[index] for index, track_id in enumerate(ids)}

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
