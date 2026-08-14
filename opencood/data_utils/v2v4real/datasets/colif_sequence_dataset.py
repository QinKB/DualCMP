"""Receiver-centric causal clips for fused-BEV CoLIF.

Every available source cloud is projected into the current receiver frame
before voxelization, matching CMP's V2V4Real intermediate-fusion contract.
Canonical identities and future trajectories are constructed in a separate
supervision branch and never enter a runtime frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from opencood.data_utils.v2v4real.datasets.colif_schema import (
    assert_runtime_is_supervision_free,
)
from opencood.data_utils.v2v4real.datasets.colif_sequence_index import (
    SceneSequenceIndex,
    SourceFrameRecord,
)
from opencood.data_utils.v2v4real.datasets.colif_target_builder import (
    CanonicalTargetBuilder,
)
from opencood.data_utils.v2v4real.pre_processor import build_preprocessor
from opencood.data_utils.v2v4real.post_processor import build_postprocessor
from opencood.utils.pcd_utils import (
    mask_ego_points,
    mask_points_by_range,
    pcd_to_np,
    shuffle_points,
)
from opencood.utils import box_utils_v2v4real


@lru_cache(maxsize=512)
def _load_yaml_cached(path: str) -> dict:
    """Cache immutable raw-frame metadata inside each DataLoader worker."""

    # V2V4Real stores NumPy-tagged values, so the unsafe Loader semantics of
    # CMP must be retained.  CLoader is API-compatible here and avoids making
    # canonical-index construction a multi-minute Python-parser bottleneck.
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.load(stream, Loader=yaml.CLoader)


@dataclass(frozen=True)
class ClipRecord:
    scene_id: str
    receiver_id: str
    start_position: int


def source_to_receiver_transform(
    source_lidar_to_world: np.ndarray,
    receiver_lidar_to_world: np.ndarray,
) -> np.ndarray:
    """Transform source-LiDAR points into receiver-LiDAR coordinates."""

    source_pose = np.asarray(source_lidar_to_world, dtype=np.float64)
    receiver_pose = np.asarray(receiver_lidar_to_world, dtype=np.float64)
    if source_pose.shape != (4, 4) or receiver_pose.shape != (4, 4):
        raise ValueError("V2V4Real lidar_pose must be a homogeneous 4x4 matrix")
    return (np.linalg.inv(receiver_pose) @ source_pose).astype(np.float32)


def points_for_split(points: np.ndarray, train: bool) -> np.ndarray:
    """Randomize training points but keep evaluation voxelization repeatable."""

    return shuffle_points(points) if train else points


class ColifSequenceDataset(Dataset):
    """Return receiver-aligned causal clips with isolated supervision."""

    def __init__(self, params, visualize=False, train=True, isSim=False):
        del isSim  # The new path records clock/delay semantics explicitly.
        self.params = params
        self.visualize = bool(visualize)
        self.train = bool(train)
        fusion_args = params.get("fusion", {}).get("args", {}) or {}

        self.runtime_only = bool(fusion_args.get("runtime_only", True))
        # Reject the retired source-local shortcut explicitly.  The fused
        # model always builds one receiver-centric target set when supervision
        # is enabled.
        self.local_targets_only = bool(fusion_args.get("local_targets_only", False))
        if self.local_targets_only:
            raise ValueError(
                "local_targets_only belongs to the retired source-local model"
            )
        if not bool(fusion_args.get("runtime_forbid_gt", True)):
            raise ValueError("CoLIF runtime_forbid_gt must remain true")

        self.frame_period_s = float(fusion_args.get("frame_period_s", 0.1))
        self.warmup_length = int(fusion_args.get("warmup_length", 0))
        self.unroll_length = int(fusion_args.get("unroll_length", 1))
        self.clip_stride = int(fusion_args.get("clip_stride", 1))
        self.future_frames = int(fusion_args.get("future_frames", 50))
        self.max_actors = int(fusion_args.get("max_actors", 128))
        if self.warmup_length < 0:
            raise ValueError("warmup_length must be non-negative")
        if self.unroll_length < 1 or self.clip_stride < 1:
            raise ValueError("unroll_length and clip_stride must be positive")
        self.runtime_clip_length = self.warmup_length + self.unroll_length
        if self.future_frames < 1 or self.max_actors < 1:
            raise ValueError("future_frames and max_actors must be positive")

        self.split_name = str(
            fusion_args.get("split_name", "train" if self.train else "val")
        )
        if self.train:
            root_dir = params["root_dir"]
        elif self.split_name == "test" and "test_dir" in params:
            root_dir = params["test_dir"]
        else:
            root_dir = params["validate_dir"]

        manifest = fusion_args.get("split_manifest") or params.get("split_manifest")
        self.index = SceneSequenceIndex(
            root_dir=root_dir,
            frame_period_s=self.frame_period_s,
            split_manifest=manifest,
            split_name=self.split_name if manifest else None,
        )
        self.canonical_actor_map = {}
        self.canonical_audit = {}
        self.pre_processor = build_preprocessor(params["preprocess"], train=self.train)
        self.lidar_range = params["preprocess"]["cav_lidar_range"]
        self.post_processor = None
        self.target_builder = None
        self.anchor_box = None
        if not self.runtime_only:
            visibility_rule = fusion_args.get("target_visibility_rule")
            if visibility_rule != "point_support":
                raise ValueError(
                    "Training requires target_visibility_rule=point_support; global GT copying is forbidden"
                )
            self.post_processor = build_postprocessor(params["postprocess"], train=self.train)
            self.anchor_box = self.post_processor.generate_anchor_box()
            self.target_builder = CanonicalTargetBuilder(
                lidar_range=self.lidar_range,
                max_objects=int(params["postprocess"].get("max_num", 100)),
                min_points=int(fusion_args.get("target_min_points", 3)),
                order=params["postprocess"].get("order", "hwl"),
            )
            self.canonical_actor_map, self.canonical_audit = (
                self._build_canonical_identity_index(
                    min_stable_frames=int(
                        fusion_args.get("canonical_min_stable_frames", 2)
                    ),
                    max_step_m=float(
                        fusion_args.get("canonical_max_step_m", 10.0)
                    ),
                )
            )
        self.max_cav = int(
            fusion_args.get("max_cav", params.get("train_params", {}).get("max_cav", 2))
        )
        if self.max_cav < 1:
            raise ValueError("max_cav must be positive")

        wild_setting = params.get("wild_setting", {})
        async_value = wild_setting.get("async", False)
        self.async_enabled = bool(async_value) and str(async_value).lower() != "false"
        self.delay_mode = str(fusion_args.get("delay_mode", "synthetic"))
        if self.delay_mode != "synthetic":
            raise NotImplementedError("Only provenance-explicit synthetic delay is implemented")
        overhead_ms = fusion_args.get(
            "async_overhead_ms", wild_setting.get("async_overhead", 0)
        )
        self.partner_delay_s = float(overhead_ms) / 1000.0 if self.async_enabled else 0.0
        if self.partner_delay_s < 0:
            raise ValueError("Synthetic delay must be non-negative")

        configured_receivers = fusion_args.get("receiver_ids", [0])
        self.receiver_ids = {str(value) for value in configured_receivers}
        self.clip_records: List[ClipRecord] = []
        for scene_id in self.index.scene_ids():
            for receiver_id in self.index.source_ids(scene_id):
                if receiver_id not in self.receiver_ids:
                    continue
                frame_count = len(self.index.receiver_frame_keys(scene_id, receiver_id))
                last_start = frame_count - self.runtime_clip_length
                for start in range(0, last_start + 1, self.clip_stride):
                    self.clip_records.append(ClipRecord(scene_id, receiver_id, start))
        if not self.clip_records:
            raise ValueError(
                "No complete clips for warmup_length=%d, unroll_length=%d"
                % (self.warmup_length, self.unroll_length)
            )

    def __len__(self) -> int:
        return len(self.clip_records)

    @staticmethod
    def _load_pose(record: SourceFrameRecord) -> np.ndarray:
        params = _load_yaml_cached(str(record.yaml_path))
        pose = np.asarray(params["lidar_pose"], dtype=np.float32)
        if pose.shape != (4, 4):
            raise ValueError("Invalid lidar_pose in %s: %s" % (record.yaml_path, pose.shape))
        return pose

    def _source_ids_for_receiver(self, scene_id: str, receiver_id: str) -> Tuple[str, ...]:
        others = [source for source in self.index.source_ids(scene_id)
                  if source != receiver_id]
        return tuple([receiver_id] + others)[: self.max_cav]

    @staticmethod
    def _object_key_as_int(object_key):
        try:
            value = int(object_key)
        except (TypeError, ValueError):
            return None
        return value if value >= 0 else None

    def _build_canonical_identity_index(
        self, min_stable_frames: int, max_step_m: float
    ):
        """Build supervision-only scene identities with conflict auditing.

        V2V4Real agent 0 commonly uses ``ass_id=-1`` while partner labels
        reference agent-0 local IDs.  The resulting mapping is never exposed
        through the runtime frame.
        """

        if min_stable_frames < 2 or max_step_m <= 0:
            raise ValueError("Invalid canonical identity quality thresholds")
        mapping = {}
        audit = {
            "stable_multi_frame_mapping_count": 0,
            "single_frame_mapping_count": 0,
            "mapping_conflict_count": 0,
            "geometry_conflict_count": 0,
            "unmapped_local_track_count": 0,
            "backfill_lengths": [],
        }
        synthetic_id = 1_000_000_000
        for scene_id in self.index.scene_ids():
            source_ids = self.index.source_ids(scene_id)
            reference_id = source_ids[0]
            used_ids = set()

            for frame_key in self.index.receiver_frame_keys(
                scene_id, reference_id
            ):
                record = self.index.source_record(
                    scene_id, reference_id, frame_key
                )
                params = _load_yaml_cached(str(record.yaml_path))
                for object_key in params.get("vehicles", {}):
                    map_key = (scene_id, str(reference_id), str(object_key))
                    if map_key in mapping:
                        continue
                    canonical_id = self._object_key_as_int(object_key)
                    if canonical_id is None or canonical_id in used_ids:
                        canonical_id = synthetic_id
                        synthetic_id += 1
                    mapping[map_key] = canonical_id
                    used_ids.add(canonical_id)

            for source_id in source_ids[1:]:
                tracks = {}
                for frame_key in self.index.receiver_frame_keys(
                    scene_id, source_id
                ):
                    record = self.index.source_record(
                        scene_id, source_id, frame_key
                    )
                    params = _load_yaml_cached(str(record.yaml_path))
                    for object_key, content in params.get("vehicles", {}).items():
                        item = tracks.setdefault(
                            str(object_key),
                            {"ids": [], "positions": [], "valid_positions": []},
                        )
                        association_id = content.get("ass_id", -1)
                        try:
                            association_id = int(association_id)
                        except (TypeError, ValueError):
                            association_id = -1
                        location = np.asarray(
                            content.get("location", [np.nan, np.nan]),
                            dtype=np.float64,
                        )[:2]
                        item["positions"].append(
                            (int(record.frame_index), location)
                        )
                        if association_id >= 0:
                            item["ids"].append(association_id)
                            item["valid_positions"].append(
                                int(record.frame_index)
                            )

                for object_key, item in tracks.items():
                    unique_ids = sorted(set(item["ids"]))
                    positions = sorted(item["positions"], key=lambda value: value[0])
                    geometry_ok = True
                    for (_, first), (_, second) in zip(
                        positions[:-1], positions[1:]
                    ):
                        if not np.isfinite(first).all() or not np.isfinite(second).all():
                            continue
                        if float(np.linalg.norm(second - first)) > max_step_m:
                            geometry_ok = False
                            break
                    stable = (
                        len(unique_ids) == 1
                        and len(item["ids"]) >= min_stable_frames
                        and geometry_ok
                    )
                    if stable:
                        canonical_id = unique_ids[0]
                        audit["stable_multi_frame_mapping_count"] += 1
                        first_valid = min(item["valid_positions"])
                        audit["backfill_lengths"].append(
                            sum(
                                int(frame_index < first_valid)
                                for frame_index, _ in positions
                            )
                        )
                    else:
                        canonical_id = synthetic_id
                        synthetic_id += 1
                        audit["unmapped_local_track_count"] += 1
                        if len(unique_ids) > 1:
                            audit["mapping_conflict_count"] += 1
                        elif len(item["ids"]) == 1:
                            audit["single_frame_mapping_count"] += 1
                        if not geometry_ok:
                            audit["geometry_conflict_count"] += 1
                    mapping[(scene_id, str(source_id), object_key)] = canonical_id

        backfills = audit.pop("backfill_lengths")
        audit["backfill_count"] = len(backfills)
        audit["backfill_length_mean"] = (
            float(np.mean(backfills)) if backfills else 0.0
        )
        audit["backfill_length_max"] = max(backfills, default=0)
        return mapping, audit

    def _load_source_capture(
        self,
        record: SourceFrameRecord,
        receiver_pose: np.ndarray,
        source_slot: int,
        delay_s: float,
        include_source_target: bool,
    ) -> Optional[dict]:
        source_yaml = _load_yaml_cached(str(record.yaml_path))
        source_pose = np.asarray(source_yaml["lidar_pose"], dtype=np.float32)
        if source_pose.shape != (4, 4):
            raise ValueError("Invalid lidar_pose in %s" % record.yaml_path)
        lidar_native = points_for_split(
            pcd_to_np(str(record.lidar_path)), self.train
        )
        lidar_native = mask_ego_points(lidar_native)
        transform = source_to_receiver_transform(source_pose, receiver_pose)
        lidar = lidar_native.copy()
        lidar[:, :3] = box_utils_v2v4real.project_points_by_matrix_torch(
            lidar[:, :3], transform
        )
        lidar = mask_points_by_range(lidar, self.lidar_range)
        if lidar.shape[0] == 0:
            return None
        processed_lidar = self.pre_processor.preprocess(lidar)
        if processed_lidar["voxel_features"].shape[0] == 0:
            return None
        capture = {
            "source_id": int(record.source_id),
            "source_slot": int(source_slot),
            "capture_frame_key": record.frame_key,
            "capture_frame_idx": int(record.frame_index),
            "capture_time_s": float(record.capture_time_s),
            "arrival_time_s": float(record.capture_time_s + delay_s),
            "delta_t_s": None,  # Filled relative to the receiver below.
            "source_pose": source_pose,
            "source_to_receiver": transform,
            "processed_lidar": processed_lidar,
        }
        if include_source_target and self.target_builder is not None:
            source_target = self.target_builder.build_source_target(
                source_yaml, lidar_native
            )
            capture["source_target"] = source_target
        return capture

    def _runtime_frame(
        self,
        scene_id: str,
        receiver_id: str,
        receiver_frame_key: str,
        include_targets: bool = True,
    ) -> dict:
        receiver_record = self.index.receiver_record(
            scene_id, receiver_id, receiver_frame_key
        )
        receiver_pose = self._load_pose(receiver_record)
        captures = []
        source_targets = []
        unavailable_slots = []
        source_ids = self._source_ids_for_receiver(scene_id, receiver_id)
        for slot, source_id in enumerate(source_ids):
            delay_s = 0.0 if source_id == receiver_id else self.partner_delay_s
            if delay_s == 0.0:
                source_record = self.index.source_record(
                    scene_id, source_id, receiver_frame_key
                )
            else:
                source_record = self.index.latest_arrived_source_record(
                    scene_id=scene_id,
                    receiver_id=receiver_id,
                    source_id=source_id,
                    receiver_frame_key=receiver_frame_key,
                    delay_s=delay_s,
                )
            if source_record is None:
                unavailable_slots.append(slot)
                continue
            capture = self._load_source_capture(
                source_record,
                receiver_pose,
                source_slot=slot,
                delay_s=delay_s,
                include_source_target=include_targets,
            )
            if capture is None:
                unavailable_slots.append(slot)
                continue
            capture["delta_t_s"] = float(
                receiver_record.capture_time_s - source_record.capture_time_s
            )
            capture["delay_source"] = (
                "receiver_current" if source_id == receiver_id else "synthetic_fixed"
            )
            if "source_target" in capture:
                source_target = capture.pop("source_target")
                source_target["source_id"] = int(source_id)
                source_target["source_slot"] = int(slot)
                source_target["capture_frame_key"] = source_record.frame_key
                source_target["capture_frame_idx"] = int(source_record.frame_index)
                source_targets.append(source_target)
            captures.append(capture)

        runtime = {
            "scene_id": scene_id,
            "receiver_id": int(receiver_id),
            "receiver_frame_key": receiver_frame_key,
            "receiver_frame_idx": int(receiver_record.frame_index),
            "receiver_time_s": float(receiver_record.capture_time_s),
            "receiver_pose": receiver_pose,
            "pose_convention": "v2v4real_lidar_to_world_homogeneous_4x4",
            "configured_source_ids": tuple(int(source) for source in source_ids),
            "unavailable_source_slots": tuple(unavailable_slots),
            "source_captures": captures,
        }
        assert_runtime_is_supervision_free(runtime)
        return runtime, {"source_targets": source_targets}

    def _canonical_id(self, scene_id, source_id, object_key):
        return self.canonical_actor_map.get(
            (str(scene_id), str(source_id), str(object_key))
        )

    def _project_yaml_actors(
        self, scene_id, source_id, yaml_params, receiver_pose
    ):
        actors = {}
        source_pose = np.asarray(yaml_params["lidar_pose"], dtype=np.float64)
        source_to_receiver = source_to_receiver_transform(
            source_pose, receiver_pose
        )
        for object_key, object_content in yaml_params.get("vehicles", {}).items():
            actor_id = self._canonical_id(scene_id, source_id, object_key)
            if actor_id is None:
                continue
            projected = {}
            box_utils_v2v4real.project_world_objects(
                {object_key: object_content},
                projected,
                source_to_receiver,
                self.lidar_range,
                order="hwl",
            )
            if object_key not in projected:
                continue
            actors[actor_id] = np.asarray(
                projected[object_key]["coord"][0], dtype=np.float32
            )
        return actors

    def _receiver_forecast_target(
        self, scene_id, receiver_id, receiver_frame_key, source_targets
    ):
        receiver_keys = self.index.receiver_frame_keys(scene_id, receiver_id)
        current_position = receiver_keys.index(receiver_frame_key)
        receiver_record = self.index.receiver_record(
            scene_id, receiver_id, receiver_frame_key
        )
        receiver_pose = self._load_pose(receiver_record)
        source_ids = self.index.source_ids(scene_id)

        def actors_at(frame_key, target_receiver_pose):
            merged = {}
            for source_id in source_ids:
                record = self.index.source_record(scene_id, source_id, frame_key)
                if record is None:
                    continue
                yaml_params = _load_yaml_cached(str(record.yaml_path))
                for actor_id, value in self._project_yaml_actors(
                    scene_id, source_id, yaml_params, target_receiver_pose
                ).items():
                    # Source order is receiver first, hence setdefault gives
                    # the receiver label priority when duplicate annotations
                    # describe the same physical actor.
                    merged.setdefault(actor_id, value)
            return merged

        current = actors_at(receiver_frame_key, receiver_pose)
        evidence = {}
        for source_target in source_targets:
            source_id = str(source_target["source_id"])
            capture_key = source_target["capture_frame_key"]
            source_slot = int(source_target["source_slot"])
            source_mask_bit = 1 << source_slot
            for target_index, object_key in enumerate(
                source_target["supported_object_keys"]
            ):
                actor_id = self._canonical_id(scene_id, source_id, object_key)
                if actor_id is None:
                    continue
                entry = evidence.setdefault(
                    actor_id,
                    {"points": 0, "source_bits": 0, "latest_frame": None},
                )
                entry["points"] += int(
                    source_target["source_point_counts"][target_index]
                )
                entry["source_bits"] |= source_mask_bit
                capture_index = int(source_target["capture_frame_idx"])
                if entry["latest_frame"] is None:
                    entry["latest_frame"] = capture_index
                else:
                    entry["latest_frame"] = max(
                        entry["latest_frame"], capture_index
                    )

        actor_keys = [key for key in current if key in evidence]
        actor_keys = actor_keys[: self.max_actors]
        current_boxes = np.zeros((self.max_actors, 7), dtype=np.float32)
        current_mask = np.zeros(self.max_actors, dtype=np.bool_)
        actor_ids = np.full(self.max_actors, -1, dtype=np.int64)
        current_velocity = np.zeros((self.max_actors, 2), dtype=np.float32)
        evidence_point_counts = np.zeros(self.max_actors, dtype=np.int32)
        evidence_source_bits = np.zeros(self.max_actors, dtype=np.int64)
        evidence_age_frames = np.zeros(self.max_actors, dtype=np.int64)
        future_xy = np.zeros(
            (self.max_actors, self.future_frames, 2), dtype=np.float32
        )
        future_mask = np.zeros(
            (self.max_actors, self.future_frames), dtype=np.bool_
        )
        fixed_receiver_future_xy = np.zeros_like(future_xy)
        for actor_index, actor_id in enumerate(actor_keys):
            current_boxes[actor_index] = current[actor_id]
            current_mask[actor_index] = True
            actor_ids[actor_index] = actor_id
            evidence_point_counts[actor_index] = evidence[actor_id]["points"]
            evidence_source_bits[actor_index] = evidence[actor_id]["source_bits"]
            evidence_age_frames[actor_index] = max(
                0,
                int(receiver_record.frame_index)
                - int(evidence[actor_id]["latest_frame"]),
            )

        for horizon in range(self.future_frames):
            future_position = current_position + horizon + 1
            if future_position >= len(receiver_keys):
                break
            future_key = receiver_keys[future_position]
            future_receiver_record = self.index.receiver_record(
                scene_id, receiver_id, future_key
            )
            future_receiver_pose = self._load_pose(future_receiver_record)
            # CMP's released MTR caches retain every trajectory state in the
            # receiver frame of that capture.  Forecast supervision follows
            # this moving-receiver contract for a fair protocol comparison.
            future = actors_at(future_key, future_receiver_pose)
            # Tracking velocity remains a spatial propagation signal in the
            # current receiver frame, so compute it separately without mixing
            # future ego motion into the reference update.
            future_fixed_receiver = actors_at(future_key, receiver_pose)
            for actor_index, actor_id in enumerate(actor_keys):
                if actor_id not in future:
                    continue
                future_box = future[actor_id]
                future_xy[actor_index, horizon] = (
                    future_box[:2] - current_boxes[actor_index, :2]
                )
                future_mask[actor_index, horizon] = True
                if actor_id in future_fixed_receiver:
                    fixed_receiver_future_xy[actor_index, horizon] = (
                        future_fixed_receiver[actor_id][:2]
                        - current_boxes[actor_index, :2]
                    )
        first_valid = future_mask[:, 0]
        current_velocity[first_valid] = (
            fixed_receiver_future_xy[first_valid, 0] / self.frame_period_s
        )
        fused_label_dict = self.post_processor.generate_label(
            gt_box_center=current_boxes,
            anchors=self.anchor_box,
            mask=current_mask.astype(np.float32),
        )
        return {
            "actor_ids": actor_ids,
            "current_boxes": current_boxes,
            "current_velocity": current_velocity,
            "current_mask": current_mask,
            "future_xy": future_xy,
            "future_mask": future_mask,
            "evidence_point_counts": evidence_point_counts,
            "evidence_source_bits": evidence_source_bits,
            "evidence_age_frames": evidence_age_frames,
            "fused_label_dict": fused_label_dict,
        }

    def __getitem__(self, index: int) -> dict:
        clip = self.clip_records[index]
        receiver_keys = self.index.receiver_frame_keys(clip.scene_id, clip.receiver_id)
        selected_keys = receiver_keys[
            clip.start_position : clip.start_position + self.runtime_clip_length
        ]
        pairs = [
            self._runtime_frame(
                clip.scene_id,
                clip.receiver_id,
                frame_key,
                include_targets=(time_index >= self.warmup_length),
            )
            for time_index, frame_key in enumerate(selected_keys)
        ]
        runtime_frames = [pair[0] for pair in pairs]
        sample = {
            "scene_receiver_key": (clip.scene_id, int(clip.receiver_id)),
            "runtime_frames": runtime_frames,
            "loss_start_index": self.warmup_length,
        }
        if not self.runtime_only:
            target_frames = []
            for frame_key, pair in zip(
                selected_keys[self.warmup_length :],
                pairs[self.warmup_length :],
            ):
                target_frame = pair[1]
                target_frame.update(
                    self._receiver_forecast_target(
                        clip.scene_id,
                        clip.receiver_id,
                        frame_key,
                        target_frame["source_targets"],
                    )
                )
                target_frames.append(target_frame)
            sample["target_frames"] = target_frames
        assert_runtime_is_supervision_free(runtime_frames)
        return sample

    def _collate_target_frame(self, targets: Sequence[dict]) -> dict:
        return {
            "fused_label_dict": self.post_processor.collate_batch(
                [target["fused_label_dict"] for target in targets]
            ),
            "actor_ids": torch.from_numpy(
                np.stack([target["actor_ids"] for target in targets])
            ).long(),
            "current_boxes": torch.from_numpy(
                np.stack([target["current_boxes"] for target in targets])
            ).float(),
            "current_mask": torch.from_numpy(
                np.stack([target["current_mask"] for target in targets])
            ).bool(),
            "current_velocity": torch.from_numpy(
                np.stack([target["current_velocity"] for target in targets])
            ).float(),
            "future_xy": torch.from_numpy(
                np.stack([target["future_xy"] for target in targets])
            ).float(),
            "future_mask": torch.from_numpy(
                np.stack([target["future_mask"] for target in targets])
            ).bool(),
            "evidence_point_counts": torch.from_numpy(
                np.stack([target["evidence_point_counts"] for target in targets])
            ).int(),
            "evidence_source_bits": torch.from_numpy(
                np.stack([target["evidence_source_bits"] for target in targets])
            ).long(),
            "evidence_age_frames": torch.from_numpy(
                np.stack([target["evidence_age_frames"] for target in targets])
            ).long(),
        }

    def _collate_runtime_frame(self, frames: Sequence[dict]) -> dict:
        batch_size = len(frames)
        max_sources = self.max_cav
        flattened_features = []
        source_to_batch = []
        source_slot = []

        source_ids = torch.full((batch_size, max_sources), -1, dtype=torch.long)
        source_mask = torch.zeros((batch_size, max_sources), dtype=torch.bool)
        source_pose = torch.eye(4, dtype=torch.float32).repeat(batch_size, max_sources, 1, 1)
        source_to_receiver = torch.eye(4, dtype=torch.float32).repeat(
            batch_size, max_sources, 1, 1
        )
        capture_frame_idx = torch.full((batch_size, max_sources), -1, dtype=torch.long)
        capture_time_s = torch.full((batch_size, max_sources), float("nan"))
        arrival_time_s = torch.full((batch_size, max_sources), float("nan"))
        delta_t_s = torch.full((batch_size, max_sources), float("nan"))
        delay_source: List[List[str]] = [
            ["unavailable"] * max_sources for _ in range(batch_size)
        ]

        for batch_index, frame in enumerate(frames):
            for capture in frame["source_captures"]:
                slot = capture["source_slot"]
                flattened_features.append(capture["processed_lidar"])
                source_to_batch.append(batch_index)
                source_slot.append(slot)
                source_ids[batch_index, slot] = capture["source_id"]
                source_mask[batch_index, slot] = True
                source_pose[batch_index, slot] = torch.from_numpy(capture["source_pose"])
                source_to_receiver[batch_index, slot] = torch.from_numpy(
                    capture["source_to_receiver"]
                )
                capture_frame_idx[batch_index, slot] = capture["capture_frame_idx"]
                capture_time_s[batch_index, slot] = capture["capture_time_s"]
                arrival_time_s[batch_index, slot] = capture["arrival_time_s"]
                delta_t_s[batch_index, slot] = capture["delta_t_s"]
                delay_source[batch_index][slot] = capture["delay_source"]

        if not flattened_features:
            raise RuntimeError("A CoLIF batch contains no valid source voxels")
        processed_lidar = self.pre_processor.collate_batch(flattened_features)
        runtime = {
            "scene_id": [frame["scene_id"] for frame in frames],
            "receiver_id": torch.tensor(
                [frame["receiver_id"] for frame in frames], dtype=torch.long
            ),
            "receiver_frame_key": [frame["receiver_frame_key"] for frame in frames],
            "receiver_frame_idx": torch.tensor(
                [frame["receiver_frame_idx"] for frame in frames], dtype=torch.long
            ),
            "receiver_time_s": torch.tensor(
                [frame["receiver_time_s"] for frame in frames], dtype=torch.float32
            ),
            "receiver_pose": torch.from_numpy(
                np.stack([frame["receiver_pose"] for frame in frames])
            ).float(),
            "pose_convention": "v2v4real_lidar_to_world_homogeneous_4x4",
            "processed_lidar": processed_lidar,
            "source_to_batch": torch.tensor(source_to_batch, dtype=torch.long),
            "source_slot": torch.tensor(source_slot, dtype=torch.long),
            "record_len": source_mask.sum(dim=1).long(),
            "source_ids": source_ids,
            "source_mask": source_mask,
            "source_pose": source_pose,
            "source_to_receiver": source_to_receiver,
            "capture_frame_idx": capture_frame_idx,
            "capture_time_s": capture_time_s,
            "arrival_time_s": arrival_time_s,
            "delta_t_s": delta_t_s,
            "delay_source": delay_source,
        }
        assert_runtime_is_supervision_free(runtime)
        return runtime

    def collate_batch_train(self, batch: Sequence[dict]) -> dict:
        if not batch:
            raise ValueError("Cannot collate an empty batch")
        clip_length = len(batch[0]["runtime_frames"])
        if any(len(sample["runtime_frames"]) != clip_length for sample in batch):
            raise ValueError("All clips in a batch must have the same unroll length")
        loss_start_index = int(batch[0]["loss_start_index"])
        if any(
            int(sample["loss_start_index"]) != loss_start_index
            for sample in batch
        ):
            raise ValueError("All clips in a batch must share loss_start_index")
        result = {
            "scene_receiver_keys": [sample["scene_receiver_key"] for sample in batch],
            "loss_start_index": loss_start_index,
            "runtime_frames": [
                self._collate_runtime_frame(
                    [sample["runtime_frames"][time_index] for sample in batch]
                )
                for time_index in range(clip_length)
            ],
        }
        if not self.runtime_only:
            target_length = len(batch[0]["target_frames"])
            if target_length != clip_length - loss_start_index:
                raise ValueError(
                    "Target clip length must equal runtime length after warmup"
                )
            if any(len(sample["target_frames"]) != target_length for sample in batch):
                raise ValueError("All target clips in a batch must have equal length")
            result["target_frames"] = [
                self._collate_target_frame(
                    [sample["target_frames"][time_index] for sample in batch]
                )
                for time_index in range(target_length)
            ]
        assert_runtime_is_supervision_free(result["runtime_frames"])
        return result

    def collate_batch_test(self, batch: Sequence[dict]) -> dict:
        return self.collate_batch_train(batch)
