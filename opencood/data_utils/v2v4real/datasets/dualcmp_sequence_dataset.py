"""Causal receiver-centric clips for the DualCMP perception/tracking path.

The returned item has a strict split: ``runtime_frames`` can be passed to an
online model unchanged, while ``target_frames`` is supervision only.  A clip
never crosses a scene or receiver timeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from opencood.data_utils.v2v4real.datasets.dualcmp_schema import assert_runtime_is_supervision_free
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_index import SceneSequenceIndex, SourceFrameRecord
from opencood.data_utils.v2v4real.datasets.dualcmp_target_builder import CausalTargetBuilder, source_to_receiver_transform
from opencood.data_utils.v2v4real.pre_processor import build_preprocessor
from opencood.data_utils.v2v4real.post_processor import build_postprocessor
from opencood.data_utils.v2v4real.augmentor.data_augmentor import DataAugmentor
from opencood.utils import box_utils_v2v4real
from opencood.utils.pcd_utils import mask_ego_points, mask_points_by_range, pcd_to_np, shuffle_points


@lru_cache(maxsize=512)
def _load_yaml_cached(path: str) -> dict:
    with open(path, encoding="utf-8") as stream:
        return yaml.load(stream, Loader=yaml.CLoader)


@dataclass(frozen=True)
class ClipRecord:
    scene_id: str
    receiver_id: str
    start_position: int


def _numeric_id(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError("V2V4Real source IDs must be integer-compatible: %r" % value) from exc


class DualCMPSequenceDataset(Dataset):
    """Dataset whose item is one independent ``(scene, receiver)`` clip."""

    valid_stages = frozenset({"detection", "tracking", "forecast"})

    def __init__(self, params, visualize=False, train=True, isSim=False):
        del isSim
        self.params, self.visualize, self.train = params, bool(visualize), bool(train)
        args = params.get("fusion", {}).get("args", {}) or {}
        self.stage = str(args.get("stage", "tracking"))
        if self.stage not in self.valid_stages:
            raise ValueError("DualCMP stage must be detection, tracking, or forecast")
        if not bool(args.get("runtime_forbid_gt", True)):
            raise ValueError("DualCMP runtime_forbid_gt must remain true")
        self.runtime_only = bool(args.get("runtime_only", not self.train))
        self.frame_period_s = float(args.get("frame_period_s", 0.1))
        self.clip_stride = int(args.get("clip_stride", 1))
        if self.frame_period_s <= 0 or self.clip_stride < 1:
            raise ValueError("frame_period_s and clip_stride must be positive")
        tracking_frames = int(args.get("tracking_frames", 3))
        if self.stage == "tracking" and tracking_frames != 3:
            raise ValueError("DualCMP tracking requires tracking_frames == 3")
        if tracking_frames < 1:
            raise ValueError("tracking_frames must be positive")
        self.forecast_history_frames = int(args.get("forecast_history_frames", tracking_frames - 1))
        self.future_frames = int(args.get("future_frames", 50))
        if self.forecast_history_frames < 0 or self.future_frames < 1:
            raise ValueError("forecast history/future lengths are invalid")
        self.clip_length = 1 if self.stage == "detection" else (tracking_frames if self.stage == "tracking" else self.forecast_history_frames + 1)
        self.loss_start_index = 0 if self.stage != "forecast" else self.forecast_history_frames
        self.max_actors = int(args.get("max_actors", 128))
        self.max_cav = int(args.get("max_cav", params.get("train_params", {}).get("max_cav", 2)))
        if self.max_actors < 1 or self.max_cav < 1:
            raise ValueError("max_actors/max_cav must be positive")
        self.shuffle_training_points = bool(args.get("shuffle_training_points", True))
        self.sequence_augmentation = bool(
            self.train and args.get("sequence_augmentation", False)
        )
        self.data_augmentor = DataAugmentor(
            params.get("data_augment", []), train=self.sequence_augmentation,
            intermediate=True,
        )
        self.lidar_range = params["preprocess"]["cav_lidar_range"]
        self.pre_processor = build_preprocessor(params["preprocess"], train=self.train)
        self.post_processor = build_postprocessor(params["postprocess"], train=self.train) if not self.runtime_only else None
        self.anchor_box = self.post_processor.generate_anchor_box() if self.post_processor is not None else None
        if not self.runtime_only and (self.post_processor is None or self.anchor_box is None):
            raise RuntimeError("DualCMP targets require postprocessor and anchor_box when runtime_only is false")
        self.target_builder = CausalTargetBuilder(self.lidar_range, self.max_actors, params["postprocess"].get("order", "hwl"))
        configured_label_root = args.get("tracking_label_root")
        self.tracking_label_root = None
        if configured_label_root is not None:
            self.tracking_label_root = str(Path(str(configured_label_root)).expanduser().resolve())
            if not Path(self.tracking_label_root).is_dir():
                raise FileNotFoundError("DualCMP tracking_label_root does not exist: %s" % self.tracking_label_root)
        split_name = str(args.get("split_name", "train" if self.train else "val"))
        root_dir = params["root_dir"] if self.train else (params.get("test_dir") if split_name == "test" and "test_dir" in params else params["validate_dir"])
        manifest = args.get("split_manifest") or params.get("split_manifest")
        self.index = SceneSequenceIndex(
            root_dir, self.frame_period_s,
            float(args.get("timeline_tolerance_s", 1e-6)),
            manifest, split_name if manifest else None,
        )
        wild = params.get("wild_setting", {})
        async_value = wild.get("async", False)
        self.async_enabled = bool(async_value) and str(async_value).lower() != "false"
        self.partner_delay_s = float(args.get("async_overhead_ms", wild.get("async_overhead", 0))) / 1000.0 if self.async_enabled else 0.0
        if self.partner_delay_s < 0: raise ValueError("Synthetic delay must be non-negative")
        configured_receivers = args.get("receiver_ids", [0])
        self.receiver_ids = {str(value) for value in configured_receivers}
        self.clip_records: List[ClipRecord] = []
        for scene_id in self.index.scene_ids():
            for receiver_id in self.index.source_ids(scene_id):
                if receiver_id not in self.receiver_ids:
                    continue
                frame_count = len(self.index.receiver_frame_keys(scene_id, receiver_id))
                # Tracking needs a real previous receiver frame for the first
                # supervision target; it never duplicates a boundary frame.
                first_start = 0 if self.stage == "detection" else (self.forecast_history_frames if self.stage == "forecast" else 1)
                for start in range(first_start, frame_count - self.clip_length + 1, self.clip_stride):
                    self.clip_records.append(ClipRecord(scene_id, receiver_id, start))
        window_entries = args.get("window")
        if window_entries is not None:
            if not isinstance(window_entries, list) or not window_entries:
                raise ValueError("DualCMP window must be a non-empty list")
            allowed = set()
            for entry in window_entries:
                if not isinstance(entry, dict) or set(entry) != {"scene_id", "receiver_id", "start_frame", "end_frame"}:
                    raise ValueError("DualCMP window entries require scene_id, receiver_id, start_frame, end_frame")
                start_frame, end_frame = int(entry["start_frame"]), int(entry["end_frame"])
                if end_frame < start_frame:
                    raise ValueError("DualCMP window end_frame must be >= start_frame")
                allowed.add((str(entry["scene_id"]), str(entry["receiver_id"]), start_frame, end_frame))
            self.clip_records = [
                record for record in self.clip_records
                if any(
                    record.scene_id == scene_id and record.receiver_id == receiver_id
                    and (
                        self.stage != "tracking"
                        or start_frame <= int(
                            self.index.receiver_frame_keys(
                                scene_id, receiver_id
                            )[record.start_position]
                        )
                    )
                    and start_frame <= int(self.index.receiver_frame_keys(scene_id, receiver_id)[record.start_position + self.clip_length - 1])
                    and int(self.index.receiver_frame_keys(scene_id, receiver_id)[record.start_position + self.clip_length - 1]) <= end_frame
                    for scene_id, receiver_id, start_frame, end_frame in allowed
                )
            ]
        if not self.clip_records:
            raise ValueError("No causal %s clips with sufficient receiver history" % self.stage)

    def __len__(self): return len(self.clip_records)

    @staticmethod
    def _pose(record: SourceFrameRecord) -> np.ndarray:
        pose = np.asarray(_load_yaml_cached(str(record.yaml_path))["lidar_pose"], dtype=np.float32)
        if pose.shape != (4, 4): raise ValueError("Invalid lidar_pose in %s" % record.yaml_path)
        return pose

    def _source_ids(self, scene_id: str, receiver_id: str) -> Tuple[str, ...]:
        return tuple([receiver_id] + [source for source in self.index.source_ids(scene_id) if source != receiver_id])[:self.max_cav]

    def _sample_augmentation(self):
        if not self.sequence_augmentation:
            return None
        flip, rotation, scale = [None, None, None], None, None
        for config in self.params.get("data_augment", []):
            if config["NAME"] == "random_world_flip":
                for index, _ in enumerate(config["ALONG_AXIS_LIST"]):
                    flip[index] = bool(np.random.choice([False, True]))
            elif config["NAME"] == "random_world_rotation":
                limits = config["WORLD_ROT_ANGLE"]
                limits = limits if isinstance(limits, list) else [-limits, limits]
                rotation = float(np.random.uniform(limits[0], limits[1]))
            elif config["NAME"] == "random_world_scaling":
                limits = config["WORLD_SCALE_RANGE"]
                scale = float(np.random.uniform(limits[0], limits[1]))
        augmentation = {"flip": flip, "noise_rotation": rotation,
                        "noise_scale": scale}
        basis = np.asarray(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            dtype=np.float32,
        )
        transformed, _ = self._augment_points_boxes(
            basis, np.zeros((0, 7), dtype=np.float32), augmentation
        )
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = (transformed[1:, :3] - transformed[0, :3]).T
        augmentation["matrix"] = matrix
        return augmentation

    def _augment_points_boxes(self, points, boxes, augmentation):
        if augmentation is None:
            return points, boxes
        result = self.data_augmentor.forward({
            "lidar_np": np.asarray(points, dtype=np.float32).copy(),
            "object_bbx_center": np.asarray(boxes, dtype=np.float32).copy(),
            "object_bbx_mask": np.ones(len(boxes), dtype=np.float32),
            "flip": augmentation["flip"],
            "noise_rotation": augmentation["noise_rotation"],
            "noise_scale": augmentation["noise_scale"],
        })
        return result["lidar_np"], result["object_bbx_center"]

    @staticmethod
    def _augmented_receiver_pose(receiver_pose, anchor_pose, augmentation):
        if augmentation is None:
            return receiver_pose
        matrix = augmentation["matrix"].astype(np.float64)
        virtual = matrix @ np.linalg.inv(anchor_pose) @ receiver_pose @ np.linalg.inv(matrix)
        return virtual.astype(np.float32)

    def _load_capture(self, record: SourceFrameRecord, receiver_id: str,
                      receiver_pose: np.ndarray, slot: int, delay_s: float,
                      augmentation=None) -> Optional[dict]:
        source_yaml = _load_yaml_cached(str(record.yaml_path))
        source_pose = np.asarray(source_yaml["lidar_pose"], dtype=np.float32)
        points = pcd_to_np(str(record.lidar_path))
        if self.train and self.shuffle_training_points: points = shuffle_points(points)
        points = mask_ego_points(points)
        transform = source_to_receiver_transform(source_pose, receiver_pose)
        points = points.copy()
        points[:, :3] = box_utils_v2v4real.project_points_by_matrix_torch(points[:, :3], transform)
        points, _ = self._augment_points_boxes(
            points, np.zeros((0, 7), dtype=np.float32), augmentation
        )
        points = mask_points_by_range(points, self.lidar_range)
        if not len(points): return None
        processed = self.pre_processor.preprocess(points)
        if processed["voxel_features"].shape[0] == 0: return None
        return {"source_id": _numeric_id(record.source_id), "source_slot": int(slot),
                "capture_frame_idx": int(record.frame_index), "capture_time_s": float(record.capture_time_s),
                "arrival_time_s": float(record.capture_time_s + delay_s), "source_pose": source_pose,
                "source_to_receiver": (
                    augmentation["matrix"] @ transform
                    if augmentation is not None else transform
                ), "processed_lidar": processed,
                "delay_source": "receiver_current" if record.source_id == receiver_id else (
                    "partner_current" if delay_s == 0 else "synthetic_fixed")}

    def _runtime_frame(self, scene_id: str, receiver_id: str, frame_key: str,
                       augmentation=None, anchor_pose=None) -> dict:
        receiver = self.index.receiver_record(scene_id, receiver_id, frame_key)
        receiver_pose, captures, unavailable = self._pose(receiver), [], []
        if anchor_pose is None:
            anchor_pose = receiver_pose
        for slot, source_id in enumerate(self._source_ids(scene_id, receiver_id)):
            delay_s = 0.0 if source_id == receiver_id else self.partner_delay_s
            record = self.index.source_record(scene_id, source_id, frame_key) if delay_s == 0 else self.index.latest_arrived_source_record(scene_id, receiver_id, source_id, frame_key, delay_s)
            if record is None:
                unavailable.append(slot); continue
            capture = self._load_capture(
                record, receiver_id, receiver_pose, slot, delay_s, augmentation
            )
            if capture is None:
                unavailable.append(slot); continue
            capture["delta_t_s"] = float(receiver.capture_time_s - record.capture_time_s)
            if capture["arrival_time_s"] > receiver.capture_time_s + 1e-9:
                raise RuntimeError("Source capture arrives after its receiver frame")
            if capture["delta_t_s"] < -1e-9:
                raise RuntimeError("Source capture is later than its receiver frame")
            captures.append(capture)
        if not any(capture["source_slot"] == 0 for capture in captures):
            raise RuntimeError("Receiver source slot 0 has no usable current LiDAR")
        runtime = {"scene_id": scene_id, "receiver_id": _numeric_id(receiver_id), "receiver_frame_key": frame_key,
                   "receiver_frame_idx": int(receiver.frame_index), "receiver_time_s": float(receiver.capture_time_s),
                   "receiver_pose": self._augmented_receiver_pose(
                       receiver_pose, anchor_pose, augmentation
                   ), "pose_convention": "v2v4real_lidar_to_world_homogeneous_4x4",
                   "configured_source_ids": tuple(_numeric_id(source) for source in self._source_ids(scene_id, receiver_id)),
                   "unavailable_source_slots": tuple(unavailable), "source_captures": captures}
        assert_runtime_is_supervision_free(runtime)
        return runtime

    def _actors(self, scene_id: str, receiver_id: str, frame_key: str,
                receiver_pose: np.ndarray, augmentation=None) -> dict:
        if self.tracking_label_root is not None:
            reference = self.index.receiver_record(scene_id, "0", frame_key)
            actors = self.target_builder.actors_from_tracking_labels(
                self.tracking_label_root, scene_id, receiver_id, int(frame_key),
                self._pose(reference), receiver_pose)
        else:
            record = self.index.receiver_record(scene_id, receiver_id, frame_key)
            yaml_params = _load_yaml_cached(str(record.yaml_path))
            actors = self.target_builder.actors_in_receiver(yaml_params, receiver_pose)
        if augmentation is None or not actors:
            return actors
        actor_ids = sorted(actors)
        _, boxes = self._augment_points_boxes(
            np.zeros((0, 4), dtype=np.float32),
            np.stack([actors[actor_id] for actor_id in actor_ids]), augmentation,
        )
        return {actor_id: boxes[index] for index, actor_id in enumerate(actor_ids)}

    def _target_frame(self, scene_id: str, receiver_id: str, frame_key: str,
                      previous_key: Optional[str], include_future: bool,
                      augmentation=None) -> dict:
        receiver = self.index.receiver_record(scene_id, receiver_id, frame_key)
        current = self._actors(
            scene_id, receiver_id, frame_key, self._pose(receiver), augmentation
        )
        previous = None
        if previous_key is not None:
            previous_record = self.index.receiver_record(scene_id, receiver_id, previous_key)
            # Reproject the previous YAML from its own receiver pose into the
            # current receiver frame; do not compare coordinates in two egos.
            previous = self._actors(
                scene_id, receiver_id, previous_key, self._pose(receiver),
                augmentation,
            )
            dt_s = receiver.capture_time_s - previous_record.capture_time_s
            if dt_s <= 0:
                raise ValueError("Causal velocity requires strictly increasing receiver times")
        else:
            dt_s = self.frame_period_s
        target = self.target_builder.build_current(current, previous, dt_s)
        target["fused_label_dict"] = self.post_processor.generate_label(gt_box_center=target["current_boxes"], anchors=self.anchor_box, mask=target["current_mask"].astype(np.float32))
        if include_future:
            keys = self.index.receiver_frame_keys(scene_id, receiver_id)
            current_position = keys.index(frame_key)
            future_xy = np.zeros((self.max_actors, self.future_frames, 2), dtype=np.float32)
            future_mask = np.zeros((self.max_actors, self.future_frames), dtype=np.bool_)
            current_ids = target["actor_ids"]
            for horizon, future_key in enumerate(keys[current_position + 1: current_position + 1 + self.future_frames]):
                future_receiver = self.index.receiver_record(scene_id, receiver_id, future_key)
                # CoLIF _receiver_forecast_target: each future target remains
                # in its own receiver capture frame; this is target-only.
                future = self._actors(
                    scene_id, receiver_id, future_key,
                    self._pose(future_receiver), augmentation,
                )
                for actor_index, actor_id in enumerate(current_ids):
                    if actor_id >= 0 and actor_id in future:
                        future_xy[actor_index, horizon] = future[actor_id][:2] - target["current_boxes"][actor_index, :2]
                        future_mask[actor_index, horizon] = True
            target.update({"future_xy": future_xy, "future_mask": future_mask})
        return target

    def __getitem__(self, index: int) -> dict:
        clip = self.clip_records[index]
        keys = self.index.receiver_frame_keys(clip.scene_id, clip.receiver_id)
        selected = keys[clip.start_position: clip.start_position + self.clip_length]
        augmentation = self._sample_augmentation()
        anchor_pose = self._pose(
            self.index.receiver_record(clip.scene_id, clip.receiver_id, selected[0])
        )
        runtime_frames = [
            self._runtime_frame(
                clip.scene_id, clip.receiver_id, key, augmentation, anchor_pose
            )
            for key in selected
        ]
        sample = {"scene_receiver_key": (clip.scene_id, _numeric_id(clip.receiver_id)), "runtime_frames": runtime_frames,
                  "loss_start_index": self.loss_start_index, "stage": self.stage}
        if not self.runtime_only:
            target_positions = range(len(selected)) if self.stage != "forecast" else range(self.loss_start_index, len(selected))
            targets = []
            for position in target_positions:
                absolute = clip.start_position + position
                previous_key = keys[absolute - 1] if absolute > 0 else None
                targets.append(self._target_frame(
                    clip.scene_id, clip.receiver_id, selected[position],
                    previous_key, self.stage == "forecast", augmentation,
                ))
            sample["target_frames"] = targets
        assert_runtime_is_supervision_free(runtime_frames)
        return sample

    def _collate_runtime_frame(self, frames: Sequence[dict]) -> dict:
        batch_size, features, source_to_batch, source_slot = len(frames), [], [], []
        source_ids = torch.full((batch_size, self.max_cav), -1, dtype=torch.long)
        source_mask = torch.zeros((batch_size, self.max_cav), dtype=torch.bool)
        source_pose = torch.eye(4).repeat(batch_size, self.max_cav, 1, 1)
        source_to_receiver = torch.eye(4).repeat(batch_size, self.max_cav, 1, 1)
        capture_idx = torch.full((batch_size, self.max_cav), -1, dtype=torch.long)
        capture_time = torch.full((batch_size, self.max_cav), float("nan"))
        arrival_time, delta_t = torch.full_like(capture_time, float("nan")), torch.full_like(capture_time, float("nan"))
        delay_source = [["unavailable"] * self.max_cav for _ in frames]
        for batch_index, frame in enumerate(frames):
            for capture in frame["source_captures"]:
                slot = capture["source_slot"]
                features.append(capture["processed_lidar"]); source_to_batch.append(batch_index); source_slot.append(slot)
                source_ids[batch_index, slot], source_mask[batch_index, slot] = capture["source_id"], True
                source_pose[batch_index, slot] = torch.from_numpy(capture["source_pose"])
                source_to_receiver[batch_index, slot] = torch.from_numpy(capture["source_to_receiver"])
                capture_idx[batch_index, slot] = capture["capture_frame_idx"]
                capture_time[batch_index, slot], arrival_time[batch_index, slot], delta_t[batch_index, slot] = capture["capture_time_s"], capture["arrival_time_s"], capture["delta_t_s"]
                delay_source[batch_index][slot] = capture["delay_source"]
        if not features: raise RuntimeError("DualCMP batch contains no valid source voxels")
        runtime = {"scene_id": [frame["scene_id"] for frame in frames], "receiver_id": torch.tensor([frame["receiver_id"] for frame in frames]),
                   "receiver_frame_key": [frame["receiver_frame_key"] for frame in frames], "receiver_frame_idx": torch.tensor([frame["receiver_frame_idx"] for frame in frames]),
                   "receiver_time_s": torch.tensor([frame["receiver_time_s"] for frame in frames]), "receiver_pose": torch.from_numpy(np.stack([frame["receiver_pose"] for frame in frames])).float(),
                   "pose_convention": "v2v4real_lidar_to_world_homogeneous_4x4", "processed_lidar": self.pre_processor.collate_batch(features),
                   "source_to_batch": torch.tensor(source_to_batch), "source_slot": torch.tensor(source_slot), "record_len": source_mask.sum(dim=1).long(),
                   "source_ids": source_ids, "source_mask": source_mask, "source_pose": source_pose, "source_to_receiver": source_to_receiver,
                   "capture_frame_idx": capture_idx, "capture_time_s": capture_time, "arrival_time_s": arrival_time, "delta_t_s": delta_t, "delay_source": delay_source}
        assert_runtime_is_supervision_free(runtime)
        return runtime

    def _collate_target_frame(self, targets: Sequence[dict]) -> dict:
        result = {"fused_label_dict": self.post_processor.collate_batch([target["fused_label_dict"] for target in targets]),
                  "actor_ids": torch.from_numpy(np.stack([target["actor_ids"] for target in targets])).long(),
                  "current_boxes": torch.from_numpy(np.stack([target["current_boxes"] for target in targets])).float(),
                  "current_mask": torch.from_numpy(np.stack([target["current_mask"] for target in targets])).bool(),
                  "current_velocity": torch.from_numpy(np.stack([target["current_velocity"] for target in targets])).float(),
                  "velocity_mask": torch.from_numpy(np.stack([target["velocity_mask"] for target in targets])).bool()}
        if "future_xy" in targets[0]:
            result["future_xy"] = torch.from_numpy(np.stack([target["future_xy"] for target in targets])).float()
            result["future_mask"] = torch.from_numpy(np.stack([target["future_mask"] for target in targets])).bool()
        return result

    def collate_batch_train(self, batch: Sequence[dict]) -> dict:
        if not batch: raise ValueError("Cannot collate an empty batch")
        length = len(batch[0]["runtime_frames"])
        if any(len(sample["runtime_frames"]) != length or sample["stage"] != batch[0]["stage"] for sample in batch): raise ValueError("Batch clips must share stage and length")
        result = {"scene_receiver_keys": [sample["scene_receiver_key"] for sample in batch], "loss_start_index": batch[0]["loss_start_index"], "stage": batch[0]["stage"],
                  "runtime_frames": [self._collate_runtime_frame([sample["runtime_frames"][time] for sample in batch]) for time in range(length)]}
        if "target_frames" in batch[0]:
            target_length = len(batch[0]["target_frames"])
            if any(len(sample.get("target_frames", [])) != target_length for sample in batch): raise ValueError("Batch target clips must have equal length")
            result["target_frames"] = [self._collate_target_frame([sample["target_frames"][time] for sample in batch]) for time in range(target_length)]
        assert_runtime_is_supervision_free(result["runtime_frames"])
        return result

    def collate_batch_test(self, batch: Sequence[dict]) -> dict:
        return self.collate_batch_train(batch)
