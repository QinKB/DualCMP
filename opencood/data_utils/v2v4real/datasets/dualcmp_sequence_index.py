"""Receiver-native causal index for V2V4Real DualCMP clips."""

from __future__ import annotations

import json
import math
import os
import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


_IGNORED_YAML_MARKERS = ("additional", "camera_gt")


@dataclass(frozen=True)
class SourceFrameRecord:
    scene_id: str
    source_id: str
    frame_key: str
    frame_index: int
    capture_time_s: float
    yaml_path: Path
    lidar_path: Path


class SceneSequenceIndex:
    """Index source-native frames without parsing point clouds or labels."""

    def __init__(self, root_dir: str, frame_period_s: float = 0.1,
                 timeline_tolerance_s: float = 1e-6,
                 split_manifest: Optional[str] = None,
                 split_name: Optional[str] = None) -> None:
        expanded = os.path.expandvars(root_dir)
        missing = sorted({match.group(1) or match.group(2) for match in re.finditer(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", expanded)})
        if missing:
            raise RuntimeError("Dataset root contains unset environment variable(s): " + ", ".join(missing))
        self.root_dir = Path(expanded).expanduser().resolve()
        self.frame_period_s = float(frame_period_s)
        self.timeline_tolerance_s = float(timeline_tolerance_s)
        if (not math.isfinite(self.frame_period_s) or self.frame_period_s <= 0
                or not math.isfinite(self.timeline_tolerance_s) or self.timeline_tolerance_s < 0):
            raise ValueError("frame_period_s must be finite and positive and timeline_tolerance_s finite and non-negative")
        if not self.root_dir.is_dir():
            raise FileNotFoundError("V2V4Real root does not exist: %s" % self.root_dir)
        self._records: Dict[str, Dict[str, Dict[str, SourceFrameRecord]]] = {}
        self._frame_keys: Dict[str, Dict[str, Tuple[str, ...]]] = {}
        self._incomplete_files: List[Tuple[str, str, str, str]] = []
        self._sequence_boundaries = {}
        self._build(self._load_allowed_scenes(split_manifest, split_name))
        if not self._records:
            raise ValueError("No complete V2V4Real scenes found under %s" % self.root_dir)

    def _load_allowed_scenes(self, manifest_path: Optional[str], split_name: Optional[str]) -> Optional[set]:
        if manifest_path is None:
            if split_name is not None:
                raise ValueError("split_name requires split_manifest")
            return None
        if split_name is None:
            raise ValueError("split_manifest requires split_name")
        with Path(manifest_path).expanduser().resolve().open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        try:
            entries = manifest["splits"][split_name]
        except KeyError as exc:
            raise KeyError("Split %r is absent from %s" % (split_name, manifest_path)) from exc
        allowed = set()
        for entry in entries:
            if isinstance(entry, str):
                allowed.add(entry)
            elif isinstance(entry, Mapping) and "scene_id" in entry:
                source_root = entry.get("source_root")
                if source_root is None or Path(source_root).expanduser().resolve() == self.root_dir:
                    allowed.add(str(entry["scene_id"]))
            else:
                raise ValueError("Invalid split entry: %r" % (entry,))
        return allowed

    @staticmethod
    def _ordered_keys(keys: Iterable[str]) -> Tuple[str, ...]:
        unique = tuple(set(keys))
        try:
            return tuple(sorted(unique, key=int))
        except ValueError:
            return tuple(sorted(unique))

    @classmethod
    def _clock_indices(cls, keys: Iterable[str]) -> Dict[str, int]:
        ordered = cls._ordered_keys(keys)
        try:
            numeric = {key: int(key) for key in ordered}
        except ValueError:
            return {key: index for index, key in enumerate(ordered)}
        origin = min(numeric.values())
        return {key: value - origin for key, value in numeric.items()}

    def _build(self, allowed_scenes: Optional[set]) -> None:
        for scene_dir in sorted(path for path in self.root_dir.iterdir() if path.is_dir()):
            if allowed_scenes is not None and scene_dir.name not in allowed_scenes:
                continue
            raw, all_keys = {}, []
            source_dirs = sorted((path for path in scene_dir.iterdir() if path.is_dir()),
                                 key=lambda path: (int(path.name) if path.name.lstrip("-").isdigit() else 10**9, path.name))
            for source_dir in source_dirs:
                frames = {}
                for yaml_path in sorted(source_dir.glob("*.yaml")):
                    if any(marker in yaml_path.name for marker in _IGNORED_YAML_MARKERS):
                        continue
                    lidar_path = source_dir / (yaml_path.stem + ".pcd")
                    if not lidar_path.is_file():
                        self._incomplete_files.append((scene_dir.name, source_dir.name, yaml_path.stem, "missing_lidar"))
                        continue
                    frames[yaml_path.stem] = (yaml_path, lidar_path)
                    all_keys.append(yaml_path.stem)
                if frames:
                    raw[source_dir.name] = frames
            if not raw:
                continue
            clock = self._clock_indices(all_keys)
            self._records[scene_dir.name] = {}
            self._frame_keys[scene_dir.name] = {}
            for source_id, frames in raw.items():
                keys = self._ordered_keys(frames)
                self._frame_keys[scene_dir.name][source_id] = keys
                self._records[scene_dir.name][source_id] = {
                    key: SourceFrameRecord(scene_dir.name, source_id, key, clock[key],
                                           clock[key] * self.frame_period_s,
                                           frames[key][0], frames[key][1]) for key in keys}
            for receiver_id, keys in self._frame_keys[scene_dir.name].items():
                self._validate_receiver_timeline(scene_dir.name, receiver_id, keys)

    def _validate_receiver_timeline(self, scene_id: str, receiver_id: str,
                                    keys: Tuple[str, ...]) -> None:
        """Reject gaps/non-monotonic clocks instead of fabricating clip time."""
        records = self._records[scene_id][receiver_id]
        for previous_key, key in zip(keys, keys[1:]):
            previous, current = records[previous_key], records[key]
            frame_delta = current.frame_index - previous.frame_index
            time_delta = current.capture_time_s - previous.capture_time_s
            if frame_delta <= 0 or time_delta <= 0:
                raise ValueError("Non-increasing receiver timeline for %s/%s" % (scene_id, receiver_id))
            if abs(time_delta - self.frame_period_s) > self.timeline_tolerance_s:
                raise ValueError(
                    "Receiver timeline interval %.9fs for %s/%s does not match %.9fs +/- %.9fs"
                    % (time_delta, scene_id, receiver_id, self.frame_period_s,
                       self.timeline_tolerance_s)
                )

    def scene_ids(self) -> Tuple[str, ...]: return tuple(sorted(self._records))
    def source_ids(self, scene_id: str) -> Tuple[str, ...]: return tuple(self._records[scene_id])
    def receiver_frame_keys(self, scene_id: str, receiver_id: str) -> Tuple[str, ...]: return self._frame_keys[scene_id][str(receiver_id)]
    def source_record(self, scene_id: str, source_id: str, frame_key: str) -> Optional[SourceFrameRecord]: return self._records[scene_id][str(source_id)].get(str(frame_key))

    def set_sequence_boundaries(self, boundaries):
        """Opt-in acquisition segments; legacy indices have no boundaries."""
        self._sequence_boundaries = {scene: tuple(sorted(set(frames)))
                                     for scene, frames in boundaries.items()}

    def segment(self, scene_id, frame_index):
        return bisect_right(self._sequence_boundaries.get(scene_id, ()), frame_index)

    def is_segment_start(self, scene_id, frame_index):
        return frame_index in self._sequence_boundaries.get(scene_id, ())
    def receiver_record(self, scene_id: str, receiver_id: str, frame_key: str) -> SourceFrameRecord:
        record = self.source_record(scene_id, receiver_id, frame_key)
        if record is None: raise KeyError((scene_id, receiver_id, frame_key))
        return record

    def latest_arrived_source_record(self, scene_id: str, receiver_id: str, source_id: str,
                                     receiver_frame_key: str, delay_s: float) -> Optional[SourceFrameRecord]:
        if not math.isfinite(delay_s) or delay_s < 0:
            raise ValueError("delay_s must be finite and non-negative")
        receiver = self.receiver_record(scene_id, receiver_id, receiver_frame_key)
        candidates = (self._records[scene_id][str(source_id)][key] for key in self._frame_keys[scene_id][str(source_id)])
        arrived = [record for record in candidates
                   if record.capture_time_s + delay_s <= receiver.capture_time_s + 1e-9
                   and self.segment(scene_id, record.frame_index) == self.segment(scene_id, receiver.frame_index)]
        return max(arrived, key=lambda record: record.capture_time_s) if arrived else None

    @property
    def incomplete_files(self) -> Tuple[Tuple[str, str, str, str], ...]: return tuple(self._incomplete_files)

    def summary(self) -> dict:
        return {"root_dir": str(self.root_dir), "clock_source": "synthetic_fixed_period",
                "frame_period_s": self.frame_period_s,
                "timeline_tolerance_s": self.timeline_tolerance_s,
                "scene_count": len(self._records),
                "source_count": sum(map(len, self._records.values())),
                "complete_frame_count": sum(len(frames) for sources in self._records.values() for frames in sources.values()),
                "incomplete_file_count": len(self._incomplete_files)}
