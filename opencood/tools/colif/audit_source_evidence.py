"""Audit whether V2V4Real labels have source-local LiDAR point support.

The output is evidence for choosing a local-positive rule.  It is not a
runtime cache: GT labels and canonical IDs are read only inside this audit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from opencood.data_utils.v2v4real.datasets.colif_sequence_index import (
    SceneSequenceIndex,
)
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils import box_utils_v2v4real, pcd_utils


DEFAULT_LIDAR_RANGE = (-102.4, -38.4, -5.0, 102.4, 38.4, 3.0)


def classify_receiver_support(receiver_id: str, supported_sources: Set[str]) -> str:
    """Classify source support without using detector/tracker output."""

    if not supported_sources:
        return "unsupported"
    receiver_supported = str(receiver_id) in supported_sources
    other_supported = bool(supported_sources - {str(receiver_id)})
    if receiver_supported and other_supported:
        return "shared"
    if receiver_supported:
        return "receiver_only"
    return "partner_only"


def uniformly_sample_keys(frame_keys: Sequence[str], maximum: Optional[int]) -> Tuple[str, ...]:
    """Deterministically cover the whole sequence when an audit is capped."""

    keys = tuple(frame_keys)
    if maximum is None or len(keys) <= maximum:
        return keys
    if maximum < 1:
        raise ValueError("max_frames_per_source must be at least one")
    if maximum == 1:
        return (keys[len(keys) // 2],)
    indices = [round(index * (len(keys) - 1) / (maximum - 1))
               for index in range(maximum)]
    return tuple(keys[index] for index in indices)


def _canonical_id(
    scene_id: str,
    source_id: str,
    object_id,
    object_content: Mapping,
    canonical_actor_map: Optional[Mapping[Tuple[str, str, str], int]] = None,
) -> Optional[int]:
    if canonical_actor_map is not None:
        return canonical_actor_map.get(
            (str(scene_id), str(source_id), str(object_id))
        )
    association_id = object_content.get("ass_id", -1)
    try:
        association_id = int(association_id)
    except (TypeError, ValueError):
        return None
    return association_id if association_id >= 0 else None


def _point_support_by_object(
    scene_id: str,
    source_id: str,
    yaml_params: Mapping,
    lidar_points: np.ndarray,
    lidar_range: Sequence[float],
    canonical_actor_map: Optional[Mapping[Tuple[str, str, str], int]] = None,
) -> Dict[object, dict]:
    vehicles = yaml_params.get("vehicles", {})
    projected = {}
    box_utils_v2v4real.project_world_objects(
        vehicles,
        projected,
        # V2V4Real YAML object locations are already expressed in this
        # source LiDAR frame.  Applying lidar_pose here would move the boxes
        # into world coordinates while the points remain source-native.
        np.eye(4, dtype=np.float64),
        list(lidar_range),
        order="hwl",
    )
    result = {}
    for object_id, projected_content in projected.items():
        box = np.asarray(projected_content["coord"], dtype=np.float32)
        corners = box_utils_v2v4real.boxes_to_corners_3d(box.copy(), order="hwl")[0]
        supported_points = box_utils_v2v4real.get_points_in_rotated_box_3d(
            lidar_points[:, :3], corners
        )
        raw_content = vehicles[object_id]
        result[object_id] = {
            "canonical_id": _canonical_id(
                scene_id,
                source_id,
                object_id,
                raw_content,
                canonical_actor_map,
            ),
            "point_count": int(supported_points.shape[0]),
        }
    return result


def audit_source_evidence(
    index: SceneSequenceIndex,
    min_points: int = 3,
    stride: int = 1,
    max_frames_per_source: Optional[int] = None,
    lidar_range: Sequence[float] = DEFAULT_LIDAR_RANGE,
    canonical_actor_map: Optional[Mapping[Tuple[str, str, str], int]] = None,
) -> dict:
    if min_points < 1:
        raise ValueError("min_points must be at least one")
    if stride < 1:
        raise ValueError("stride must be at least one")

    totals = Counter()
    point_histogram = Counter()
    per_source = defaultdict(Counter)
    frame_support: Dict[Tuple[str, str], Dict[str, Set[int]]] = defaultdict(dict)
    frame_canonical_labels: Dict[Tuple[str, str], Dict[str, Set[int]]] = defaultdict(dict)

    for scene_id in index.scene_ids():
        for source_id in index.source_ids(scene_id):
            frame_keys = index.receiver_frame_keys(scene_id, source_id)[::stride]
            frame_keys = uniformly_sample_keys(frame_keys, max_frames_per_source)
            for frame_key in frame_keys:
                record = index.receiver_record(scene_id, source_id, frame_key)
                yaml_params = load_yaml(str(record.yaml_path))
                lidar_points = pcd_utils.pcd_to_np(str(record.lidar_path))
                lidar_points = pcd_utils.mask_ego_points(lidar_points)
                support = _point_support_by_object(
                    scene_id,
                    source_id,
                    yaml_params,
                    lidar_points,
                    lidar_range,
                    canonical_actor_map,
                )

                totals["source_frames"] += 1
                totals["lidar_points"] += int(lidar_points.shape[0])
                supported_ids = set()
                canonical_label_ids = set()
                for object_support in support.values():
                    point_count = object_support["point_count"]
                    canonical_id = object_support["canonical_id"]
                    totals["projected_labeled_objects"] += 1
                    per_source[source_id]["projected_labeled_objects"] += 1
                    if canonical_id is None:
                        totals["objects_without_canonical_id"] += 1
                    else:
                        totals["objects_with_canonical_id"] += 1
                        canonical_label_ids.add(canonical_id)
                    if point_count == 0:
                        bucket = "0"
                    elif point_count <= 2:
                        bucket = "1-2"
                    elif point_count <= 4:
                        bucket = "3-4"
                    elif point_count <= 9:
                        bucket = "5-9"
                    else:
                        bucket = "10+"
                    point_histogram[bucket] += 1
                    if point_count >= min_points:
                        totals["supported_objects"] += 1
                        per_source[source_id]["supported_objects"] += 1
                        if canonical_id is not None:
                            supported_ids.add(canonical_id)
                frame_support[(scene_id, frame_key)][source_id] = supported_ids
                frame_canonical_labels[(scene_id, frame_key)][source_id] = canonical_label_ids

    receiver_state_counts = Counter()
    raw_receiver_state_counts = Counter()
    raw_source_multiplicity = Counter()
    supported_source_multiplicity = Counter()
    transition_counts = Counter()
    previous_state = {}
    for (scene_id, frame_key), by_source in sorted(frame_support.items()):
        all_actor_ids = set().union(*by_source.values()) if by_source else set()
        receivers = tuple(sorted(by_source))
        for receiver_id in receivers:
            for actor_id in all_actor_ids:
                supported_sources = {
                    source_id
                    for source_id, actor_ids in by_source.items()
                    if actor_id in actor_ids
                }
                state = classify_receiver_support(receiver_id, supported_sources)
                receiver_state_counts[state] += 1
                lifecycle_key = (scene_id, receiver_id, actor_id)
                if lifecycle_key in previous_state and previous_state[lifecycle_key] != state:
                    transition_counts[
                        previous_state[lifecycle_key] + "->" + state
                    ] += 1
                previous_state[lifecycle_key] = state

        for actor_id in all_actor_ids:
            supported_source_multiplicity[str(sum(
                actor_id in actor_ids for actor_ids in by_source.values()
            ))] += 1

    for _frame_id, by_source in sorted(frame_canonical_labels.items()):
        all_actor_ids = set().union(*by_source.values()) if by_source else set()
        for actor_id in all_actor_ids:
            labeled_sources = {
                source_id
                for source_id, actor_ids in by_source.items()
                if actor_id in actor_ids
            }
            raw_source_multiplicity[str(len(labeled_sources))] += 1
            for receiver_id in by_source:
                raw_receiver_state_counts[
                    classify_receiver_support(receiver_id, labeled_sources)
                ] += 1

    projected = totals["projected_labeled_objects"]
    supported = totals["supported_objects"]
    return {
        "index": index.summary(),
        "evidence_rule": {
            "coordinate_frame": "source_native_lidar",
            "minimum_points_in_3d_box": min_points,
            "lidar_range": [float(value) for value in lidar_range],
            "ego_points_removed": True,
        },
        "sampling": {
            "stride": stride,
            "max_frames_per_source": max_frames_per_source,
            "transition_counts_are_sampled": stride != 1 or max_frames_per_source is not None,
        },
        "totals": dict(sorted(totals.items())),
        "point_count_histogram": dict(sorted(point_histogram.items())),
        "per_source_id": {
            source_id: dict(sorted(counts.items()))
            for source_id, counts in sorted(per_source.items())
        },
        "raw_canonical_source_multiplicity": dict(sorted(raw_source_multiplicity.items())),
        "supported_canonical_source_multiplicity": dict(
            sorted(supported_source_multiplicity.items())
        ),
        "raw_receiver_label_states": dict(sorted(raw_receiver_state_counts.items())),
        "receiver_support_states": dict(sorted(receiver_state_counts.items())),
        "sampled_state_transitions": dict(sorted(transition_counts.items())),
        "supported_object_fraction": (supported / projected if projected else None),
        "interpretation_warning": (
            "Point support is a reproducible weak visibility proxy, not an occlusion label. "
            + (
                "Canonical identities use the supervision-only stable mapping from "
                "the sequence dataset."
                if canonical_actor_map is not None
                else "Canonical IDs with ass_id=-1 are excluded from cross-source state counts."
            )
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    parser.add_argument("--hypes-yaml")
    parser.add_argument("--manifest")
    parser.add_argument("--split")
    parser.add_argument("--frame-period-s", type=float, default=0.1)
    parser.add_argument("--min-points", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames-per-source", type=int)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    canonical_actor_map = None
    canonical_audit = None
    if args.hypes_yaml:
        from opencood.data_utils.v2v4real.datasets import build_v2v4real_dataset

        hypes = load_yaml(args.hypes_yaml)
        if args.root:
            hypes["validate_dir"] = args.root
            hypes["test_dir"] = args.root
        dataset = build_v2v4real_dataset(hypes, train=False)
        index = dataset.index
        canonical_actor_map = dataset.canonical_actor_map
        canonical_audit = dataset.canonical_audit
    else:
        if not args.root:
            raise ValueError("Either --root or --hypes-yaml is required")
        index = SceneSequenceIndex(
            args.root,
            frame_period_s=args.frame_period_s,
            split_manifest=args.manifest,
            split_name=args.split,
        )
    report = audit_source_evidence(
        index,
        min_points=args.min_points,
        stride=args.stride,
        max_frames_per_source=args.max_frames_per_source,
        canonical_actor_map=canonical_actor_map,
    )
    if canonical_audit is not None:
        report["canonical_identity_audit"] = canonical_audit
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({
        "output": str(output_path),
        "source_frames": report["totals"].get("source_frames", 0),
        "projected_labeled_objects": report["totals"].get(
            "projected_labeled_objects", 0
        ),
        "supported_object_fraction": report["supported_object_fraction"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
