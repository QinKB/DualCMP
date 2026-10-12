"""Verify new CMP IDs without changing native geometry or sample membership."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset, _load_yaml_cached
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_index import SceneSequenceIndex
from opencood.data_utils.v2v4real.datasets import GT_RANGE
from opencood.data_utils.v2v4real.datasets.dualcmp_target_builder import load_tracking_label_file
from opencood.tools.dualcmp.aligned_protocol import ALIGNED_FORECAST_WINDOW
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES, load_checkpoint_hypes
from opencood.utils import box_utils_v2v4real


def is_recovered_crop_alias(index, scene, ordinal, old_identity):
    """Prove native +100 dedup discarded a second, in-range physical car.

    Native dedup precedes the final all-corners crop. In a collided namespace,
    an out-of-range first-CAV box can also erase the in-range second-CAV box.
    """
    key = index.receiver_frame_keys(scene, "0")[ordinal]
    raw = [_load_yaml_cached(str(index.receiver_record(scene, str(source), key).yaml_path))
           for source in (0, 1)]
    partner_id = old_identity - 100
    if (old_identity not in raw[0].get("vehicles", {}) or
            partner_id not in raw[1].get("vehicles", {})):
        return False
    if any(int(raw[source]["vehicles"][identifier].get("ass_id", -1)) != -1
           for source, identifier in ((0, old_identity), (1, partner_id))):
        return False
    projected = {}
    box_utils_v2v4real.project_world_objects(
        {old_identity: raw[0]["vehicles"][old_identity]}, projected, np.eye(4), GT_RANGE, "hwl")
    if old_identity not in projected:
        return False
    corners = box_utils_v2v4real.boxes_to_corners_3d(projected[old_identity]["coord"].copy(), "hwl")
    return not bool(box_utils_v2v4real.get_mask_for_boxes_within_range_torch(
        torch.from_numpy(corners).float())[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--legacy-labels", required=True)
    parser.add_argument("--tracker-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite label audit evidence")
    manifest_path = Path(args.labels).parent / "aligned_labels_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    index = SceneSequenceIndex(args.root)
    extra_rows = mismatched_ids = 0
    explained_crop_aliases = []
    unexplained_mismatches = 0
    max_geometry_error = 0.
    for scene in manifest["scenes"]:
        fresh = load_tracking_label_file(str(Path(args.labels) / scene["file"]))
        legacy = load_tracking_label_file(str(Path(args.legacy_labels) / scene["file"]))
        for frame in range(scene["frames"]):
            restored = fresh.get(frame, {})
            original = legacy.get(frame, {})
            legacy_mapping = {}
            for identity, box in sorted(restored.items()):
                # All audited train raw/association IDs are <1000, CAVs=0/1.
                old_identity = identity - 900 if identity >= 1000 else identity
                legacy_mapping.setdefault(old_identity, box)
            removed = set(original) - set(legacy_mapping)
            added = set(legacy_mapping) - set(original)
            mismatched_ids += len(removed) + len(added)
            unexplained_mismatches += len(removed)
            for identity in added:
                if is_recovered_crop_alias(index, scene["scene"], frame, identity):
                    explained_crop_aliases.append({"scene": scene["scene"], "frame_ordinal": frame,
                                                   "legacy_aliased_id": identity})
                else:
                    unexplained_mismatches += 1
            extra_rows += len(restored) - len(original)
            for identity in set(legacy_mapping) & set(original):
                error = np.abs(legacy_mapping[identity] - original[identity])
                error[6] = abs(np.arctan2(np.sin(error[6]), np.cos(error[6])))
                max_geometry_error = max(max_geometry_error, float(error.max()))
    hypes, _ = load_checkpoint_hypes(DEFAULT_HYPES, torch.load(args.tracker_checkpoint, map_location="cpu"))
    hypes.update(root_dir=args.root, validate_dir=args.root)
    hypes["fusion"] = {"args": {
        "stage": "forecast", "forecast_history_frames": 10, "future_frames": 50,
        "runtime_only": False, "runtime_forbid_gt": True, "split_name": "train",
        "receiver_ids": [0, 1], "max_cav": 2, "max_actors": 128,
        "tracking_label_root": args.labels, "tracking_label_layout": "dmstrack_train",
        "forecast_coordinate_frame": "current_receiver", "shuffle_training_points": False,
        "window": list(ALIGNED_FORECAST_WINDOW),
    }}
    dataset = DualCMPSequenceDataset(hypes, train=True)
    window = []
    for clip in dataset.clip_records:
        keys = dataset.index.receiver_frame_keys(clip.scene_id, clip.receiver_id)
        position = clip.start_position + 10
        frame = keys[position]
        pose = dataset._pose(dataset.index.receiver_record(clip.scene_id, clip.receiver_id, frame))
        current = dataset._actors(clip.scene_id, clip.receiver_id, frame, pose)
        future = [dataset._actors(clip.scene_id, clip.receiver_id, key, pose, range_crop=False)
                  for key in keys[position + 1:position + 51]]
        actors = []
        for identity in sorted(current):
            if not all(identity in item for item in future):
                continue
            trajectory = np.stack([current[identity][:2]] + [item[identity][:2] for item in future])
            actors.append({"identity": identity,
                           "endpoint_displacement_m": float(np.linalg.norm(trajectory[-1] - trajectory[0])),
                           "max_step_m": float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).max())})
        window.append({"scene": clip.scene_id, "receiver": int(clip.receiver_id), "frame": int(frame),
                       "complete_actors": actors})
    report = {"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "sequence_policy": dataset.sequence_policy, "restored_rows": extra_rows,
              "legacy_mapping_id_mismatches": mismatched_ids,
              "explained_native_dedup_before_crop_aliases": explained_crop_aliases,
              "unexplained_id_mismatches": unexplained_mismatches,
              "max_native_geometry_error": max_geometry_error, "window": window,
              "complete_targets": sum(len(item["complete_actors"]) for item in window)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    assert unexplained_mismatches == 0 and max_geometry_error < 1e-3, report


if __name__ == "__main__":
    main()
