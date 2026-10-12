"""Build train-only MTR endpoint clusters for the explicit aligned protocol.

Uses MTR's clustered intention interface and sklearn KMeans, as in upstream
motion-prior preparation. Does not use test labels or change legacy clusters.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np
from sklearn.cluster import KMeans

from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES, load_checkpoint_hypes
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--exclude-scenes-manifest",
                        help="opt-in research split-overlap audit; legacy preparation is unchanged")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite an intention artifact")
    payload = torch.load(args.tracker_checkpoint, map_location="cpu")
    hypes, _ = load_checkpoint_hypes(DEFAULT_HYPES, payload)
    root = str(Path(os.environ["V2V4REAL_ROOT"]) / "train")
    hypes.update(root_dir=root, validate_dir=root)
    hypes["fusion"] = {"args": {
        "stage": "forecast", "forecast_history_frames": 10, "future_frames": 50,
        "runtime_only": False, "runtime_forbid_gt": True, "split_name": "train",
        "receiver_ids": [0, 1], "max_cav": 2, "max_actors": 128,
        "tracking_label_root": os.environ["DUALCMP_TRAIN_TRACKING_LABEL_ROOT"],
        "tracking_label_layout": "dmstrack_train", "shuffle_training_points": False,
        "forecast_coordinate_frame": "current_receiver",
    }}
    dataset = DualCMPSequenceDataset(hypes, train=True)
    excluded = set()
    if args.exclude_scenes_manifest:
        excluded = set(json.loads(Path(args.exclude_scenes_manifest).read_text())["shared_scene_names"])
    eligible = [index for index, clip in enumerate(dataset.clip_records) if clip.scene_id not in excluded]
    if not eligible:
        raise ValueError("no train clips remain after explicit scene exclusion")
    indices = np.asarray(eligible)[np.linspace(0, len(eligible) - 1, min(args.samples, len(eligible)), dtype=int)]
    endpoints = []
    selected_keys = []
    for index in indices:
        clip = dataset.clip_records[int(index)]
        keys = dataset.index.receiver_frame_keys(clip.scene_id, clip.receiver_id)
        if clip.start_position + 60 >= len(keys):
            continue  # No five-second endpoint: never cluster a shorter horizon.
        frame = keys[clip.start_position + 10]
        future_frame = keys[clip.start_position + 60]
        pose = dataset._pose(dataset.index.receiver_record(clip.scene_id, clip.receiver_id, frame))
        current = dataset._actors(clip.scene_id, clip.receiver_id, frame, pose)
        future = dataset._actors(clip.scene_id, clip.receiver_id, future_frame, pose, range_crop=False)
        selected_keys.append((clip.scene_id, clip.receiver_id, frame))
        for actor in sorted(set(current) & set(future)):
            delta = future[actor][:2] - current[actor][:2]
            yaw = current[actor][6]
            endpoints.append([np.cos(yaw) * delta[0] + np.sin(yaw) * delta[1],
                              -np.sin(yaw) * delta[0] + np.cos(yaw) * delta[1]])
    endpoints = np.asarray(endpoints, dtype=np.float32)
    if len(endpoints) < 64 or len(np.unique(endpoints, axis=0)) < 64 or not np.isfinite(endpoints).all():
        raise RuntimeError("insufficient finite distinct train endpoints for 64 clusters")
    clusters = KMeans(n_clusters=64, random_state=20260930, n_init=10).fit(endpoints).cluster_centers_.astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = pickle.dumps({"TYPE_VEHICLE": clusters}, protocol=4)
    output.write_bytes(data)
    metadata = {"split": "train", "coordinate_frame": "current_receiver", "heading_unit": "radians",
                "clusters": 64, "endpoints": len(endpoints), "samples": len(indices),
                "selection": "evenly spaced train clip indices, full five-second endpoint presence required",
                "selected_keys": selected_keys, "sha256": hashlib.sha256(data).hexdigest(),
                "tracker_checkpoint": args.tracker_checkpoint,
                "label_root": os.environ["DUALCMP_TRAIN_TRACKING_LABEL_ROOT"],
                "seed": 20260930, "n_init": 10}
    metadata["sequence_policy"] = dataset.sequence_policy
    if args.exclude_scenes_manifest:
        metadata.update(excluded_scenes=sorted(excluded),
                        eligible_train_clips=len(eligible),
                        excluded_scenes_manifest=args.exclude_scenes_manifest,
                        excluded_scenes_manifest_sha256=hashlib.sha256(Path(args.exclude_scenes_manifest).read_bytes()).hexdigest())
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metadata.items() if key != "selected_keys"}))


if __name__ == "__main__":
    main()
