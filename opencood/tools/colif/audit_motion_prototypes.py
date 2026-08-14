"""Fit train-only endpoint residual prototypes without loading point clouds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from opencood.data_utils.v2v4real.datasets import build_v2v4real_dataset
import opencood.data_utils.v2v4real.datasets.colif_sequence_dataset as sequence_data
from opencood.hypes_yaml.yaml_utils import load_yaml


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypes-yaml", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=5)
    return parser.parse_args()


def _native_receiver_actors(dataset, scene_id, receiver_id, frame_key):
    receiver_record = dataset.index.receiver_record(
        scene_id, receiver_id, frame_key
    )
    receiver_pose = dataset._load_pose(receiver_record)
    merged = {}
    for source_id in dataset._source_ids_for_receiver(
        scene_id, receiver_id
    ):
        record = dataset.index.source_record(
            scene_id, source_id, frame_key
        )
        if record is None:
            continue
        params = sequence_data._load_yaml_cached(str(record.yaml_path))
        for actor_id, box in dataset._project_yaml_actors(
            scene_id, source_id, params, receiver_pose
        ).items():
            merged.setdefault(actor_id, box)
    return merged


def collect_local_cv_residuals(dataset, frame_stride: int) -> np.ndarray:
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    residuals = []
    for scene_id in dataset.index.scene_ids():
        receiver_id = "0"
        if receiver_id not in dataset.index.source_ids(scene_id):
            continue
        keys = dataset.index.receiver_frame_keys(scene_id, receiver_id)
        native = [
            _native_receiver_actors(
                dataset, scene_id, receiver_id, frame_key
            )
            for frame_key in keys
        ]
        for frame_index in range(
            1, len(keys) - dataset.future_frames, frame_stride
        ):
            current = native[frame_index]
            future = native[frame_index + dataset.future_frames]
            for actor_id in current.keys() & future.keys():
                history_indices = [
                    index
                    for index in range(max(0, frame_index - 10), frame_index + 1)
                    if actor_id in native[index]
                ]
                if len(history_indices) < 2:
                    continue
                pair_velocity = []
                for earlier_offset, earlier in enumerate(
                    history_indices[:-1]
                ):
                    for later in history_indices[earlier_offset + 1 :]:
                        elapsed = (later - earlier) * dataset.frame_period_s
                        pair_velocity.append(
                            (
                                native[later][actor_id][:2]
                                - native[earlier][actor_id][:2]
                            ) / elapsed
                        )
                velocity = np.median(
                    np.asarray(pair_velocity, dtype=np.float32), axis=0
                )
                goal = future[actor_id][:2] - current[actor_id][:2]
                heading = current[actor_id][6]
                speed = float(np.linalg.norm(velocity))
                direction = np.asarray(
                    [np.cos(heading), np.sin(heading)], dtype=np.float32
                )
                if speed > 0.5:
                    direction = velocity / speed
                perpendicular = np.asarray(
                    [-direction[1], direction[0]], dtype=np.float32
                )
                horizon_s = dataset.future_frames * dataset.frame_period_s
                receiver_residual = goal - velocity * horizon_s
                residuals.append(
                    np.asarray(
                        [[
                            float(receiver_residual @ direction),
                            float(receiver_residual @ perpendicular),
                        ]],
                        dtype=np.float32,
                    )
                )
    if not residuals:
        raise RuntimeError("No full-horizon training residuals were found")
    return np.concatenate(residuals, axis=0).astype(np.float32)


def main():
    args = _arguments()
    hypes = load_yaml(args.hypes_yaml)
    fusion = hypes["fusion"]["args"]
    fusion["runtime_only"] = False
    fusion["warmup_length"] = 0
    fusion["unroll_length"] = 1
    fusion["clip_stride"] = 1
    dataset = build_v2v4real_dataset(hypes, train=True)
    residuals = collect_local_cv_residuals(dataset, int(args.frame_stride))
    model = MiniBatchKMeans(
        n_clusters=int(args.clusters),
        random_state=int(args.seed),
        batch_size=4096,
        n_init=10,
        max_iter=300,
        reassignment_ratio=0.01,
    ).fit(residuals)
    centers = model.cluster_centers_
    order = np.argsort(np.arctan2(centers[:, 1], centers[:, 0]))
    centers = centers[order]
    nearest = np.linalg.norm(
        residuals[:, None, :] - centers[None, :, :], axis=-1
    ).min(axis=1)
    report = {
        "split": "train",
        "future_frames": dataset.future_frames,
        "frame_period_s": dataset.frame_period_s,
        "sample_count": int(residuals.shape[0]),
        "clusters": int(args.clusters),
        "seed": int(args.seed),
        "frame_stride": int(args.frame_stride),
        "coordinate_frame": (
            "velocity-aligned residual from native capture-time receiver "
            "history, matching forecast_history_box"
        ),
        "centers_longitudinal_lateral_m": centers.tolist(),
        "residual_median_m": np.median(residuals, axis=0).tolist(),
        "residual_p10_m": np.percentile(residuals, 10, axis=0).tolist(),
        "residual_p90_m": np.percentile(residuals, 90, axis=0).tolist(),
        "train_oracle_mean_endpoint_residual_m": float(nearest.mean()),
        "train_oracle_p90_endpoint_residual_m": float(
            np.percentile(nearest, 90)
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
