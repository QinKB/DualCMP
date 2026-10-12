"""Read-only raw-pose continuity evidence, with no data filtering or repair.

Distance bins describe the observed data. They are NOT approved validity
thresholds, a metric change, or permission to exclude samples.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import _load_yaml_cached
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_index import SceneSequenceIndex


def audit_source(records):
    poses = []
    digest = hashlib.sha256()
    for record in records:
        data = record.yaml_path.read_bytes()
        digest.update(record.frame_key.encode() + b"\0" + data)
        pose = np.asarray(_load_yaml_cached(str(record.yaml_path))["lidar_pose"], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("non-finite/non-4x4 raw pose: %s" % record.yaml_path)
        poses.append(pose)
    poses = np.stack(poses)
    translation = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=-1)
    delta = np.diff([record.capture_time_s for record in records])
    yaw = np.arctan2(poses[:, 1, 0], poses[:, 0, 0])
    yaw_delta = np.arctan2(np.sin(np.diff(yaw)), np.cos(np.diff(yaw)))
    edges = []
    for index in np.argsort(translation)[-10:][::-1]:
        edges.append({"previous_frame": records[index].frame_key,
                      "frame": records[index + 1].frame_key,
                      "synthetic_interval_s": float(delta[index]),
                      "translation_m": float(translation[index]),
                      "implied_speed_mps": float(translation[index] / delta[index]),
                      "yaw_change_rad": float(yaw_delta[index])})
    bins = {}
    # A forecasting clip consumes 11 history + 50 future frames, i.e. 60
    # consecutive edges. Count crossing clips without excluding any of them.
    windows = np.lib.stride_tricks.sliding_window_view(translation, 60) if len(translation) >= 60 else np.zeros((0, 60))
    for boundary in (5., 10., 25., 50.):
        bins[str(boundary)] = {"edges": int((translation > boundary).sum()),
                              "61_frame_clips_crossing": int((windows > boundary).any(-1).sum())}
    return {"scene": records[0].scene_id, "source": records[0].source_id,
            "frames": len(records), "pose_yaml_sha256": digest.hexdigest(),
            "distance_quantiles_m": dict(zip(("p50", "p95", "p99", "max"),
                                             np.quantile(translation, [.5, .95, .99, 1.]).tolist())),
            "distance_bins_m": bins, "largest_edges": edges,
            "forecast_clips": len(windows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite pose-audit evidence")
    splits = {}
    for split in ("train", "test"):
        index = SceneSequenceIndex(str(Path(args.root) / split))
        groups = [[index.source_record(scene, source, key)
                   for key in index.receiver_frame_keys(scene, source)]
                  for scene in index.scene_ids() for source in index.source_ids(scene)]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            sources = list(executor.map(audit_source, groups))
        splits[split] = {"index": index.summary(), "sources": sources,
                         "forecast_clips": sum(source["forecast_clips"] for source in sources),
                         "distance_bins_m": {
                             str(boundary): {key: sum(source["distance_bins_m"][str(boundary)][key]
                                                    for source in sources)
                                             for key in ("edges", "61_frame_clips_crossing")}
                             for boundary in (5., 10., 25., 50.)}}
    report = {"scope": "all complete raw train/test CAV pose YAMLs; no sample filtering",
              "clock_caveat": "frame filenames define a synthetic 0.1s clock; no hardware timestamps were recovered",
              "bins_caveat": "descriptive distance bins, not approved validity/exclusion thresholds",
              "splits": splits, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({split: {key: value for key, value in data.items() if key not in ("sources", "index")}
                      for split, data in splits.items()}))


if __name__ == "__main__":
    main()
