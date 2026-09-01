"""Two-GPU wrapper around the existing fixed AB3DMOT tracking evaluator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import (
    DualCMPSequenceDataset,
)
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools.dualcmp.ab3dmot_metrics import compute_ab3dmot_metrics
from opencood.tools.dualcmp.online import (
    OnlineDualCMP,
    _checkpoint_provenance,
    _gt_conditioned_audit,
    _model_from_checkpoint,
    _prediction_in_reference,
    _target_lifecycle_audit,
)
from opencood.tools.dualcmp.protocol import (
    AUDIT_THRESHOLDS,
    DETECTOR_COMPRESSION,
    METRIC_THRESHOLDS,
    WINDOW,
    WINDOW_FRAMES,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("tracking evaluation requires two torchrun processes")
    root = os.environ.get("V2V4REAL_ROOT")
    labels = os.environ.get("DUALCMP_TRACKING_LABEL_ROOT")
    if not root or not labels:
        raise EnvironmentError(
            "V2V4REAL_ROOT and DUALCMP_TRACKING_LABEL_ROOT are required"
        )
    dist.init_process_group("nccl")
    rank, local_rank = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    entry = WINDOW[rank]
    h = load_yaml(
        "opencood/hypes_yaml/point_pillar_cobevt_multiego_1x_v2v4real.yaml"
    )
    h["root_dir"] = str(Path(root) / "test")
    h["validate_dir"] = h["root_dir"]
    h["model"]["args"]["compression"] = DETECTOR_COMPRESSION
    h["fusion"] = {"core_method": "DualCMPSequenceDataset", "args": {
        "stage": "tracking", "tracking_frames": 3, "runtime_only": True,
        "runtime_forbid_gt": True, "receiver_ids": [entry["receiver_id"]],
        "frame_period_s": 0.1, "max_actors": 128, "max_cav": 2,
        "tracking_label_root": labels,
    }}
    dataset = DualCMPSequenceDataset(h, train=False)
    keys = [
        key
        for key in dataset.index.receiver_frame_keys(
            entry["scene_id"], str(entry["receiver_id"])
        )
        if entry["start_frame"] <= int(key) <= entry["end_frame"]
    ]
    if len(keys) != WINDOW_FRAMES:
        raise RuntimeError("fixed tracking window must contain 24 frames")
    payload = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_state = payload.get(
        "model", payload.get("model_state_dict", payload)
    )
    has_forecast_head = any(
        (key[7:] if key.startswith("module.") else key).startswith(
            "forecast_head."
        )
        for key in checkpoint_state
    )
    forecast_args = None
    if has_forecast_head:
        forecast_args = {
            "history_frames": 11, "future_frames": 50, "modes": 6,
            "intention_points_file":
                "preprocessed_data/v2v4real/"
                "v2v4real_cluster_64_center_dict.pkl",
        }
    online = OnlineDualCMP(
        _model_from_checkpoint(
            h, args.checkpoint, device,
            forecast_args=forecast_args,
            allow_dirty_smoke=True,
        ),
        device,
    )
    frames = []
    for key in keys:
        runtime = dataset._runtime_frame(
            entry["scene_id"], str(entry["receiver_id"]), key
        )
        prediction = online.step(dataset._collate_runtime_frame([runtime]))
        reference_pose = dataset._pose(
            dataset.index.receiver_record(entry["scene_id"], "0", key)
        )
        frames.append(
            _prediction_in_reference(
                prediction, runtime["receiver_pose"], reference_pose
            )
        )
    local = {(entry["scene_id"], entry["receiver_id"]): frames}
    gathered = [None, None]
    dist.all_gather_object(gathered, local)
    if rank == 0:
        predictions = {}
        for group in gathered:
            predictions.update(group)
        lifecycle = _target_lifecycle_audit(h, predictions, ".")
        duplicate = _gt_conditioned_audit(".", predictions)
        duplicate_count, duplicate_denominator = map(
            int, duplicate["duplicate"].split("/")
        )
        lifecycle.update({"duplicate": duplicate_count,
                          "duplicate_denominator": duplicate_denominator})
        rates = {
            "birth": lifecycle["birth_tp"] / lifecycle["birth"],
            "keep": lifecycle["keep_tp"] / lifecycle["keep"],
            "delete": lifecycle["delete_tp"] / lifecycle["delete"],
        }
        passed = {
            "birth": lifecycle["birth_tp"] >= AUDIT_THRESHOLDS["birth_tp"],
            "keep": lifecycle["keep_tp"] >= AUDIT_THRESHOLDS["keep_same_runtime"],
            "delete": lifecycle["delete_tp"] >= AUDIT_THRESHOLDS["delete_tp"],
            "duplicate": duplicate_count <= AUDIT_THRESHOLDS["duplicate_max"],
        }
        metrics = compute_ab3dmot_metrics(
            predictions, repository_root=".", ground_truth_root=labels
        )
        metric_pass = {
            key: metrics[key] >= threshold
            for key, threshold in METRIC_THRESHOLDS.items()
            if key != "identity_switches_max"
        }
        metric_pass["identity_switches"] = (
            metrics["identity_switches"]
            <= METRIC_THRESHOLDS["identity_switches_max"]
        )
        passed.update(metric_pass)
        report = {
            "provenance": _checkpoint_provenance(
                args.checkpoint, allow_dirty_smoke=True
            ),
            "metrics": metrics,
            "audit": {"numerator": lifecycle, "rate": rates,
                      "pass": passed, "overall_pass": all(passed.values())},
        }
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
