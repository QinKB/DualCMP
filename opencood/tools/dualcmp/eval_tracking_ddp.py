"""Multi-GPU wrapper around the existing AB3DMOT tracking evaluator."""

from __future__ import annotations

import argparse
import json
import os
import traceback
import math
from pathlib import Path

import torch
import torch.distributed as dist

from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import (
    DualCMPSequenceDataset,
)
from opencood.tools.dualcmp.ab3dmot_metrics import (
    compute_ab3dmot_metrics,
    distributed_ab3dmot_metrics,
)
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
    METRIC_THRESHOLDS,
    WINDOW,
    WINDOW_FRAMES,
    checkpoint_detector_compression,
)
from opencood.tools.dualcmp.temporal_protocol import (
    checkpoint_forecast_contract, validate_evaluation_delay_contract,
)
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES, load_checkpoint_hypes


def _module_modes(model):
    return [(module, module.training) for module in model.modules()]


def _restore_module_modes(modes):
    for module, training in modes:
        module.train(training)
    if any(module.training != training for module, training in modes):
        raise RuntimeError("tracking evaluator failed to restore module training modes")


def evaluate_tracking_model_ddp(model, hypes, device, root, labels,
                                partner_delay_ms=0.0):
    """Evaluate the current DDP model on the frozen V2V4Real *test* window."""
    if not dist.is_initialized():
        raise RuntimeError("tracking evaluation requires initialized DDP")
    if not root or not labels:
        raise EnvironmentError("V2V4REAL_ROOT and DUALCMP_TRACKING_LABEL_ROOT are required")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    modes = _module_modes(model)
    model.eval()
    entries = WINDOW[rank::world_size]
    h = dict(hypes)
    h["root_dir"] = str(Path(root) / "test")
    h["validate_dir"] = h["root_dir"]
    h["fusion"] = {"core_method": "DualCMPSequenceDataset", "args": {
        "stage": "tracking", "tracking_frames": 3, "runtime_only": True,
        "runtime_forbid_gt": True,
        "receiver_ids": [entry["receiver_id"] for entry in entries],
        "frame_period_s": 0.1, "max_actors": 128, "max_cav": 2,
        "partner_delay_ms": float(partner_delay_ms),
        "tracking_label_root": labels,
    }}
    local = {}
    if entries:
        dataset = DualCMPSequenceDataset(h, train=False)
    for entry in entries:
        keys = [
            key
            for key in dataset.index.receiver_frame_keys(
                entry["scene_id"], str(entry["receiver_id"])
            )
            if entry["start_frame"] <= int(key) <= entry["end_frame"]
        ]
        if len(keys) != WINDOW_FRAMES:
            raise RuntimeError("fixed tracking window must contain 24 frames")
        online = OnlineDualCMP(
            model.module if hasattr(model, "module") else model, device
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
        local[(entry["scene_id"], entry["receiver_id"])] = frames
    gathered = [None] * world_size
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
            "metrics": metrics,
            "audit": {"numerator": lifecycle, "rate": rates,
                      "pass": passed, "overall_pass": all(passed.values())},
            "partner_delay_ms": float(partner_delay_ms),
        }
    dist.barrier()
    _restore_module_modes(modes)
    return report if rank == 0 else None


def evaluate_tracking_test_model_ddp(model, hypes, device, root, labels,
                                     partner_delay_ms=0.0):
    """Run strict-online tracking over every V2V4Real test timeline once."""
    if not dist.is_initialized():
        raise RuntimeError("tracking evaluation requires initialized DDP")
    if not root or not labels:
        raise EnvironmentError("V2V4REAL_ROOT and DUALCMP_TRACKING_LABEL_ROOT are required")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    modes = _module_modes(model)
    model.eval()
    h = dict(hypes)
    h["root_dir"] = str(Path(root) / "test")
    h["validate_dir"] = h["root_dir"]
    h["fusion"] = {"core_method": "DualCMPSequenceDataset", "args": {
        "stage": "tracking", "tracking_frames": 3, "runtime_only": True,
        "runtime_forbid_gt": True, "receiver_ids": [0, 1],
        "frame_period_s": 0.1, "max_actors": 128, "max_cav": 2,
        "partner_delay_ms": float(partner_delay_ms),
        "tracking_label_root": labels, "split_name": "test",
    }}
    dataset = DualCMPSequenceDataset(h, train=False)
    candidates = [
        (scene_id, int(receiver_id))
        for scene_id in dataset.index.scene_ids()
        for receiver_id in dataset.index.source_ids(scene_id)
        if str(receiver_id) in dataset.receiver_ids
    ]
    timelines = []
    for scene_id, receiver_id in candidates:
        label = Path(labels) / (str(scene_id) + "-" + str(receiver_id) + ".txt")
        if label.is_file():
            timelines.append((scene_id, receiver_id))
    if len(timelines) != 18:
        raise RuntimeError("complete tracking test requires exactly 18 labeled timelines, found %d" % len(timelines))
    local = {}
    for timeline_index, (scene_id, receiver_id) in enumerate(timelines):
        if timeline_index % world_size != rank:
            continue
        online = OnlineDualCMP(model.module if hasattr(model, "module") else model, device)
        frames = []
        with torch.no_grad():
            for key in dataset.index.receiver_frame_keys(scene_id, str(receiver_id)):
                runtime = dataset._runtime_frame(scene_id, str(receiver_id), key)
                prediction = online.step(dataset._collate_runtime_frame([runtime]))
                reference_pose = dataset._pose(dataset.index.receiver_record(scene_id, "0", key))
                frames.append(_prediction_in_reference(prediction, runtime["receiver_pose"], reference_pose))
        local[(scene_id, receiver_id)] = frames
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local)
    status = None
    if rank == 0:
        try:
            merged = {}
            for group in gathered:
                overlap = set(merged).intersection(group)
                if overlap:
                    raise RuntimeError("tracking timelines overlap across ranks: %s" % sorted(overlap))
                merged.update(group)
            status = {"metrics": compute_ab3dmot_metrics(merged, repository_root=".",
                ground_truth_root=labels), "error": None}
        except Exception:
            status = {"metrics": None, "error": traceback.format_exc()}
    payload = [status]
    dist.broadcast_object_list(payload, src=0)
    if payload[0]["error"]:
        raise RuntimeError("complete tracking test failed:\n%s" % payload[0]["error"])
    metrics = payload[0]["metrics"]
    counts = torch.tensor([len(local), sum(len(value) for value in local.values())],
                          device=device, dtype=torch.long)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    _restore_module_modes(modes)
    return {"metrics": metrics, "timeline_count": int(counts[0]),
            "frame_count": int(counts[1]), "ground_truth_root": str(Path(labels).resolve()),
            "partner_delay_ms": float(partner_delay_ms)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--partner-delay-ms", type=float, default=0.0)
    args = parser.parse_args()
    if not math.isfinite(args.partner_delay_ms) or args.partner_delay_ms < 0:
        raise ValueError("partner delay must be finite and non-negative")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("tracking evaluation requires torchrun")
    root = os.environ.get("V2V4REAL_ROOT")
    labels = os.environ.get("DUALCMP_TRACKING_LABEL_ROOT")
    if not root or not labels:
        raise EnvironmentError("V2V4REAL_ROOT and DUALCMP_TRACKING_LABEL_ROOT are required")
    payload = torch.load(args.checkpoint, map_location="cpu")
    state = payload.get("model", payload.get("model_state_dict", payload))
    has_forecast = any(
        (key[7:] if key.startswith("module.") else key).startswith("forecast_head.")
        for key in state
    )
    if has_forecast:
        ablation = checkpoint_forecast_contract(payload.get("provenance"))
        validate_evaluation_delay_contract(
            payload.get("provenance"), args.partner_delay_ms,
            forecast_representation=ablation["forecast_representation"],
            capture_alignment=ablation["capture_alignment"],
            age_encoding=ablation["age_encoding"],
            mean_source_aggregation=ablation["mean_source_aggregation"],
            context="tracking evaluation",
        )
    dist.init_process_group("nccl")
    rank, local_rank = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    h, _ = load_checkpoint_hypes(DEFAULT_HYPES, payload)
    h["model"]["args"]["compression"] = checkpoint_detector_compression(payload)
    forecast_args = None
    if any((key[7:] if key.startswith("module.") else key).startswith("forecast_head.") for key in state):
        forecast_args = {"history_frames": 11, "future_frames": 50, "modes": 6,
                         "intention_points_file": "preprocessed_data/v2v4real/v2v4real_cluster_64_center_dict.pkl",
                         "representation": ablation["forecast_representation"],
                         "capture_alignment": ablation["capture_alignment"],
                         "age_encoding": ablation["age_encoding"],
                         "mean_source_aggregation": (
                             ablation["mean_source_aggregation"])}
    model = _model_from_checkpoint(h, args.checkpoint, device, forecast_args=forecast_args,
                                   allow_dirty_smoke=True)
    evaluator = evaluate_tracking_test_model_ddp if args.full_test else evaluate_tracking_model_ddp
    report = evaluator(
        model, h, device, root, labels,
        partner_delay_ms=args.partner_delay_ms)
    if rank == 0:
        report["provenance"] = _checkpoint_provenance(args.checkpoint, allow_dirty_smoke=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
