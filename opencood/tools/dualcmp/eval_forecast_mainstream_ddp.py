"""CMP-metric GT-free adapter; the historical UniAD evaluator stays explicit."""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from opencood.data_utils.dualcmp_dataloader import build_dataloader
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
from opencood.tools.dualcmp.aligned_protocol import ALIGNED_FORECAST_WINDOW
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES, load_checkpoint_hypes
from opencood.tools.dualcmp.eval_forecast_ddp import _rank_zero_evaluation
from opencood.tools.dualcmp.mainstream_forecast_metrics import forecast_frame, mainstream_report
from opencood.tools.dualcmp.cmp_adapted_forecast_metrics import (
    forecast_frame as cmp_forecast_frame, cmp_adapted_report, PRIMARY_METRICS,
)
from opencood.tools.dualcmp.online import _checkpoint_provenance, _model_from_checkpoint
from opencood.tools.dualcmp.protocol import FORECAST_WINDOW, checkpoint_detector_compression
from opencood.tools.dualcmp.temporal_protocol import checkpoint_forecast_contract, validate_evaluation_delay_contract
from opencood.tools.train_utils import to_device


def _evaluate_native_forecast_ddp(model, hypes, device, root, labels,
                                     batch_size=1, workers=2, full_test=False,
                                     partner_delay_ms=100.0, train_window=False,
                                     frame_exporter=forecast_frame,
                                     report_builder=mainstream_report):
    """Same native dataset and model forward; independently export all actors."""
    if not dist.is_initialized():
        raise RuntimeError("mainstream evaluation requires initialized DDP")
    implementation = model.module if hasattr(model, "module") else model
    if implementation.forecast_head.forecast_coordinate_frame != "current_receiver":
        raise ValueError("use the historical evaluator for legacy_capture checkpoints")
    split = "train" if train_window else "test"
    h = dict(hypes)
    h["root_dir"] = h["validate_dir"] = str(Path(root) / split)
    h["fusion"] = {"args": {
        "stage": "forecast", "tracking_frames": 3, "forecast_history_frames": 10,
        "future_frames": 50, "runtime_only": False, "runtime_forbid_gt": True,
        "receiver_ids": [0, 1], "max_actors": 128, "max_cav": 2,
        "tracking_label_root": labels, "split_name": split,
        "partner_delay_ms": float(partner_delay_ms), "shuffle_training_points": False,
        "forecast_coordinate_frame": "current_receiver",
    }}
    if train_window:
        h["fusion"]["args"].update(window=list(ALIGNED_FORECAST_WINDOW),
                                  tracking_label_layout="dmstrack_train")
    elif not full_test:
        h["fusion"]["args"]["window"] = list(FORECAST_WINDOW)
    dataset = DualCMPSequenceDataset(h, train=False)
    expected = 4 if train_window else 16
    if not full_test and len(dataset) != expected:
        raise RuntimeError("unexpected frozen evaluation sample set")
    _, loader, _ = build_dataloader(dataset, batch_size, True, workers=workers,
        rank=dist.get_rank(), world_size=dist.get_world_size(), seed=0, training=False,
        collate_fn=dataset.collate_batch_test, pin_memory=True)
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    frames = []
    started = time.monotonic()
    try:
        with torch.no_grad():
            for batch in loader:
                keys = [(str(scene), int(receiver), int(frame))
                        for (scene, receiver), frame in zip(batch["scene_receiver_keys"],
                            batch["runtime_frames"][-1]["receiver_frame_idx"])]
                batch = to_device(batch, device)
                outputs, _ = model(batch["runtime_frames"], run_forecast=True)
                frames.extend(frame_exporter(outputs[-1], batch["target_frames"][-1], row, key)
                              for row, key in enumerate(keys))
        torch.cuda.synchronize(device)
        usage = {"rank": dist.get_rank(), "per_gpu_batch": batch_size,
                 "peak_scope": "whole_process_including_prior_training",
                 "samples": len(frames), "seconds": time.monotonic() - started,
                 "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                 "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20}
        usage["clips_per_second"] = usage["samples"] / max(usage["seconds"], 1e-9)
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (frames, usage))
        def report():
            result = report_builder([frame for group in gathered for frame in group[0]])
            result.update(dataset_samples=len(dataset), world_size=dist.get_world_size(),
                per_gpu_batch=batch_size, resources=[group[1] for group in gathered],
                split=split, full_test=full_test, sequence_policy=dataset.sequence_policy,
                partner_delay_ms=float(partner_delay_ms),
                forecast_representation=implementation.forecast_representation,
                aligned_evidence_policy=getattr(implementation, "aligned_evidence_policy", None))
            return result
        return _rank_zero_evaluation(report)
    finally:
        for module, training in modes:
            module.train(training)


def evaluate_mainstream_forecast_ddp(model, hypes, device, root, labels,
                                    batch_size=1, workers=2, full_test=False,
                                    partner_delay_ms=100.0, train_window=False):
    """Retain the historical UniAD protocol without relabeling its results."""
    return _evaluate_native_forecast_ddp(model, hypes, device, root, labels,
        batch_size, workers, full_test, partner_delay_ms, train_window)


def evaluate_cmp_adapted_forecast_ddp(model, hypes, device, root, labels,
                                    batch_size=1, workers=2, full_test=False,
                                    partner_delay_ms=100.0, train_window=False):
    """GT is exported only after native, GT-free model inference completes."""
    return _evaluate_native_forecast_ddp(model, hypes, device, root, labels,
        batch_size, workers, full_test, partner_delay_ms, train_window,
        frame_exporter=cmp_forecast_frame, report_builder=cmp_adapted_report)


def evaluate_forecast_training_ddp(model, hypes, device, root, labels,
                                    batch_size=1, workers=2, full_test=True,
                                    partner_delay_ms=0.0):
    """Formal B/C share the primary evaluator; old representations stay legacy."""
    implementation = model.module if hasattr(model, "module") else model
    if implementation.forecast_head.forecast_coordinate_frame != "current_receiver":
        from opencood.tools.dualcmp.eval_forecast_ddp import evaluate_forecast_test_model_ddp
        return evaluate_forecast_test_model_ddp(model, hypes, device, root, labels,
            batch_size, workers, full_test, partner_delay_ms)
    return evaluate_cmp_adapted_forecast_ddp(model, hypes, device, root, labels,
        batch_size, workers, full_test, partner_delay_ms)


def _training_bn_hypes(hypes, payload, root, labels):
    """Use the checkpoint's training observations, never the evaluation split."""
    provenance = payload["provenance"]
    scope = provenance.get("tracking_label_scope", {})
    if scope.get("split") != "train" or not labels:
        raise ValueError("external BN refresh requires recorded native train labels")
    result = copy.deepcopy(hypes)
    result["root_dir"] = result["validate_dir"] = str(Path(root) / "train")
    result["fusion"] = {"args": {
        "stage": "forecast", "tracking_frames": 3, "forecast_history_frames": 10,
        "future_frames": 50, "runtime_only": False, "runtime_forbid_gt": True,
        "receiver_ids": [0, 1], "max_actors": 128, "max_cav": 2,
        "tracking_label_root": labels, "tracking_label_layout": scope["layout"],
        "split_name": "train", "shuffle_training_points": False,
        "forecast_coordinate_frame": "current_receiver",
        "partner_delay_ms": provenance["partner_delay_ms"],
        "partner_delay_choices_ms": provenance.get("train_delays_ms", [provenance["partner_delay_ms"]]),
        "partner_delay_seed": provenance.get("delay_seed", 20260901),
        "excluded_scene_ids": list(provenance.get("excluded_scene_ids", [])),
    }}
    if scope.get("window"):
        result["fusion"]["args"]["window"] = copy.deepcopy(scope["window"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--full-test", action="store_true")
    scope.add_argument("--train-window", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--partner-delay-ms", type=float, default=100.0)
    parser.add_argument("--refresh-train-bn", action="store_true",
                        help="refresh forecast BN from recorded train observations only; checkpoint stays untouched")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("invalid evaluation batch size or worker count")
    if not math.isfinite(args.partner_delay_ms) or args.partner_delay_ms < 0:
        raise ValueError("partner delay must be finite and nonnegative")
    destination = Path(args.output)
    csv_destination = destination.with_suffix(".csv")
    if destination.exists() or csv_destination.exists():
        raise FileExistsError("evaluation must not overwrite an existing result")
    payload = torch.load(args.checkpoint, map_location="cpu")
    from opencood.tools.dualcmp.train_forecast_ddp import _configure_deterministic_runtime
    deterministic = bool(payload.get("provenance", {}).get("deterministic_runtime", {}).get("deterministic_algorithms", False))
    runtime = _configure_deterministic_runtime(deterministic)
    provenance = _checkpoint_provenance(args.checkpoint, payload, allow_dirty_smoke=True)
    contract = checkpoint_forecast_contract(payload["provenance"])
    validate_evaluation_delay_contract(payload["provenance"], args.partner_delay_ms,
        context="mainstream evaluation", **{key: contract[key] for key in (
            "forecast_representation", "capture_alignment", "age_encoding", "mean_source_aggregation")})
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch mainstream evaluation with torchrun")
    dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    try:
        hypes, _ = load_checkpoint_hypes(DEFAULT_HYPES, payload)
        hypes["model"]["args"]["compression"] = checkpoint_detector_compression(payload)
        model = _model_from_checkpoint(hypes, args.checkpoint, device, allow_dirty_smoke=True)
        calibration = None
        if args.refresh_train_bn:
            from opencood.tools.dualcmp.forecast_bn_refresh import refresh_forecast_bn_ddp
            train_hypes = _training_bn_hypes(hypes, payload, os.environ["V2V4REAL_ROOT"],
                                           os.environ.get("DUALCMP_TRAIN_TRACKING_LABEL_ROOT"))
            calibration = refresh_forecast_bn_ddp(model, train_hypes, device,
                int(payload["provenance"]["batch_contract"]["global_batch"]), args.workers)
        report = evaluate_cmp_adapted_forecast_ddp(model, hypes, device,
            os.environ["V2V4REAL_ROOT"], os.environ[
                "DUALCMP_TRAIN_TRACKING_LABEL_ROOT" if args.train_window else "DUALCMP_TRACKING_LABEL_ROOT"],
            args.batch_size, args.workers, args.full_test, args.partner_delay_ms, args.train_window)
        def save():
            report["checkpoint_provenance"] = provenance
            report["checkpoint_epoch"] = int(payload["epoch"])
            report["checkpoint_step"] = int(payload["step"])
            migration = payload["provenance"].get("evaluation_migration", {})
            report["checkpoint_training_evaluation_protocol"] = migration.get(
                "original_evaluation_protocol", payload["provenance"].get("forecast_evaluation"))
            report["original_checkpoint_sha256"] = migration.get(
                "original_checkpoint_sha256", provenance["checkpoint_sha256"])
            report["deterministic_runtime"] = runtime
            if calibration is not None:
                if _checkpoint_provenance(args.checkpoint, payload, allow_dirty_smoke=True)["checkpoint_sha256"] != provenance["checkpoint_sha256"]:
                    raise RuntimeError("external BN refresh must not modify the training checkpoint")
                report["external_train_bn_calibration"] = dict(calibration,
                    checkpoint_unchanged=True, buffers_scope="in-memory forecast BN only")
            report["training_forecast_association"] = payload["provenance"].get("forecast_association")
            report["evaluation_source_sha256"] = {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("cmp_adapted_forecast_metrics.py", "eval_forecast_mainstream_ddp.py")}
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(destination)
            fields = ("evaluation_protocol", "checkpoint_epoch", "checkpoint_step") + PRIMARY_METRICS
            with csv_destination.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict({"evaluation_protocol": report["metric_version"],
                    "checkpoint_epoch": report["checkpoint_epoch"], "checkpoint_step": report["checkpoint_step"]},
                    **{name: report["model"][name] for name in PRIMARY_METRICS}))
            print(json.dumps({"output": str(destination), "csv": str(csv_destination),
                              "kind": "forecast_evaluation", "report": report}), flush=True)
            return report
        _rank_zero_evaluation(save)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
