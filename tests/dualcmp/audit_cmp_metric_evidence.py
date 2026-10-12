"""Bounded, read-only native inference audit; no production source is changed."""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from opencood.tools.dualcmp import eval_forecast_mainstream_ddp as native
from opencood.tools.dualcmp.cmp_adapted_forecast_metrics import (
    forecast_frame, cmp_adapted_report, _validated_frames, _pair_frames,
)
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES, load_checkpoint_hypes
from opencood.tools.dualcmp.online import _model_from_checkpoint
from opencood.tools.dualcmp.protocol import checkpoint_detector_compression
from opencood.tools.dualcmp.train_forecast_ddp import _configure_deterministic_runtime


class StratifiedDataset(native.DualCMPSequenceDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        strata = defaultdict(list)
        for record in self.clip_records:
            strata[(record.scene_id, record.receiver_id)].append(record)
        self.original_samples = len(self.clip_records)
        selected = []
        for key in sorted(strata):
            records = strata[key]
            selected.extend(records[index] for index in sorted(set(
                [len(records) // 3, 2 * len(records) // 3])))
        self.clip_records = selected
        if len(strata) != 18 or len(selected) != 36:
            raise RuntimeError("expected nine scenes, two receivers, two clips per stratum")


def arithmetic_metrics(prediction, target):
    distances = np.linalg.norm(np.asarray(prediction, dtype=np.float64)
                               - np.asarray(target, dtype=np.float64)[None], axis=-1)
    result = {}
    for seconds, prefix, endpoint in ((1, 10, 10), (3, 30, 30), (5, 49, 49)):
        result["minADE_%ds" % seconds] = float(distances[:, :prefix].mean(-1).min())
        result["minFDE_%ds" % seconds] = float(distances[:, endpoint].min())
    for threshold in (3.6, 3.0, 2.0, 1.8):
        result["MR_%.1f" % threshold] = float(distances[:, 49].min() > threshold)
    return result


def mean_rows(rows):
    if not rows:
        return {"count": 0}
    return dict({key: float(np.mean([row[key] for row in rows])) for key in rows[0]}, count=len(rows))


def evidence_report(frames):
    frames = _validated_frames(frames)
    result = cmp_adapted_report(frames)
    pairs, _ = _pair_frames(frames)
    model_rows, stationary_rows, oracle_stationary_rows = [], [], []
    groups = defaultdict(lambda: {"model": [], "stationary": []})
    complete_gt_groups = defaultdict(int)
    for frame in frames:
        for gt_slot in np.flatnonzero(frame["gt_mask"].all(-1)):
            movement = float(np.linalg.norm(frame["gt_future"][gt_slot]
                - frame["gt_boxes"][gt_slot, None, :2], axis=-1).max())
            label = "max_movement_lt_1m" if movement < 1 else (
                "max_movement_1_to_5m" if movement < 5 else "max_movement_ge_5m")
            complete_gt_groups[label] += 1
    records = []
    for frame, pred_slot, gt_slot in pairs:
        target = frame["gt_future"][gt_slot]
        current_gt = frame["gt_boxes"][gt_slot, :2]
        current_pred = frame["pred_boxes"][pred_slot, :2]
        movement = float(np.linalg.norm(target - current_gt[None], axis=-1).max())
        label = "max_movement_lt_1m" if movement < 1 else (
            "max_movement_1_to_5m" if movement < 5 else "max_movement_ge_5m")
        model = arithmetic_metrics(frame["pred_future"][pred_slot], target)
        stationary = arithmetic_metrics(np.broadcast_to(current_pred, (6, 50, 2)), target)
        oracle = arithmetic_metrics(np.broadcast_to(current_gt, (6, 50, 2)), target)
        model_rows.append(model)
        stationary_rows.append(stationary)
        oracle_stationary_rows.append(oracle)
        groups[label]["model"].append(model)
        groups[label]["stationary"].append(stationary)
        records.append({"key": list(frame["key"]), "gt_id": int(frame["gt_ids"][gt_slot]),
            "runtime_id": int(frame["pred_ids"][pred_slot]), "max_gt_movement_m": movement,
            "current_center_error_m": float(np.linalg.norm(current_pred - current_gt)),
            "model": model, "stationary_runtime_anchor": stationary})
    independent = mean_rows(model_rows)
    differences = {key: abs(value - independent[key]) for key, value in result["model"].items()}
    if max(differences.values()) > 1e-5:
        raise RuntimeError("independent metric arithmetic differs from official function")
    result["diagnostics"] = {
        "sample_policy": "two chronological-tertile clips per scene/receiver, selected without reading GT",
        "scope": "36 stratified clips; do not extrapolate group proportions to the full test population",
        "motion_bins": "diagnostic maximum GT displacement from current GT center, not an official filter",
        "complete_gt_motion_counts": dict(complete_gt_groups),
        "eligible_fraction_per_motion_bin": {key: len(groups[key]["model"]) / count
                                            for key, count in complete_gt_groups.items()},
        "independent_arithmetic": independent, "official_max_difference": max(differences.values()),
        "stationary_runtime_anchor": mean_rows(stationary_rows),
        "stationary_gt_anchor_oracle_diagnostic": mean_rows(oracle_stationary_rows),
        "groups": {key: {name: mean_rows(rows) for name, rows in values.items()}
                   for key, values in groups.items()}, "matched_records": records}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    payload = torch.load(args.checkpoint, map_location="cpu")
    _configure_deterministic_runtime(True)
    dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    try:
        hypes, _ = load_checkpoint_hypes(DEFAULT_HYPES, payload)
        hypes["model"]["args"]["compression"] = checkpoint_detector_compression(payload)
        model = _model_from_checkpoint(hypes, args.checkpoint, device, allow_dirty_smoke=True)
        native.DualCMPSequenceDataset = StratifiedDataset
        report = native._evaluate_native_forecast_ddp(model, hypes, device,
            os.environ["V2V4REAL_ROOT"], os.environ["DUALCMP_TRACKING_LABEL_ROOT"],
            batch_size=2, workers=2, full_test=True, partner_delay_ms=100,
            frame_exporter=forecast_frame, report_builder=evidence_report)
        report["full_test"] = False
        report["available_test_clips"] = 2906
        report["checkpoint_epoch"], report["checkpoint_step"] = payload["epoch"], payload["step"]
        if dist.get_rank() == 0:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
            print(json.dumps({"output": str(destination), "model": report["model"],
                "diagnostics": {key: value for key, value in report["diagnostics"].items()
                                if key != "matched_records"}}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
