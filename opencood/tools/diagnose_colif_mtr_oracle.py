"""GT-history diagnostic for isolating CoLIF's MTR adapter.

This script is deliberately non-deployable: ground-truth identity, current
state and history replace the persistent tracker.  It must never be reported
as a strict-online model result.  Its only purpose is to distinguish a broken
MTR input contract from errors in learned query histories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.colif.persistent_query_mtr import PersistentQueryMTRAdapter
from opencood.tools import train_utils
from opencood.tools.colif.eval_utils import build_colif_evaluation_loader


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypes-yaml", required=True)
    parser.add_argument("--mtr-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def main():
    args = _arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hypes = load_yaml(args.hypes_yaml)
    forecast_args = dict(hypes["model"]["args"]["forecast"])
    forecast_args["enabled"] = True
    adapter = PersistentQueryMTRAdapter(
        int(hypes["model"]["args"]["persistent_query"]["hidden_dim"]),
        forecast_args,
    ).to(device)
    load_report = adapter.load_cmp_checkpoint(args.mtr_checkpoint)
    adapter.eval()
    _, loader = build_colif_evaluation_loader(
        hypes, split="test", workers=args.workers
    )

    horizons = (10, 30, 50)
    sums = {h: {"ade": 0.0, "fde": 0.0, "count": 0} for h in horizons}
    cv_sums = {h: {"ade": 0.0, "fde": 0.0, "count": 0} for h in horizons}
    stationary_sums = {
        h: {"ade": 0.0, "fde": 0.0, "count": 0} for h in horizons
    }
    future_step_statistics = {
        "valid": 0,
        "sum_speed": 0.0,
        "max_speed": 0.0,
        "over_20mps": 0,
        "over_50mps": 0,
    }
    history_steps = {age: {"fde": 0.0, "count": 0} for age in (1, 2, 3, 4, 11)}
    history_store = {}
    previous_key = None
    frame_count = 0
    scene_statistics = {}

    with torch.no_grad():
        for frame_index, batch in enumerate(loader):
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            if frame_index % 100 == 0:
                print(
                    "GT-history MTR diagnostic frame=%d/%d"
                    % (frame_index, len(loader)),
                    flush=True,
                )
            key = batch["scene_receiver_keys"][0]
            scene_name = str(key[0])
            scene_record = scene_statistics.setdefault(
                scene_name,
                {
                    "mtr_fde_5s": 0.0,
                    "cv_fde_5s": 0.0,
                    "stationary_fde_5s": 0.0,
                    "count_5s": 0,
                },
            )
            runtime = train_utils.to_device(batch["runtime_frames"][0], device)
            targets = train_utils.to_device(batch["target_frames"][0], device)
            current_pose = runtime["receiver_pose"]
            if key != previous_key:
                history_store = {}
                previous_key = key
            else:
                # Match CMP's released MTR cache exactly: each stored state
                # remains in the receiver-local coordinates of its own
                # capture.  This diagnostic deliberately does not use the
                # ego-motion-compensated history required by tracking.
                for record in history_store.values():
                    record["boxes"] = torch.roll(record["boxes"], -1, dims=0)
                    record["mask"] = torch.roll(record["mask"], -1, dims=0)
                    record["boxes"][-1].zero_()
                    record["mask"][-1] = False

            current_indices = torch.where(targets["current_mask"][0])[0]
            actor_ids = [
                int(targets["actor_ids"][0, index]) for index in current_indices
            ]
            for target_index, actor_id in zip(current_indices.tolist(), actor_ids):
                if actor_id not in history_store:
                    history_store[actor_id] = {
                        "boxes": torch.zeros(11, 7, device=device),
                        "mask": torch.zeros(11, dtype=torch.bool, device=device),
                    }
                history_store[actor_id]["boxes"][-1] = targets[
                    "current_boxes"
                ][0, target_index]
                history_store[actor_id]["mask"][-1] = True

            count = len(actor_ids)
            if count:
                current_boxes = targets["current_boxes"][0, current_indices].unsqueeze(0)
                history_box = torch.stack(
                    [history_store[actor_id]["boxes"] for actor_id in actor_ids]
                ).unsqueeze(0)
                history_mask = torch.stack(
                    [history_store[actor_id]["mask"] for actor_id in actor_ids]
                ).unsqueeze(0)
                query = torch.zeros(
                    1, count, adapter.query_dim, device=device
                )
                valid = torch.ones(1, count, dtype=torch.bool, device=device)
                prediction = adapter(
                    query,
                    current_boxes,
                    valid,
                    history_box,
                    history_mask,
                    runtime["receiver_time_s"],
                )["pred_trajs"][0, :, :, :, :2]
                for local_index, target_index in enumerate(current_indices.tolist()):
                    mask_full = targets["future_mask"][0, target_index]
                    target = targets["future_xy"][0, target_index]
                    age = int(history_mask[0, local_index].sum())
                    valid_history = torch.where(history_mask[0, local_index])[0]
                    if valid_history.numel() >= 2:
                        previous_index = int(valid_history[-2])
                        elapsed = (10 - previous_index) * float(
                            forecast_args.get("frame_period_s", 0.1)
                        )
                        velocity = (
                            history_box[0, local_index, -1, :2]
                            - history_box[0, local_index, previous_index, :2]
                        ) / elapsed
                    else:
                        velocity = target.new_zeros(2)
                    future_times = torch.arange(
                        1,
                        target.shape[0] + 1,
                        device=device,
                        dtype=target.dtype,
                    ) * float(forecast_args.get("frame_period_s", 0.1))
                    cv_prediction = future_times[:, None] * velocity[None]
                    stationary_prediction = torch.zeros_like(target)

                    positions = torch.cat(
                        (target.new_zeros(1, 2), target), dim=0
                    )
                    valid_positions = torch.cat(
                        (torch.ones(1, dtype=torch.bool, device=device), mask_full)
                    )
                    valid_steps = valid_positions[:-1] & valid_positions[1:]
                    if valid_steps.any():
                        speeds = torch.norm(
                            positions[1:] - positions[:-1], dim=-1
                        )[valid_steps] / float(
                            forecast_args.get("frame_period_s", 0.1)
                        )
                        future_step_statistics["valid"] += int(speeds.numel())
                        future_step_statistics["sum_speed"] += float(speeds.sum())
                        future_step_statistics["max_speed"] = max(
                            future_step_statistics["max_speed"], float(speeds.max())
                        )
                        future_step_statistics["over_20mps"] += int(
                            (speeds > 20.0).sum()
                        )
                        future_step_statistics["over_50mps"] += int(
                            (speeds > 50.0).sum()
                        )
                    for horizon in horizons:
                        mask = mask_full[:horizon]
                        if not mask.any():
                            continue
                        errors = torch.norm(
                            prediction[local_index, :, :horizon]
                            - target[:horizon].unsqueeze(0),
                            dim=-1,
                        )
                        mode_ade = (
                            errors * mask[None].to(errors.dtype)
                        ).sum(-1) / mask.sum()
                        last = int(torch.where(mask)[0][-1])
                        sums[horizon]["ade"] += float(mode_ade.min())
                        sums[horizon]["fde"] += float(errors[:, last].min())
                        sums[horizon]["count"] += 1
                        for baseline, baseline_sums in (
                            (cv_prediction, cv_sums),
                            (stationary_prediction, stationary_sums),
                        ):
                            baseline_errors = torch.norm(
                                baseline[:horizon] - target[:horizon], dim=-1
                            )
                            baseline_sums[horizon]["ade"] += float(
                                (
                                    baseline_errors
                                    * mask.to(baseline_errors.dtype)
                                ).sum()
                                / mask.sum()
                            )
                            baseline_sums[horizon]["fde"] += float(
                                baseline_errors[last]
                            )
                            baseline_sums[horizon]["count"] += 1
                        if horizon == 50:
                            scene_record["mtr_fde_5s"] += float(
                                errors[:, last].min()
                            )
                            scene_record["cv_fde_5s"] += float(
                                torch.norm(
                                    cv_prediction[last] - target[last]
                                )
                            )
                            scene_record["stationary_fde_5s"] += float(
                                torch.norm(target[last])
                            )
                            scene_record["count_5s"] += 1
                            bucket = age if age <= 4 else 11
                            history_steps[bucket]["fde"] += float(
                                errors[:, last].min()
                            )
                            history_steps[bucket]["count"] += 1
            frame_count += 1

    metrics = {
        "diagnostic_only": True,
        "strict_online": False,
        "runtime_uses_ground_truth": True,
        "not_deployable": True,
        "purpose": "isolate MTR adapter with GT identity/current/history",
        "frame_count": frame_count,
        "mtr_checkpoint": str(Path(args.mtr_checkpoint).resolve()),
        "mtr_load_report": load_report,
    }
    for horizon in horizons:
        count = sums[horizon]["count"]
        seconds = horizon / 10
        metrics["oracle_actor_count_%.0fs" % seconds] = count
        metrics["oracle_minADE_%.0fs" % seconds] = (
            sums[horizon]["ade"] / count if count else None
        )
        metrics["oracle_minFDE_%.0fs" % seconds] = (
            sums[horizon]["fde"] / count if count else None
        )
        cv_count = cv_sums[horizon]["count"]
        stationary_count = stationary_sums[horizon]["count"]
        metrics["gt_history_cv_minADE_%.0fs" % seconds] = (
            cv_sums[horizon]["ade"] / cv_count if cv_count else None
        )
        metrics["gt_history_cv_minFDE_%.0fs" % seconds] = (
            cv_sums[horizon]["fde"] / cv_count if cv_count else None
        )
        metrics["stationary_minADE_%.0fs" % seconds] = (
            stationary_sums[horizon]["ade"] / stationary_count
            if stationary_count
            else None
        )
        metrics["stationary_minFDE_%.0fs" % seconds] = (
            stationary_sums[horizon]["fde"] / stationary_count
            if stationary_count
            else None
        )
    for age, values in history_steps.items():
        metrics["oracle_history_%s_count_5s" % age] = values["count"]
        metrics["oracle_history_%s_minFDE_5s" % age] = (
            values["fde"] / values["count"] if values["count"] else None
        )
    step_count = future_step_statistics.pop("valid")
    metrics["future_valid_step_count"] = step_count
    metrics["future_mean_speed_mps"] = (
        future_step_statistics.pop("sum_speed") / step_count
        if step_count
        else None
    )
    metrics.update(
        {"future_%s" % key: value for key, value in future_step_statistics.items()}
    )
    metrics["per_scene_5s"] = {
        scene: {
            "actor_count": values["count_5s"],
            "oracle_mtr_minFDE": (
                values["mtr_fde_5s"] / values["count_5s"]
                if values["count_5s"]
                else None
            ),
            "gt_history_cv_FDE": (
                values["cv_fde_5s"] / values["count_5s"]
                if values["count_5s"]
                else None
            ),
            "stationary_FDE": (
                values["stationary_fde_5s"] / values["count_5s"]
                if values["count_5s"]
                else None
            ),
        }
        for scene, values in scene_statistics.items()
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
