"""Fresh two-rank CMP forecast evaluation for the frozen 16-sample protocol."""

from __future__ import annotations
import argparse, json, os, hashlib
from pathlib import Path
import numpy as np
import torch, torch.distributed as dist
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import (
    DualCMPSequenceDataset,
)
from opencood.loss.dualcmp_loss import update_identity_bindings
from opencood.loss.forecast_association import (
    LEGACY_ASSOCIATION, VALIDATED_ASSOCIATION, ASSOCIATION_STATUS,
    association_contract, associate_forecast_clip,
    forecast_metric_version,
)
from opencood.data_utils.dualcmp_dataloader import build_dataloader
from opencood.tools.train_utils import to_device
from opencood.tools.dualcmp.forecast_metrics import cmp_metrics, horizon_metrics, FORECAST_METRIC_KEYS
from opencood.tools.dualcmp.online import (
    OnlineDualCMP,
    _checkpoint_provenance,
    _model_from_checkpoint,
)
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES, load_checkpoint_hypes


def _forecast_rows(output, target, keys):
    """Existing target-only binding and CMP row extraction, keyed for dedupe."""
    if output["forecast"].get("forecast_coordinate_frame", "legacy_capture") != target.get("forecast_coordinate_frame", "legacy_capture"):
        raise RuntimeError("evaluation forecast/target coordinate contracts disagree")
    assigned, _ = update_identity_bindings(output, target, [dict() for _ in keys])
    forecast, rows, coverage = output["forecast"], [], []
    for row, key in enumerate(keys):
        complete = set(torch.nonzero(target["future_mask"][row].all(-1), as_tuple=False).flatten().tolist())
        eligible = set()
        for slot in torch.nonzero(assigned[row] >= 0, as_tuple=False).flatten().tolist():
            index = int(assigned[row, slot])
            if index in complete and bool(forecast["forecast_valid_mask"][row, slot]):
                eligible.add(index)
        coverage.append({"scene": key[0], "receiver": key[1], "prediction_frame": key[2],
                         "complete": len(complete), "eligible_complete": len(eligible)})
        for slot in torch.nonzero((assigned[row] >= 0) & forecast["forecast_valid_mask"][row], as_tuple=False).flatten().tolist():
            index = int(assigned[row, slot])
            mask = target["future_mask"][row, index].detach().cpu().numpy()
            if mask.all():
                rows.append((key + (int(target["actor_ids"][row, index]),),
                    forecast["pred_trajs"][row, slot, :, :, :2].detach().cpu().numpy(),
                    target["future_xy"][row, index].detach().cpu().numpy(), mask,
                    target["current_velocity"][row, index].detach().cpu().numpy(),
                    output["track_box"][row, slot, :2].detach().cpu().numpy(),
                    target["current_boxes"][row, index, :2].detach().cpu().numpy()))
    return rows, coverage


def _forecast_report(rows, coverage):
    unique = {}
    for row in rows:
        unique.setdefault(row[0], row)  # DistributedSampler may pad one sample.
    values = list(unique.values())
    prediction, target, mask = [x[1] for x in values], [x[2] for x in values], [x[3] for x in values]
    stationary = [np.zeros((6, 50, 2), np.float32) for _ in values]
    cv = [np.repeat((np.arange(1, 51, dtype=np.float32)[:, None] * x[4] * 0.1)[None], 6, axis=0) for x in values]
    coverage_unique = {(x["scene"], x["receiver"], x["prediction_frame"]): x for x in coverage}
    report = {"coverage": {"prediction_keys": len(coverage_unique), "complete": sum(x["complete"] for x in coverage_unique.values()),
                           "eligible_complete": sum(x["eligible_complete"] for x in coverage_unique.values())},
            "model": cmp_metrics(prediction, target, mask, prediction_centers=[x[5] for x in values], target_centers=[x[6] for x in values]),
            "stationary": cmp_metrics(stationary, target, mask), "cv": cmp_metrics(cv, target, mask)}
    pred_centers, target_centers = [x[5] for x in values], [x[6] for x in values]
    center_error = np.asarray([np.linalg.norm(x[5] - x[6]) for x in values])
    report["diagnostics"] = {
        "legacy_baseline_note": "stationary/cv use GT current centers; CV velocity is ego-motion compensated",
        "model_start_center_error_m": {"mean": float(center_error.mean()),
                                        "median": float(np.median(center_error)),
                                        "p95": float(np.percentile(center_error, 95)),
                                        "maximum": float(center_error.max())},
        "matched_beyond_3_6m": int((center_error > 3.6).sum()),
        "baseline_stationary_track_anchor": cmp_metrics(stationary, target, mask,
                                                           pred_centers, target_centers),
        "baseline_cv_track_anchor_gt_velocity": cmp_metrics(cv, target, mask,
                                                              pred_centers, target_centers),
        "exact_horizon_model": horizon_metrics(prediction, target, mask, pred_centers, target_centers),
        "exact_horizon_top1": horizon_metrics([x[:1] for x in prediction], target, mask,
                                               pred_centers, target_centers),
        "metric_note": "CMP legacy labels stay unchanged; exact horizons include endpoints and use t+0.1 as index zero",
    }
    # This is a disclosed diagnostic filter, not a replacement matching cost or
    # an end-to-end benchmark. Coverage excludes misses and incomplete futures.
    bounded = [x for x in values if np.linalg.norm(x[5] - x[6]) <= 3.6]
    if bounded:
        report["diagnostics"]["center_error_le_3_6m_model"] = cmp_metrics(
            [x[1] for x in bounded], [x[2] for x in bounded], [x[3] for x in bounded],
            [x[5] for x in bounded], [x[6] for x in bounded])
    return report


def _validated_forecast_rows(outputs, batch, keys):
    """A causal, target-only pairing with explicit admitted/rejected population."""
    targets = batch["association_target_frames"]
    output, target = outputs[-1], batch["target_frames"][-1]
    if output["forecast"].get("forecast_coordinate_frame", "legacy_capture") != target.get("forecast_coordinate_frame", "legacy_capture"):
        raise RuntimeError("evaluation forecast/target coordinate contracts disagree")
    if (not torch.equal(targets[-1]["actor_ids"], target["actor_ids"])
            or not torch.equal(targets[-1]["current_mask"], target["current_mask"])):
        raise RuntimeError("evaluation causal and final target identities disagree")
    assigned, history = associate_forecast_clip(outputs, targets)
    diagnostics, forecast = history[-1], output["forecast"]
    # Learned quality can emit a Kalman coast without a detection. Runtime
    # history length alone is not evidence of eleven measured observations.
    observed_steps = torch.stack([
        item["state"].valid_mask & item["state"].active_mask
        & (item["state"].runtime_id == output["state"].runtime_id)
        & (item["state"].miss_count == 0) for item in outputs]).sum(0)
    rows, coverage, records = [], [], []
    for row, key in enumerate(keys):
        frame_records = []
        current = set(torch.nonzero(target["current_mask"][row], as_tuple=False).flatten().tolist())
        complete = set(torch.nonzero(target["future_mask"][row].all(-1) & target["current_mask"][row], as_tuple=False).flatten().tolist())
        valid_slots = torch.nonzero(forecast["forecast_valid_mask"][row], as_tuple=False).flatten().tolist()
        eligible, strict = set(), set()
        matched_current = 0
        for slot in valid_slots:
            index = int(assigned[row, slot])
            steps = int(forecast["forecast_history_steps"][row, slot])
            detections = int(observed_steps[row, slot])
            if index >= 0:
                matched_current += 1
            record = {"scene": key[0], "receiver": key[1], "prediction_frame": key[2],
                "runtime_id": int(output["state"].runtime_id[row, slot]),
                "gt_actor_id": int(target["actor_ids"][row, index]) if index >= 0 else None,
                "history_steps": steps, "match_status": ASSOCIATION_STATUS[int(diagnostics["status"][row, slot])],
                "detection_observation_steps": detections,
                "matched_iou": float(diagnostics["matched_iou"][row, slot]),
                "best_current_iou": float(diagnostics["best_iou"][row, slot]),
                "prediction_center": output["track_box"][row, slot, :2].detach().cpu().tolist(),
                "gt_center": target["current_boxes"][row, index, :2].detach().cpu().tolist() if index >= 0 else None,
                "complete_future": bool(index in complete),
                "strict_observed_history": bool(steps == 11 and detections == 11)}
            records.append(record)
            frame_records.append(record)
            if index not in complete:
                continue
            eligible.add(index)
            if steps == 11 and detections == 11:
                strict.add(index)
            rows.append((key + (int(target["actor_ids"][row, index]),),
                forecast["pred_trajs"][row, slot, :, :, :2].detach().cpu().numpy(),
                target["future_xy"][row, index].detach().cpu().numpy(),
                target["future_mask"][row, index].detach().cpu().numpy(),
                target["current_velocity"][row, index].detach().cpu().numpy(),
                output["track_box"][row, slot, :2].detach().cpu().numpy(),
                target["current_boxes"][row, index, :2].detach().cpu().numpy()))
        coverage.append({"scene": key[0], "receiver": key[1], "prediction_frame": key[2],
            "complete": len(complete), "eligible_complete": len(eligible),
            "current_gt": len(current), "predicted_forecasts": len(valid_slots),
            "matched_current": matched_current,
            "unmatched_predictions": len(valid_slots) - matched_current,
            "unmatched_current_gt": len(current) - matched_current,
            "missed_complete": len(complete - eligible), "strict_history_complete": len(strict),
            "missed_complete_actor_ids": [int(target["actor_ids"][row, index]) for index in sorted(complete - eligible)],
            "identity_conflicts": sum(x["match_status"] == "identity_conflict" for x in frame_records)})
    return rows, coverage, records, assigned


def _validated_forecast_report(rows, coverage, records):
    """Quality is conditional; failures remain visible rather than becoming fake ADE."""
    if rows:
        report = _forecast_report(rows, coverage)
        report["cmp_legacy_same_population_model"] = report["model"]
        report["model"] = report["diagnostics"]["exact_horizon_model"]
        values = list({x[0]: x for x in rows}.values())
        targets, masks = [x[2] for x in values], [x[3] for x in values]
        centers, gt_centers = [x[5] for x in values], [x[6] for x in values]
        stationary = [np.zeros((6, 50, 2), np.float32) for _ in values]
        cv = [np.repeat((np.arange(1, 51, dtype=np.float32)[:, None] * x[4] * .1)[None], 6, axis=0) for x in values]
        report["stationary"] = horizon_metrics(stationary, targets, masks)
        report["cv"] = horizon_metrics(cv, targets, masks)
        report["diagnostics"]["baseline_stationary_track_anchor"] = horizon_metrics(stationary, targets, masks, centers, gt_centers)
        report["diagnostics"]["baseline_cv_track_anchor_gt_velocity"] = horizon_metrics(cv, targets, masks, centers, gt_centers)
        report["diagnostics"]["metric_note"] = "headline1/3/5s uses indices9/29/49 and includes ADE endpoints; CMP legacy retained separately"
    else:
        report = {"model": dict({name: None for name in FORECAST_METRIC_KEYS}, count=0),
                  "diagnostics": {"exact_horizon_model": dict({name: None for name in FORECAST_METRIC_KEYS}, count=0)},
                  "coverage": {"prediction_keys": len({(x["scene"], x["receiver"], x["prediction_frame"]) for x in coverage}),
                               "complete": 0, "eligible_complete": 0},
                  "metric_status": "no_valid_forecast_pairs"}
    unique = {(x["scene"], x["receiver"], x["prediction_frame"]): x for x in coverage}
    fields = ("complete", "eligible_complete", "current_gt", "predicted_forecasts", "matched_current",
              "unmatched_predictions", "unmatched_current_gt", "missed_complete",
              "strict_history_complete", "identity_conflicts")
    counts = {name: sum(x[name] for x in unique.values()) for name in fields}
    report["coverage"].update({name: counts[name] for name in ("complete", "eligible_complete")})
    report["association"] = dict(counts,
        forecast_precision=counts["matched_current"] / max(counts["predicted_forecasts"], 1),
        current_actor_recall=counts["matched_current"] / max(counts["current_gt"], 1),
        complete_future_coverage=counts["eligible_complete"] / max(counts["complete"], 1))
    unique_records = {(x["scene"], x["receiver"], x["prediction_frame"], x["runtime_id"]): x for x in records}
    report["association_records"] = list(unique_records.values())
    strict_keys = {(x["scene"], x["receiver"], x["prediction_frame"], x["gt_actor_id"])
                   for x in unique_records.values() if x["complete_future"] and x["strict_observed_history"]}
    strict_rows = [x for x in rows if x[0] in strict_keys]
    report["strict_observed_history_model"] = (_forecast_report(strict_rows, coverage)["diagnostics"]["exact_horizon_model"]
        if strict_rows else dict({name: None for name in FORECAST_METRIC_KEYS}, count=0))
    report["forecast_association"] = association_contract(VALIDATED_ASSOCIATION)
    report["metric_version"] = forecast_metric_version(VALIDATED_ASSOCIATION)
    report["metric_scope"] = "IoU-admitted causal-identity forecast quality, with explicit failure/coverage counts"
    report["cmp_comparison_note"] = "strict11 requires eleven emitted detection observations, excludes Kalman-only coasts, and is not CMP's oracle-filtered cache population"
    if report["model"]["count"] != counts["eligible_complete"]:
        raise RuntimeError("forecast metric population disagrees with admitted complete targets")
    return report


def _forecast_rows_by_policy(outputs, batch, keys, policy):
    association_contract(policy)
    if policy == VALIDATED_ASSOCIATION:
        return _validated_forecast_rows(outputs, batch, keys)
    rows, coverage = _forecast_rows(outputs[-1], batch["target_frames"][-1], keys)
    assigned, _ = update_identity_bindings(outputs[-1], batch["target_frames"][-1], [dict() for _ in keys])
    return rows, coverage, [], assigned


def _rank_zero_evaluation(build_report):
    """Broadcast metric failures too, so no peer waits for a dead root."""
    payload = [None]
    if dist.get_rank() == 0:
        try:
            payload[0] = {"report": build_report(), "error": None}
        except Exception as exc:
            payload[0] = {"report": None,
                          "error": "%s: %s" % (type(exc).__name__, exc)}
    dist.broadcast_object_list(payload, src=0)
    if payload[0]["error"] is not None:
        raise RuntimeError("forecast evaluation failed on rank 0: " + payload[0]["error"])
    return payload[0]["report"]


def evaluate_forecast_test_model_ddp(model, hypes, device, root, labels,
                                     batch_size=1, workers=0, full_test=True,
                                     partner_delay_ms=0.0):
    """Evaluate the current model over full test or the frozen smoke window."""
    if not dist.is_initialized():
        raise RuntimeError("forecast evaluation requires initialized DDP")
    h = dict(hypes)
    implementation = model.module if hasattr(model, "module") else model
    policy = getattr(implementation, "forecast_association_policy", LEGACY_ASSOCIATION)
    h["root_dir"], h["validate_dir"] = str(Path(root) / "test"), str(Path(root) / "test")
    h["fusion"] = {"args": {"stage": "forecast", "tracking_frames": 3,
        "forecast_history_frames": 10, "future_frames": 50, "runtime_only": False,
        "runtime_forbid_gt": True, "receiver_ids": [0, 1], "max_actors": 128,
        "max_cav": 2, "tracking_label_root": labels, "split_name": "test",
        "partner_delay_ms": float(partner_delay_ms),
        "shuffle_training_points": False,
        "include_forecast_association_history": policy == VALIDATED_ASSOCIATION,
        "forecast_coordinate_frame": (model.module if hasattr(model, "module") else model).forecast_head.forecast_coordinate_frame}}
    if not full_test:
        h["fusion"]["args"]["window"] = list(FORECAST_WINDOW)
    dataset = DualCMPSequenceDataset(h, train=False)
    if not full_test and len(dataset) != FORECAST_SAMPLES:
        raise RuntimeError("fixed forecast evaluation requires 16 samples")
    _, loader, _ = build_dataloader(dataset, batch_size, True, workers=workers,
        rank=dist.get_rank(), world_size=dist.get_world_size(), seed=0, training=False,
        collate_fn=dataset.collate_batch_test, pin_memory=True)
    modes = [(module, module.training) for module in model.modules()]
    model.eval(); rows, coverage, records, legacy_rows, legacy_coverage = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            keys = [(str(scene), int(receiver), int(frame)) for (scene, receiver), frame in zip(
                batch["scene_receiver_keys"], batch["runtime_frames"][-1]["receiver_frame_idx"])]
            batch = to_device(batch, device)
            outputs, _ = model(batch["runtime_frames"], run_forecast=True)
            local_rows, local_coverage, local_records, _ = _forecast_rows_by_policy(outputs, batch, keys, policy)
            rows.extend(local_rows); coverage.extend(local_coverage)
            records.extend(local_records)
            if policy == VALIDATED_ASSOCIATION:
                raw_rows, raw_coverage = _forecast_rows(outputs[-1], batch["target_frames"][-1], keys)
                legacy_rows.extend(raw_rows); legacy_coverage.extend(raw_coverage)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (rows, coverage, records, legacy_rows, legacy_coverage))
    def build_report():
        all_rows, all_coverage = [x for group in gathered for x in group[0]], [x for group in gathered for x in group[1]]
        if policy == VALIDATED_ASSOCIATION:
            report = _validated_forecast_report(all_rows, all_coverage, [x for group in gathered for x in group[2]])
            raw_rows = [x for group in gathered for x in group[3]]
            report["legacy_unbounded_diagnostic"] = (_forecast_report(raw_rows, [x for group in gathered for x in group[4]])
                if raw_rows else {"model": dict({name: None for name in FORECAST_METRIC_KEYS}, count=0)})
        else:
            report = _forecast_report(all_rows, all_coverage)
            report["forecast_association"] = association_contract(policy)
            report["metric_version"] = forecast_metric_version(policy)
            report["metric_scope"] = "legacy unbounded current-frame assignment; not a validated identity benchmark"
        implementation = model.module if hasattr(model, "module") else model
        report.update({
            "forecast_supervision": implementation.forecast_supervision,
            "forecast_variance": implementation.forecast_head.forecast_variance,
            "partner_delay_ms": float(partner_delay_ms),
            "forecast_representation": implementation.forecast_representation,
            "capture_alignment": bool(
                implementation.capture_alignment_enabled),
            "age_encoding": bool(implementation.age_encoding_enabled),
            "mean_source_aggregation": bool(
                implementation.mean_source_aggregation_enabled),
            "capture_time_evidence": bool(
                implementation.capture_time_evidence_enabled),
            "forecast_coordinate_frame": implementation.forecast_head.forecast_coordinate_frame,
            "sequence_policy": dataset.sequence_policy,
        })
        if implementation.forecast_representation == "source_temporal_aligned":
            report.update(aligned_evidence_policy=implementation.aligned_evidence_policy,
                          source_feature_encoder=implementation.source_feature_encoder,
                          forecast_normalization=implementation.forecast_head.forecast_normalization)
        return report
    try:
        return _rank_zero_evaluation(build_report)
    finally:
        for module, training in modes:
            module.train(training)
        if any(module.training != training for module, training in modes):
            raise RuntimeError("forecast evaluator failed to restore module training modes")
from opencood.tools.dualcmp.protocol import (
    FORECAST_WINDOW,
    FORECAST_SAMPLES,
    checkpoint_detector_compression,
)
from opencood.tools.dualcmp.temporal_protocol import (
    checkpoint_forecast_contract, validate_evaluation_delay_contract,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--full-test", action="store_true")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--partner-delay-ms", type=float, default=0.0)
    a = p.parse_args()
    if not np.isfinite(a.partner_delay_ms) or a.partner_delay_ms < 0:
        raise ValueError("partner delay must be finite and non-negative")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("forecast evaluation requires torchrun")
    checkpoint_provenance = _checkpoint_provenance(
        a.checkpoint, allow_dirty_smoke=True
    )
    checkpoint_payload = torch.load(a.checkpoint, map_location="cpu")
    ablation = checkpoint_forecast_contract(
        checkpoint_payload.get("provenance"))
    validate_evaluation_delay_contract(
        checkpoint_payload.get("provenance"), a.partner_delay_ms,
        context="forecast evaluation",
        forecast_representation=ablation["forecast_representation"],
        capture_alignment=ablation["capture_alignment"],
        age_encoding=ablation["age_encoding"],
        mean_source_aggregation=ablation["mean_source_aggregation"],
    )
    dist.init_process_group("nccl")
    rank, local = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local)
    h, _ = load_checkpoint_hypes(DEFAULT_HYPES, checkpoint_payload)
    h["root_dir"] = str(Path(os.environ["V2V4REAL_ROOT"]) / "test")
    h["validate_dir"] = h["root_dir"]
    h["model"]["args"]["compression"] = checkpoint_detector_compression(
        checkpoint_payload
    )
    h["fusion"] = {
        "args": {
            "stage": "forecast",
            "tracking_frames": 3,
            "forecast_history_frames": 10,
            "future_frames": 50,
            "runtime_only": False,
            "runtime_forbid_gt": True,
            "receiver_ids": [0, 1],
            "max_actors": 128,
            "max_cav": 2,
            "partner_delay_ms": a.partner_delay_ms,
            "tracking_label_root": os.environ["DUALCMP_TRACKING_LABEL_ROOT"],
            "window": list(FORECAST_WINDOW),
            "shuffle_training_points": False,
            "forecast_coordinate_frame": "current_receiver" if ablation["forecast_representation"] == "source_temporal_aligned" else "legacy_capture",
        }
    }
    d = DualCMPSequenceDataset(h, train=True)
    assert len(d) == FORECAST_SAMPLES
    fa = {
        "history_frames": 11,
        "future_frames": 50,
        "modes": 6,
        "intention_points_file": "preprocessed_data/v2v4real/v2v4real_cluster_64_center_dict.pkl",
        "representation": ablation["forecast_representation"],
        "capture_alignment": ablation["capture_alignment"],
        "age_encoding": ablation["age_encoding"],
        "mean_source_aggregation": ablation["mean_source_aggregation"],
    }
    model = _model_from_checkpoint(
        h, a.checkpoint, torch.device("cuda", local), forecast_args=fa,
        allow_dirty_smoke=True,
    )
    if a.full_test or model.forecast_association_policy == VALIDATED_ASSOCIATION:
        report = evaluate_forecast_test_model_ddp(
            model, h, torch.device("cuda", local),
            os.environ["V2V4REAL_ROOT"],
            os.environ["DUALCMP_TRACKING_LABEL_ROOT"],
            batch_size=a.batch_size, workers=a.workers, full_test=a.full_test,
            partner_delay_ms=a.partner_delay_ms,
        )
        def save_report():
            report["checkpoint_provenance"] = checkpoint_provenance
            Path(a.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report))
            return report
        _rank_zero_evaluation(save_report)
        dist.barrier(); dist.destroy_process_group()
        return
    online = OnlineDualCMP(model, torch.device("cuda", local))
    rows = []
    coverage = []
    for i in range(rank, len(d), world_size):
        sample = d[i]
        prediction = None
        online.reset_scene(sample["scene_receiver_key"][0])
        for frame_index, frame in enumerate(sample["runtime_frames"]):
            prediction = online.step(
                d._collate_runtime_frame([frame]),
                run_forecast=(frame_index == len(sample["runtime_frames"]) - 1),
            )
        target = d._collate_target_frame([sample["target_frames"][0]])
        target = {
            key: (
                value.to(torch.device("cuda", local))
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in target.items()
        }
        bindings = [{}]
        assigned, _ = update_identity_bindings(online._last_output, target, bindings)
        forecast = online._last_output["forecast"]
        complete = set(
            torch.nonzero(target["future_mask"][0].all(-1), as_tuple=False)
            .flatten()
            .tolist()
        )
        assigned_complete = set()
        eligible_complete = set()
        runtime_by_target = {}
        for slot in torch.nonzero(assigned[0] >= 0, as_tuple=False).flatten().tolist():
            target_index = int(assigned[0, slot])
            runtime = int(online._last_output["state"].runtime_id[0, slot])
            if target_index in complete:
                assigned_complete.add(target_index)
                runtime_by_target[target_index] = runtime
                if bool(forecast["forecast_valid_mask"][0, slot]):
                    eligible_complete.add(target_index)
        missing = sorted(complete - eligible_complete)
        slot_by_target = {
            int(assigned[0, slot]): slot
            for slot in torch.nonzero(assigned[0] >= 0, as_tuple=False)
            .flatten()
            .tolist()
        }
        coverage.append(
            {
                "scene": sample["scene_receiver_key"][0],
                "receiver": sample["scene_receiver_key"][1],
                "prediction_frame": sample["runtime_frames"][-1]["receiver_frame_idx"],
                "complete": len(complete),
                "assigned_complete": len(assigned_complete),
                "eligible_complete": len(eligible_complete),
                "missing": [
                    {
                        "actor_id": int(target["actor_ids"][0, index]),
                        "runtime_id": runtime_by_target.get(index),
                        "stage": (
                            "forecast_invalid"
                            if index in assigned_complete
                            else "unassigned"
                        ),
                        "active": (
                            bool(
                                online._last_output["state"].active_mask[
                                    0, slot_by_target[index]
                                ]
                            )
                            if index in slot_by_target
                            else None
                        ),
                        "forecast_history_steps": (
                            int(
                                forecast["forecast_history_steps"][
                                    0, slot_by_target[index]
                                ]
                            )
                            if index in slot_by_target
                            else None
                        ),
                    }
                    for index in missing
                ],
            }
        )
        for slot in torch.nonzero(
            (assigned >= 0) & forecast["forecast_valid_mask"], as_tuple=False
        ):
            row, actor = slot.tolist()
            index = int(assigned[row, actor])
            mask = target["future_mask"][row, index].cpu().numpy()
            if mask.all():
                traj = (
                    forecast["pred_trajs"][row, actor, :, :, :2].detach().cpu().numpy()
                )
                future = target["future_xy"][row, index].cpu().numpy()
                velocity = target["current_velocity"][row, index].cpu().numpy()
                track_center = (
                    online._last_output["track_box"][row, actor, :2]
                    .detach()
                    .cpu()
                    .numpy()
                )
                gt_center = target["current_boxes"][row, index, :2].cpu().numpy()
                rows.append((traj, future, mask, velocity, track_center, gt_center))
    gathered = [None] * world_size
    gathered_coverage = [None] * world_size
    dist.all_gather_object(gathered, rows)
    dist.all_gather_object(gathered_coverage, coverage)
    def build_report():
        rows = [x for group in gathered for x in group]
        pred = [x[0] for x in rows]
        target = [x[1] for x in rows]
        mask = [x[2] for x in rows]
        track_centers = [x[4] for x in rows]
        gt_centers = [x[5] for x in rows]
        stationary = [np.zeros((6, 50, 2), np.float32) for _ in rows]
        cv = [
            np.repeat(
                (np.arange(1, 51, dtype=np.float32)[:, None] * x[3] * 0.1)[None],
                6,
                axis=0,
            )
            for x in rows
        ]
        root = Path(__file__).resolve().parents[3]
        digest = hashlib.sha256()
        for relative in sorted(
            (
                "opencood/tools/dualcmp/eval_forecast_ddp.py",
                "opencood/tools/dualcmp/forecast_metrics.py",
                "opencood/tools/dualcmp/online.py",
            )
        ):
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update((root / relative).read_bytes())
        coverage = [item for group in gathered_coverage for item in group]
        report = {
            "checkpoint_provenance": checkpoint_provenance,
            "partner_delay_ms": a.partner_delay_ms,
            "forecast_representation": ablation["forecast_representation"],
            "capture_alignment": ablation["capture_alignment"],
            "age_encoding": ablation["age_encoding"],
            "mean_source_aggregation": ablation["mean_source_aggregation"],
            "evaluator_tree_sha256": digest.hexdigest(),
            "coverage": coverage,
            "coverage_totals": {
                "complete": sum(item["complete"] for item in coverage),
                "assigned_complete": sum(
                    item["assigned_complete"] for item in coverage
                ),
                "eligible_complete": sum(
                    item["eligible_complete"] for item in coverage
                ),
            },
            "model": cmp_metrics(
                pred,
                target,
                mask,
                prediction_centers=track_centers,
                target_centers=gt_centers,
            ),
            "stationary": cmp_metrics(stationary, target, mask),
            "cv": cmp_metrics(cv, target, mask),
        }
        if model.forecast_representation == "source_temporal_aligned":
            report.update(aligned_evidence_policy=model.aligned_evidence_policy,
                          source_feature_encoder=model.source_feature_encoder,
                          forecast_normalization=model.forecast_head.forecast_normalization)
        Path(a.output).write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
        return report
    report = _rank_zero_evaluation(build_report)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
