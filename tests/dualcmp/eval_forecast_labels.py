"""Fresh two-GPU train-window evaluation using the existing CMP evaluator.

Only dataset selection differs from eval_forecast_ddp's test path. This is
development learning evidence, never a substitute for held-out test metrics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from opencood.data_utils.dualcmp_dataloader import build_dataloader
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES, load_checkpoint_hypes
from opencood.tools.dualcmp.eval_forecast_ddp import (
    _forecast_rows, _forecast_report, _rank_zero_evaluation,
    _forecast_rows_by_policy, _validated_forecast_report,
)
from opencood.loss.forecast_association import LEGACY_ASSOCIATION, VALIDATED_ASSOCIATION, ASSOCIATION_POLICIES, association_contract, forecast_metric_version
from opencood.tools.dualcmp.forecast_metrics import horizon_metrics
from opencood.tools.dualcmp.online import _checkpoint_provenance, _model_from_checkpoint
from opencood.loss.dualcmp_loss import update_identity_bindings
from opencood.tools.dualcmp.protocol import (
    DELAY_FORECAST_COMPLETE_TARGETS,
    DELAY_FORECAST_SAMPLES,
    DELAY_FORECAST_WINDOW,
    TRAIN_LABEL_WINDOW,
)
from opencood.tools.dualcmp.temporal_protocol import (
    checkpoint_forecast_contract, validate_evaluation_delay_contract,
)
from opencood.tools.train_utils import to_device
from opencood.tools.dualcmp.aligned_protocol import ALIGNED_FORECAST_WINDOW


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--delay-window", action="store_true")
    parser.add_argument("--shuffle-source-time", action="store_true")
    parser.add_argument("--partner-delay-ms", type=float, default=0.0)
    parser.add_argument("--evaluation-association-policy", choices=ASSOCIATION_POLICIES,
                        help="explicit diagnostic evaluator override; never changes checkpoint training provenance")
    parser.add_argument("--export-actor-metrics", action="store_true",
                        help="retain actor IDs and exact metrics for matched-cohort regression comparisons")
    args = parser.parse_args()
    if not np.isfinite(args.partner_delay_ms) or args.partner_delay_ms < 0:
        raise ValueError("partner delay must be finite and non-negative")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("native forecast evaluation requires torchrun")
    provenance = _checkpoint_provenance(args.checkpoint, allow_dirty_smoke=True)
    payload = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_run = payload.get("provenance", {})
    training_association_policy = checkpoint_run.get("forecast_association", association_contract(LEGACY_ASSOCIATION))["policy"]
    association_policy = args.evaluation_association_policy or training_association_policy
    association_contract(association_policy)
    ablation = checkpoint_forecast_contract(checkpoint_run)
    validate_evaluation_delay_contract(
        checkpoint_run, args.partner_delay_ms,
        context="development forecast evaluation",
        forecast_representation=ablation["forecast_representation"],
        capture_alignment=ablation["capture_alignment"],
        age_encoding=ablation["age_encoding"],
        mean_source_aggregation=ablation["mean_source_aggregation"],
    )
    hypes, _ = load_checkpoint_hypes(DEFAULT_HYPES, payload)
    hypes["root_dir"] = str(Path(os.environ["V2V4REAL_ROOT"]) / "train")
    hypes["validate_dir"] = hypes["root_dir"]
    hypes["fusion"] = {"args": {
        "stage": "forecast", "tracking_frames": 3,
        "forecast_history_frames": 10, "future_frames": 50,
        "runtime_only": False, "runtime_forbid_gt": True,
        "receiver_ids": [0, 1], "max_actors": 128, "max_cav": 2,
        "tracking_label_root": os.environ["DUALCMP_TRAIN_TRACKING_LABEL_ROOT"],
        "tracking_label_layout": "dmstrack_train", "split_name": "train",
        "partner_delay_ms": args.partner_delay_ms,
        "window": list((ALIGNED_FORECAST_WINDOW if ablation["forecast_representation"] == "source_temporal_aligned"
                        else DELAY_FORECAST_WINDOW) if args.delay_window
                       else TRAIN_LABEL_WINDOW),
        "shuffle_training_points": False,
        "include_forecast_association_history": association_policy == VALIDATED_ASSOCIATION,
        "forecast_coordinate_frame": "current_receiver" if ablation["forecast_representation"] == "source_temporal_aligned" else "legacy_capture",
    }}
    dataset = DualCMPSequenceDataset(hypes, train=False)
    expected_samples = DELAY_FORECAST_SAMPLES if args.delay_window else 10
    if len(dataset) != expected_samples:
        raise RuntimeError(
            "native forecast evaluation requires exactly %d samples"
            % expected_samples)
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    _, loader, _ = build_dataloader(
        dataset, args.batch_size, True, workers=args.workers, rank=dist.get_rank(),
        world_size=world_size, seed=0, training=False,
        collate_fn=dataset.collate_batch_test, pin_memory=True)
    model = _model_from_checkpoint(hypes, args.checkpoint, device, forecast_args={
        "history_frames": 11, "future_frames": 50, "modes": 6,
        "intention_points_file": "preprocessed_data/v2v4real/v2v4real_cluster_64_center_dict.pkl",
        "representation": ablation["forecast_representation"],
        "capture_alignment": ablation["capture_alignment"],
        "age_encoding": ablation["age_encoding"],
        "mean_source_aggregation": ablation["mean_source_aggregation"],
    }, allow_dirty_smoke=True)
    rows, coverage, temporal_history, mode_diagnostics, association_records, legacy_rows, legacy_coverage = [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            keys = [(str(scene), int(receiver), int(frame))
                    for (scene, receiver), frame in zip(
                        batch["scene_receiver_keys"],
                        batch["runtime_frames"][-1]["receiver_frame_idx"])]
            batch = to_device(batch, device)
            outputs, _ = model(
                batch["runtime_frames"], run_forecast=True,
                shuffle_source_time=args.shuffle_source_time)
            local_rows, local_coverage, local_records, assigned = _forecast_rows_by_policy(
                outputs, batch, keys, association_policy)
            rows.extend(local_rows)
            coverage.extend(local_coverage)
            association_records.extend(local_records)
            if association_policy == VALIDATED_ASSOCIATION:
                raw_rows, raw_coverage = _forecast_rows(outputs[-1], batch["target_frames"][-1], keys)
                legacy_rows.extend(raw_rows); legacy_coverage.extend(raw_coverage)
            forecast = outputs[-1]["forecast"]
            if "source_temporal_token_mask" in forecast:
                age = forecast["source_temporal_age_s"]
                mask = forecast["source_temporal_token_mask"].reshape(
                    *age.shape, -1
                ).any(dim=-1)
                for row, slot in torch.nonzero(
                        forecast["forecast_valid_mask"], as_tuple=False).tolist():
                    temporal_history.append({
                        "valid_steps_per_source": mask[row, slot].sum(-1).cpu().tolist(),
                        "valid_sources": int(mask[row, slot].any(-1).sum().item()),
                    })
            target = batch["target_frames"][-1]
            for row, slot in torch.nonzero(
                    (assigned >= 0) & forecast["forecast_valid_mask"],
                    as_tuple=False).tolist():
                target_index = int(assigned[row, slot])
                if not bool(target["future_mask"][row, target_index].all()):
                    continue
                receiver_target = (
                    target["current_boxes"][row, target_index, None, :2]
                    + target["future_xy"][row, target_index]
                    - outputs[-1]["track_box"][row, slot, None, :2]
                )
                heading = outputs[-1]["track_box"][row, slot, 6]
                if model.forecast_head.forecast_coordinate_frame == "legacy_capture":
                    heading = torch.rad2deg(heading)
                cosine, sine = torch.cos(heading), torch.sin(heading)
                actor_target = torch.stack((
                    cosine * receiver_target[:, 0]
                    + sine * receiver_target[:, 1],
                    -sine * receiver_target[:, 0]
                    + cosine * receiver_target[:, 1],
                ), dim=-1)
                raw = forecast["loss_pred_trajs_local"][row, slot, ..., :2]
                selected = forecast["pred_trajs_local"][row, slot, ..., :2]
                raw_ade = torch.linalg.vector_norm(
                    raw - actor_target[None], dim=-1).mean(-1)
                selected_ade = torch.linalg.vector_norm(
                    selected - actor_target[None], dim=-1).mean(-1)
                top = int(forecast["loss_pred_scores"][row, slot].argmax())
                mode_diagnostics.append({
                    "best_raw64_ade_5s": float(raw_ade.min()),
                    "top_raw_ade_5s": float(raw_ade[top]),
                    "best_selected6_ade_5s": float(selected_ade.min()),
                    "top_raw_mean_std": float(forecast[
                        "loss_pred_trajs_local"][row, slot, top, :, 2:4]
                        .clamp(-1.609, 5.0).exp().mean()),
                })
    gathered = [None] * world_size
    dist.all_gather_object(
        gathered, (rows, coverage, temporal_history, mode_diagnostics,
                   association_records, legacy_rows, legacy_coverage))
    def build_report():
        all_rows, all_coverage = [row for group in gathered for row in group[0]], [row for group in gathered for row in group[1]]
        if association_policy == VALIDATED_ASSOCIATION:
            report = _validated_forecast_report(all_rows, all_coverage, [row for group in gathered for row in group[4]])
            raw_rows = [row for group in gathered for row in group[5]]
            if raw_rows:
                report["legacy_unbounded_diagnostic"] = _forecast_report(raw_rows, [row for group in gathered for row in group[6]])
        else:
            report = _forecast_report(all_rows, all_coverage)
            report["forecast_association"] = association_contract(association_policy)
            report["metric_version"] = forecast_metric_version(association_policy)
        repository = Path(__file__).resolve().parents[2]
        evaluator_digest = hashlib.sha256()
        for relative in (
                "opencood/tools/dualcmp/eval_forecast_ddp.py",
                "opencood/tools/dualcmp/forecast_metrics.py",
                "opencood/loss/forecast_association.py",
                "tests/dualcmp/eval_forecast_labels.py"):
            evaluator_digest.update(relative.encode("utf-8") + b"\0")
            evaluator_digest.update((repository / relative).read_bytes())
        window = ((ALIGNED_FORECAST_WINDOW if ablation["forecast_representation"] == "source_temporal_aligned"
                   else DELAY_FORECAST_WINDOW) if args.delay_window else TRAIN_LABEL_WINDOW)
        report["sequence_policy"] = dataset.sequence_policy
        report["forecast_supervision"] = model.forecast_supervision
        report["forecast_variance"] = model.forecast_head.forecast_variance
        report["training_forecast_association"] = association_contract(training_association_policy)
        report["evaluation_association_override"] = args.evaluation_association_policy
        if args.export_actor_metrics:
            def actor_metrics(values):
                unique = {item[0]: item for item in values}
                return [{"key": list(key),
                         "prediction_center": item[5].tolist(),
                         "target_center": item[6].tolist(),
                         "current_velocity": item[4].tolist(),
                         "metrics": horizon_metrics([item[1]], [item[2]], [item[3]], [item[5]], [item[6]])}
                        for key, item in sorted(unique.items())]
            report["actor_metrics"] = actor_metrics(all_rows)
            report["legacy_actor_metrics"] = actor_metrics(
                [item for group in gathered for item in group[5]]
                if association_policy == VALIDATED_ASSOCIATION else all_rows)
        if model.forecast_representation == "source_temporal_aligned":
            report.update(aligned_evidence_policy=model.aligned_evidence_policy,
                          source_feature_encoder=model.source_feature_encoder,
                          forecast_normalization=model.forecast_head.forecast_normalization)
        report.update({"scope": "delay-window" if args.delay_window else "train-window",
                       "window": list(window),
                       "dataset_samples": len(dataset), "world_size": world_size,
                       "per_rank_batch": args.batch_size,
                       "partner_delay_ms": args.partner_delay_ms,
                       "forecast_representation": ablation["forecast_representation"],
                       "capture_alignment": ablation["capture_alignment"],
                       "age_encoding": ablation["age_encoding"],
                       "mean_source_aggregation": (
                           ablation["mean_source_aggregation"]),
                       "shuffle_source_time": args.shuffle_source_time,
                       "evaluator_tree_sha256": evaluator_digest.hexdigest(),
                       "checkpoint_provenance": provenance})
        histories = [item for group in gathered for item in group[2]]
        if histories:
            report["source_temporal_history"] = {
                "forecast_actors": len(histories),
                "actors_with_multiple_steps": sum(
                    max(item["valid_steps_per_source"], default=0) > 1
                    for item in histories
                ),
                "max_valid_steps": max(
                    max(item["valid_steps_per_source"], default=0)
                    for item in histories
                ),
                "min_valid_sources": min(
                    item["valid_sources"] for item in histories
                ),
                "max_valid_sources": max(
                    item["valid_sources"] for item in histories
                ),
            }
        diagnostics = [item for group in gathered for item in group[3]]
        if diagnostics:
            report["mode_diagnostics"] = {"actors": len(diagnostics)}
            report["mode_diagnostics"].update({
                key: float(np.mean([item[key] for item in diagnostics]))
                for key in diagnostics[0]
            })
        if report["coverage"]["prediction_keys"] != len(dataset):
            raise RuntimeError("fresh evaluation missed native train-window samples")
        if (args.delay_window and ablation["forecast_representation"] != "source_temporal_aligned"
                and report["coverage"]["complete"]
                != DELAY_FORECAST_COMPLETE_TARGETS):
            raise RuntimeError("delay-window complete-target contract changed")
        text = json.dumps(report, indent=2, allow_nan=False)
        Path(args.output).write_text(text, encoding="utf-8")
        print(text, flush=True)
        return report
    _rank_zero_evaluation(build_report)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
