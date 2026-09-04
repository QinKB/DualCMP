"""Fresh two-rank CMP forecast evaluation for the frozen 16-sample protocol."""

from __future__ import annotations
import argparse, json, os, hashlib
from pathlib import Path
import numpy as np
import torch, torch.distributed as dist
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import (
    DualCMPSequenceDataset,
)
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.loss.dualcmp_loss import update_identity_bindings
from opencood.data_utils.dualcmp_dataloader import build_dataloader
from opencood.tools.train_utils import to_device
from opencood.tools.dualcmp.forecast_metrics import cmp_metrics
from opencood.tools.dualcmp.online import (
    OnlineDualCMP,
    _checkpoint_provenance,
    _model_from_checkpoint,
)


def _forecast_rows(output, target, keys):
    """Existing target-only binding and CMP row extraction, keyed for dedupe."""
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
    return {"coverage": {"prediction_keys": len(coverage_unique), "complete": sum(x["complete"] for x in coverage_unique.values()),
                           "eligible_complete": sum(x["eligible_complete"] for x in coverage_unique.values())},
            "model": cmp_metrics(prediction, target, mask, prediction_centers=[x[5] for x in values], target_centers=[x[6] for x in values]),
            "stationary": cmp_metrics(stationary, target, mask), "cv": cmp_metrics(cv, target, mask)}


def evaluate_forecast_test_model_ddp(model, hypes, device, root, labels,
                                     batch_size=1, workers=0, full_test=True):
    """Evaluate the current model over full test or the frozen smoke window."""
    if not dist.is_initialized() or dist.get_world_size() != 2:
        raise RuntimeError("forecast evaluation requires two initialized DDP processes")
    h = dict(hypes)
    h["root_dir"], h["validate_dir"] = str(Path(root) / "test"), str(Path(root) / "test")
    h["fusion"] = {"args": {"stage": "forecast", "tracking_frames": 3,
        "forecast_history_frames": 10, "future_frames": 50, "runtime_only": False,
        "runtime_forbid_gt": True, "receiver_ids": [0, 1], "max_actors": 128,
        "max_cav": 2, "tracking_label_root": labels, "split_name": "test",
        "shuffle_training_points": False}}
    if not full_test:
        h["fusion"]["args"]["window"] = list(FORECAST_WINDOW)
    dataset = DualCMPSequenceDataset(h, train=False)
    if not full_test and len(dataset) != FORECAST_SAMPLES:
        raise RuntimeError("fixed forecast evaluation requires 16 samples")
    _, loader, _ = build_dataloader(dataset, batch_size, True, workers=workers,
        rank=dist.get_rank(), world_size=dist.get_world_size(), seed=0, training=False,
        collate_fn=dataset.collate_batch_test, pin_memory=True)
    modes = [(module, module.training) for module in model.modules()]
    model.eval(); rows, coverage = [], []
    with torch.no_grad():
        for batch in loader:
            keys = [(str(scene), int(receiver), int(frame)) for (scene, receiver), frame in zip(
                batch["scene_receiver_keys"], batch["runtime_frames"][-1]["receiver_frame_idx"])]
            batch = to_device(batch, device)
            outputs, _ = model(batch["runtime_frames"], run_forecast=True)
            local_rows, local_coverage = _forecast_rows(outputs[-1], batch["target_frames"][-1], keys)
            rows.extend(local_rows); coverage.extend(local_coverage)
    gathered, gathered_coverage = [None] * dist.get_world_size(), [None] * dist.get_world_size()
    dist.all_gather_object(gathered, rows); dist.all_gather_object(gathered_coverage, coverage)
    if dist.get_rank() == 0:
        report = _forecast_report([x for group in gathered for x in group], [x for group in gathered_coverage for x in group])
    else:
        report = None
    payload = [report]
    dist.broadcast_object_list(payload, src=0)
    for module, training in modes:
        module.train(training)
    if any(module.training != training for module, training in modes):
        raise RuntimeError("forecast evaluator failed to restore module training modes")
    return payload[0]
from opencood.tools.dualcmp.protocol import (
    FORECAST_WINDOW,
    FORECAST_SAMPLES,
    checkpoint_detector_compression,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--disable-source-evidence", action="store_true")
    p.add_argument("--full-test", action="store_true")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    checkpoint_provenance = _checkpoint_provenance(
        a.checkpoint, allow_dirty_smoke=True
    )
    checkpoint_payload = torch.load(a.checkpoint, map_location="cpu")
    dist.init_process_group("nccl")
    rank, local = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    h = load_yaml("opencood/hypes_yaml/point_pillar_cobevt_multiego_1x_v2v4real.yaml")
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
            "tracking_label_root": os.environ["DUALCMP_TRACKING_LABEL_ROOT"],
            "window": list(FORECAST_WINDOW),
            "shuffle_training_points": False,
        }
    }
    d = DualCMPSequenceDataset(h, train=True)
    assert len(d) == FORECAST_SAMPLES
    fa = {
        "history_frames": 11,
        "future_frames": 50,
        "modes": 6,
        "intention_points_file": "preprocessed_data/v2v4real/v2v4real_cluster_64_center_dict.pkl",
        "source_evidence": not a.disable_source_evidence,
    }
    model = _model_from_checkpoint(
        h, a.checkpoint, torch.device("cuda", local), forecast_args=fa,
        allow_dirty_smoke=True,
    )
    if a.full_test:
        report = evaluate_forecast_test_model_ddp(
            model, h, torch.device("cuda", local),
            os.environ["V2V4REAL_ROOT"],
            os.environ["DUALCMP_TRACKING_LABEL_ROOT"],
            batch_size=a.batch_size, workers=a.workers, full_test=True,
        )
        if rank == 0:
            report["checkpoint_provenance"] = checkpoint_provenance
            Path(a.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report))
        dist.barrier(); dist.destroy_process_group()
        return
    online = OnlineDualCMP(model, torch.device("cuda", local))
    rows = []
    coverage = []
    for i in range(rank, len(d), 2):
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
    gathered = [None, None]
    gathered_coverage = [None, None]
    dist.all_gather_object(gathered, rows)
    dist.all_gather_object(gathered_coverage, coverage)
    if rank == 0:
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
        Path(a.output).write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
