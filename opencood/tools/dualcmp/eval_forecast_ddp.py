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
from opencood.tools.dualcmp.forecast_metrics import cmp_metrics
from opencood.tools.dualcmp.online import OnlineDualCMP, _model_from_checkpoint
from opencood.tools.dualcmp.protocol import (
    DETECTOR_COMPRESSION,
    FORECAST_WINDOW,
    FORECAST_SAMPLES,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    dist.init_process_group("nccl")
    rank, local = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    h = load_yaml("opencood/hypes_yaml/point_pillar_cobevt_multiego_1x_v2v4real.yaml")
    h["root_dir"] = str(Path(os.environ["V2V4REAL_ROOT"]) / "test")
    h["validate_dir"] = h["root_dir"]
    h["model"]["args"]["compression"] = DETECTOR_COMPRESSION
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
    }
    online = OnlineDualCMP(
        _model_from_checkpoint(
            h,
            a.checkpoint,
            torch.device("cuda", local),
            forecast_args=fa,
            allow_dirty_smoke=True,
        ),
        torch.device("cuda", local),
    )
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
                rows.append((traj, future, mask, velocity))
    gathered = [None, None]
    gathered_coverage = [None, None]
    dist.all_gather_object(gathered, rows)
    dist.all_gather_object(gathered_coverage, coverage)
    if rank == 0:
        rows = [x for group in gathered for x in group]
        pred = [x[0] for x in rows]
        target = [x[1] for x in rows]
        mask = [x[2] for x in rows]
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
            "model": cmp_metrics(pred, target, mask),
            "stationary": cmp_metrics(stationary, target, mask),
            "cv": cmp_metrics(cv, target, mask),
        }
        Path(a.output).write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
