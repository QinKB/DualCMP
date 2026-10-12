"""Frozen-checkpoint training/eval normalization diagnostic; no optimization."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from opencood.data_utils.dualcmp_dataloader import build_dataloader
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
from opencood.tools.dualcmp.aligned_protocol import ALIGNED_FORECAST_WINDOW
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES, load_checkpoint_hypes
from opencood.tools.dualcmp.eval_forecast_ddp import (
    _forecast_report, _forecast_rows_by_policy, _validated_forecast_report,
)
from opencood.loss.forecast_association import LEGACY_ASSOCIATION, VALIDATED_ASSOCIATION, association_contract
from opencood.tools.dualcmp.train_forecast_ddp import _snapshot_training_normalization, _restore_training_normalization
from opencood.tools.dualcmp.online import _model_from_checkpoint
from opencood.tools.train_utils import to_device
from opencood.loss.dualcmp_loss import update_identity_bindings
from audit_task_fitness_ddp import capture_forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("refusing to overwrite learning diagnostic evidence")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("diagnostic requires two GPUs")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    payload = torch.load(args.checkpoint, map_location="cpu")
    policy = payload['provenance'].get('forecast_association', association_contract(LEGACY_ASSOCIATION))['policy']
    h, _ = load_checkpoint_hypes(DEFAULT_HYPES, payload)
    h.update(root_dir=str(Path(os.environ["V2V4REAL_ROOT"]) / "train"),
             validate_dir=str(Path(os.environ["V2V4REAL_ROOT"]) / "train"))
    h["fusion"] = {"args": dict(stage="forecast", forecast_history_frames=10,
        future_frames=50, runtime_only=False, runtime_forbid_gt=True,
        receiver_ids=[0, 1], max_actors=128, max_cav=2, split_name="train",
        tracking_label_root=os.environ["DUALCMP_TRAIN_TRACKING_LABEL_ROOT"],
        tracking_label_layout="dmstrack_train", partner_delay_ms=100.,
        shuffle_training_points=False, forecast_coordinate_frame="current_receiver",
        include_forecast_association_history=policy == VALIDATED_ASSOCIATION,
        window=list(ALIGNED_FORECAST_WINDOW))}
    dataset = DualCMPSequenceDataset(h, train=False)
    _, loader, _ = build_dataloader(dataset, 2, True, workers=2, rank=rank,
        world_size=2, training=False, collate_fn=dataset.collate_batch_test)
    model = _model_from_checkpoint(h, args.checkpoint, device, allow_dirty_smoke=True)
    rows = {name: [] for name in ("checkpoint_running_stats", "local_batch_stats")}
    coverage = []
    association_records = []
    uncertainty = []
    with torch.no_grad():
        for batch in loader:
            keys = [(str(scene), int(receiver), int(frame)) for (scene, receiver), frame in zip(
                batch["scene_receiver_keys"], batch["runtime_frames"][-1]["receiver_frame_idx"])]
            batch = to_device(batch, device)
            captured = []
            original = capture_forward(model.forecast_head, captured)
            try:
                outputs, _ = model(batch["runtime_frames"], run_forecast=True)
            finally:
                model.forecast_head.forward = original
            normal, local_coverage, local_records, assigned = _forecast_rows_by_policy(outputs, batch, keys, policy)
            rows["checkpoint_running_stats"].extend(normal)
            coverage.extend(local_coverage)
            association_records.extend(local_records)
            target, output = batch["target_frames"][-1], outputs[-1]
            forecast = output["forecast"]
            for row, slot in torch.nonzero((assigned >= 0) & forecast["forecast_valid_mask"], as_tuple=False).tolist():
                gt_index = int(assigned[row, slot])
                if not bool(target["future_mask"][row, gt_index].all()):
                    continue
                physical = (target["future_xy"][row, gt_index]
                            + target["current_boxes"][row, gt_index, None, :2]
                            - output["track_box"][row, slot, None, :2])
                heading = output["track_box"][row, slot, 6]
                c, s = heading.cos(), heading.sin()
                actor_target = torch.stack((c * physical[:, 0] + s * physical[:, 1],
                                            -s * physical[:, 0] + c * physical[:, 1]), dim=-1)
                layers = []
                for layer in range(forecast["loss_pred_trajs_local_layers"].shape[2]):
                    points = forecast["loss_intention_points_layers"][row, slot, layer]
                    mode = int(torch.linalg.vector_norm(points - actor_target[-1], dim=-1).argmin())
                    pred = forecast["loss_pred_trajs_local_layers"][row, slot, layer, mode]
                    error = torch.linalg.vector_norm(pred[:, :2] - actor_target, dim=-1)
                    logs = pred[:, 2:4]
                    layers.append({"layer": layer, "supervised_mode": mode,
                                   "ade_5s": float(error.mean()), "fde_5s": float(error[-1]),
                                   "max_clamped_std_fraction": float((logs >= 5.).float().mean()),
                                   "mean_std_m": float(logs.clamp(-1.609, 5.).exp().mean()),
                                   "max_raw_log_std": float(logs.max())})
                uncertainty.append({"key": keys[row] + (int(target["actor_ids"][row, gt_index]),),
                                    "layers": layers})
            # Only change forecast BN's statistics source. Keep dropout off,
            # detector/tracker/Source memory/GT pairs and all weights fixed.
            modules = [(module, module.training) for module in model.forecast_head.modules()
                       if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
            for module, _ in modules:
                module.train()
            snapshot = _snapshot_training_normalization(model.forecast_head)
            try:
                changed = model.forecast_head(*captured[0], **captured[1])
            finally:
                _restore_training_normalization(snapshot)
                for module, training in modules:
                    module.train(training)
            changed_outputs = list(outputs)
            changed_outputs[-1] = dict(outputs[-1], forecast=changed)
            modified, _, _, _ = _forecast_rows_by_policy(changed_outputs, batch, keys, policy)
            rows["local_batch_stats"].extend(modified)
    gathered = [None, None]
    dist.all_gather_object(gathered, (rows, coverage, uncertainty, association_records))
    if rank == 0:
        result = {"scope": "same checkpoint/input/GT pairs; BN-statistics-only inference intervention, not deployable validation",
                  "checkpoint": args.checkpoint, "world_size": 2, "per_rank_batch": 2, "variants": {},
                  "supervised_mode_uncertainty": [item for group in gathered for item in group[2]]}
        all_coverage = [row for group in gathered for row in group[1]]
        for name in rows:
            values = [row for group in gathered for row in group[0][name]]
            report = (_validated_forecast_report(values, all_coverage, [row for group in gathered for row in group[3]])
                      if policy == VALIDATED_ASSOCIATION else _forecast_report(values, all_coverage))
            details = []
            for row in values:
                error = np.linalg.norm(row[1] + row[5][None, None] - row[2][None] - row[6][None, None], axis=-1)
                mode = int(error.mean(-1).argmin())
                details.append({"key": row[0], "future_endpoint_norm_m": float(np.linalg.norm(row[2][-1])),
                                "gt_max_step_m": float(np.linalg.norm(np.diff(row[2], axis=0), axis=-1).max()),
                                "best_ade_mode_error_curve_m": error[mode].tolist()})
            result["variants"][name] = {"report": report, "actors": details}
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({name: data["report"]["diagnostics"]["exact_horizon_model"]
                          for name, data in result["variants"].items()}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
