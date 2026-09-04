"""Two-GPU Phase-C joint fine-tuning from an accepted Phase-B checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

from opencood.data_utils.dualcmp_dataloader import build_dataloader
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import (
    DualCMPSequenceDataset,
)
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.loss.dualcmp_loss import DualCmpLoss
from opencood.models.dualcmp.model import DualCmpClipModel
from opencood.tools.dualcmp.protocol import (
    FORECAST_SAMPLES,
    FORECAST_WINDOW,
    checkpoint_detector_compression,
)
from opencood.tools.dualcmp.train_tracking_ddp import _assert_sync, _sha256_file
from opencood.tools.dualcmp.train_tracking_ddp import _append_csv, _atomic_save, _trim_csv_after_checkpoint
from opencood.tools.dualcmp.eval_tracking_ddp import evaluate_tracking_model_ddp, evaluate_tracking_test_model_ddp
from opencood.tools.dualcmp.eval_forecast_ddp import evaluate_forecast_test_model_ddp
from opencood.tools.train_utils import to_device


SOURCES = (
    "opencood/data_utils/v2v4real/datasets/dualcmp_sequence_dataset.py",
    "opencood/loss/dualcmp_loss.py",
    "opencood/models/dualcmp/model.py",
    "opencood/models/point_pillar_fax.py",
    "opencood/tools/dualcmp/ab3dmot_metrics.py",
    "opencood/tools/dualcmp/eval_forecast_ddp.py",
    "opencood/tools/dualcmp/eval_tracking_ddp.py",
    "opencood/tools/dualcmp/forecast_metrics.py",
    "opencood/tools/dualcmp/protocol.py",
    "opencood/tools/dualcmp/train_joint_ddp.py",
)
FORECAST_METRICS = ("minADE_1s", "minADE_3s", "minADE_5s",
                    "minFDE_1s", "minFDE_3s", "minFDE_5s",
                    "MR_3.6", "count")
TRACKING_METRICS = ("MOTA", "AMOTA", "sAMOTA", "tracking_recall",
                    "tracking_precision", "identity_switches", "fragments")
CSV_FIELDS = (("kind", "split", "epoch", "global_step", "global_samples",
               "lr_detector", "lr_tracking", "lr_forecast",
               "L_det", "L_track", "L_pred", "L_total")
              + tuple("test_track_" + key for key in TRACKING_METRICS)
              + ("test_track_timelines", "test_track_frames",)
              + tuple("test_forecast_" + key for key in FORECAST_METRICS)
              + ("test_prediction_keys", "test_complete", "test_eligible_complete"))


def _provenance():
    root = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root)).decode().strip()
    code_sha = os.environ.get("DUALCMP_CODE_SHA")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *SOURCES], cwd=str(root)
    ).decode().strip()
    if code_sha:
        if not re.match(r"^[0-9a-fA-F]{40}$", code_sha) or code_sha.lower() != head.lower() or dirty:
            raise RuntimeError("DUALCMP_CODE_SHA requires matching clean Phase-C sources")
        return {"source_state": "clean", "source_sha": code_sha.lower(),
                "protocol_sha256": _sha256_file(Path(__file__).with_name("protocol.py"))}
    digest = hashlib.sha256()
    for relative in sorted(SOURCES):
        digest.update(relative.encode() + b"\0")
        digest.update((root / relative).read_bytes())
    return {"source_state": "dirty-smoke", "base_sha": head,
            "source_tree_sha256": digest.hexdigest(),
            "protocol_sha256": _sha256_file(Path(__file__).with_name("protocol.py"))}


def _hypes(path, split, full_data):
    root = os.environ.get("V2V4REAL_ROOT")
    weights = os.environ.get("DUALCMP_INIT_WEIGHTS")
    labels = os.environ.get("DUALCMP_TRACKING_LABEL_ROOT")
    if not root or not weights or not labels:
        raise EnvironmentError(
            "V2V4REAL_ROOT, DUALCMP_INIT_WEIGHTS, and "
            "DUALCMP_TRACKING_LABEL_ROOT are required"
        )
    payload = torch.load(str(weights), map_location="cpu")
    detector_compression = checkpoint_detector_compression(payload)
    h = load_yaml(path)
    h["root_dir"] = str(Path(root) / split)
    h["validate_dir"] = h["root_dir"]
    h["model"]["args"]["compression"] = detector_compression
    args = {
        "stage": "forecast", "tracking_frames": 3,
        "forecast_history_frames": 10, "future_frames": 50,
        "runtime_only": False, "runtime_forbid_gt": True,
        "receiver_ids": [0, 1], "max_actors": 128, "max_cav": 2,
        "split_name": split, "shuffle_training_points": False,
    }
    if labels:
        args["tracking_label_root"] = labels
    if not full_data:
        if split != "test":
            raise ValueError("the frozen Phase-C acceptance subset belongs to test")
        args["window"] = list(FORECAST_WINDOW)
    h["fusion"] = {"core_method": "DualCMPSequenceDataset", "args": args}
    return h, Path(weights), payload, detector_compression


def _build_optimizer(model, base_lr, detector_scale, forecast_scale, weight_decay):
    groups = {"detector": [], "tracking": [], "forecast": []}
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise AssertionError("optimizer parameter appears twice: %s" % name)
        seen.add(id(parameter))
        if name.startswith("detector."):
            group = "detector"
        elif name.startswith("forecast_head.") or name.startswith("actor_source_tokenizer."):
            group = "forecast"
        else:
            group = "tracking"
        groups[group].append(parameter)
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if seen != expected or not all(groups.values()):
        raise AssertionError("joint optimizer groups do not cover the trainable model")
    parameter_groups = [
        {"params": groups["detector"], "lr": base_lr * detector_scale,
         "group_name": "detector"},
        {"params": groups["tracking"], "lr": base_lr,
         "group_name": "tracking"},
        {"params": groups["forecast"], "lr": base_lr * forecast_scale,
         "group_name": "forecast"},
    ]
    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay), {
        name: sum(p.numel() for p in parameters) for name, parameters in groups.items()
    }


def main():
    parser = argparse.ArgumentParser(description="two-process DualCMP Phase-C trainer")
    parser.add_argument("--hypes", default="opencood/hypes_yaml/point_pillar_cobevt_multiego_1x_v2v4real.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--full-data", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=6,
                        help="independent clips per GPU")
    parser.add_argument("--workers", type=int, default=8,
                        help="DataLoader workers per GPU process")
    parser.add_argument("--base-lr", type=float, default=1e-5)
    parser.add_argument("--detector-lr-scale", type=float, default=0.1)
    parser.add_argument("--forecast-lr-scale", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lambda-det", type=float, default=1.0)
    parser.add_argument("--lambda-track", type=float, default=1.0)
    parser.add_argument("--lambda-pred", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--min-lr-ratio", type=float, default=1e-3)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if args.log_every < 1 or args.save_every < 1 or args.eval_every < 1:
        raise ValueError("Phase C logging, saving, and evaluation intervals must be positive")
    if args.full_data and (args.save_every != 1 or args.eval_every != 1):
        raise ValueError("formal Phase C requires save_every=1 and eval_every=1")
    if args.resume and Path(args.resume).resolve().parent != Path(args.output_dir).resolve():
        raise ValueError("--resume checkpoint must belong to --output-dir")
    if args.full_data and not os.environ.get("DUALCMP_CODE_SHA"):
        raise EnvironmentError("formal full-data Phase C requires DUALCMP_CODE_SHA")
    if min(args.base_lr, args.detector_lr_scale, args.forecast_lr_scale,
           args.lambda_det, args.lambda_track, args.lambda_pred) <= 0:
        raise ValueError("learning rates, scales, and loss weights must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("Phase C requires torchrun --nproc_per_node=2")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dist.init_process_group("nccl")
    rank, local_rank = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    h, init_weights, payload, detector_compression = _hypes(
        args.hypes, args.data_split, args.full_data
    )
    dataset = DualCMPSequenceDataset(h, train=True)
    if not args.full_data and len(dataset) != FORECAST_SAMPLES:
        raise RuntimeError("fixed Phase-C protocol requires 16 samples")
    _, loader, sampler = build_dataloader(
        dataset, args.batch_size, True, workers=args.workers, rank=rank,
        world_size=2, seed=args.seed, training=True,
        collate_fn=dataset.collate_batch_train, pin_memory=True,
    )
    forecast_args = {
        "history_frames": 11, "future_frames": 50, "modes": 6,
        "intention_points_file": "preprocessed_data/v2v4real/v2v4real_cluster_64_center_dict.pkl",
    }
    model = DualCmpClipModel(
        h["model"]["args"], dataset.anchor_box,
        h["preprocess"]["cav_lidar_range"], forecast_args=forecast_args,
    ).to(device)
    state = payload.get("model", payload.get("model_state_dict", payload))
    state = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    optimizer, parameter_counts = _build_optimizer(
        model, args.base_lr, args.detector_lr_scale,
        args.forecast_lr_scale, args.weight_decay,
    )
    model.train()
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    criterion = DualCmpLoss({"detector": h["loss"]["args"]}).to(device)
    run_provenance = _provenance()
    total_steps = (
        args.epochs * len(loader)
        if args.full_data else (args.max_steps or args.epochs * len(loader))
    )
    warmup_steps = min(args.warmup_steps, max(total_steps - 1, 0))

    def multiplier(step):
        if warmup_steps and step < warmup_steps:
            return 1.0 / 3.0 + (2.0 / 3.0) * step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    output = Path(args.output_dir)
    start_epoch = step = global_samples = 0
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu")
        if not isinstance(resume_payload, dict) or any(key not in resume_payload for key in ("model", "optimizer", "scheduler", "epoch", "step", "provenance")):
            raise RuntimeError("resume checkpoint must contain model, optimizer, scheduler, epoch, step, and provenance")
        if int(resume_payload["provenance"].get("total_steps", -1)) != total_steps:
            raise RuntimeError("resume scheduler budget disagrees with requested 20-epoch schedule")
        if int(resume_payload["scheduler"].get("last_epoch", -1)) != int(resume_payload["step"]):
            raise RuntimeError("resume scheduler step does not match checkpoint step")
        previous = resume_payload["provenance"]
        source_key = "source_sha" if "source_sha" in run_provenance else "source_tree_sha256"
        if previous.get(source_key) != run_provenance.get(source_key):
            raise RuntimeError("Phase C resume requires the original source state")
        if previous.get("init_checkpoint_sha256") != _sha256_file(init_weights):
            raise RuntimeError("Phase C resume requires the original Phase-B checkpoint")
        model.module.load_state_dict(resume_payload["model"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"]); scheduler.load_state_dict(resume_payload["scheduler"])
        start_epoch, step, global_samples = int(resume_payload["epoch"]), int(resume_payload["step"]), int(resume_payload.get("global_samples", 0))
    if rank == 0:
        output.mkdir(parents=True, exist_ok=bool(args.resume))
        if args.resume:
            _trim_csv_after_checkpoint(output / "metrics.csv", step, CSV_FIELDS)
    dist.barrier()
    epoch = start_epoch
    started = time.monotonic()
    last = None
    forecast_to_source_grad = None
    joint_detector_grad = None
    while epoch < args.epochs:
        sampler.set_epoch(epoch)
        epoch_totals = np.zeros(4, dtype=np.float64); epoch_count = 0
        log_totals = np.zeros(4, dtype=np.float64); log_count = 0
        for batch in loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs, _ = model(batch["runtime_frames"], detector_autocast=args.amp,
                               run_forecast=True, current_frame_only_grad=True)
            loss = criterion([outputs[-1]], batch["target_frames"])
            counts = torch.stack((loss["C_gmm"], loss["C_mode"], loss["C_dense_gmm"])).detach()
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            pred = dist.get_world_size() * (
                loss["S_gmm"] / counts[0].clamp_min(1.0)
                + loss["S_mode"] / counts[1].clamp_min(1.0)
                + loss["S_dense_gmm"] / counts[2].clamp_min(1.0)
            )
            track = loss["L_exist"] + loss["L_box"] + loss["L_velocity"]
            total = args.lambda_det * loss["L_det"] + args.lambda_track * track + args.lambda_pred * pred
            if step == 0:
                source_grad = torch.autograd.grad(
                    pred, outputs[-1]["forecast_source_bev"], retain_graph=True
                )[0]
                source_value = float(source_grad.detach().float().norm())
                source_min = torch.tensor(source_value, device=device)
                dist.all_reduce(source_min, op=dist.ReduceOp.MIN)
                forecast_to_source_grad = float(source_min)
                if not forecast_to_source_grad > 0:
                    raise RuntimeError(
                        "forecast loss gradient to source BEV is zero: %.8g"
                        % forecast_to_source_grad
                    )
            total.backward()
            if step == 0:
                detector_value = sum(
                    float(parameter.grad.detach().float().norm())
                    for parameter in model.module.detector.parameters()
                    if parameter.grad is not None
                )
                detector_min = torch.tensor(detector_value, device=device)
                dist.all_reduce(detector_min, op=dist.ReduceOp.MIN)
                joint_detector_grad = float(detector_min)
                if not joint_detector_grad > 0:
                    raise RuntimeError(
                        "joint loss gradient to detector parameters is zero: %.8g"
                        % joint_detector_grad
                    )
            grad_norm = clip_grad_norm_(model.parameters(), 1000.0)
            optimizer.step(); scheduler.step(); step += 1
            values = torch.stack((loss["L_det"].detach(), track.detach(), pred.detach(), total.detach())).float()
            if not torch.isfinite(values).all():
                raise FloatingPointError("non-finite Phase-C loss")
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            last = (values / dist.get_world_size()).cpu().tolist()
            batch_count = len(batch["scene_receiver_keys"]) * dist.get_world_size()
            epoch_totals += np.asarray(last) * batch_count; epoch_count += batch_count
            log_totals += np.asarray(last) * batch_count; log_count += batch_count
            global_samples += batch_count
            if step % args.log_every == 0 and rank == 0:
                row = dict(zip(("L_det", "L_track", "L_pred", "L_total"), (log_totals / log_count).tolist()))
                row.update({"kind": "step", "split": "train", "epoch": epoch + 1, "global_step": step,
                            "global_samples": global_samples, "lr_detector": optimizer.param_groups[0]["lr"],
                            "lr_tracking": optimizer.param_groups[1]["lr"], "lr_forecast": optimizer.param_groups[2]["lr"]})
                _append_csv(output / "metrics.csv", row, CSV_FIELDS); print(json.dumps(row), flush=True)
                log_totals[:] = 0; log_count = 0
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
        provenance = dict(run_provenance); provenance.update({"detector_compression": detector_compression,
            "init_checkpoint_sha256": _sha256_file(init_weights), "total_steps": total_steps})
        checkpoint = {"model": model.module.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "epoch": epoch + 1, "step": step, "global_samples": global_samples, "provenance": provenance}
        if rank == 0 and (epoch + 1) % args.save_every == 0:
            _atomic_save(checkpoint, output / ("checkpoint_epoch_%03d.pt" % (epoch + 1)))
            _atomic_save(checkpoint, output / "checkpoint.pt")
        dist.barrier()
        tracking = forecast = None
        if (epoch + 1) % args.eval_every == 0:
            tracking = (evaluate_tracking_test_model_ddp if args.full_data else evaluate_tracking_model_ddp)(model, h, device, os.environ["V2V4REAL_ROOT"], os.environ["DUALCMP_TRACKING_LABEL_ROOT"])
            forecast = evaluate_forecast_test_model_ddp(
                model, h, device, os.environ["V2V4REAL_ROOT"],
                os.environ["DUALCMP_TRACKING_LABEL_ROOT"],
                batch_size=args.batch_size, workers=args.workers,
                full_test=args.full_data,
            )
        if rank == 0:
            row = dict(zip(("L_det", "L_track", "L_pred", "L_total"), (epoch_totals / epoch_count).tolist()))
            row.update({"kind": "epoch", "split": "train+test", "epoch": epoch + 1, "global_step": step,
                        "global_samples": global_samples, "lr_detector": optimizer.param_groups[0]["lr"],
                        "lr_tracking": optimizer.param_groups[1]["lr"], "lr_forecast": optimizer.param_groups[2]["lr"]})
            if tracking is not None:
                row.update({"test_track_" + key: tracking["metrics"][key]
                            for key in TRACKING_METRICS})
                row.update({"test_track_timelines": tracking.get("timeline_count", ""),
                            "test_track_frames": tracking.get("frame_count", "")})
                row.update({"test_forecast_" + key: forecast["model"][key]
                            for key in FORECAST_METRICS})
                row.update({"test_prediction_keys": forecast["coverage"]["prediction_keys"],
                            "test_complete": forecast["coverage"]["complete"],
                            "test_eligible_complete": forecast["coverage"]["eligible_complete"]})
                (output / ("test_tracking_epoch_%03d.json" % (epoch + 1))).write_text(json.dumps(tracking, indent=2), encoding="utf-8")
                (output / ("test_forecast_epoch_%03d.json" % (epoch + 1))).write_text(json.dumps(forecast, indent=2), encoding="utf-8")
            _append_csv(output / "metrics.csv", row, CSV_FIELDS); print(json.dumps(row), flush=True)
        epoch += 1
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            break
    _assert_sync(model.module)
    elapsed = max(time.monotonic() - started, 1e-9)
    provenance = dict(run_provenance)
    provenance.update({"detector_compression": detector_compression,
                       "init_checkpoint_sha256": _sha256_file(init_weights),
                       "init_provenance": payload.get("provenance") if isinstance(payload, dict) else None})
    report = {
        "steps": step, "epochs_completed": epoch, "dataset_samples": len(dataset),
        "loss": dict(zip(("L_det", "L_track", "L_pred", "L_total"), last or [])),
        "per_rank_batch": args.batch_size, "global_batch": args.batch_size * 2,
        "workers_per_rank": args.workers, "parameter_counts": parameter_counts,
        "optimizer": "AdamW", "base_lr": args.base_lr,
        "detector_compression": detector_compression,
        "group_lrs_initial": {"detector": args.base_lr * args.detector_lr_scale,
                              "tracking": args.base_lr,
                              "forecast": args.base_lr * args.forecast_lr_scale},
        "loss_weights": {"det": args.lambda_det, "track": args.lambda_track,
                         "pred": args.lambda_pred},
        "weight_decay": args.weight_decay,
        "scheduler": "linear-warmup-cosine",
        "warmup_steps": warmup_steps, "min_lr_ratio": args.min_lr_ratio,
        "last_group_lrs": {group["group_name"]: group["lr"] for group in optimizer.param_groups},
        "forecast_to_source_bev_grad_norm_min_rank": forecast_to_source_grad,
        "joint_detector_grad_norm_min_rank": joint_detector_grad,
        "last_grad_norm": float(grad_norm), "ddp_sync": True,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
        "clips_per_second_per_rank": step * args.batch_size / elapsed,
        "provenance": provenance,
    }
    if rank == 0:
        checkpoint = output / "checkpoint.pt"
        report["checkpoint_sha256"] = _sha256_file(checkpoint)
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
