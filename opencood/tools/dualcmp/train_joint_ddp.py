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
from opencood.tools.train_utils import to_device


SOURCES = (
    "opencood/data_utils/v2v4real/datasets/dualcmp_sequence_dataset.py",
    "opencood/loss/dualcmp_loss.py",
    "opencood/models/dualcmp/model.py",
    "opencood/models/point_pillar_fax.py",
    "opencood/tools/dualcmp/train_joint_ddp.py",
)


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
        return {"source_state": "clean", "source_sha": code_sha.lower()}
    digest = hashlib.sha256()
    for relative in sorted(SOURCES):
        digest.update(relative.encode() + b"\0")
        digest.update((root / relative).read_bytes())
    return {"source_state": "dirty-smoke", "base_sha": head,
            "source_tree_sha256": digest.hexdigest()}


def _hypes(path, split, full_data):
    root = os.environ.get("V2V4REAL_ROOT")
    weights = os.environ.get("DUALCMP_INIT_WEIGHTS")
    labels = os.environ.get("DUALCMP_TRACKING_LABEL_ROOT")
    if not root or not weights:
        raise EnvironmentError("V2V4REAL_ROOT and DUALCMP_INIT_WEIGHTS are required")
    if not full_data and not labels:
        raise EnvironmentError("fixed acceptance also requires DUALCMP_TRACKING_LABEL_ROOT")
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
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
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
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    step = epoch = 0
    started = time.monotonic()
    last = None
    forecast_to_source_grad = None
    joint_detector_grad = None
    while (not args.max_steps) or step < args.max_steps:
        sampler.set_epoch(epoch)
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
            if args.max_steps and step >= args.max_steps:
                break
        epoch += 1
        if rank == 0 and args.save_every and epoch % args.save_every == 0:
            torch.save(
                {"model": model.module.state_dict(), "epoch": epoch,
                 "step": step, "provenance": _provenance()},
                output / ("checkpoint_epoch_%03d.pt" % epoch),
            )
        if (args.max_steps and step >= args.max_steps) or (not args.max_steps and epoch >= args.epochs):
            break
    _assert_sync(model.module)
    elapsed = max(time.monotonic() - started, 1e-9)
    provenance = _provenance()
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
        torch.save({"model": model.module.state_dict(), "provenance": provenance}, checkpoint)
        report["checkpoint_sha256"] = _sha256_file(checkpoint)
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
