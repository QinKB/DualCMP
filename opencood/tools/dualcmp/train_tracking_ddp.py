"""Multi-process Phase-A DDP trainer for fixed acceptance or full T=3 data.

Launch with ``torchrun --nproc_per_node=N``.  ``batch_size`` is clips per GPU;
the frozen window below has 44 receiver-centric clips (22 per receiver).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

from opencood.data_utils.dualcmp_dataloader import build_dataloader
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.loss.dualcmp_loss import DualCmpLoss
from opencood.models.dualcmp.model import DualCmpClipModel
from opencood.tools.train_utils import to_device
from opencood.tools.dualcmp.protocol import CLIPS_PER_RECEIVER, DETECTOR_COMPRESSION, TOTAL_CLIPS, WINDOW, WINDOW_FRAMES, TRAIN_LABEL_WINDOW
from opencood.tools.dualcmp.detector_config import (
    DEFAULT_HYPES, detector_provenance, inherited_detector_provenance,
    load_checkpoint_hypes,
)
from opencood.tools.dualcmp.eval_tracking_ddp import (
    evaluate_tracking_model_ddp,
    evaluate_tracking_test_model_ddp,
)

LOSS_KEYS = ("L_det", "L_exist", "L_box", "L_velocity")
OFFICIAL_DETECTOR_ACTIVE_KEYS = 243
OFFICIAL_DETECTOR_SHA256 = "5740a36f8a8613128f29049c789b81e669a815000f4e3a1cc362199e9c84ad6e"
PHASE_A_SOURCES = (
    "opencood/data_utils/dualcmp_dataloader.py",
    "opencood/data_utils/v2v4real/datasets/dualcmp_sequence_dataset.py",
    "opencood/data_utils/v2v4real/datasets/dualcmp_target_builder.py",
    "opencood/loss/dualcmp_loss.py",
    "opencood/loss/point_pillar_loss.py",
    "opencood/models/dualcmp/model.py",
    "opencood/models/dualcmp/box_refinement.py",
    "opencood/models/dualcmp/kalman.py",
    "opencood/models/dualcmp/state.py",
    "opencood/models/dualcmp/tracker.py",
    "opencood/models/point_pillar_fax.py",
    "opencood/tools/dualcmp/ab3dmot_metrics.py",
    "opencood/tools/dualcmp/eval_tracking_ddp.py",
    "opencood/tools/dualcmp/online.py",
    "opencood/tools/dualcmp/detector_config.py",
    "opencood/tools/dualcmp/protocol.py",
    "opencood/tools/dualcmp/train_tracking_ddp.py",
    "opencood/hypes_yaml/point_pillar_v2v4real.yaml",
)
CSV_FIELDS = ("kind", "split", "epoch", "global_step", "global_samples", "lr_trunk", "lr_heads",
              "L_det", "L_exist", "L_box", "L_velocity", "L_total",
              "test_MOTA", "test_AMOTA", "test_sAMOTA", "test_recall", "test_precision",
              "test_IDS", "test_fragments", "test_overall_pass")
_SHA1_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol_sha256():
    return _sha256_file(Path(__file__).resolve().with_name("protocol.py"))


def _required_code_sha(value=None):
    source_sha = os.environ.get("DUALCMP_CODE_SHA") if value is None else value
    if not source_sha or not _SHA1_RE.match(source_sha):
        raise EnvironmentError("DUALCMP_CODE_SHA must be a 40-character hexadecimal checkpoint commit SHA")
    root = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(root)
    ).decode("utf-8").strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *PHASE_A_SOURCES],
        cwd=str(root),
    ).decode("utf-8").strip()
    if source_sha.lower() != head.lower() or dirty:
        raise RuntimeError(
            "formal Phase A requires DUALCMP_CODE_SHA=HEAD and clean Phase-A sources"
        )
    return source_sha.lower()


def _run_provenance(args, world_size, gpu_names):
    """Record the explicit checkpoint source, never a dirty working-tree HEAD."""
    return {"source_sha": _required_code_sha(), "protocol_sha256": _protocol_sha256(), "stage": "tracking",
            "argv": list(sys.argv), "world_size": int(world_size),
            "gpu_names": list(gpu_names), "seed": int(args.seed),
            "epochs": int(args.epochs), "lr": float(args.lr), "amp": bool(args.amp),
            "per_rank_batch": int(args.batch_size), "global_batch": int(args.batch_size * world_size),
            "workers_per_rank": int(args.workers),
            "window_frames": WINDOW_FRAMES, "clips_per_receiver": CLIPS_PER_RECEIVER,
            "total_clips": TOTAL_CLIPS, "window": list(WINDOW)}


def _dirty_provenance(args, world_size, gpu_names):
    root = Path(__file__).resolve().parents[3]
    digest = _source_tree_sha256()
    return {"source_state": "dirty-smoke",
            "base_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(root)
            ).decode().strip(),
            "source_tree_sha256": digest,
            "protocol_sha256": _protocol_sha256(), "argv": list(sys.argv),
            "world_size": world_size, "gpu_names": gpu_names, "stage": "tracking",
            "seed": int(args.seed)}


def _source_tree_sha256():
    root = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for relative in sorted(PHASE_A_SOURCES):
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def _init_distributed():
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("DualCMP tracking training requires torchrun")
    dist.init_process_group(backend="nccl")
    rank, local_rank = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def _hypes(path, args=None):
    root = os.environ.get("V2V4REAL_ROOT")
    phase_a_weights = os.environ.get("DUALCMP_PHASE_A_INIT_WEIGHTS")
    detector_weights = os.environ.get("DUALCMP_DETECTOR_WEIGHTS")
    label_root = os.environ.get("DUALCMP_TRACKING_LABEL_ROOT")
    train_label_root = os.environ.get("DUALCMP_TRAIN_TRACKING_LABEL_ROOT")
    full_data = bool(args is not None and args.full_data)
    resume = getattr(args, "resume", None)
    detector_compression = int(getattr(args, "detector_compression", 32))
    if phase_a_weights:
        raise RuntimeError("Phase A rejects DUALCMP_PHASE_A_INIT_WEIGHTS; use official detector weights or same-run --resume")
    if not root or not label_root or not detector_weights:
        raise EnvironmentError("V2V4REAL_ROOT, DUALCMP_TRACKING_LABEL_ROOT, and DUALCMP_DETECTOR_WEIGHTS are required")
    if resume:
        if not detector_weights or phase_a_weights:
            raise EnvironmentError("Phase A resume requires DUALCMP_DETECTOR_WEIGHTS")
        if _sha256_file(detector_weights) != OFFICIAL_DETECTOR_SHA256:
            raise RuntimeError("official V2V4Real detector checkpoint SHA256 mismatch")
        resume_payload = torch.load(resume, map_location="cpu")
        inherited = inherited_detector_provenance(resume_payload)
        if not inherited:
            raise RuntimeError("Phase A cannot resume a legacy CMP detector checkpoint")
        if resume_payload["provenance"].get("detector_checkpoint_sha256") != _sha256_file(detector_weights):
            raise RuntimeError("Phase A resume detector checkpoint SHA256 mismatch")
    if not resume and detector_compression != 32:
        raise RuntimeError("new Phase A detector initialization requires compression 32")
    smoke = bool(getattr(args, "train_label_smoke", False))
    split = "train" if smoke else (args.data_split if full_data else "test")
    if (full_data or smoke) and split == "train" and not train_label_root:
        raise EnvironmentError("formal train split requires DUALCMP_TRAIN_TRACKING_LABEL_ROOT")
    if resume:
        hypes, _ = load_checkpoint_hypes(path, resume_payload)
    else:
        hypes = load_yaml(path)
    hypes["root_dir"] = str(Path(root) / split)
    hypes["validate_dir"] = hypes["root_dir"]
    if not resume:
        hypes["model"]["args"]["compression"] = detector_compression
    if int(hypes["model"]["args"]["compression"]) != 32:
        raise RuntimeError("Phase A detector configuration requires compression 32")
    hypes["fusion"] = {"core_method": "DualCMPSequenceDataset", "args": {
        "stage": "tracking", "tracking_frames": 3, "runtime_only": False,
        "runtime_forbid_gt": True, "receiver_ids": [0, 1], "frame_period_s": 0.1,
        "max_actors": 128, "max_cav": 2,
        "shuffle_training_points": full_data,
        "sequence_augmentation": full_data,
    }}
    if full_data or smoke:
        hypes["fusion"]["args"]["split_name"] = split
        if split == "train":
            hypes["fusion"]["args"].update({"tracking_label_root": train_label_root,
                                                "tracking_label_layout": "dmstrack_train"})
            if smoke:
                hypes["fusion"]["args"]["window"] = list(TRAIN_LABEL_WINDOW)
        elif split == "test" and label_root:
            hypes["fusion"]["args"]["tracking_label_root"] = label_root
    else:
        hypes["fusion"]["args"].update({
            "window": list(WINDOW), "tracking_label_root": label_root,
        })
    return hypes, Path(detector_weights)


def _load_official_detector(detector, checkpoint):
    if _sha256_file(checkpoint) != OFFICIAL_DETECTOR_SHA256:
        raise RuntimeError("official V2V4Real detector checkpoint SHA256 mismatch")
    payload = torch.load(str(checkpoint), map_location="cpu")
    state = payload.get("model", payload.get("model_state_dict", payload))
    if not isinstance(state, dict):
        raise RuntimeError("official detector checkpoint must contain a state dictionary")
    state = {(key[7:] if key.startswith("module.") else key): value
             for key, value in state.items()}
    if len(state) != OFFICIAL_DETECTOR_ACTIVE_KEYS:
        raise RuntimeError("official detector must contain exactly 243 state-dict keys")
    detector.load_state_dict(state, strict=True)
    return len(state)


def _optimizer(model, initial_lr, max_lr, weight_decay):
    trunk, heads = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("detector.") and not name.startswith(
            ("detector.cls_head.", "detector.reg_head.")
        ):
            trunk.append(parameter)
        else:
            heads.append(parameter)
    if not trunk or not heads:
        raise RuntimeError("Phase-A optimizer groups are incomplete")
    optimizer = torch.optim.AdamW(
        [
            {"params": trunk, "lr": initial_lr * 0.1, "group_name": "trunk"},
            {"params": heads, "lr": initial_lr, "group_name": "heads"},
        ],
        weight_decay=weight_decay, betas=(0.95, 0.99),
    )
    return optimizer, [max_lr * 0.1, max_lr]


def _assert_sync(model):
    # Two independent moments over every trainable tensor catch rank-local
    # optimizer drift without materializing a second full model copy.
    values = torch.zeros(2, device=next(model.parameters()).device)
    for parameter in model.parameters():
        flat = parameter.detach().float()
        values[0] += flat.sum()
        values[1] += flat.square().sum()
    gathered = [torch.zeros_like(values) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, values)
    if not all(torch.allclose(values, other) for other in gathered):
        raise RuntimeError("DDP parameters diverged across ranks")


def _sync_buffers(model):
    if not dist.is_initialized():
        raise RuntimeError("buffer synchronization requires initialized DDP")
    module = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for buffer in module.buffers():
            dist.broadcast(buffer, src=0)


def _global_loss(loss, batch_size, device):
    values = torch.stack([loss[key].detach().float() for key in LOSS_KEYS + ("total_loss",)])
    if not torch.isfinite(values).all():
        raise FloatingPointError("non-finite DualCMP loss")
    totals = torch.cat((values * float(batch_size), values.new_tensor([float(batch_size)])))
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return totals[:-1].cpu().tolist(), int(totals[-1].item())


def _append_csv(path, row, fieldnames=CSV_FIELDS):
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _trim_csv_after_checkpoint(path, step, fieldnames=CSV_FIELDS):
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([row for row in rows if int(row.get("global_step") or 0) <= step])


def _atomic_save(payload, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(str(temporary), str(path))


def train(args):
    if not args.dirty_smoke:
        _required_code_sha()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rank, local_rank, world_size = _init_distributed()
    contract = "dmstrack_train" if (args.full_data and args.data_split == "train") or args.train_label_smoke else "legacy"
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu")
        if resumed.get("provenance", {}).get("tracking_label_contract") != contract:
            raise RuntimeError("resume tracking label contract disagrees with requested training split")
    device = torch.device("cuda", local_rank)
    gpu_names = [None for _ in range(world_size)]
    dist.all_gather_object(gpu_names, torch.cuda.get_device_name(local_rank))
    provenance = (_dirty_provenance(args, world_size, gpu_names)
                  if args.dirty_smoke else _run_provenance(args, world_size, gpu_names))
    hypes, init_weights = _hypes(args.hypes, args)
    dataset = DualCMPSequenceDataset(hypes, train=True)
    per_receiver = {entry["receiver_id"]: 0 for entry in WINDOW}
    for record in dataset.clip_records:
        per_receiver[int(record.receiver_id)] += 1
    if (not args.full_data and (len(dataset) != TOTAL_CLIPS or
            any(value != CLIPS_PER_RECEIVER for value in per_receiver.values()))):
        raise RuntimeError("fixed protocol requires 24 frames/22 clips per receiver and 44 total: %s" % per_receiver)
    _, loader, sampler = build_dataloader(dataset, args.batch_size, True, workers=args.workers, rank=rank,
                                           world_size=world_size, seed=args.seed,
                                           training=True, collate_fn=dataset.collate_batch_train,
                                           pin_memory=True)
    model = DualCmpClipModel(
        hypes["model"]["args"], dataset.anchor_box,
        hypes["preprocess"]["cav_lidar_range"],
        proposal_score_threshold=hypes["postprocess"]["target_args"]["score_threshold"],
        proposal_nms_thresh=hypes["postprocess"]["nms_thresh"],
        emission_threshold=hypes["postprocess"]["target_args"]["score_threshold"],
    ).to(device)
    loaded_detector_keys = 0
    if not args.resume:
        loaded_detector_keys = _load_official_detector(model.detector, init_weights)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                broadcast_buffers=False)
    criterion = DualCmpLoss({"detector": hypes["loss"]["args"]}).to(device)
    optimizer, max_lrs = _optimizer(
        model.module, args.lr, args.max_lr, args.weight_decay
    )
    total_steps = (
        args.epochs * len(loader)
        if args.full_data else (args.max_steps or args.epochs * len(loader))
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lrs, total_steps=total_steps, pct_start=0.4,
        anneal_strategy="cos", div_factor=args.max_lr / args.lr,
        final_div_factor=1e4,
    )
    output = Path(args.output_dir)
    start_epoch = 0
    step = 0
    resume_payload = None
    if args.resume:
        resume_path = Path(args.resume)
        resume_payload = torch.load(str(resume_path), map_location="cpu")
        if not isinstance(resume_payload, dict) or "optimizer" not in resume_payload or "scheduler" not in resume_payload:
            raise RuntimeError("resume checkpoint must contain model, optimizer, and scheduler state")
        model.module.load_state_dict(resume_payload["model"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"])
        if int(resume_payload["scheduler"].get("total_steps", -1)) != total_steps:
            raise RuntimeError("resume scheduler total_steps disagrees with the requested formal epoch budget")
        scheduler.load_state_dict(resume_payload["scheduler"])
        start_epoch, step = int(resume_payload["epoch"]), int(resume_payload["step"])
        previous = resume_payload.get("provenance")
        source_keys = (("source_sha",) if provenance.get("source_state") != "dirty-smoke"
                       else ("base_sha", "source_tree_sha256"))
        required = source_keys + ("detector_checkpoint_sha256", "detector_compression",
                                  "active_loaded_keys", "ignored_unused_keys",
                                  "tracker_initialization", "detector_source",
                                  "detector_source_revision", "detector_config",
                                  "detector_configuration_sha256")
        if not isinstance(previous, dict) or any(key not in previous for key in required):
            raise RuntimeError("resume checkpoint is missing formal detector initialization provenance")
        if any(previous[key] != provenance[key] for key in source_keys):
            raise RuntimeError("Phase A resume requires the original source state")
        provenance.update({key: previous[key] for key in required})
    if rank == 0:
        output.mkdir(parents=True, exist_ok=bool(args.resume))
        provenance.update({"dataset_samples": len(dataset), "steps_per_epoch": len(loader),
                           "per_rank_batch": args.batch_size, "global_batch": args.batch_size * world_size,
                           "workers_per_rank": args.workers, "initial_lr": args.lr, "max_lr": args.max_lr,
                           "init_kind": "resume" if args.resume else "official-detector",
                           "tracking_label_contract": contract})
        if not args.resume:
            provenance.update({"detector_compression": int(args.detector_compression),
                               "detector_checkpoint_sha256": _sha256_file(init_weights),
                               "active_loaded_keys": loaded_detector_keys,
                               "ignored_unused_keys": [],
                               "tracker_initialization": "random"})
            provenance.update(detector_provenance(hypes))
        if args.resume:
            provenance["resume_checkpoint_sha256"] = _sha256_file(args.resume)
        (output / "config.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        if args.resume:
            _trim_csv_after_checkpoint(output / "metrics.csv", step)
    dist.barrier()
    started, last = time.monotonic(), None
    run_start_step = step
    global_samples = int(resume_payload.get("global_samples", step * args.batch_size * world_size)) if resume_payload else 0
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        epoch_totals = np.zeros(len(LOSS_KEYS) + 1, dtype=np.float64)
        epoch_count = 0
        log_totals = np.zeros(len(LOSS_KEYS) + 1, dtype=np.float64)
        log_count = 0
        for batch in loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            # Only the dense detector is autocast; ActorState/lifecycle are FP32.
            outputs, state = model(batch["runtime_frames"], detector_autocast=args.amp)
            loss = criterion(outputs, batch["target_frames"])
            loss["total_loss"].backward()
            grad_norm = clip_grad_norm_(model.parameters(), 35.0)
            optimizer.step()
            scheduler.step()
            state = state.detached()
            local_batch = len(batch["scene_receiver_keys"])
            global_totals, global_count = _global_loss(loss, local_batch, device)
            last = [value / global_count for value in global_totals]
            epoch_totals += np.asarray(global_totals)
            epoch_count += global_count
            log_totals += np.asarray(global_totals)
            log_count += global_count
            step += 1
            global_samples += global_count
            if step % args.log_every == 0:
                if rank == 0:
                    row = dict(zip(LOSS_KEYS + ("L_total",), [value / log_count for value in log_totals]))
                    row.update({"kind": "step", "split": "train", "epoch": epoch + 1, "global_step": step,
                                "global_samples": global_samples, "lr_trunk": optimizer.param_groups[0]["lr"],
                                "lr_heads": optimizer.param_groups[1]["lr"]})
                    _append_csv(output / "metrics.csv", row)
                    print(json.dumps(row), flush=True)
                log_totals[:] = 0.0
                log_count = 0
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
        epoch_row = dict(zip(LOSS_KEYS + ("L_total",), [value / epoch_count for value in epoch_totals]))
        epoch_row.update({"kind": "epoch", "split": "train+test", "epoch": epoch + 1, "global_step": step,
                          "global_samples": global_samples, "lr_trunk": optimizer.param_groups[0]["lr"],
                          "lr_heads": optimizer.param_groups[1]["lr"]})
        _sync_buffers(model)
        checkpoint_payload = {"model": model.module.state_dict(), "optimizer": optimizer.state_dict(),
                              "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "step": step,
                              "global_samples": global_samples, "provenance": provenance}
        if rank == 0:
            _atomic_save(checkpoint_payload, output / ("checkpoint_epoch_%03d.pt" % (epoch + 1)))
            _atomic_save(checkpoint_payload, output / "checkpoint.pt")
        dist.barrier()
        if (epoch + 1) % args.eval_every == 0:
            evaluator = evaluate_tracking_test_model_ddp if args.full_data else evaluate_tracking_model_ddp
            test_report = evaluator(model, hypes, device, os.environ["V2V4REAL_ROOT"],
                                    os.environ["DUALCMP_TRACKING_LABEL_ROOT"])
            if rank == 0:
                metrics = test_report["metrics"]
                epoch_row.update({"test_MOTA": metrics["MOTA"], "test_AMOTA": metrics["AMOTA"],
                                  "test_sAMOTA": metrics["sAMOTA"], "test_recall": metrics["tracking_recall"],
                                  "test_precision": metrics["tracking_precision"], "test_IDS": metrics["identity_switches"],
                                  "test_fragments": metrics["fragments"], "test_overall_pass": test_report.get("audit", {}).get("overall_pass", "")})
                (output / ("test_metrics_epoch_%03d.json" % (epoch + 1))).write_text(json.dumps(test_report, indent=2), encoding="utf-8")
        if rank == 0:
            _append_csv(output / "metrics.csv", epoch_row)
            print(json.dumps(epoch_row), flush=True)
        if args.stop_after_epoch and epoch + 1 >= args.stop_after_epoch:
            break
    _assert_sync(model.module)
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    elapsed = time.monotonic() - started
    provenance.update({"seed": args.seed, "epochs": args.epochs,
                       "dataset_samples": len(dataset),
                       "per_rank_batch": args.batch_size,
                       "global_batch": args.batch_size * world_size,
                       "workers_per_rank": args.workers,
                       "initial_lr": args.lr, "max_lr": args.max_lr,
                       "weight_decay": args.weight_decay,
                       "sequence_augmentation": args.full_data,
                       "init_kind": "resume" if args.resume else "official-detector"})
    report = {"steps": step,
              "loss": dict(zip(LOSS_KEYS + ("L_total",), last or [])),
              "provenance": provenance, "peak_memory_mib": peak_memory,
              "clips_per_second_per_rank": (step - run_start_step) * args.batch_size / max(elapsed, 1e-9),
              "scheduler": "JDT3D-one-cycle-cosine",
              "steps_per_epoch": len(loader), "scheduler_total_steps": total_steps,
              "cosine_rise_fraction": 0.4,
              "last_group_lrs": {group["group_name"]: group["lr"] for group in optimizer.param_groups},
              "last_grad_norm": float(grad_norm), "ddp_sync": True}
    if rank == 0:
        checkpoint = output / "checkpoint.pt"
        report["checkpoint_sha256"] = (
            _sha256_file(checkpoint) if checkpoint.is_file() else None
        )
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="two-process DualCMP T=3 DDP trainer")
    parser.add_argument("--hypes", default=DEFAULT_HYPES)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="JDT3D-aligned initial tracking-head LR")
    parser.add_argument("--max-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--detector-compression", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="independent T=3 clips per GPU")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--full-data", action="store_true")
    parser.add_argument("--train-label-smoke", action="store_true")
    parser.add_argument("--data-split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-epoch", type=int, default=0)
    parser.add_argument("--dirty-smoke", action="store_true")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if args.train_label_smoke and args.full_data:
        raise ValueError("--train-label-smoke cannot be combined with --full-data")
    if args.log_every < 1 or args.save_every != 1 or args.eval_every != 1:
        raise ValueError("Phase A requires log_every>0, save_every=1, and eval_every=1")
    if args.resume and Path(args.resume).resolve().parent != Path(args.output_dir).resolve():
        raise ValueError("--resume checkpoint must belong to --output-dir")
    train(args)


if __name__ == "__main__":
    main()
