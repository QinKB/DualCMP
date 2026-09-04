"""Two-process Phase-B source-preserved DualCMP forecast trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time, random, re
import numpy as np
import subprocess

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
from opencood.tools.train_utils import to_device
from opencood.tools.dualcmp.protocol import (
    FORECAST_WINDOW,
    FORECAST_SAMPLES,
    checkpoint_detector_compression,
)
from opencood.tools.dualcmp.train_tracking_ddp import _sha256_file, _assert_sync
from opencood.tools.dualcmp.train_tracking_ddp import _append_csv, _atomic_save, _trim_csv_after_checkpoint
from opencood.tools.dualcmp.eval_forecast_ddp import evaluate_forecast_test_model_ddp

FORECAST_METRICS = ("minADE_1s", "minADE_3s", "minADE_5s",
                    "minFDE_1s", "minFDE_3s", "minFDE_5s",
                    "MR_3.6", "count")
CSV_FIELDS = (("kind", "split", "epoch", "global_step", "global_samples", "lr",
               "L_gmm", "L_mode", "L_dense_gmm", "L_total")
              + tuple("test_" + key for key in FORECAST_METRICS)
              + ("test_prediction_keys", "test_complete", "test_eligible_complete"))

PHASE_B_SOURCES = (
    "opencood/models/dualcmp/model.py",
    "opencood/models/dualcmp/state.py",
    "opencood/models/dualcmp/tracker.py",
    "opencood/models/dualcmp/actor_source_tokenizer.py",
    "opencood/models/dualcmp/mtr_actor_adapter.py",
    "opencood/models/dualcmp/mtr_vendor.py",
    "opencood/models/dualcmp/intention_query_forecast.py",
    "opencood/loss/dualcmp_loss.py",
    "opencood/loss/dualcmp_gmm_loss.py",
    "opencood/data_utils/v2v4real/datasets/dualcmp_sequence_dataset.py",
    "opencood/tools/dualcmp/protocol.py",
    "opencood/tools/dualcmp/eval_forecast_ddp.py",
    "opencood/tools/dualcmp/forecast_metrics.py",
    "opencood/tools/dualcmp/train_forecast_ddp.py",
    "MTR/mtr/models_v2v4real/context_encoder/mtr_encoder.py",
    "MTR/mtr/models_v2v4real/motion_decoder/mtr_decoder.py",
    "MTR/mtr/utils/common_utils.py",
)


def _provenance():
    root = Path(__file__).resolve().parents[3]
    code_sha = os.environ.get("DUALCMP_CODE_SHA")
    if code_sha:
        if not re.match(r"^[0-9a-fA-F]{40}$", code_sha):
            raise EnvironmentError("DUALCMP_CODE_SHA must be 40 hexadecimal characters")
        head = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root))
            .decode("utf-8")
            .strip()
        )
        dirty = (
            subprocess.check_output(
                ["git", "status", "--porcelain", "--", *PHASE_B_SOURCES], cwd=str(root)
            )
            .decode("utf-8")
            .strip()
        )
        if code_sha.lower() != head.lower() or dirty:
            raise RuntimeError(
                "clean source SHA requires HEAD match and no Phase-B source changes"
            )
        return {
            "source_state": "clean",
            "source_sha": code_sha.lower(),
            "protocol_sha256": _sha256_file(Path(__file__).with_name("protocol.py")),
        }
    base_sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root))
        .decode("utf-8")
        .strip()
    )
    digest = hashlib.sha256()
    for relative in sorted(PHASE_B_SOURCES):
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update((root / relative).read_bytes())
    return {
        "source_state": "dirty-smoke",
        "base_sha": base_sha,
        "source_tree_sha256": digest.hexdigest(),
        "protocol_sha256": _sha256_file(Path(__file__).with_name("protocol.py")),
    }


def _hypes(path, args=None):
    root, labels, tracker_weights = (
        os.environ.get(key)
        for key in (
            "V2V4REAL_ROOT",
            "DUALCMP_TRACKING_LABEL_ROOT",
            "DUALCMP_TRACKER_WEIGHTS",
        )
    )
    if os.environ.get("DUALCMP_MTR_WEIGHTS"):
        raise RuntimeError("formal Phase B rejects pretrained MTR weights")
    if not root or not labels or not tracker_weights:
        raise EnvironmentError(
            "V2V4REAL_ROOT, DUALCMP_TRACKING_LABEL_ROOT, "
            "and DUALCMP_TRACKER_WEIGHTS are required"
        )
    full_data = bool(args is not None and args.full_data)
    split = args.data_split if full_data else "test"
    tracker_payload = torch.load(tracker_weights, map_location="cpu")
    detector_compression = checkpoint_detector_compression(tracker_payload)
    h = load_yaml(path)
    h["root_dir"] = str(Path(root) / split)
    h["validate_dir"] = h["root_dir"]
    h["model"]["args"]["compression"] = detector_compression
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
            "shuffle_training_points": False,
        }
    }
    if full_data:
        h["fusion"]["args"]["split_name"] = split
        if split == "test":
            h["fusion"]["args"]["tracking_label_root"] = labels
    else:
        h["fusion"]["args"].update({
            "tracking_label_root": labels, "window": list(FORECAST_WINDOW),
        })
    return h, tracker_weights, tracker_payload, detector_compression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hypes",
        default="opencood/hypes_yaml/point_pillar_cobevt_multiego_1x_v2v4real.yaml",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--full-data", action="store_true")
    parser.add_argument("--data-split", choices=("train", "val", "test"),
                        default="train")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-epoch", type=int, default=0)
    args = parser.parse_args()
    if args.log_every < 1 or args.save_every < 1 or args.eval_every < 1:
        raise ValueError("Phase B logging, saving, and evaluation intervals must be positive")
    if args.full_data and (args.save_every != 1 or args.eval_every != 1):
        raise ValueError("formal Phase B requires save_every=1 and eval_every=1")
    if args.resume and Path(args.resume).resolve().parent != Path(args.output_dir).resolve():
        raise ValueError("--resume checkpoint must belong to --output-dir")
    if args.full_data and not os.environ.get("DUALCMP_CODE_SHA"):
        raise EnvironmentError("formal full-data Phase B requires DUALCMP_CODE_SHA")
    random.seed(20260901)
    np.random.seed(20260901)
    torch.manual_seed(20260901)
    torch.cuda.manual_seed_all(20260901)
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("requires torchrun --nproc_per_node=2")
    dist.init_process_group("nccl")
    rank, local_rank = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    h, tracker_weights, payload, detector_compression = _hypes(
        args.hypes, args
    )
    dataset = DualCMPSequenceDataset(h, train=True)
    if not args.full_data and len(dataset) != FORECAST_SAMPLES:
        raise RuntimeError("forecast protocol requires exactly 16 samples")
    _, loader, sampler = build_dataloader(
        dataset,
        args.batch_size,
        True,
        workers=args.workers,
        rank=rank,
        world_size=2,
        seed=20260901,
        training=True,
        collate_fn=dataset.collate_batch_train,
        pin_memory=True,
    )
    forecast_args = {
        "history_frames": 11,
        "future_frames": 50,
        "modes": 6,
        "intention_points_file": "preprocessed_data/v2v4real/v2v4real_cluster_64_center_dict.pkl",
    }
    model = DualCmpClipModel(
        h["model"]["args"],
        dataset.anchor_box,
        h["preprocess"]["cav_lidar_range"],
        forecast_args=forecast_args,
    ).to(device)
    state = payload.get("model", payload.get("model_state_dict", payload))
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    forbidden = tuple(
        key for key in state
        if key.startswith(("forecast_head.", "actor_source_tokenizer."))
    )
    if forbidden:
        raise RuntimeError("Phase-A checkpoint contains forbidden forecast weights")
    loaded = model.load_state_dict(state, strict=False)
    allowed = ("forecast_head.", "actor_source_tokenizer.")
    invalid = [key for key in loaded.missing_keys if not key.startswith(allowed)]
    if loaded.unexpected_keys or invalid:
        raise RuntimeError(
            "Phase-A tracker checkpoint mismatch: missing=%s unexpected=%s"
            % (invalid, loaded.unexpected_keys)
        )
    for parameter in (
        list(model.detector.parameters())
        + list(model.tracker.parameters())
        + list(model.temporal_attention.parameters())
        + list(model.track_update.parameters())
        + list(model.track_exist.parameters())
        + list(model.track_box.parameters())
        + list(model.track_velocity.parameters())
    ):
        parameter.requires_grad = False
    frozen_temporal_attention = {
        key: value.detach().cpu().clone()
        for key, value in model.temporal_attention.state_dict().items()
    }
    model.train()
    for module in (
        model.detector,
        model.tracker,
        model.roi,
        model.temporal_attention,
        model.track_update,
        model.track_exist,
        model.track_box,
        model.track_velocity,
    ):
        module.eval()
    model.forecast_head.train()
    model.actor_source_tokenizer.train()
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    total_budget_steps = (
        args.epochs * len(loader)
        if args.full_data else (args.max_steps or args.epochs * len(loader))
    )
    decay_steps = [
        round(total_budget_steps * epoch / 30.0) for epoch in (22, 24, 26, 28)
    ]

    def lr_multiplier(current_step):
        decay = 1.0
        for decay_step in decay_steps:
            if current_step >= decay_step:
                decay *= 0.5
        return max(decay, 1e-6 / args.lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    criterion = DualCmpLoss({"detector": h["loss"]["args"]}).to(device)
    run_provenance = _provenance()
    output = Path(args.output_dir)
    start_epoch = step = global_samples = 0
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu")
        if not isinstance(resume_payload, dict) or any(key not in resume_payload for key in ("model", "optimizer", "scheduler", "epoch", "step")):
            raise RuntimeError("resume checkpoint must contain model, optimizer, scheduler, epoch, and step")
        if int(resume_payload["scheduler"].get("last_epoch", -1)) != int(resume_payload["step"]):
            raise RuntimeError("resume scheduler step does not match checkpoint step")
        if int(resume_payload["provenance"].get("total_budget_steps", -1)) != total_budget_steps:
            raise RuntimeError("resume scheduler budget disagrees with requested 30-epoch schedule")
        previous = resume_payload["provenance"]
        source_key = "source_sha" if "source_sha" in run_provenance else "source_tree_sha256"
        if previous.get(source_key) != run_provenance.get(source_key):
            raise RuntimeError("Phase B resume requires the original source state")
        if previous.get("phase_a_checkpoint_sha256") != _sha256_file(tracker_weights):
            raise RuntimeError("Phase B resume requires the original Phase-A checkpoint")
        model.module.load_state_dict(resume_payload["model"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"]); scheduler.load_state_dict(resume_payload["scheduler"])
        start_epoch, step = int(resume_payload["epoch"]), int(resume_payload["step"])
        global_samples = int(resume_payload.get("global_samples", 0))
    if rank == 0:
        output.mkdir(parents=True, exist_ok=bool(args.resume))
        if args.resume:
            _trim_csv_after_checkpoint(output / "metrics.csv", step, CSV_FIELDS)
    dist.barrier()
    started = time.monotonic()
    last = None
    last_local_forecast_counts = None
    epoch = start_epoch
    while epoch < args.epochs:
        sampler.set_epoch(epoch)
        epoch_totals = np.zeros(3, dtype=np.float64); epoch_counts = np.zeros(3, dtype=np.float64)
        log_totals = np.zeros(3, dtype=np.float64); log_counts = np.zeros(3, dtype=np.float64)
        for batch in loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs, _ = model(batch["runtime_frames"], run_forecast=True)
            loss = criterion([outputs[-1]], batch["target_frames"])
            global_counts = torch.stack(
                (
                    loss["C_gmm"],
                    loss["C_mode"],
                    loss["C_dense_gmm"],
                )
            ).detach()
            last_local_forecast_counts = global_counts.cpu().tolist()
            dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
            forecast_loss = dist.get_world_size() * (
                loss["S_gmm"] / global_counts[0].clamp_min(1.0)
                + loss["S_mode"] / global_counts[1].clamp_min(1.0)
                + loss["S_dense_gmm"] / global_counts[2].clamp_min(1.0)
            )
            forecast_loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), 1000.0)
            optimizer.step()
            scheduler.step()
            sums_and_counts = torch.stack(
                (
                    loss["S_gmm"].detach(), loss["C_gmm"],
                    loss["S_mode"].detach(), loss["C_mode"],
                    loss["S_dense_gmm"].detach(), loss["C_dense_gmm"],
                )
            ).float()
            dist.all_reduce(sums_and_counts, op=dist.ReduceOp.SUM)
            last = {
                "L_gmm": float(sums_and_counts[0] / sums_and_counts[1].clamp_min(1.0)),
                "L_mode": float(sums_and_counts[2] / sums_and_counts[3].clamp_min(1.0)),
                "L_dense_gmm": float(sums_and_counts[4] / sums_and_counts[5].clamp_min(1.0)),
            }
            step += 1
            global_samples += len(batch["scene_receiver_keys"]) * dist.get_world_size()
            sums = sums_and_counts.detach().cpu().numpy()
            epoch_totals += sums[[0, 2, 4]]; epoch_counts += sums[[1, 3, 5]]
            log_totals += sums[[0, 2, 4]]; log_counts += sums[[1, 3, 5]]
            if step % args.log_every == 0 and rank == 0:
                row = {"kind": "step", "split": "train", "epoch": epoch + 1, "global_step": step,
                       "global_samples": global_samples, "lr": optimizer.param_groups[0]["lr"],
                       "L_gmm": float(log_totals[0] / max(log_counts[0], 1)),
                       "L_mode": float(log_totals[1] / max(log_counts[1], 1)),
                       "L_dense_gmm": float(log_totals[2] / max(log_counts[2], 1))}
                row["L_total"] = row["L_gmm"] + row["L_mode"] + row["L_dense_gmm"]
                _append_csv(output / "metrics.csv", row, CSV_FIELDS); print(json.dumps(row), flush=True)
                log_totals[:] = 0; log_counts[:] = 0
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
        provenance = dict(run_provenance); provenance.update({"detector_compression": detector_compression,
            "phase_a_checkpoint_sha256": _sha256_file(tracker_weights), "total_budget_steps": total_budget_steps,
            "forecast_initialization": "random", "evidence_initialization": "random"})
        checkpoint = {"model": model.module.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "epoch": epoch + 1, "step": step, "global_samples": global_samples, "provenance": provenance}
        if rank == 0 and (epoch + 1) % args.save_every == 0:
            _atomic_save(checkpoint, output / ("checkpoint_epoch_%03d.pt" % (epoch + 1)))
            _atomic_save(checkpoint, output / "checkpoint.pt")
        dist.barrier()
        test_report = None
        if (epoch + 1) % args.eval_every == 0:
            test_report = evaluate_forecast_test_model_ddp(
                model, h, device, os.environ["V2V4REAL_ROOT"],
                os.environ["DUALCMP_TRACKING_LABEL_ROOT"],
                batch_size=args.batch_size, workers=args.workers,
                full_test=args.full_data,
            )
        if rank == 0:
            row = {"kind": "epoch", "split": "train+test", "epoch": epoch + 1, "global_step": step,
                   "global_samples": global_samples, "lr": optimizer.param_groups[0]["lr"],
                   "L_gmm": float(epoch_totals[0] / max(epoch_counts[0], 1)), "L_mode": float(epoch_totals[1] / max(epoch_counts[1], 1)),
                   "L_dense_gmm": float(epoch_totals[2] / max(epoch_counts[2], 1))}
            row["L_total"] = row["L_gmm"] + row["L_mode"] + row["L_dense_gmm"]
            if test_report is not None:
                row.update({"test_" + key: test_report["model"][key]
                            for key in FORECAST_METRICS})
                row.update({"test_prediction_keys": test_report["coverage"]["prediction_keys"],
                            "test_complete": test_report["coverage"]["complete"],
                            "test_eligible_complete": test_report["coverage"]["eligible_complete"]})
                (output / ("test_forecast_epoch_%03d.json" % (epoch + 1))).write_text(json.dumps(test_report, indent=2), encoding="utf-8")
            _append_csv(output / "metrics.csv", row, CSV_FIELDS); print(json.dumps(row), flush=True)
        epoch += 1
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            break
    _assert_sync(model.module)
    gathered_local_counts = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_local_counts, last_local_forecast_counts)
    if any(
        not torch.equal(
            value.detach().cpu(), frozen_temporal_attention[key]
        )
        for key, value in model.module.temporal_attention.state_dict().items()
    ):
        raise RuntimeError("Phase-B temporal attention changed while frozen")
    elapsed = max(time.monotonic() - started, 1e-9)
    report = {
        "steps": step,
        "epochs_completed": epoch,
        "dataset_samples": len(dataset),
        "loss": last,
        "per_rank_batch": args.batch_size,
        "global_batch": args.batch_size * 2,
        "workers": args.workers,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "clips_per_second_per_rank": step * args.batch_size / elapsed,
        "steps_per_second_per_rank": step / elapsed,
        "ddp_sync": True,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "grad_norm_clip": 1000.0,
        "last_grad_norm": float(grad_norm),
        "last_lr": float(optimizer.param_groups[0]["lr"]),
        "lr_decay_steps": decay_steps,
        "last_local_forecast_counts": gathered_local_counts,
        "forecast_initialization": "random",
        "forecast_initialization_seed": 20260901,
        "evidence_initialization": "random",
        "temporal_attention_unchanged": True,
        "detector_compression": detector_compression,
    }
    provenance = dict(run_provenance)
    provenance["detector_compression"] = detector_compression
    provenance["phase_a_checkpoint_sha256"] = _sha256_file(tracker_weights)
    provenance["phase_a_source_provenance"] = (
        payload.get("provenance", {}).get("source_sha")
        if isinstance(payload, dict)
        else None
    )
    report["provenance"] = provenance
    if rank == 0:
        checkpoint = output / "checkpoint.pt"
        report["checkpoint_sha256"] = _sha256_file(checkpoint)
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
