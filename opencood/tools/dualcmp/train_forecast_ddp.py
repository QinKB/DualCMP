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

from opencood.data_utils.dualcmp_dataloader import build_dataloader
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import (
    DualCMPSequenceDataset,
)
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.loss.dualcmp_loss import DualCmpLoss
from opencood.models.dualcmp.model import DualCmpClipModel
from opencood.tools.train_utils import to_device
from opencood.tools.dualcmp.protocol import (
    DETECTOR_COMPRESSION,
    FORECAST_WINDOW,
    FORECAST_SAMPLES,
)
from opencood.tools.dualcmp.train_tracking_ddp import _sha256_file, _assert_sync

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


def _hypes(path):
    root, labels, tracker_weights = (
        os.environ.get(key)
        for key in (
            "V2V4REAL_ROOT",
            "DUALCMP_TRACKING_LABEL_ROOT",
            "DUALCMP_TRACKER_WEIGHTS",
        )
    )
    if not root or not labels or not tracker_weights:
        raise EnvironmentError(
            "V2V4REAL_ROOT, DUALCMP_TRACKING_LABEL_ROOT, and DUALCMP_TRACKER_WEIGHTS are required"
        )
    h = load_yaml(path)
    h["root_dir"] = str(Path(root) / "test")
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
            "tracking_label_root": labels,
            "window": list(FORECAST_WINDOW),
            "shuffle_training_points": False,
        }
    }
    return h, tracker_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hypes",
        default="opencood/hypes_yaml/point_pillar_cobevt_multiego_1x_v2v4real.yaml",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
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
    h, tracker_weights = _hypes(args.hypes)
    dataset = DualCMPSequenceDataset(h, train=True)
    if len(dataset) != FORECAST_SAMPLES:
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
    payload = torch.load(tracker_weights, map_location="cpu")
    state = payload.get("model", payload.get("model_state_dict", payload))
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
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
        + list(model.track_update.parameters())
        + list(model.track_exist.parameters())
        + list(model.track_box.parameters())
        + list(model.track_velocity.parameters())
    ):
        parameter.requires_grad = False
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
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    criterion = DualCmpLoss({"detector": h["loss"]["args"]}).to(device)
    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    step = 0
    started = time.monotonic()
    last = None
    epoch = 0
    while (not args.max_steps) or step < args.max_steps:
        sampler.set_epoch(epoch)
        for batch in loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs, _ = model(batch["runtime_frames"], run_forecast=True)
            loss = criterion([outputs[-1]], batch["target_frames"])
            loss["total_loss"].backward()
            optimizer.step()
            values = torch.stack(
                (loss["L_gmm"].detach(), loss["L_mode"].detach())
            ).float()
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values /= 2
            last = {"L_gmm": float(values[0]), "L_mode": float(values[1])}
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
        epoch += 1
        if not args.max_steps and epoch >= args.epochs:
            break
    _assert_sync(model.module)
    elapsed = max(time.monotonic() - started, 1e-9)
    report = {
        "steps": step,
        "loss": last,
        "per_rank_batch": args.batch_size,
        "global_batch": args.batch_size * 2,
        "workers": args.workers,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "clips_per_second_per_rank": step * args.batch_size / elapsed,
        "steps_per_second_per_rank": step / elapsed,
        "ddp_sync": True,
    }
    provenance = _provenance()
    provenance["phase_a_checkpoint_sha256"] = _sha256_file(tracker_weights)
    provenance["phase_a_source_provenance"] = (
        payload.get("provenance", {}).get("source_sha")
        if isinstance(payload, dict)
        else None
    )
    report["provenance"] = provenance
    if rank == 0:
        checkpoint = output / "checkpoint.pt"
        torch.save(
            {"model": model.module.state_dict(), "provenance": provenance}, checkpoint
        )
        report["checkpoint_sha256"] = _sha256_file(checkpoint)
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
