"""Dedicated two-process DDP trainer for the fixed DualCMP T=3 protocol.

Launch only with ``torchrun --nproc_per_node=2``.  ``batch_size`` is clips per
GPU; the frozen window below has 44 receiver-centric clips (22 per receiver).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from opencood.data_utils.dualcmp_dataloader import build_dataloader
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.loss.dualcmp_loss import DualCmpLoss
from opencood.models.dualcmp.model import DualCmpClipModel
from opencood.tools.train_utils import to_device
from opencood.tools.dualcmp.protocol import CLIPS_PER_RECEIVER, DETECTOR_COMPRESSION, TOTAL_CLIPS, WINDOW, WINDOW_FRAMES

LOSS_KEYS = ("L_det", "L_exist", "L_box", "L_velocity")


def _init_distributed():
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("DualCMP tracking training requires torchrun --nproc_per_node=2")
    dist.init_process_group(backend="nccl")
    rank, local_rank = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def _hypes(path):
    root = os.environ.get("V2V4REAL_ROOT")
    weights = os.environ.get("DUALCMP_DETECTOR_WEIGHTS")
    label_root = os.environ.get("DUALCMP_TRACKING_LABEL_ROOT")
    if not root or not weights or not label_root:
        raise EnvironmentError("V2V4REAL_ROOT, DUALCMP_DETECTOR_WEIGHTS, and DUALCMP_TRACKING_LABEL_ROOT are required")
    hypes = load_yaml(path)
    hypes["root_dir"] = str(Path(root) / "test")
    hypes["validate_dir"] = hypes["root_dir"]
    hypes["model"]["args"]["compression"] = DETECTOR_COMPRESSION
    hypes["fusion"] = {"core_method": "DualCMPSequenceDataset", "args": {
        "stage": "tracking", "tracking_frames": 3, "runtime_only": False,
        "runtime_forbid_gt": True, "receiver_ids": [0, 1], "frame_period_s": 0.1,
        "max_actors": 128, "max_cav": 2, "window": list(WINDOW), "tracking_label_root": label_root,
        "shuffle_training_points": False,
    }}
    return hypes, weights


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


def train(args):
    rank, local_rank, world_size = _init_distributed()
    device = torch.device("cuda", local_rank)
    hypes, detector_weights = _hypes(args.hypes)
    dataset = DualCMPSequenceDataset(hypes, train=True)
    per_receiver = {entry["receiver_id"]: 0 for entry in WINDOW}
    for record in dataset.clip_records:
        per_receiver[int(record.receiver_id)] += 1
    if (len(dataset) != TOTAL_CLIPS or
            any(value != CLIPS_PER_RECEIVER for value in per_receiver.values())):
        raise RuntimeError("fixed protocol requires 24 frames/22 clips per receiver and 44 total: %s" % per_receiver)
    _, loader, sampler = build_dataloader(dataset, 1, True, workers=0, rank=rank,
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
    model.detector.load_state_dict(torch.load(detector_weights, map_location="cpu"), strict=True)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                broadcast_buffers=False)
    criterion = DualCmpLoss({"detector": hypes["loss"]["args"]}).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-10,
                                 weight_decay=1e-4)
    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.json").write_text(json.dumps({"seed": args.seed, "epochs": args.epochs,
            "per_rank_batch": 1, "global_batch": world_size, "workers": 0,
            "window_frames": WINDOW_FRAMES, "clips_per_receiver": CLIPS_PER_RECEIVER,
            "total_clips": TOTAL_CLIPS, "window": WINDOW}, indent=2), encoding="utf-8")
    dist.barrier()
    step, started, last = 0, time.monotonic(), None
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            # Only the dense detector is autocast; ActorState/lifecycle are FP32.
            outputs, state = model(batch["runtime_frames"], detector_autocast=args.amp)
            loss = criterion(outputs, batch["target_frames"])
            loss["total_loss"].backward()
            optimizer.step()
            state = state.detached()
            values = torch.stack([loss[key].detach().float() for key in LOSS_KEYS])
            if not torch.isfinite(values).all():
                raise FloatingPointError("non-finite DualCMP loss")
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            last = (values / world_size).cpu().tolist()
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
    _assert_sync(model.module)
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    elapsed = time.monotonic() - started
    report = {"steps": step, "loss": dict(zip(LOSS_KEYS, last or [])), "per_rank_batch": 1,
              "global_batch": world_size, "workers": 0, "peak_memory_mib": peak_memory,
              "clips_per_second_per_rank": step / max(elapsed, 1e-9)}
    if rank == 0:
        torch.save({"model": model.module.state_dict(), "epoch": epoch, "step": step}, output / "checkpoint.pt")
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="two-process DualCMP T=3 DDP trainer")
    parser.add_argument("--hypes", default="opencood/hypes_yaml/point_pillar_cobevt_multiego_1x_v2v4real.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--amp", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
