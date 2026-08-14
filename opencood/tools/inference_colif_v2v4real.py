"""GT-free chronological CoLIF inference on V2V4Real."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from opencood.data_utils.v2v4real.datasets import build_v2v4real_dataset
from opencood.data_utils.v2v4real.datasets.colif_schema import (
    assert_runtime_is_supervision_free,
)
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils
from opencood.tools.colif.receiver_protocol import apply_receiver_override


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypes_yaml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--receiver-ids",
        help="Comma-separated CAV IDs; use 0 for one ego or 0,1 for CMP-style multi-ego",
    )
    return parser.parse_args()


def main():
    args = _arguments()
    requested_hypes = load_yaml(args.hypes_yaml)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    # Reuse checkpoint runtime semantics such as thresholds and lifecycle TTL.
    hypes = checkpoint.get("config", requested_hypes)
    receiver_ids = apply_receiver_override(hypes, args.receiver_ids)
    fusion_args = hypes["fusion"]["args"]
    fusion_args["runtime_only"] = True
    fusion_args["warmup_length"] = 0
    fusion_args["unroll_length"] = 1
    fusion_args["clip_stride"] = 1
    fusion_args["split_name"] = args.split
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = build_v2v4real_dataset(hypes, train=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=dataset.collate_batch_test,
    )
    model = train_utils.create_model(hypes).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    state = None
    previous_key = None
    records = []
    with torch.no_grad():
        for frame_index, batch in enumerate(loader):
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            assert_runtime_is_supervision_free(batch["runtime_frames"])
            scene_receiver_key = batch["scene_receiver_keys"][0]
            if scene_receiver_key != previous_key:
                state = None
                previous_key = scene_receiver_key
            runtime = train_utils.to_device(batch["runtime_frames"][0], device)
            outputs, state, _ = model(runtime, state)
            active = outputs["active_mask"][0]
            records.append({
                "scene_id": scene_receiver_key[0],
                "receiver_id": int(scene_receiver_key[1]),
                "frame_key": runtime["receiver_frame_key"][0],
                "receiver_time_s": float(runtime["receiver_time_s"][0].cpu()),
                "runtime_id": outputs["runtime_ids"][0, active].cpu(),
                "boxes": outputs["track_boxes"][0, active].cpu(),
                "exist_scores": outputs["exist_scores"][0, active].cpu(),
                "query_age": outputs["query_age"][0, active].cpu(),
                "miss_count": outputs["miss_count"][0, active].cpu(),
                "forecast_scores": outputs["pred_scores"][0, active].cpu(),
                "forecast_trajs": outputs["pred_trajs"][0, active].cpu(),
                "source_mask": runtime["source_mask"][0].cpu(),
                "capture_time_s": runtime["capture_time_s"][0].cpu(),
            })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "colif_runtime_v1",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "split": args.split,
            "receiver_ids": list(receiver_ids),
            "records": records,
        },
        output,
    )
    print({"output": str(output.resolve()), "record_count": len(records)})


if __name__ == "__main__":
    main()
