"""Paired, frozen-input Source diagnostics on predeclared real test clips.

This is an inference intervention, not a trained ablation or a test-set model
selection procedure. All variants share detector/tracker outputs and GT pairs.
Run with torchrun --nproc_per_node=2; no model weights are modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Subset

from opencood.data_utils.dualcmp_dataloader import build_dataloader
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
from opencood.loss.dualcmp_loss import update_identity_bindings
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES, load_checkpoint_hypes
from opencood.tools.dualcmp.eval_forecast_ddp import _forecast_report, _forecast_rows
from opencood.tools.dualcmp.online import OnlineDualCMP, _model_from_checkpoint
from opencood.tools.dualcmp.temporal_protocol import checkpoint_forecast_contract
from opencood.tools.train_utils import to_device


def capture_forward(module, captured):
    original = module.forward

    def forward(*args, **kwargs):
        captured[:] = [args, kwargs]
        return original(*args, **kwargs)

    module.forward = forward
    return original


def intervene_temporal(model, encoder_input, head_input, variant):
    tokens, positions, mask, ages, poses = encoder_input[0]
    if variant == "shuffle_time":
        tokens = tokens.flip(3)
    elif variant == "latest_only":
        mask = mask.clone()
        mask[:, :, :, :-1] = False
    elif variant == "receiver_only":
        mask = mask.clone()
        mask[:, :, 1:] = False
    elif variant == "zero_bev":
        tokens = torch.zeros_like(tokens)
    encoded = model.source_temporal_encoder(tokens, positions, mask, ages, poses)
    batch, actors, sources, times, patches, channels = tokens.shape
    kwargs = dict(head_input[1])
    kwargs.update(
        source_tokens=encoded["tokens"].reshape(batch, actors, sources * times, patches, channels),
        source_positions=encoded["positions"].reshape(batch, actors, sources * times, patches, 3),
        source_token_mask=encoded["token_mask"].reshape(batch, actors, sources * times, patches),
    )
    return model.forecast_head(*head_input[0], **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--partner-delay-ms", type=float, default=100.0)
    parser.add_argument("--allow-dirty-smoke", action="store_true")
    parser.add_argument("--boundary-stream", action="store_true")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("audit requires both GPUs via two processes")
    dist.init_process_group("nccl")
    rank, local = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    payload = torch.load(args.checkpoint, map_location="cpu")
    contract = checkpoint_forecast_contract(payload["provenance"])
    hypes, _ = load_checkpoint_hypes(DEFAULT_HYPES, payload)
    root = str(Path(os.environ["V2V4REAL_ROOT"]) / "test")
    hypes.update(root_dir=root, validate_dir=root)
    hypes["fusion"] = {"args": {
        "stage": "forecast", "forecast_history_frames": 10, "future_frames": 50,
        "runtime_only": False, "runtime_forbid_gt": True, "receiver_ids": [0, 1],
        "max_actors": 128, "max_cav": 2, "split_name": "test",
        "tracking_label_root": os.environ["DUALCMP_TRACKING_LABEL_ROOT"],
        "partner_delay_ms": args.partner_delay_ms, "shuffle_training_points": False,
        "forecast_coordinate_frame": "current_receiver" if contract["forecast_representation"] == "source_temporal_aligned" else "legacy_capture",
    }}
    dataset = DualCMPSequenceDataset(hypes, train=False)
    indices = np.linspace(0, len(dataset) - 1, args.samples, dtype=int).tolist()
    if len(set(indices)) != args.samples:
        raise ValueError("sample selection has duplicates")
    selection = Subset(dataset, indices)
    _, loader, _ = build_dataloader(
        selection, args.batch_size, True, workers=args.workers,
        rank=rank, world_size=2, seed=0, training=False,
        collate_fn=dataset.collate_batch_test, pin_memory=True,
    )
    forecast_args = dict(representation=contract["forecast_representation"],
                         capture_alignment=contract["capture_alignment"],
                         age_encoding=contract["age_encoding"],
                         mean_source_aggregation=contract["mean_source_aggregation"])
    model = _model_from_checkpoint(hypes, args.checkpoint, device, forecast_args,
                                   allow_dirty_smoke=args.allow_dirty_smoke)
    temporal = model.source_temporal_encoder is not None
    variants = ["normal", "no_evidence", "zero_memory"]
    if temporal:
        variants += ["shuffle_time", "latest_only", "receiver_only", "zero_bev"]
    rows = {name: [] for name in variants}
    coverage = []
    diagnostics, keys_seen = [], []
    begin = time.monotonic()
    with torch.no_grad():
        for batch in loader:
            keys = [(str(scene), int(receiver), int(frame)) for (scene, receiver), frame in zip(
                batch["scene_receiver_keys"], batch["runtime_frames"][-1]["receiver_frame_idx"])]
            keys_seen.extend(keys)
            batch = to_device(batch, device)
            encoder_input, head_input = [], []
            head_forward = capture_forward(model.forecast_head, head_input)
            encoder_forward = capture_forward(model.source_temporal_encoder, encoder_input) if temporal else None
            try:
                outputs, _ = model(batch["runtime_frames"], run_forecast=True)
            finally:
                model.forecast_head.forward = head_forward
                if temporal:
                    model.source_temporal_encoder.forward = encoder_forward
            output, target = outputs[-1], batch["target_frames"][-1]
            normal_rows, local_coverage = _forecast_rows(output, target, keys)
            rows["normal"].extend(normal_rows)
            coverage.extend(local_coverage)
            for variant in variants[1:]:
                kwargs = dict(head_input[1])
                if variant == "no_evidence":
                    for prefix in ("source", "consensus"):
                        for suffix in ("tokens", "positions", "token_mask"):
                            kwargs[prefix + "_" + suffix] = None
                    kwargs["source_age_s"] = None
                    forecast = model.forecast_head(*head_input[0], **kwargs)
                elif variant == "zero_memory":
                    for prefix in ("source", "consensus"):
                        if kwargs.get(prefix + "_tokens") is not None:
                            kwargs[prefix + "_tokens"] = torch.zeros_like(kwargs[prefix + "_tokens"])
                    forecast = model.forecast_head(*head_input[0], **kwargs)
                else:
                    forecast = intervene_temporal(model, encoder_input, head_input, variant)
                changed_rows, _ = _forecast_rows(dict(output, forecast=forecast), target, keys)
                rows[variant].extend(changed_rows)
            assigned, _ = update_identity_bindings(output, target, [dict() for _ in keys])
            for row, slot in torch.nonzero((assigned >= 0) & output["forecast"]["forecast_valid_mask"], as_tuple=False).tolist():
                index = int(assigned[row, slot])
                if not bool(target["future_mask"][row, index].all()):
                    continue
                item = {"key": keys[row] + (int(target["actor_ids"][row, index]),),
                        "center_error_m": float(torch.linalg.vector_norm(output["track_box"][row, slot, :2] - target["current_boxes"][row, index, :2]))}
                if temporal:
                    tok, _, mask, age, _ = encoder_input[0]
                    item.update(valid_steps=mask[row, slot].any(-1).sum(-1).cpu().tolist(),
                                age_s=age[row, slot].cpu().tolist(),
                                bev_token_norm=float(tok[row, slot][mask[row, slot]].norm(dim=-1).mean()),
                                attention_mass=output["forecast"]["mode_source_attention_weights"][row, slot].reshape(6, tok.shape[2], tok.shape[3], tok.shape[4]).sum((0, 3)).cpu().tolist())
                diagnostics.append(item)
            del outputs, output, batch, encoder_input, head_input
        # Exercise the actual single-frame online API, with a fresh state owner.
        sample = dataset[indices[rank]]
        online = OnlineDualCMP(model, device)
        online_encoder_input = []
        encoder_forward = capture_forward(model.source_temporal_encoder, online_encoder_input) if temporal else None
        try:
            for frame_index, frame in enumerate(sample["runtime_frames"]):
                online.step(dataset._collate_runtime_frame([frame]),
                            run_forecast=(frame_index == len(sample["runtime_frames"]) - 1))
        finally:
            if temporal:
                model.source_temporal_encoder.forward = encoder_forward
        # Move runtime tensors only; clip identity tuples are metadata and the
        # legacy generic device mover does not support a direct tuple batch.
        runtime_one = [to_device(dataset._collate_runtime_frame([frame]), device)
                       for frame in sample["runtime_frames"]]
        clip_output, _ = model(runtime_one, run_forecast=True)
        streaming_error = float((clip_output[-1]["forecast"]["pred_trajs"]
                                 - online._last_output["forecast"]["pred_trajs"]).abs().max())
        streaming = {"rank": rank, "source_times": int(online_encoder_input[0][0].shape[3]) if temporal else None,
                     "clip_online_max_abs_error": streaming_error}
        if args.boundary_stream:
            if contract["forecast_representation"] != "source_temporal_aligned":
                raise ValueError("boundary streaming requires aligned representation")
            boundary_hypes = dict(hypes)
            boundary_hypes.update(root_dir=str(Path(os.environ["V2V4REAL_ROOT"]) / "train"),
                                  validate_dir=str(Path(os.environ["V2V4REAL_ROOT"]) / "train"))
            boundary_hypes["fusion"] = {"args": dict(hypes["fusion"]["args"], runtime_only=True,
                                                      split_name="train", tracking_label_root=None)}
            boundary_dataset = DualCMPSequenceDataset(boundary_hypes, train=False)
            scene = "testoutput_CAV_data_2022-03-17-10-43-13"
            frames = [boundary_dataset._collate_runtime_frame([
                boundary_dataset._runtime_frame(scene, "0", "%06d" % key)]) for key in range(185, 193)]
            boundary_online = OnlineDualCMP(model, device)
            for offset, frame in enumerate(frames):
                boundary_online.step(frame, run_forecast=(offset == len(frames) - 1))
            boundary_batch, _ = model([to_device(frame, device) for frame in frames], run_forecast=True)
            boundary_error = float((boundary_batch[-1]["forecast"]["pred_trajs"]
                                    - boundary_online._last_output["forecast"]["pred_trajs"]).abs().max())
            max_history = int(boundary_online._last_output["state"].forecast_history_mask.sum(-1).max())
            assert max_history <= 4 and boundary_error < 1e-3, (max_history, boundary_error)
            streaming.update(boundary_stream_max_history=max_history,
                             boundary_stream_online_error=boundary_error)
    gathered = [None, None]
    dist.all_gather_object(gathered, (rows, coverage, diagnostics, keys_seen, streaming, torch.cuda.max_memory_allocated() / 2 ** 20))
    if rank == 0:
        all_coverage = [item for group in gathered for item in group[1]]
        report = {"scope": "paired inference interventions; not trained ablations",
                  "checkpoint": args.checkpoint, "checkpoint_provenance": payload["provenance"],
                  "dataset_samples": len(dataset), "selected_indices": indices,
                  "selected_keys": sorted(set(tuple(key) for group in gathered for key in group[3])),
                  "partner_delay_ms": args.partner_delay_ms, "world_size": 2,
                  "global_batch": 2 * args.batch_size, "elapsed_s": time.monotonic() - begin,
                  "peak_memory_mib": [group[5] for group in gathered],
                  "streaming": [group[4] for group in gathered],
                  "diagnostics": [item for group in gathered for item in group[2]],
                  "variants": {name: _forecast_report([item for group in gathered for item in group[0][name]], all_coverage) for name in variants}}
        digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report["diagnostic_script_sha256"] = digest
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"output": args.output, "elapsed_s": report["elapsed_s"]}))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
