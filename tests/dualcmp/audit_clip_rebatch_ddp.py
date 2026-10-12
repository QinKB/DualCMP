"""Diagnose a completed checkpoint's batch-composition parity failure.

No training, target changes, confirmation data, or acceptance relaxation.
The original absolute selected-trajectory tolerance remains 1e-3.
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from cached_clip_batching import collate_cached_clips, slice_clip, split_cached_chunk
from train_information_ablation_ddp import (CachedForecast, evaluate, full_replay_batch,
    load_chunks, move, verify_full_replay)
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES, load_checkpoint_hypes
from opencood.tools.dualcmp.eval_forecast_ddp import _forecast_rows, _forecast_report
from opencood.tools.dualcmp.online import _model_from_checkpoint


def tensor_error(first, second):
    if first.shape != second.shape:
        raise RuntimeError("parity tensors have different shapes")
    if first.dtype == torch.bool or not first.is_floating_point():
        return dict(different=int((first != second).sum()), count=first.numel())
    delta = (first - second).abs()
    return dict(maximum=float(delta.max()), mean=float(delta.mean()),
                count=first.numel(), above_tolerance=int((delta > 1e-3).sum()))


@torch.no_grad()
def diagnose(model, chunks, device):
    wrapper = CachedForecast(model).eval()
    captured = []
    def capture(module, inputs, output):
        captured.append(output["tokens"].detach().cpu())
    hook = model.source_temporal_encoder.register_forward_hook(capture)
    originals = [chunks[0], chunks[-1]]
    clips = [split_cached_chunk(chunk)[0] for chunk in originals]
    merged = move(collate_cached_clips(clips), device)
    combined = wrapper(merged)
    combined_source = captured.pop()
    combined_repeat = wrapper(merged)
    captured.clear()
    repeat_error = tensor_error(combined["forecast"]["pred_trajs"],
                                combined_repeat["forecast"]["pred_trajs"])
    del combined_repeat
    combined_rows, combined_coverage = _forecast_rows(
        combined, merged["target"], merged["keys"])
    records, original_rows, original_coverage = [], [], []
    keys = ("pred_trajs", "pred_scores", "selected_mode_indices", "loss_pred_trajs_local_layers",
            "loss_pred_scores_layers", "forecast_valid_mask")
    for row, chunk in enumerate(originals):
        data = move(chunk, device)
        reference = wrapper(data)
        reference_source = captured.pop()
        slots = torch.searchsorted(merged["original_slots"], data["original_slots"])
        differences = {key: tensor_error(combined["forecast"][key][row, slots],
                                         reference["forecast"][key][0]) for key in keys}
        # Comparing unordered selected trajectories identifies a permutation
        # separately from a change of selected mode; neither bypasses the gate.
        first = combined["forecast"]["pred_trajs"][row, slots, :, :, :2]
        second = reference["forecast"]["pred_trajs"][0, :, :, :, :2]
        distances = (first[:, :, None] - second[:, None, :]).abs().amax((-1, -2))
        unordered = torch.maximum(distances.amin(1).amax(1), distances.amin(2).amax(1))
        records.append(dict(key=clips[row]["keys"][0], errors=differences,
                            same_batch_repeat=repeat_error,
                            source_feature_error=tensor_error(
                                combined_source[row, chunk["original_slots"]],
                                reference_source[0, chunk["original_slots"]]),
                            unordered_selected_maximum=float(unordered.max()),
                            original_actor_slots=len(data["original_slots"]),
                            merged_actor_slots=len(merged["original_slots"])))
        local, coverage = _forecast_rows(slice_clip(reference, 0),
            move(clips[row]["target"], device), clips[row]["keys"])
        original_rows.extend(local)
        original_coverage.extend(coverage)
    hook.remove()
    return records, combined_rows, combined_coverage, original_rows, original_coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--linear-fp64-diagnostic", action="store_true",
                        help="diagnostic only: promote head Linear accumulation, preserve stored weights")
    parser.add_argument("--complete-reports", action="store_true",
                        help="original precision only: final train/dev and real full-model replay")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("rebatch audit requires both GPU ranks")
    if Path(args.output).exists():
        raise FileExistsError("refusing to overwrite diagnostic evidence")
    if args.linear_fp64_diagnostic and args.complete_reports:
        raise ValueError("precision intervention must not be presented as normal final metrics")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    cache = Path(args.cache_dir)
    manifest = json.loads((cache / "manifest.json").read_text())
    payload = torch.load(args.checkpoint, map_location="cpu")
    hypes, _ = load_checkpoint_hypes(DEFAULT_HYPES, payload)
    provenance = payload["provenance"]
    model = _model_from_checkpoint(hypes, args.checkpoint, device, allow_dirty_smoke=True)
    chunks = load_chunks(cache, manifest, "train", rank)
    if args.linear_fp64_diagnostic:
        # Reuse torch.nn.Linear.forward's exact F.linear operation. Only the
        # temporary accumulation precision changes; no model/checkpoint writes.
        for module in model.forecast_head.modules():
            if isinstance(module, torch.nn.Linear):
                def linear_forward(value, layer=module):
                    bias = layer.bias.double() if layer.bias is not None else None
                    return torch.nn.functional.linear(value.double(), layer.weight.double(), bias).to(value.dtype)
                module.forward = linear_forward
    diagnosed = diagnose(model, chunks, device)
    complete_reports = None
    if args.complete_reports:
        development = load_chunks(cache, manifest, "development", rank)
        runtime_batch = full_replay_batch(hypes, manifest, rank)
        full_replay = verify_full_replay(model, development[0], runtime_batch, device)
        complete_reports = dict(full_replay=full_replay,
            training=evaluate(CachedForecast(model), chunks, device),
            development=evaluate(CachedForecast(model), development, device))
    gathered = [None, None]
    dist.all_gather_object(gathered, diagnosed)
    if rank == 0:
        records = [record for group in gathered for record in group[0]]
        result = dict(status="diagnosed", checkpoint=args.checkpoint,
            updates=provenance["research_updates"],
            policy=provenance["aligned_evidence_policy"],
            scope="numerical precision intervention only" if args.linear_fp64_diagnostic else
                  "completed-checkpoint diagnosis; original gate remains authoritative",
            linear_accumulation="float64 diagnostic" if args.linear_fp64_diagnostic else "original float32",
            complete_reports=complete_reports,
            absolute_tolerance=1e-3,
            original_gate_passed=all(record["errors"]["pred_trajs"]["maximum"] <= 1e-3
                                     for record in records),
            float32_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            records=records,
            original_report=_forecast_report([v for group in gathered for v in group[3]],
                [v for group in gathered for v in group[4]]),
            combined_report=_forecast_report([v for group in gathered for v in group[1]],
                [v for group in gathered for v in group[2]]))
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps(dict(status="diagnosed", original_gate_passed=result["original_gate_passed"])))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
