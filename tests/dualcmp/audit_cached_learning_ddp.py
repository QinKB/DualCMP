"""Diagnose completed paired learning; no optimizer or target filtering.

Batch-statistics intervention freezes dropout and restores every running buffer
after each batch. It is a diagnostic, not a deployable evaluation protocol.
Confirmation samples are never read by this script.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from train_information_ablation_ddp import CachedForecast, load_chunks, move
from opencood.loss.dualcmp_loss import DualCmpLoss, update_identity_bindings
from opencood.loss.dualcmp_gmm_loss import nearest_mode_gmm_loss
from opencood.models.dualcmp.mtr_actor_adapter import pack_mtr_actor_polylines
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES, load_checkpoint_hypes
from opencood.tools.dualcmp.eval_forecast_ddp import _forecast_rows, _forecast_report
from opencood.tools.dualcmp.online import _model_from_checkpoint


def diversity(output, target):
    assigned, _ = update_identity_bindings(output, target, [dict() for _ in range(target["current_mask"].shape[0])])
    forecast = output["forecast"]
    rows, slots = torch.nonzero((assigned >= 0) & forecast["forecast_valid_mask"], as_tuple=True)
    if not len(rows):
        return []
    indices = assigned[rows, slots]
    complete = target["future_mask"][rows, indices].all(-1)
    rows, slots, indices = rows[complete], slots[complete], indices[complete]
    trajectories = forecast["loss_pred_trajs_local_layers"][rows, slots, -1]
    logits = forecast["loss_pred_scores_layers"][rows, slots, -1]
    span = (trajectories[:, :, -1, :2].amax(1) - trajectories[:, :, -1, :2].amin(1)).norm(dim=-1)
    std = trajectories[..., :2].std(1, unbiased=False).mean(dim=(1, 2))
    center = (output["track_box"][rows, slots, :2] - target["current_boxes"][rows, indices, :2]).norm(dim=-1)
    log_span = logits.amax(1) - logits.amin(1)
    max_std_fraction = (trajectories[..., 2:4] >= 5.).float().mean(dim=(1, 2, 3))
    return torch.stack((center, span, std, log_span, max_std_fraction), dim=-1).cpu().tolist()


def summarize(values):
    names = ("center_error_m", "raw64_endpoint_span_m", "raw64_xy_std_m", "raw64_logit_span", "max_clamped_std_fraction")
    array = np.asarray(values)
    return {name: dict(mean=float(array[:, index].mean()),
                      quantiles=np.quantile(array[:, index], [0., .5, .95, 1.]).tolist())
            for index, name in enumerate(names)}


def supervised_modes(output, target):
    """Read the actual supervised modes; never substitute oracle supervision."""
    assigned, _ = update_identity_bindings(output, target, [dict() for _ in range(target["current_mask"].shape[0])])
    forecast = output["forecast"]
    rows, slots = torch.nonzero((assigned >= 0) & forecast["forecast_valid_mask"], as_tuple=True)
    if not len(rows):
        return []
    gt = assigned[rows, slots]
    masks = target["future_mask"][rows, gt]
    receiver = DualCmpLoss._future_relative_to_track(target["future_xy"][rows, gt],
        target["current_boxes"][rows, gt, None, :2], output["track_box"][rows, slots, None, :2])
    actor = DualCmpLoss._receiver_displacement_to_actor(receiver,
        output["track_box"][rows, slots, 6, None], legacy_heading=False)
    predictions = forecast["loss_pred_trajs_local_layers"][rows, slots, -1]
    scores = forecast["loss_pred_scores_layers"][rows, slots, -1]
    losses = nearest_mode_gmm_loss(scores, predictions, actor, masks,
                                  forecast["loss_intention_points_layers"][rows, slots, -1])
    modes = losses["selected_mode"]
    points = torch.arange(len(rows), device=rows.device)
    xy = predictions[points, modes, :, :2]
    complete = masks.all(-1)
    return [dict(target_endpoint_m=float(actor[i, -1].norm()),
                 supervised_prediction_endpoint_m=float(xy[i, -1].norm()),
                 supervised_ade5_m=float((xy[i] - actor[i]).norm(dim=-1).mean()),
                 supervised_mode=int(modes[i])) for i in torch.where(complete)[0]]


def motion_input_summary(chunk, model):
    head = model.forecast_head
    packed = pack_mtr_actor_polylines(*chunk["head_args"][1:],
        legacy_heading=False, min_forecast_history_steps=head.min_forecast_history_steps,
        min_context_history_steps=head.min_context_history_steps,
        frame_period_s=head.frame_period_s, timestamp_scale=head.timestamp_scale)
    if packed is None:
        return None
    values = packed.batch_dict["input_dict"]["obj_trajs"]
    valid = values[packed.batch_dict["input_dict"]["obj_trajs_mask"]]
    return dict(count=len(valid), abs_mean=valid.abs().mean(0).cpu().tolist(),
                abs_max=valid.abs().amax(0).cpu().tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interventions", nargs="+", choices=("normal", "batch_statistics_only", "batch_context", "batch_dense", "batch_motion", "batch_intention", "unit_object_memory"),
                        default=["normal", "batch_statistics_only", "unit_object_memory"])
    parser.add_argument("--training-checkpoint-subset", action="store_true")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("refusing to overwrite diagnostic evidence")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("learning audit requires both GPU ranks")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    cache = Path(args.cache_dir)
    manifest = json.loads((cache / "manifest.json").read_text())
    chunks = {split: load_chunks(cache, manifest, split, rank) for split in ("train", "development")}
    results = []
    with torch.no_grad():
        for checkpoint in args.checkpoints:
            payload = torch.load(checkpoint, map_location="cpu")
            hypes, _ = load_checkpoint_hypes(DEFAULT_HYPES, payload)
            model = _model_from_checkpoint(hypes, checkpoint, device, allow_dirty_smoke=True)
            wrapper = CachedForecast(model).eval()
            batchnorms = [(name, module) for name, module in model.forecast_head.named_modules()
                          if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
            bn_summary = [{"name": name, "mean_abs": float(module.running_mean.abs().mean()),
                "var_min": float(module.running_var.min()), "var_max": float(module.running_var.max()),
                "batches": int(module.num_batches_tracked)} for name, module in batchnorms]
            record = dict(checkpoint=checkpoint, batchnorm=bn_summary, splits={})
            for split, batches in chunks.items():
                if split == "train" and args.training_checkpoint_subset:
                    indices = payload["provenance"]["paired_train_chunk_indices"]
                    batches = [batches[index] for index in indices]
                input_summary = []
                for intervention in args.interventions:
                    rows, raw_rows, coverage, diversity_rows, attention_trace, supervised = [], [], [], [], [], []
                    for chunk in batches:
                        data = move(chunk, device)
                        if intervention == "normal":
                            input_summary.append(motion_input_summary(data, model))
                        prefixes = {
                            "batch_context": ("context_encoder.",),
                            "batch_dense": ("decoder.dense_future_head.",),
                            "batch_motion": ("decoder.motion_reg_heads.", "decoder.motion_cls_heads.",
                                             "decoder.query_feature_fusion_layers."),
                            "batch_intention": ("decoder.intention_query_mlps.",),
                        }
                        selected = batchnorms if intervention == "batch_statistics_only" else [
                            (name, module) for name, module in batchnorms
                            if name.startswith(prefixes.get(intervention, ("__none__",)))]
                        saved = [(module, module.running_mean.clone(), module.running_var.clone(), module.num_batches_tracked.clone())
                                 for _, module in selected]
                        for module, _, _, _ in saved:
                            module.train()
                        decoder = model.forecast_head.decoder
                        original_attention = decoder.apply_cross_attention
                        def trace_attention(**kwargs):
                            is_object = kwargs["attention_layer"] in decoder.obj_decoder_layers
                            memory = kwargs["kv_feature"][kwargs["kv_mask"]]
                            original_norm = float(memory.norm(dim=-1).mean())
                            if intervention == "unit_object_memory" and is_object:
                                kwargs["kv_feature"] = torch.nn.functional.layer_norm(
                                    kwargs["kv_feature"], (kwargs["kv_feature"].shape[-1],))
                            result = original_attention(**kwargs)
                            attention_trace.append(dict(kind="object" if is_object else "source",
                                layer=kwargs["layer_idx"], memory_norm=original_norm,
                                query_std=float(result.std(dim=0, unbiased=False).mean())))
                            return result
                        decoder.apply_cross_attention = trace_attention
                        try:
                            output = wrapper(data)
                        finally:
                            decoder.apply_cross_attention = original_attention
                            for module, mean, variance, count in saved:
                                module.running_mean.copy_(mean)
                                module.running_var.copy_(variance)
                                module.num_batches_tracked.copy_(count)
                                module.eval()
                        local, cov = _forecast_rows(output, data["target"], data["keys"])
                        forecast = output["forecast"]
                        raw = forecast["loss_pred_trajs_local_layers"][:, :, -1]
                        shape = raw.shape
                        rotated = model.forecast_head._local_gmm_to_receiver(
                            raw.reshape(-1, *shape[2:]), output["track_box"][..., 6].reshape(-1)).reshape_as(raw)
                        raw_forecast = dict(forecast, pred_trajs=rotated)
                        raw_local, _ = _forecast_rows(dict(output, forecast=raw_forecast), data["target"], data["keys"])
                        rows.extend(local)
                        raw_rows.extend(raw_local)
                        coverage.extend(cov)
                        diversity_rows.extend(diversity(output, data["target"]))
                        supervised.extend(supervised_modes(output, data["target"]))
                    gathered = [None, None]
                    dist.all_gather_object(gathered, (rows, coverage, diversity_rows, input_summary, attention_trace, raw_rows, supervised))
                    if rank == 0:
                        record["splits"].setdefault(split, {})[intervention] = dict(
                            report=_forecast_report([v for group in gathered for v in group[0]],
                                [v for group in gathered for v in group[1]]),
                            raw64_oracle_diagnostic=_forecast_report([v for group in gathered for v in group[5]],
                                [v for group in gathered for v in group[1]]),
                            diversity=summarize([v for group in gathered for v in group[2]]),
                            supervised_complete_modes=[v for group in gathered for v in group[6]],
                            attention=[dict(kind=kind, layer=layer,
                                memory_norm_mean=float(np.mean([v["memory_norm"] for group in gathered for v in group[4] if v["kind"] == kind and v["layer"] == layer])),
                                query_std_mean=float(np.mean([v["query_std"] for group in gathered for v in group[4] if v["kind"] == kind and v["layer"] == layer])))
                                for kind in ("object", "source") for layer in range(6)
                                if any(v["kind"] == kind and v["layer"] == layer for group in gathered for v in group[4])])
                        if intervention == "normal":
                            valid = [v for group in gathered for v in group[3] if v]
                            count = sum(v["count"] for v in valid)
                            record["splits"][split]["motion_input"] = dict(count=count,
                                abs_mean=(sum(np.asarray(v["abs_mean"]) * v["count"] for v in valid) / count).tolist(),
                                abs_max=np.asarray([v["abs_max"] for v in valid]).max(0).tolist())
                dist.barrier()
            if rank == 0:
                results.append(record)
            del model, wrapper, payload, batchnorms
            torch.cuda.empty_cache()
    if rank == 0:
        Path(args.output).write_text(json.dumps({"status": "complete", "scope": "completed-checkpoint diagnosis, not trained ablations", "results": results}, indent=2))
        print(json.dumps({"status": "complete", "checkpoints": args.checkpoints}))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
