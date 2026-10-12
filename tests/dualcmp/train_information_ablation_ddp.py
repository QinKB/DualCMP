"""Paired real-forecast learning over verified frozen-perception inputs.

Completed research budgets are evaluation-only, not formal resume checkpoints.
Every learned feature path is also implemented by DualCmpClipModel.forward.
"""
import argparse
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset

from opencood.loss.dualcmp_loss import DualCmpLoss
from opencood.models.dualcmp.model import DualCmpClipModel
from opencood.models.dualcmp.source_evidence_policy import apply_evidence_policy
from opencood.models.dualcmp.forecast_normalization import FORECAST_NORMALIZATIONS
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES, load_checkpoint_hypes
from opencood.data_utils.dualcmp_dataloader import build_dataloader
from opencood.tools.dualcmp.eval_forecast_ddp import _forecast_rows, _forecast_report
from opencood.tools.dualcmp.online import _model_from_checkpoint, _sha256_file
from opencood.tools.dualcmp.train_forecast_ddp import _provenance, _ensure_forecast_ddp_backward
from opencood.tools.dualcmp.train_tracking_ddp import _assert_sync, _sync_buffers
from opencood.data_utils.v2v4real.post_processor import build_postprocessor
from opencood.tools.train_utils import to_device
from cached_clip_batching import collate_cached_clips, split_cached_chunk

VARIANTS = {
    "history": ("history", "legacy_additive_v1"),
    "fused_temporal": ("fused_temporal", "legacy_additive_v1"),
    "source_current": ("source_current", "legacy_additive_v1"),
    "source_temporal": ("source_temporal", "legacy_additive_v1"),
    "geometry_temporal": ("geometry_temporal", "legacy_additive_v1"),
    "receiver_temporal": ("receiver_temporal", "legacy_additive_v1"),
    "fused_spatiotemporal": ("fused_temporal", "factorized_qk_v1"),
    "source_spatiotemporal": ("source_temporal", "factorized_qk_v1"),
    "source_spatial_additive": ("source_temporal", "factorized_additive_v1"),
    "source_temporal_qk": ("source_temporal", "temporal_qk_v1"),
    "fused_temporal_qk": ("fused_temporal", "temporal_qk_v1"),
}


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(move(item, device) for item in value)
    if isinstance(value, SimpleNamespace):
        return SimpleNamespace(**move(vars(value), device))
    return value


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_model(hypes, stage_a, prior, variant, device, normalization="batch_norm_v1"):
    policy, feature = VARIANTS[variant]
    post = build_postprocessor(hypes["postprocess"], train=False)
    score = hypes["postprocess"]["target_args"]["score_threshold"]
    model = DualCmpClipModel(hypes["model"]["args"], post.generate_anchor_box(),
        hypes["preprocess"]["cav_lidar_range"], proposal_score_threshold=score,
        proposal_nms_thresh=hypes["postprocess"]["nms_thresh"], emission_threshold=score,
        forecast_args=dict(representation="source_temporal_aligned",
            intention_points_file=prior, aligned_evidence_policy=policy,
            source_feature_encoder=feature, forecast_normalization=normalization,
            capture_alignment=True, age_encoding=True))
    loaded = model.load_state_dict(stage_a, strict=False)
    if loaded.unexpected_keys or any(not key.startswith(("forecast_head.", "source_temporal_encoder.", "actor_source_tokenizer.")) for key in loaded.missing_keys):
        raise RuntimeError("Phase-A state cannot initialize the actual model")
    for name, parameter in model.named_parameters():
        if not name.startswith(("forecast_head.", "source_temporal_encoder.")):
            parameter.requires_grad = False
    return model.to(device)


def shared_source_initialization(model, source_initial):
    candidate = model.source_temporal_encoder.state_dict()
    for key in candidate:
        reference = key
        if key.startswith("spatial_encoder."):
            reference = key.replace("spatial_encoder.", "temporal_encoder.")
        if reference.startswith("temporal_encoder.0."):
            reference = reference.replace("temporal_encoder.0.", "temporal_encoder.layers.0.")
        if reference in source_initial and candidate[key].shape == source_initial[reference].shape:
            candidate[key] = source_initial[reference]
    model.source_temporal_encoder.load_state_dict(candidate, strict=True)


class CachedForecast(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        policy = self.model.aligned_evidence_policy
        kwargs = {}
        if policy != "history":
            raw = batch["fused_inputs"] if policy == "fused_temporal" else batch["source_inputs"]
            # Preserve the real actor axis before learned projections for
            # full-path shape/numerical parity. Empty rows remain masked and
            # never become observations or evidence.
            slots = batch["original_slots"]
            padded = []
            for value in raw:
                shape = (value.shape[0], self.model.max_actors) + tuple(value.shape[2:])
                expanded = value.new_zeros(shape)
                expanded[:, slots] = value
                padded.append(expanded)
            encoded = self.model.source_temporal_encoder(*apply_evidence_policy(tuple(padded), policy))
            encoded = {key: value[:, slots] for key, value in encoded.items()}
            b, a, l, t, k, c = encoded["tokens"].shape
            kwargs = dict(source_tokens=encoded["tokens"].reshape(b, a, l * t, k, c),
                          source_positions=encoded["positions"].reshape(b, a, l * t, k, 3),
                          source_token_mask=encoded["token_mask"].reshape(b, a, l * t, k))
        forecast = self.model.forecast_head(*batch["head_args"], **kwargs)
        output = dict(batch["output"], forecast=forecast)
        return output


def load_chunks(cache, manifest, split, rank):
    names = [name for name in manifest["selection"][split]["files"] if "_rank%d_" % rank in name]
    return [torch.load(cache / name, map_location="cpu") for name in sorted(names)]


def evaluate(wrapper, chunks, device):
    wrapper.eval()
    rows, coverage = [], []
    with torch.no_grad():
        for chunk in chunks:
            data = move(chunk, device)
            output = wrapper(data)
            local, cov = _forecast_rows(output, data["target"], data["keys"])
            rows.extend(local)
            coverage.extend(cov)
    gathered = [None, None]
    dist.all_gather_object(gathered, (rows, coverage))
    return _forecast_report([r for group in gathered for r in group[0]],
                            [r for group in gathered for r in group[1]])


@torch.no_grad()
def verify_clip_rebatch(model, chunks, device, enforce=True):
    model.eval()
    clips = [split_cached_chunk(chunk)[0] for chunk in (chunks[0], chunks[-1])]
    merged = collate_cached_clips(clips)
    wrapper = CachedForecast(model).eval()
    combined = wrapper(move(merged, device))["forecast"]["pred_trajs"]
    errors = []
    for row, chunk in enumerate((chunks[0], chunks[-1])):
        reference = wrapper(move(chunk, device))["forecast"]["pred_trajs"][0]
        indices = torch.searchsorted(merged["original_slots"], chunk["original_slots"]).to(device)
        errors.append(float((combined[row, indices] - reference).abs().max()))
    passed = max(errors) <= 1e-3
    if enforce and not passed:
        raise RuntimeError("clip rebatching changed inference beyond numerical tolerance: %g" % max(errors))
    return dict(max_forecast_error=max(errors), passed=passed, absolute_tolerance=1e-3, keys=merged["keys"],
                target_protocol="unchanged fixed-128 GT axes and original runtime actor slots")


def gradient_norm(module):
    values = [parameter.grad.float().norm() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.stack(values).sum()) if values else 0.


def full_replay_batch(hypes, manifest, rank):
    from prepare_information_cache_ddp import make_dataset
    dataset, _ = make_dataset(hypes, "test")
    # First four ordered indices give the same first two clips on each rank
    # as the original cache's unshuffled DistributedSampler.
    _, loader, _ = build_dataloader(Subset(dataset, manifest["selection"]["development"]["indices"][:4]),
        2, True, workers=2, rank=rank, world_size=2, training=False,
        collate_fn=dataset.collate_batch_test, pin_memory=True)
    return next(iter(loader))


@torch.no_grad()
def verify_full_replay(model, cached, runtime_batch, device):
    model.eval()
    keys = [(str(scene), int(receiver), int(frame)) for (scene, receiver), frame in zip(
        runtime_batch["scene_receiver_keys"], runtime_batch["runtime_frames"][-1]["receiver_frame_idx"])]
    if keys != cached["keys"]:
        raise RuntimeError("full replay did not select the cache's exact clip ordering")
    real = to_device(runtime_batch, device)
    captured = []
    from audit_task_fitness_ddp import capture_forward
    original = capture_forward(model.source_temporal_encoder, captured)
    try:
        outputs, _ = model(real["runtime_frames"], run_forecast=True)
    finally:
        model.source_temporal_encoder.forward = original
    data = move(cached, device)
    replay = CachedForecast(model).eval()(data)
    slots = data["original_slots"]
    full = outputs[-1]["forecast"]["pred_trajs"][:, slots]
    error = float((full - replay["forecast"]["pred_trajs"]).abs().max()) if slots.numel() else 0.
    if error > 1e-3:
        raise RuntimeError("new trained full-model/cache disagreement: %g" % error)
    full_rows, full_coverage = _forecast_rows(outputs[-1], real["target_frames"][-1], keys)
    cached_rows, cached_coverage = _forecast_rows(replay, data["target"], keys)
    full_report = _forecast_report(full_rows, full_coverage)
    cached_report = _forecast_report(cached_rows, cached_coverage)
    if full_report["coverage"] != cached_report["coverage"] or any(
            abs(full_report["model"][key] - cached_report["model"][key]) > 1e-6
            for key in full_report["model"]):
        raw = data["fused_inputs"] if model.aligned_evidence_policy == "fused_temporal" else data["source_inputs"]
        expected = apply_evidence_policy(raw, model.aligned_evidence_policy)
        input_errors = [float((a[:, slots].float() - b.float()).abs().max()) for a, b in zip(captured[0], expected)]
        raise RuntimeError("new full-model/cache metrics or population disagree: %s" % json.dumps(dict(
            prediction_error=error, full_coverage=full_report["coverage"], cached_coverage=cached_report["coverage"],
            full_metrics=full_report["model"], cached_metrics=cached_report["model"],
            encoder_input_errors=input_errors,
            box_error=float((outputs[-1]["track_box"][:, slots] - data["output"]["track_box"]).abs().max()),
            score_error=float((outputs[-1]["forecast"]["pred_scores"][:, slots] - replay["forecast"]["pred_scores"]).abs().max()))))
    return dict(max_forecast_error=error, coverage=full_report["coverage"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--tracker-checkpoint", required=True)
    parser.add_argument("--intention-points-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260901, 20261003])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--budget-id", choices=("frozen800", "shuffled16_overfit3200_v1", "shuffled128_information3200_v1"), default="frozen800")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--full-replay", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--schedule", choices=("constant", "phase_b_ratio"), default="constant")
    parser.add_argument("--training-report", action="store_true")
    parser.add_argument("--train-chunks-per-rank", type=int)
    parser.add_argument("--forecast-mean-weight", type=float, default=5.)
    parser.add_argument("--forecast-normalization", choices=FORECAST_NORMALIZATIONS, default="batch_norm_v1")
    parser.add_argument("--clip-sampling", choices=("fixed_cache_chunks", "shuffled_clips"), default="fixed_cache_chunks")
    parser.add_argument("--batch-size", type=int, default=2, help="clips per GPU, shuffled-clip path only")
    parser.add_argument("--budget-group", choices=("all", "baseline", "information", "information_remaining"), default="all")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("paired learning requires both GPUs")
    expected_steps = 800 if args.budget_id == "frozen800" else 3200
    if not 0 < args.steps <= expected_steps or (args.steps != expected_steps and not args.preflight):
        raise ValueError("steps disagree with the explicitly frozen research budget")
    if args.budget_id == "shuffled16_overfit3200_v1" and (
            args.clip_sampling != "shuffled_clips" or args.train_chunks_per_rank != 4
            or args.variants != ["source_temporal"] or args.seeds != [20260901]
            or args.forecast_normalization != "batch_norm_v1"
            or args.lr != 1e-4 or args.schedule != "phase_b_ratio"
            or args.forecast_mean_weight != 250. or args.batch_size != 2):
        raise ValueError("shuffled16 overfit budget must retain its frozen data/model/optimizer contract")
    registered_groups = {
        "baseline": ["source_temporal", "fused_temporal"],
        "information": ["source_current", "receiver_temporal", "geometry_temporal",
                        "source_spatiotemporal", "fused_spatiotemporal"],
    }
    registered_groups["all"] = registered_groups["baseline"] + registered_groups["information"]
    # The first information model already completed its frozen training budget.
    # It failed a post-training numerical gate and is retained, not retrained.
    registered_groups["information_remaining"] = registered_groups["information"][1:]
    if args.budget_id == "shuffled128_information3200_v1" and (
            args.clip_sampling != "shuffled_clips" or args.train_chunks_per_rank is not None
            or args.variants != registered_groups[args.budget_group]
            or args.seeds != [20260901] or args.forecast_normalization != "batch_norm_v1"
            or args.lr != 1e-4 or args.schedule != "phase_b_ratio"
            or args.forecast_mean_weight != 250. or args.batch_size != 4):
        raise ValueError("shuffled128 information budget must retain its frozen paired contract")
    if args.batch_size < 1 or (args.clip_sampling == "fixed_cache_chunks" and args.batch_size != 2):
        raise ValueError("cached fixed chunks require their original batch size")
    if not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("learning rate must be finite and positive")
    if not np.isfinite(args.forecast_mean_weight) or args.forecast_mean_weight < 0:
        raise ValueError("forecast mean weight must be finite and non-negative")
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError("refusing to overwrite a paired experiment")
    cache = Path(args.cache_dir)
    manifest = json.loads((cache / "manifest.json").read_text())
    prior_meta = json.loads(Path(args.intention_points_file).with_suffix(".json").read_text())
    if sorted(manifest.get("excluded_scenes", [])) != sorted(prior_meta.get("excluded_scenes", [])):
        raise RuntimeError("cache/prior split exclusions disagree")
    if prior_meta["sha256"] != _sha256_file(args.intention_points_file):
        raise RuntimeError("prior hash mismatch")
    if any(manifest["cache_replay"][name]["max_forecast_error"] > 1e-3 for name in manifest["selection"]):
        raise RuntimeError("cache never passed real-model replay")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    if rank == 0:
        out.mkdir(parents=True)
    dist.barrier()
    payload = torch.load(args.tracker_checkpoint, map_location="cpu")
    hypes, detector_provenance = load_checkpoint_hypes(DEFAULT_HYPES, payload)
    state = payload.get("model", payload.get("model_state_dict", payload))
    state = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
    if any(key.startswith("forecast_head.") for key in state):
        raise RuntimeError("paired experiments must initialize from Phase A, not test-trained forecast weights")
    teacher = torch.load(manifest["teacher_checkpoint"], map_location="cpu")
    if any(not torch.equal(value, teacher["model"][key]) for key, value in state.items()):
        raise RuntimeError("cache perception differs from Phase A")
    del teacher
    train = load_chunks(cache, manifest, "train", rank)
    train_indices = list(range(len(train)))
    if args.train_chunks_per_rank is not None:
        if not 1 <= args.train_chunks_per_rank <= len(train):
            raise ValueError("invalid frozen train chunk count")
        train_indices = np.linspace(0, len(train) - 1, args.train_chunks_per_rank, dtype=int).tolist()
        train = [train[index] for index in train_indices]
    development = load_chunks(cache, manifest, "development", rank)
    clip_pool = None
    if args.clip_sampling == "shuffled_clips":
        other = load_chunks(cache, manifest, "train", 1 - rank)
        selected_other = [other[index] for index in train_indices]
        clip_pool = [clip for chunk in train + selected_other for clip in split_cached_chunk(chunk)]
        clip_pool.sort(key=lambda clip: clip["keys"][0])
        if len({clip["keys"][0] for clip in clip_pool}) != len(clip_pool):
            raise RuntimeError("clip-level sampler contains duplicates")
        del other, selected_other
    runtime_batch = full_replay_batch(hypes, manifest, rank) if args.full_replay else None
    if args.preflight:
        development = development[:1]
    # Confirmation data are not loaded or evaluated during model selection.
    results = []
    inference_consistency_passed = True
    for seed in args.seeds:
        seed_all(seed)
        prototype = create_model(hypes, state, args.intention_points_file, "source_temporal", device, args.forecast_normalization)
        head_initial = {key: value.detach().cpu().clone() for key, value in prototype.forecast_head.state_dict().items()}
        source_initial = {key: value.detach().cpu().clone() for key, value in prototype.source_temporal_encoder.state_dict().items()}
        del prototype
        torch.cuda.empty_cache()
        for variant in args.variants:
            seed_all(seed)
            model = create_model(hypes, state, args.intention_points_file, variant, device, args.forecast_normalization)
            model.forecast_head.load_state_dict(head_initial, strict=True)
            shared_source_initialization(model, source_initial)
            _, train_loader, clip_sampler = build_dataloader(clip_pool, args.batch_size, True,
                workers=2, rank=rank, world_size=2, seed=seed, training=True,
                collate_fn=collate_cached_clips, pin_memory=True) if clip_pool is not None else (None, None, None)
            wrapper = CachedForecast(model).train()
            wrapper.model.forecast_head.train()
            wrapper = DDP(wrapper, device_ids=[rank], broadcast_buffers=False)
            optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
            # Same proportions and lower bound as the real Phase-B entry.
            decay_steps = [round(args.steps * epoch / 30.) for epoch in (22, 24, 26, 28)]
            def lr_multiplier(current_step):
                count = sum(current_step >= threshold for threshold in decay_steps) if args.schedule == "phase_b_ratio" else 0
                return max(.5 ** count, 1e-6 / args.lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
            criterion = DualCmpLoss({"detector": hypes["loss"]["args"], "forecast_mean_weight": args.forecast_mean_weight}).to(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            max_source_grad = max_spatial_grad = max_temporal_grad = 0.
            order = []
            for step in range(args.steps):
                if train_loader is not None:
                    if step % len(train_loader) == 0:
                        clip_sampler.set_epoch(step // len(train_loader))
                        train_iterator = iter(train_loader)
                    data = move(next(train_iterator), device)
                else:
                    if step % len(train) == 0:
                        order = np.random.RandomState(seed + step // len(train)).permutation(len(train)).tolist()
                    data = move(train[order[step % len(train)]], device)
                optimizer.zero_grad(set_to_none=True)
                output = wrapper(data)
                loss = criterion([output], [data["target"]])
                counts = torch.stack([loss[key] for key in ("C_gmm", "C_mode", "C_mean_reg", "C_dense_gmm")]).detach()
                dist.all_reduce(counts)
                objective = 2 * sum(loss[key] * weight / count.clamp_min(1.) for key, weight, count in zip(
                    ("S_gmm", "S_mode", "S_mean_reg", "S_dense_gmm"), (1., 1., args.forecast_mean_weight, 1.), counts))
                objective = _ensure_forecast_ddp_backward(wrapper, objective, loss, output)
                objective.backward()
                encoder = wrapper.module.model.source_temporal_encoder
                max_source_grad = max(max_source_grad, gradient_norm(encoder))
                max_temporal_grad = max(max_temporal_grad, gradient_norm(encoder.temporal_encoder))
                if encoder.spatial_encoder is not None:
                    max_spatial_grad = max(max_spatial_grad, gradient_norm(encoder.spatial_encoder))
                norm = torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1000.)
                if not torch.isfinite(norm):
                    raise RuntimeError("paired experiment rejected non-finite optimizer step")
                optimizer.step()
                scheduler.step()
            elapsed = time.monotonic() - started
            if train_loader is not None:
                del train_iterator
            del train_loader, clip_sampler
            _sync_buffers(wrapper)
            _assert_sync(wrapper.module.model)
            record = dict(variant=variant, seed=seed, steps=args.steps, global_batch=2 * args.batch_size,
                elapsed_s=elapsed, peak_memory_mib=torch.cuda.max_memory_allocated(device) / 2 ** 20,
                clips_per_second_per_rank=args.steps * args.batch_size / elapsed,
                max_source_grad=max_source_grad, max_temporal_grad=max_temporal_grad,
                max_spatial_grad=max_spatial_grad, last_gradient_norm=float(norm),
                last_loss={key: float(loss[key]) for key in ("L_gmm", "L_mode", "L_mean_reg", "L_dense_gmm")})
            ranks = [None, None]
            dist.all_gather_object(ranks, record)
            policy, feature = VARIANTS[variant]
            provenance = _provenance()
            provenance.update(detector_provenance)
            provenance.update(forecast_representation="source_temporal_aligned",
                aligned_evidence_policy=policy, source_feature_encoder=feature,
                forecast_normalization=args.forecast_normalization,
                capture_alignment=True, age_encoding=True, mean_source_aggregation=False,
                motion_input_contract="causal_velocity_acceleration_and_predicted_dense_velocity_v1",
                decoder_query_contract="persistent_static_intention_position_v1",
                intention_points_file=args.intention_points_file,
                intention_points_sha256=_sha256_file(args.intention_points_file),
                forecast_coordinate_frame="current_receiver", sequence_policy=prior_meta["sequence_policy"],
                excluded_scene_ids=manifest.get("excluded_scenes", []),
                information_cache_manifest_sha256=_sha256_file(cache / "manifest.json"),
                information_experiment_script_sha256=_sha256_file(__file__),
                phase_a_checkpoint_sha256=_sha256_file(args.tracker_checkpoint),
                stage="forecast", partial_epoch=True, research_budget_complete=not args.preflight,
                partner_delay_ms=100., train_delays_ms=[100.], delay_seed=20260901,
                forecast_mean_weight=args.forecast_mean_weight, initial_lr=args.lr, weight_decay=1e-4)
            provenance.update(initialization_seed=seed, research_updates=args.steps,
                paired_global_batch=2 * args.batch_size, paired_optimizer="AdamW", paired_schedule=args.schedule,
                paired_variants=args.variants, paired_train_chunk_indices=train_indices,
                paired_clip_sampling=args.clip_sampling,
                research_budget_id=args.budget_id,
                research_budget_group=args.budget_group,
                paired_clip_batching_sha256=_sha256_file(Path(__file__).with_name("cached_clip_batching.py")))
            checkpoint = out / ("%s_seed%d.pt" % (variant, seed))
            if rank == 0:
                torch.save(dict(model=wrapper.module.model.state_dict(), provenance=provenance,
                    epoch=0, step=args.steps, global_samples=args.steps * 2 * args.batch_size), checkpoint)
            dist.barrier()
            del wrapper, model, optimizer, criterion, output, loss, objective, data, encoder
            torch.cuda.empty_cache()
            restored = _model_from_checkpoint(hypes, checkpoint, device, allow_dirty_smoke=True)
            replay = CachedForecast(restored)
            rebatch = verify_clip_rebatch(restored, train, device, enforce=False) if clip_pool is not None else None
            if rebatch is not None:
                rank_rebatch = [None, None]
                dist.all_gather_object(rank_rebatch, rebatch)
                rebatch = dict(rebatch, rank_results=rank_rebatch,
                    max_forecast_error=max(item["max_forecast_error"] for item in rank_rebatch),
                    passed=all(item["passed"] for item in rank_rebatch))
                inference_consistency_passed = inference_consistency_passed and rebatch["passed"]
            full_replay = verify_full_replay(restored, development[0], runtime_batch, device) if args.full_replay else None
            report = evaluate(replay, development, device)
            training_report = evaluate(replay, train, device) if args.training_report else None
            result = dict(variant=variant, seed=seed, checkpoint=str(checkpoint),
                ranks=ranks, development=report, training=training_report, full_replay=full_replay,
                clip_rebatch=rebatch,
                inference_consistency_status="passed" if rebatch is None or rebatch["passed"] else "rejected",
                scope="preflight only" if args.preflight else "paired learning, fixed completed budget")
            if rank == 0:
                (out / ("%s_seed%d.json" % (variant, seed))).write_text(json.dumps(result, indent=2))
                results.append(result)
            del restored, replay
            torch.cuda.empty_cache()
    if rank == 0:
        status = "complete" if inference_consistency_passed else "acceptance_failed"
        (out / "results.json").write_text(json.dumps({"status": status, "preflight": args.preflight,
            "cache_dir": args.cache_dir, "results": results}, indent=2))
        print(json.dumps({"status": status, "variants": args.variants, "seeds": args.seeds, "steps": args.steps}))
    dist.barrier()
    dist.destroy_process_group()
    if not inference_consistency_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
