"""Phase-aware recurrent CoLIF trainer with single- and multi-GPU support."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from opencood.data_utils.v2v4real.datasets import build_v2v4real_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils
from opencood.tools.colif_train_utils import (
    build_colif_optimizer,
    config_sha256,
    configure_trainable_phase,
    load_colif_weights,
    load_cobevt_trunk,
    load_mtr_decoder,
    resume_colif_checkpoint,
    save_colif_checkpoint,
    set_colif_train_mode,
)
from opencood.tools.colif.eval_utils import (
    build_colif_evaluation_loader,
    evaluate_colif_strict_online,
)
from opencood.tools.colif.receiver_protocol import apply_receiver_override


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypes_yaml", required=True)
    parser.add_argument("--output-dir", default="opencood/logs/colif_smoke")
    parser.add_argument("--init-cobevt-checkpoint")
    parser.add_argument("--init-mtr-checkpoint")
    parser.add_argument("--init-colif-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--smoke-batches", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detect-anomaly", action="store_true")
    parser.add_argument("--smoke-force-birth", action="store_true")
    parser.add_argument("--phase", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--receiver-ids",
        help="Comma-separated CAV IDs; use 0 for one ego or 0,1 for CMP-style multi-ego",
    )
    parser.add_argument(
        "--observation-score-threshold", type=float,
        help="Override lifecycle keep threshold; birth remains no lower than keep",
    )
    parser.add_argument(
        "--eval-split", choices=("test",), default="test",
        help="CMP-compatible protocol: evaluate the official test split only",
    )
    parser.add_argument("--eval-workers", type=int, default=2)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--disable-actor-cooperative-fusion", action="store_true")
    parser.add_argument("--disable-temporal-history", action="store_true")
    parser.add_argument("--disable-proposal-refinement", action="store_true")
    parser.add_argument(
        "--forecast-refinement-type",
        choices=("cross_mode", "trajectory_source"),
    )
    parser.add_argument(
        "--forecast-inference-layer",
        choices=("proposal", "refined"),
    )
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_data_worker(_worker_id: int) -> None:
    """Keep dataset randomness independent of model parameter creation."""

    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _ddp_parameter_sync_report(model, device):
    """Return a cheap epoch-boundary proof that every rank has equal weights."""

    checksum = torch.zeros(3, dtype=torch.float64, device=device)
    for parameter in model.parameters():
        value = parameter.detach().double()
        checksum[0] += value.sum()
        checksum[1] += value.square().sum()
        checksum[2] += value.numel()
    if not (dist.is_available() and dist.is_initialized()):
        return {"checked": False, "max_checksum_delta": 0.0}
    gathered = [torch.zeros_like(checksum) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, checksum)
    stacked = torch.stack(gathered)
    delta = (stacked - stacked[0]).abs().max()
    if float(delta) > 1e-8:
        raise RuntimeError("DDP parameter checksum diverged across ranks")
    return {"checked": True, "max_checksum_delta": float(delta)}


def _synchronize_ddp_buffers(model, device):
    """Make the frozen/evaluated model identical on every rank.

    DDP broadcasts buffers at the beginning of a wrapped training forward.
    BatchNorm running statistics can therefore differ after the final local
    batch.  Explicit epoch-boundary synchronization is required before rank 0
    saves a checkpoint and before ranks evaluate separate test scenes.
    """

    if not (dist.is_available() and dist.is_initialized()):
        return {"checked": False, "max_checksum_delta": 0.0}
    for buffer in model.buffers():
        dist.broadcast(buffer, src=0)
    checksum = torch.zeros(3, dtype=torch.float64, device=device)
    for buffer in model.buffers():
        value = buffer.detach().double()
        checksum[0] += value.sum()
        checksum[1] += value.square().sum()
        checksum[2] += value.numel()
    gathered = [torch.zeros_like(checksum) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, checksum)
    delta = (torch.stack(gathered) - gathered[0]).abs().max()
    if float(delta) > 1e-8:
        raise RuntimeError("DDP model buffers diverged after synchronization")
    return {"checked": True, "max_checksum_delta": float(delta)}


def main() -> None:
    args = _arguments()
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    hypes = load_yaml(args.hypes_yaml)
    receiver_ids = apply_receiver_override(hypes, args.receiver_ids)
    if args.disable_actor_cooperative_fusion:
        hypes["model"]["args"]["actor_cooperative_fusion"]["enabled"] = False
    if args.disable_temporal_history:
        hypes["model"]["args"]["forecast"]["temporal_history"] = False
    if args.disable_proposal_refinement:
        hypes["model"]["args"]["forecast"]["proposal_refinement"] = False
    if args.forecast_refinement_type is not None:
        hypes["model"]["args"]["forecast"]["refinement_type"] = (
            args.forecast_refinement_type
        )
    if args.forecast_inference_layer is not None:
        hypes["model"]["args"]["forecast"]["inference_layer"] = (
            args.forecast_inference_layer
        )
    if args.observation_score_threshold is not None:
        threshold = float(args.observation_score_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("observation score threshold must be in [0, 1]")
        hypes["model"]["args"]["state"]["keep_threshold"] = threshold
        hypes["model"]["args"]["state"]["birth_threshold"] = max(
            threshold,
            float(hypes["model"]["args"]["state"].get("birth_threshold", threshold)),
        )
    if args.smoke_force_birth:
        if args.smoke_batches is None:
            raise ValueError("--smoke-force-birth is allowed only with --smoke-batches")
        state_config = hypes["model"]["args"]["state"]
        state_config["keep_threshold"] = 0.001
        state_config["birth_threshold"] = 0.005
    phase = int(hypes.get("train_params", {}).get("phase", 1))
    if args.phase is not None:
        phase = int(args.phase)
        hypes["train_params"]["phase"] = phase
    if phase not in (1, 2):
        raise ValueError(
            "Fused CoLIF now uses two stages only: phase 1 tracker, "
            "phase 2 forecast-conditioned recurrence"
        )
    epochs = (
        int(args.epochs)
        if args.epochs is not None
        else int(hypes["train_params"]["epoches"])
    )
    if epochs < 1:
        raise ValueError("epochs must be positive")
    # The CLI duration and scheduler horizon are one contract.  Previously a
    # 10-epoch command silently kept the YAML's 20-epoch cosine schedule.
    hypes["train_params"]["epoches"] = epochs
    hypes["lr_scheduler"]["epoches"] = epochs
    phase2_future_prior_mode = str(
        hypes["train_params"].get("phase2_future_prior_mode", "none")
    )
    valid_future_prior_modes = (
        "none",
        "association",
        "propagation_detached",
        "propagation_bptt",
        "detached",
        "recurrent_bptt",
    )
    if phase2_future_prior_mode not in valid_future_prior_modes:
        raise ValueError("Invalid train_params.phase2_future_prior_mode")
    active_future_prior_mode = (
        phase2_future_prior_mode if phase == 2 else "none"
    )
    hypes["model"]["args"]["future_adapter"]["mode"] = "none"
    hypes["model"]["args"]["future_adapter"]["phase2_mode"] = "none"
    hypes["model"]["args"]["forecast_motion_prior"]["mode"] = (
        active_future_prior_mode
    )
    hypes["model"]["args"]["forecast_motion_prior"]["phase2_mode"] = (
        phase2_future_prior_mode
    )
    hypes["model"]["args"]["forecast"]["enabled"] = phase >= 2
    warmup_key = "phase2_warmup_length" if phase == 2 else "phase1_warmup_length"
    causal_warmup = int(hypes["train_params"].get(warmup_key, 0))
    if causal_warmup < 0:
        raise ValueError("train_params.%s must be non-negative" % warmup_key)
    hypes["fusion"]["args"]["warmup_length"] = causal_warmup
    unroll_key = "phase2_unroll_length" if phase == 2 else "phase1_unroll_length"
    phase_unroll = int(
        hypes["train_params"].get(
            unroll_key, hypes["fusion"]["args"].get("unroll_length", 1)
        )
    )
    if phase_unroll < 1:
        raise ValueError("train_params.%s must be positive" % unroll_key)
    hypes["fusion"]["args"]["unroll_length"] = phase_unroll
    world_size = int(__import__("os").environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", "0"))
    rank = int(__import__("os").environ.get("RANK", "0"))
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    def log_record(payload):
        if rank != 0:
            return
        serialized = json.dumps(payload, sort_keys=True)
        print(serialized)
        with (output_dir / "train_log.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
    seed = int(hypes.get("train_params", {}).get("seed", 20))
    _seed_everything(seed + rank)

    device = torch.device(
        "cuda:%d" % local_rank if distributed else (
            args.device if torch.cuda.is_available() else "cpu"
        )
    )
    dataset = build_v2v4real_dataset(hypes, train=True)
    workers = args.workers
    if workers is None:
        workers = int(hypes["train_params"].get("num_workers", 0))
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
    ) if distributed else None
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = int(hypes["train_params"].get("batch_size", 1))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    hypes["train_params"]["batch_size"] = batch_size
    data_generator = torch.Generator()
    data_generator.manual_seed(seed + rank)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        collate_fn=dataset.collate_batch_train,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=_seed_data_worker,
        generator=data_generator,
    )

    model = train_utils.create_model(hypes).to(device)
    # Keep CMP/MTR BatchNorm modules local.  MTR executes conditionally only
    # when a rank has active actors; converting it to SyncBatchNorm would make
    # ranks enter different collective-call counts and deadlock in forward.
    # DDP broadcasts rank-0 running buffers at each clip boundary instead.
    phase_report = configure_trainable_phase(model, phase)
    log_record({"phase_configuration": phase_report})
    criterion = train_utils.create_loss(hypes).to(device)
    optimizer = build_colif_optimizer(hypes, model)
    grad_accum_steps = args.grad_accum_steps
    if grad_accum_steps is None:
        grad_accum_steps = int(hypes["train_params"].get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be positive")
    updates_per_epoch = math.ceil(len(loader) / grad_accum_steps)
    scheduler = train_utils.setup_lr_schedular(
        hypes, optimizer, updates_per_epoch
    )
    log_record({"training_data": {
            "dataset_records": len(dataset),
            "batches_per_rank": len(loader),
            "batch_size_per_rank": batch_size,
            "runtime_clip_length": int(dataset.runtime_clip_length),
            "loss_clip_length": int(dataset.unroll_length),
            "warmup_length": int(dataset.warmup_length),
            "phase2_future_prior_mode": active_future_prior_mode,
            "grad_accum_steps": grad_accum_steps,
            "updates_per_epoch": updates_per_epoch,
            "world_size": world_size,
            "receiver_ids": list(receiver_ids),
        }})
    evaluation_dataset = None
    evaluation_loader = None
    if not args.no_eval and args.smoke_batches is None:
        evaluation_dataset, evaluation_loader = build_colif_evaluation_loader(
            hypes,
            split=args.eval_split,
            rank=rank,
            world_size=world_size,
            workers=args.eval_workers,
        )
        log_record({"evaluation_data": {
                "evaluation_split": args.eval_split,
                "evaluation_records_global": len(evaluation_dataset),
                "evaluation_records_rank0": len(evaluation_loader),
                "evaluation_sequence_sharding": "whole_scene_receiver",
                "evaluation_metric_sync": "all_reduce_raw_sums_and_counts",
                "receiver_ids": list(receiver_ids),
            }})
    amp_phases = tuple(
        int(value)
        for value in hypes["train_params"].get("amp_phases", (1, 2))
    )
    amp_enabled = (
        bool(hypes["train_params"].get("amp", True))
        and device.type == "cuda"
        and phase in amp_phases
    )
    log_record({"precision": {
        "amp_enabled": amp_enabled,
        "amp_phases": list(amp_phases),
        "reason": (
            "phase_requires_lidar_or_recurrent_fp32"
            if phase not in amp_phases and not amp_enabled else "configured_amp"
        ),
    }})
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled,
        init_scale=float(hypes["train_params"].get("amp_init_scale", 4096.0)),
    )
    start_epoch = 0
    global_step = 0

    if args.resume and any(
        (args.init_cobevt_checkpoint, args.init_colif_checkpoint, args.init_mtr_checkpoint)
    ):
        raise ValueError(
            "--resume cannot be combined with initialization checkpoints"
        )
    if args.init_cobevt_checkpoint and args.init_colif_checkpoint:
        raise ValueError("Use CoBEVT initialization or a CoLIF phase checkpoint, not both")
    if args.init_cobevt_checkpoint:
        report = load_cobevt_trunk(model, args.init_cobevt_checkpoint)
        log_record({"cobevt_initialization": report})
    if args.init_colif_checkpoint:
        report = load_colif_weights(model, args.init_colif_checkpoint)
        log_record({"colif_initialization": report})
    if args.init_mtr_checkpoint:
        report = load_mtr_decoder(model, args.init_mtr_checkpoint)
        log_record({"mtr_initialization": report})
    if args.resume:
        checkpoint = resume_colif_checkpoint(
            args.resume, model, optimizer, scaler, scheduler
        )
        expected_hash = checkpoint.get("config_hash")
        current_hash = config_sha256(hypes)
        if expected_hash is not None and expected_hash != current_hash:
            raise ValueError(
                "Resume config mismatch: checkpoint=%s current=%s"
                % (expected_hash, current_hash)
            )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])

    training_model = model
    if distributed:
        training_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            broadcast_buffers=True,
        )

    max_grad_norm_key = (
        "phase2_max_grad_norm" if phase == 2 else "max_grad_norm"
    )
    max_grad_norm = float(
        hypes["train_params"].get(
            max_grad_norm_key,
            hypes["train_params"].get("max_grad_norm", 5.0),
        )
    )
    if max_grad_norm <= 0.0:
        raise ValueError("gradient clipping norm must be positive")
    log_record({"gradient_clipping": {
        "max_grad_norm": max_grad_norm,
        "phase": phase,
    }})
    for epoch in range(start_epoch, epochs):
        training_model.train()
        set_colif_train_mode(model, phase)
        if sampler is not None:
            sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        epoch_statistics = torch.zeros(6, dtype=torch.float64, device=device)
        accumulated_has_signal = False
        for batch_index, batch in enumerate(loader):
            profile_first_batch = batch_index == 0 and rank == 0
            if profile_first_batch and device.type == "cuda":
                torch.cuda.synchronize(device)
            profile_start = time.perf_counter()
            runtime_frames = train_utils.to_device(batch["runtime_frames"], device)
            target_frames = train_utils.to_device(batch["target_frames"], device)
            loss_start_index = int(batch.get("loss_start_index", 0))
            if len(target_frames) != len(runtime_frames) - loss_start_index:
                raise ValueError("Runtime/target clip lengths violate warmup contract")
            if profile_first_batch and device.type == "cuda":
                torch.cuda.synchronize(device)
            profile_after_transfer = time.perf_counter()
            if profile_first_batch:
                log_record({"first_batch_stage": {
                    "stage": "device_transfer_complete",
                    "elapsed_s": profile_after_transfer - profile_start,
                }})
            is_last_batch = batch_index + 1 == len(loader)
            should_step = (batch_index + 1) % grad_accum_steps == 0 or is_last_batch
            group_start = (batch_index // grad_accum_steps) * grad_accum_steps
            group_size = min(grad_accum_steps, len(loader) - group_start)
            sync_context = (
                contextlib.nullcontext()
                if not distributed or should_step
                else training_model.no_sync()
            )
            with sync_context:
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    total_loss = torch.zeros((), device=device)
                    local_signal_count = torch.zeros((), device=device)
                    supervision_state = None
                    bptt_len = int(
                        hypes["train_params"].get("bptt_len", len(runtime_frames))
                    )
                    # One DDP forward owns the entire causal clip.  This is
                    # required when conditional forecast branches differ
                    # across ranks; several wrapper forwards followed by one
                    # backward leave the reducer prepared only for the last
                    # frame and can stall indefinitely.
                    outputs_by_time, state, aux_by_time = training_model(
                        runtime_frames,
                        None,
                        bptt_len,
                        loss_start_index,
                    )
                    if profile_first_batch and device.type == "cuda":
                        torch.cuda.synchronize(device)
                    profile_after_forward = time.perf_counter()
                    if profile_first_batch:
                        log_record({"first_batch_stage": {
                            "stage": "model_forward_complete",
                            "elapsed_s": profile_after_forward - profile_after_transfer,
                        }})
                    for outputs, aux, target_frame in zip(
                        outputs_by_time[loss_start_index:],
                        aux_by_time[loss_start_index:],
                        target_frames,
                    ):
                        loss_dict, count_dict, supervision_state = criterion.forward_frame(
                            outputs, aux, target_frame, supervision_state
                        )
                        total_loss = total_loss + criterion.weighted_sum(
                            loss_dict,
                            count_dict,
                            {
                                "phase": phase,
                                "future_prior_mode": active_future_prior_mode,
                            },
                        )
                        if phase == 2:
                            local_signal_count = (
                                local_signal_count
                                + count_dict["gmm"].detach().float()
                            )
                    total_loss = (
                        total_loss
                        + outputs_by_time[-1]["training_anchor"]
                    )
                    if phase == 2:
                        global_signal_count = local_signal_count.clone()
                        if distributed:
                            dist.all_reduce(
                                global_signal_count, op=dist.ReduceOp.SUM
                            )
                        batch_has_signal = bool(global_signal_count > 0)
                    else:
                        batch_has_signal = True
                    accumulated_has_signal = (
                        accumulated_has_signal or batch_has_signal
                    )
                    last_loss_components = {
                        name: float(
                            value.detach().float()
                            / count_dict.get(name, 1.0).detach().float().clamp_min(1.0)
                        )
                        for name, value in loss_dict.items()
                    }
                    total_loss = total_loss / (len(target_frames) * group_size)
                    reported_loss = total_loss.detach() * group_size
                if profile_first_batch and device.type == "cuda":
                    torch.cuda.synchronize(device)
                profile_after_loss = time.perf_counter()
                if profile_first_batch:
                    log_record({"first_batch_stage": {
                        "stage": "criterion_and_anchor_complete",
                        "elapsed_s": profile_after_loss - profile_after_forward,
                    }})
                if not torch.isfinite(total_loss):
                    raise FloatingPointError(
                        "Non-finite CoLIF loss at epoch=%d batch=%d" % (epoch, batch_index)
                    )
                scaler.scale(total_loss).backward()
                if profile_first_batch and device.type == "cuda":
                    torch.cuda.synchronize(device)
                profile_after_backward = time.perf_counter()

            if batch_index == 0 and rank == 0:
                unused_trainable = [
                    name
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and parameter.grad is None
                ]
                log_record({"first_backward_parameter_coverage": {
                    "unused_trainable_count": len(unused_trainable),
                    "unused_trainable_parameters_sample": unused_trainable[:32],
                }})
                if active_future_prior_mode != "none":
                    prior_aux = aux_by_time[loss_start_index:]
                    log_record({"first_backward_future_prior_coverage": {
                        "valid_query_frames": int(sum(
                            int(item["future_prior_valid_mask"].sum())
                            for item in prior_aux
                        )),
                        "selected_forecast_query_frames": int(sum(
                            int(item[
                                "future_prior_selected_forecast_mask"
                            ].sum())
                            for item in prior_aux
                        )),
                    }})
                log_record({"first_batch_timing_s": {
                    "device_transfer": profile_after_transfer - profile_start,
                    "model_forward": profile_after_forward - profile_after_transfer,
                    "criterion_and_anchor": profile_after_loss - profile_after_forward,
                    "backward_and_ddp": profile_after_backward - profile_after_loss,
                }})

            epoch_statistics[0] += reported_loss.double()
            epoch_statistics[1] += 1.0
            epoch_statistics[2] += outputs["active_mask"].sum().double()
            epoch_statistics[3] += aux["birth_mask"].sum().double()

            gradient_norm = torch.zeros((), device=device)
            optimizer_step_skipped = False
            optimizer_step_no_signal = False
            if should_step:
                if not accumulated_has_signal:
                    # Complete DDP backward above, but do not change Adam
                    # moments, AMP scale or LR when the global clip has no
                    # forecast target at all.
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step_no_signal = True
                    epoch_statistics[5] += 1.0
                else:
                    scaler.unscale_(optimizer)
                    gradient_terms = [
                        parameter.grad.detach().float().norm(2)
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    ]
                    gradient_norm = (
                        torch.stack(gradient_terms).norm(2)
                        if gradient_terms else torch.zeros((), device=device)
                    )
                    gradients_finite = bool(torch.isfinite(gradient_norm))
                    if gradients_finite:
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_grad_norm,
                            error_if_nonfinite=True,
                        )
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_step_skipped = scaler.get_scale() < scale_before
                    if not gradients_finite and not optimizer_step_skipped:
                        raise FloatingPointError(
                            "Non-finite gradients were not rejected by GradScaler"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    if optimizer_step_skipped:
                        epoch_statistics[4] += 1.0
                    else:
                        global_step += 1
                        if hypes["lr_scheduler"]["core_method"] == "cosineannealwarm":
                            scheduler.step_update(global_step)
                accumulated_has_signal = False
            if rank == 0 and (
                batch_index % args.log_every == 0
                or should_step and is_last_batch
                or optimizer_step_skipped
                or optimizer_step_no_signal
            ):
                log_record({"training_step": {
                    "epoch": epoch,
                    "batch": batch_index,
                    "global_step": global_step,
                    "loss": float(total_loss.detach()),
                    "gradient_norm": (
                        float(gradient_norm) if torch.isfinite(gradient_norm) else None
                    ),
                    "world_size": world_size,
                    "optimizer_step": bool(should_step),
                    "optimizer_step_skipped": bool(optimizer_step_skipped),
                    "optimizer_step_no_signal": bool(optimizer_step_no_signal),
                    "amp_scale": float(scaler.get_scale()),
                    "learning_rates": [
                        float(group["lr"]) for group in optimizer.param_groups
                    ],
                    "active_queries": int(outputs["active_mask"].sum()),
                    "birth_observations": int(aux["birth_mask"].sum()),
                    "measurement_gain_mean": (
                        float(outputs["measurement_gain"][
                            outputs["measurement_observed_mask"]
                        ].mean())
                        if bool(outputs["measurement_observed_mask"].any())
                        else None
                    ),
                    "last_frame_loss_components": last_loss_components,
                }})
            if args.smoke_batches is not None and batch_index + 1 >= args.smoke_batches:
                break

        if hypes["lr_scheduler"]["core_method"] != "cosineannealwarm":
            scheduler.step()
        if distributed:
            dist.all_reduce(epoch_statistics, op=dist.ReduceOp.SUM)
        buffer_synchronization = _synchronize_ddp_buffers(model, device)
        synchronization = _ddp_parameter_sync_report(model, device)
        epoch_summary = {
            "epoch": epoch,
            "epoch_complete": args.smoke_batches is None,
            "global_micro_batches": int(epoch_statistics[1].item()),
            "mean_clip_loss": float(
                epoch_statistics[0] / epoch_statistics[1].clamp_min(1.0)
            ),
            "mean_active_queries_per_last_frame": float(
                epoch_statistics[2] / epoch_statistics[1].clamp_min(1.0)
            ),
            "birth_observations": int(epoch_statistics[3].item()),
            "global_step": global_step,
            "world_size": world_size,
            "amp_overflow_skipped_steps": int(
                epoch_statistics[4].item() / world_size
            ),
            "no_signal_skipped_steps": int(
                epoch_statistics[5].item() / world_size
            ),
            "ddp_parameter_synchronization": synchronization,
            "ddp_buffer_synchronization": buffer_synchronization,
        }
        log_record({"epoch_summary": epoch_summary})
        # Persist the completed epoch before the more expensive online test
        # pass.  A test-side failure must never discard hours of valid training.
        if rank == 0:
            checkpoint_path = output_dir / ("checkpoint_epoch_%03d.pth" % epoch)
            save_colif_checkpoint(
                str(checkpoint_path), model, optimizer, scaler, scheduler,
                epoch, global_step, hypes
            )
        if distributed:
            dist.barrier()
        evaluation_metrics = None
        eval_frequency = int(hypes["train_params"].get("eval_freq", 1))
        if (
            evaluation_loader is not None
            and (epoch + 1) % eval_frequency == 0
        ):
            evaluation_metrics = evaluate_colif_strict_online(
                model,
                evaluation_loader,
                device,
                match_distance=float(
                    hypes["loss"]["args"].get("matcher", {}).get(
                        "max_center_distance", 12.0
                    )
                ),
            )
            evaluation_metrics.update({
                "split": args.eval_split,
                "epoch": epoch,
                "global_step": global_step,
                "world_size": world_size,
            })
            if rank == 0:
                log_record({"evaluation": evaluation_metrics})
                metrics_path = output_dir / ("metrics_%s.jsonl" % args.eval_split)
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(evaluation_metrics, sort_keys=True) + "\n")
        if distributed:
            dist.barrier()
        if args.smoke_batches is not None:
            break
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
