"""Training utilities specific to recurrent CoLIF models."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Dict, Optional

import torch


def _set_batch_norm_statistics_frozen(module: torch.nn.Module) -> int:
    """Use pretrained running statistics while retaining affine gradients."""

    count = 0
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()
            count += 1
    return count


def _tracker_modules(model):
    modules = [
        model.cobevt,
        model.query_decoder,
        model.temporal_memory,
    ]
    if hasattr(model, "measurement_state_update"):
        modules.append(model.measurement_state_update)
    return modules


def clip_forecast_gradient_groups(model, max_grad_norm: float) -> dict:
    """Clip v2 base/source gradients independently and return pre-clip norms."""

    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    group_builder = getattr(
        getattr(model, "forecast_head", None),
        "gradient_parameter_groups",
        None,
    )
    if group_builder is None:
        parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not parameters:
            return {}
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, max_grad_norm, error_if_nonfinite=True
        )
        return {"forecast": float(norm.detach())}

    raw_groups = group_builder()
    groups = {
        name: [
            parameter
            for parameter in parameters
            if parameter.requires_grad and parameter.grad is not None
        ]
        for name, parameters in raw_groups.items()
    }
    actor_parameters = [
        parameter
        for parameter in model.actor_cooperative_fusion.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    groups.setdefault("base", []).extend(actor_parameters)
    seen = set()
    for name, parameters in groups.items():
        for parameter in parameters:
            identifier = id(parameter)
            if identifier in seen:
                raise ValueError(
                    "forecast gradient groups overlap at group %s" % name
                )
            seen.add(identifier)
    norms = {}
    for name, parameters in groups.items():
        if not parameters:
            continue
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, max_grad_norm, error_if_nonfinite=True
        )
        norms[name] = float(norm.detach())
    return norms


def configure_trainable_phase(model, phase: int) -> dict:
    """Apply CoLIF's two-stage optimization contract."""

    if phase not in (1, 2):
        raise ValueError("fused CoLIF uses phase 1 (tracker) and phase 2 (forecast recurrence)")
    for parameter in model.parameters():
        parameter.requires_grad = False

    tracker_modules = [
        model.cobevt.pillar_vfe,
        model.cobevt.backbone,
        model.cobevt.fusion_net,
        model.cobevt.cls_head,
        model.cobevt.reg_head,
        model.query_decoder,
        model.temporal_memory,
    ]
    if hasattr(model, "measurement_state_update"):
        tracker_modules.append(model.measurement_state_update)
    if model.cobevt.shrink_flag:
        tracker_modules.append(model.cobevt.shrink_conv)
    if model.cobevt.compression:
        tracker_modules.append(model.cobevt.naive_compressor)
    if phase == 1:
        trainable_modules = tracker_modules
        model.forecast_head.enabled = False
        model.future_adapter.mode = "none"
        model.forecast_motion_prior.mode = "none"
    else:
        # The previous latent-content adapter is retired.  The forecast head is
        # trained either independently or through the explicit spatial prior.
        future_prior_mode = str(
            getattr(model, "phase2_future_prior_mode", "none")
        )
        if future_prior_mode not in model.forecast_motion_prior.MODES:
            raise ValueError("Invalid Phase-2 forecast motion-prior mode")
        trainable_modules = [
            model.forecast_head,
        ]
        if getattr(model, "actor_cooperative_fusion_enabled", True):
            trainable_modules.append(model.actor_cooperative_fusion)
        model.forecast_head.enabled = True
        model.future_adapter.mode = "none"
        model.forecast_motion_prior.mode = future_prior_mode
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    if getattr(model, "actor_cooperative_fusion_enabled", True):
        freeze_actor_fusion = getattr(
            model.actor_cooperative_fusion,
            "freeze_inactive_parameters",
            None,
        )
        if freeze_actor_fusion is not None:
            freeze_actor_fusion()

    # The V2V4Real decoder's later cross-attention layers deliberately reuse
    # the first layer's query-position projection, while only the final mode
    # classifier is consumed by our online adapter.  Do not leave these known
    # dead branches trainable: asymmetric active-actor counts otherwise force
    # DDP to reason about parameters that can never receive a gradient.
    decoder = getattr(model.forecast_head, "decoder", None)
    if decoder is not None:
        for layer in decoder.obj_decoder_layers[1:]:
            for parameter in layer.ca_qpos_proj.parameters():
                parameter.requires_grad = False
        for head in decoder.motion_cls_heads[:-1]:
            for parameter in head.parameters():
                parameter.requires_grad = False
    # CoLIF always supplies metric reference points explicitly; MOTR's
    # optional learned reference-point layer is bypassed in this contract.
    transformer = getattr(model.query_decoder, "transformer", None)
    learned_reference = getattr(transformer, "reference_points", None)
    if learned_reference is not None:
        for parameter in learned_reference.parameters():
            parameter.requires_grad = False
    if phase == 1:
        model.memory_scale.requires_grad = True
    else:
        # These projections existed only for the retired latent-content cue.
        # The spatial recurrence consumes explicit GMM means/variances/scores.
        forecast_modules = [model.forecast_head]
        nested_motion_base = getattr(model.forecast_head, "motion_base", None)
        if nested_motion_base is not None:
            forecast_modules.append(nested_motion_base)
        for forecast_module in forecast_modules:
            for name in ("latent_projection", "future_token"):
                projection = getattr(forecast_module, name, None)
                if projection is not None:
                    for parameter in projection.parameters():
                        parameter.requires_grad = False
    model.forecast_head.freeze_unused_map_parameters()
    freeze_inactive = getattr(
        model.forecast_head, "freeze_inactive_parameters", None
    )
    if freeze_inactive is not None:
        freeze_inactive(
            actor_cooperative_fusion_enabled=getattr(
                model, "actor_cooperative_fusion_enabled", True
            )
        )
    return {
        "phase": phase,
        "forecast_enabled": model.forecast_head.enabled,
        "future_adapter_mode": model.future_adapter.mode,
        "future_prior_mode": model.forecast_motion_prior.mode,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def set_colif_train_mode(model, phase: int) -> None:
    """Protect frozen Stage-2 state while retaining recurrent backpropagation."""

    model.train()
    if phase == 2:
        for module in _tracker_modules(model):
            module.eval()
        # Gradients from frame t+1 must cross the frozen temporal GRU to reach
        # the future cue written at frame t.  cuDNN rejects GRU backward after
        # an eval-mode forward even when the GRU weights themselves are frozen.
        # TemporalQueryMemory has no BatchNorm or dropout, so train mode does
        # not mutate its state or make its output stochastic.
        if model.forecast_motion_prior.mode in (
            "propagation_bptt", "recurrent_bptt"
        ):
            model.temporal_memory.train()
        model.forecast_head.train()
        if getattr(model, "actor_cooperative_fusion_enabled", True):
            model.actor_cooperative_fusion.train()
        else:
            model.actor_cooperative_fusion.eval()
        _set_batch_norm_statistics_frozen(model.forecast_head)
        model.future_adapter.eval()


def build_colif_optimizer(hypes: dict, model: torch.nn.Module):
    config = hypes["optimizer"]
    optimizer_class = getattr(torch.optim, config["core_method"], None)
    if optimizer_class is None:
        raise ValueError("Unsupported optimizer: %s" % config["core_method"])
    base_lr = float(config["lr"])
    trunk_scale = float(config.get("trunk_lr_scale", 0.1))

    forecast_scale = float(config.get("forecast_lr_scale", 1.0))
    groups = {"cobevt": [], "forecast": [], "colif_heads": []}
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen:
            raise AssertionError("Parameter appears more than once: %s" % name)
        seen.add(parameter_id)
        if name.startswith("cobevt."):
            group_name = "cobevt"
        elif name.startswith("forecast_head."):
            group_name = "forecast"
        else:
            group_name = "colif_heads"
        groups[group_name].append(parameter)

    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if seen != expected:
        raise AssertionError("Optimizer parameter coverage is incomplete")
    parameter_groups = []
    if groups["cobevt"]:
        parameter_groups.append(
            {"params": groups["cobevt"], "lr": base_lr * trunk_scale,
             "group_name": "cobevt"}
        )
    if groups["forecast"]:
        parameter_groups.append(
            {"params": groups["forecast"], "lr": base_lr * forecast_scale,
             "group_name": "forecast"}
        )
    if groups["colif_heads"]:
        parameter_groups.append(
            {"params": groups["colif_heads"], "lr": base_lr,
             "group_name": "colif_heads"}
        )
    return optimizer_class(parameter_groups, **config.get("args", {}))


def load_cobevt_trunk(model, checkpoint_path: str) -> dict:
    """Strictly warm-start CMP's complete CoBEVT encoder and auxiliary heads."""

    payload = torch.load(checkpoint_path, map_location="cpu")
    source_state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    source_state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in source_state.items()
    }
    result = model.cobevt.load_state_dict(source_state, strict=True)
    return {
        "loaded_count": len(source_state),
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
    }


def load_mtr_decoder(model, checkpoint_path: str) -> dict:
    """Warm-start the semantically identical bundled V2V4Real MTR decoder."""

    return model.forecast_head.load_cmp_checkpoint(checkpoint_path)


def load_colif_weights(model, checkpoint_path: str) -> dict:
    """Initialize a later phase from parameters trained by the source phase.

    Modules frozen throughout the source phase retain the current model's
    deterministic initialization instead of importing arbitrary unused random
    tensors from an older checkpoint.
    """

    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload["model_state"] if "model_state" in payload else payload
    source_phase = payload.get("phase") if isinstance(payload, dict) else None
    retained_prefixes = ()
    if source_phase == 1 and model.forecast_head.enabled:
        retained_prefixes = (
            "forecast_head.",
            "future_adapter.",
            "actor_cooperative_fusion.",
        )
    loaded_state = {
        key: value
        for key, value in state.items()
        if not key.startswith(retained_prefixes)
    }
    result = model.load_state_dict(loaded_state, strict=False)
    retained_initialization = [
        key for key in model.state_dict() if key.startswith(retained_prefixes)
    ]
    invalid_missing = [
        key for key in result.missing_keys if not key.startswith(retained_prefixes)
    ]
    if result.unexpected_keys or invalid_missing:
        raise RuntimeError(
            "CoLIF cross-phase checkpoint mismatch: missing=%s unexpected=%s"
            % (invalid_missing, result.unexpected_keys)
        )
    loaded_count = len(loaded_state)
    return {
        "loaded": str(Path(checkpoint_path).resolve()),
        "source_phase": source_phase,
        "source_epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        "loaded_count": loaded_count,
        "retained_current_initialization_count": len(retained_initialization),
        "retained_current_initialization": retained_initialization[:16],
    }


def config_sha256(config: dict) -> str:
    serialized = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def save_colif_checkpoint(
    path: str,
    model,
    optimizer,
    scaler,
    scheduler,
    epoch: int,
    global_step: int,
    hypes: dict,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    repository_root = Path(__file__).resolve().parents[2]
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository_root,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty_status = subprocess.check_output(
            ["git", "status", "--short"], cwd=repository_root,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit, dirty_status = None, None
    payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "phase": int(hypes.get("train_params", {}).get("phase", 1)),
            "config": hypes,
            "config_hash": config_sha256(hypes),
            "git_commit": git_commit,
            "git_dirty_status": dirty_status,
            "split_manifest_hash": _optional_file_sha256(
                hypes.get("split_manifest")
            ),
            "source_evidence_rule": hypes.get("fusion", {}).get("args", {}).get(
                "target_visibility_rule"
            ),
            "frame_period_s": hypes.get("fusion", {}).get("args", {}).get(
                "frame_period_s"
            ),
            "delay_convention": {
                "mode": hypes.get("fusion", {}).get("args", {}).get("delay_mode"),
                "async_overhead_ms": hypes.get("fusion", {}).get("args", {}).get(
                    "async_overhead_ms"
                ),
            },
            "seed": hypes.get("train_params", {}).get("seed", 20),
        }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)


def _optional_file_sha256(path) -> Optional[str]:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resume_colif_checkpoint(
    path: str, model, optimizer=None, scaler=None, scheduler=None
) -> dict:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    return payload
