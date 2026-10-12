"""Capture per-layer intention gradients when the native trainer guard fails."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import traceback

import torch
import torch.distributed as dist

from opencood.tools.dualcmp import train_forecast_ddp


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _tensor_range(value):
    if not torch.is_tensor(value) or value.numel() == 0:
        return None
    data = value.detach().float()
    finite = torch.isfinite(data)
    if not bool(finite.any()):
        return {"min": None, "max": None, "finite": False, "elements": int(data.numel())}
    selected = data[finite]
    return {"min": _number(selected.min().item()), "max": _number(selected.max().item()),
            "finite": bool(finite.all()), "elements": int(data.numel())}


def _module_gradient(module):
    if module is None:
        return {"present": False, "parameters": 0, "with_grad": 0, "l2_norm": None,
                "finite": True}
    squared = None
    parameters = with_grad = 0
    finite = True
    for parameter in module.parameters():
        parameters += 1
        if parameter.grad is None:
            continue
        with_grad += 1
        grad = parameter.grad.detach().float()
        finite = finite and bool(torch.isfinite(grad).all())
        contribution = grad.square().sum()
        squared = contribution if squared is None else squared + contribution
    return {"present": True, "parameters": parameters, "with_grad": with_grad,
            "l2_norm": _number(math.sqrt(float(squared.item()))) if squared is not None else 0.0,
            "finite": finite}


def _projection_gradients(head):
    result = []
    groups = (("obj_decoder_layers", getattr(head.decoder, "obj_decoder_layers", ())),
              ("evidence_decoder_layers", getattr(head, "evidence_decoder_layers", ())))
    for group_name, layers in groups:
        for index, layer in enumerate(layers):
            projection = getattr(layer, "ca_qpos_proj", None)
            if projection is None:
                continue
            params = {}
            for name, parameter in projection.named_parameters():
                grad = parameter.grad
                params[name] = {
                    "requires_grad": bool(parameter.requires_grad),
                    "grad_norm": _number(grad.detach().float().norm().item()) if grad is not None else None,
                    "grad_finite": bool(torch.isfinite(grad).all()) if grad is not None else None,
                }
            result.append({"group": group_name, "layer": index,
                           "active": any(item["requires_grad"] for item in params.values()),
                           "parameters": params})
    return result


def _forecast_summary(forecast_output):
    if not isinstance(forecast_output, dict):
        return None
    trajectories = forecast_output.get("loss_pred_trajs_local_layers")
    scores = forecast_output.get("loss_pred_scores_layers")
    per_layer = []
    if torch.is_tensor(trajectories) and trajectories.ndim == 6:
        # Native MTR layout is [batch, actor, decoder-layer, mode, time, xy/log-std/rho].
        for layer_index in range(trajectories.shape[2]):
            prediction = trajectories[:, :, layer_index]
            means = prediction[..., :2]
            mode_spread = None
            if means.shape[-3] > 1:
                pair_delta = means.unsqueeze(-4) - means.unsqueeze(-3)
                pair_distance = pair_delta.square().sum(-1).sqrt()
                upper = torch.triu(torch.ones(pair_distance.shape[-3:-1], dtype=torch.bool,
                                              device=pair_distance.device), diagonal=1)
                pair_values = pair_distance[..., upper, :]
                mode_spread = _tensor_range(pair_values.mean(-1))
            per_layer.append({
                "layer": layer_index,
                "predicted_xy_range": _tensor_range(means),
                "log_std_xy_range": _tensor_range(prediction[..., 2:4]),
                "rho_range": _tensor_range(prediction[..., 4:5]),
                "mean_pairwise_mode_spread_m": mode_spread,
                "scores_range": _tensor_range(scores[:, :, layer_index])
                if torch.is_tensor(scores) and scores.ndim >= 4 else None,
            })
    uncertainty = forecast_output.get("future_uncertainty")
    return {"decoder_layers": per_layer,
            "future_uncertainty_range": _tensor_range(uncertainty)}


def _loss_summary(loss):
    if not isinstance(loss, dict):
        return None
    keys = ("S_gmm", "C_gmm", "S_mode", "C_mode", "S_mean_reg", "C_mean_reg",
            "S_dense_gmm", "C_dense_gmm")
    return {key: _number(loss[key].detach().float().item())
            if torch.is_tensor(loss.get(key)) and loss[key].numel() == 1 else loss.get(key)
            for key in keys if key in loss}


def _main_frame(exc):
    tb = exc.__traceback__
    candidate = None
    while tb is not None:
        frame = tb.tb_frame
        if frame.f_code.co_name == "main" and frame.f_code.co_filename.endswith("train_forecast_ddp.py"):
            candidate = frame
        tb = tb.tb_next
    return candidate


def _write_diagnostic(frame, exc, output_dir):
    local = frame.f_locals if frame is not None else {}
    model = local.get("model")
    network = getattr(model, "module", model)
    head = getattr(network, "forecast_head", None)
    rank = int(local.get("rank", os.environ.get("RANK", 0)))
    step = int(local.get("step", -1))
    epoch = int(local.get("epoch", -1))
    source_encoder = getattr(network, "source_temporal_encoder", None)
    provenance = local.get("run_provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    report = {
        "diagnostic_only": True,
        "status": "expected_gradient_guard_failure",
        "exception": "%s: %s" % (type(exc).__name__, exc),
        "failure_step": step,
        "failure_epoch_zero_based": epoch,
        "rank": rank,
        "world_size": int(local.get("world_size", os.environ.get("WORLD_SIZE", 1))),
        "local_forecast_counts": local.get("last_local_forecast_counts"),
        "global_forecast_counts_at_failure": (local.get("global_counts").detach().cpu().tolist()
                                               if torch.is_tensor(local.get("global_counts")) else None),
        "last_persistent_query_grad_norm_min_rank": local.get("last_persistent_query_grad"),
        "loss_sums_and_counts": _loss_summary(local.get("loss")),
        "forecast_output_summary": _forecast_summary(local.get("forecast_output")),
        "source_motion_contract": provenance.get("source_motion_contract",
                                                   getattr(train_forecast_ddp, "MOTION_EVIDENCE_CONTRACT", None)),
        "forecast_normalization": provenance.get("forecast_normalization"),
        "forecast_variance": provenance.get("forecast_variance"),
        "ca_qpos_projection_gradients": _projection_gradients(head) if head is not None else [],
        "aggregate_gradient_norms": {
            "source_encoder": _module_gradient(source_encoder),
            "source_context_fusion": _module_gradient(getattr(head, "source_context_fusion", None)),
            "original_context_encoder": _module_gradient(getattr(head, "context_encoder", None)),
            "original_decoder": _module_gradient(getattr(head, "decoder", None)),
        },
        "checkpoint_local_present": isinstance(local.get("checkpoint"), dict),
        "checkpoint_local_step": (local.get("checkpoint", {}).get("step")
                                  if isinstance(local.get("checkpoint"), dict) else None),
        "checkpoint_note": "Any saved checkpoint is diagnostic only; its metadata may predate failure. The failed backward performs no optimizer update.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0 and isinstance(local.get("checkpoint"), dict):
        checkpoint = dict(local["checkpoint"])
        checkpoint["diagnostic_only"] = True
        checkpoint["diagnostic_failure_step"] = step
        checkpoint["diagnostic_failure_epoch_zero_based"] = epoch
        torch.save(checkpoint, output_dir / "diagnostic_checkpoint.pt")
        report["diagnostic_checkpoint"] = "diagnostic_checkpoint.pt"
    path = output_dir / ("diagnostic-rank%d.json" % rank)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")


def main():
    diagnostic_dir = Path(os.environ.get("DUALCMP_INTENTION_DIAGNOSTIC_DIR", "runs/diagnose-intention-gradient"))
    try:
        train_forecast_ddp.main()
    except RuntimeError as exc:
        if "learning subset requires non-zero persistent intention gradients" in str(exc):
            try:
                frame = _main_frame(exc)
                _write_diagnostic(frame, exc, diagnostic_dir)
            except Exception as diagnostic_error:
                print("diagnostic capture failed: %s: %s" %
                      (type(diagnostic_error).__name__, diagnostic_error), file=sys.stderr)
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        raise
    else:
        rank = int(os.environ.get("RANK", 0))
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        (diagnostic_dir / ("diagnostic-rank%d.json" % rank)).write_text(
            json.dumps({"diagnostic_only": True,
                        "status": "completed_without_expected_failure",
                        "note": "Native trainer stop-after-epoch cap completed before the persistent intention gradient guard reproduced."},
                       indent=2), encoding="utf-8")
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
