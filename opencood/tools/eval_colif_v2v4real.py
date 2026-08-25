"""Distributed strict-online CoLIF evaluation on V2V4Real."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils
from opencood.tools.colif.eval_utils import (
    FORECAST_METRIC_PROTOCOLS,
    build_colif_evaluation_loader,
    evaluate_colif_strict_online,
)
from opencood.tools.colif.receiver_protocol import apply_receiver_override


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypes_yaml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--receiver-ids",
        help="Comma-separated CAV IDs; use 0 for one ego or 0,1 for CMP-style multi-ego",
    )
    parser.add_argument("--max-frames-per-rank", type=int)
    parser.add_argument(
        "--forecast-metric-protocol",
        choices=FORECAST_METRIC_PROTOCOLS,
        default="online_masked",
        help=(
            "online_masked preserves variable-validity online metrics; "
            "cmp_fixed reports corrected fixed 1/3/5 s metrics using FDE "
            "indices 9/29/49 and complete 5 s targets"
        ),
    )
    parser.add_argument(
        "--force-measurement-gain",
        type=float,
        help=(
            "Diagnostic-only fixed predict-correct gain in [0,1]; does not "
            "change targets, matching, or metric computation"
        ),
    )
    parser.add_argument(
        "--disable-future-feedback",
        action="store_true",
        help="Legacy diagnostic switch; latent feedback is disabled by default",
    )
    parser.add_argument(
        "--future-prior-mode",
        choices=(
            "checkpoint",
            "none",
            "association",
            "propagation_detached",
            "propagation_bptt",
            "detached",
            "recurrent_bptt",
        ),
        default="checkpoint",
        help=(
            "Same-checkpoint spatial-prior ablation; does not alter targets or "
            "metric computation"
        ),
    )
    parser.add_argument(
        "--diagnostic-reload-mtr-checkpoint",
        help=(
            "Diagnostic-only: replace the checkpoint's forecast weights with "
            "an original CMP/MTR checkpoint while preserving the same online "
            "tracker, targets, matching, coverage, and metric implementation"
        ),
    )
    parser.add_argument("--diagnostic-min-forecast-history", type=int)
    parser.add_argument("--diagnostic-min-context-history", type=int)
    parser.add_argument("--diagnostic-heading-smoothing-steps", type=int)
    parser.add_argument("--diagnostic-disable-actor-cooperative-fusion", action="store_true")
    parser.add_argument("--diagnostic-disable-temporal-history", action="store_true")
    parser.add_argument(
        "--diagnostic-forecast-layer",
        choices=("checkpoint", "refined", "proposal"),
        default="checkpoint",
        help=(
            "Diagnostic-only output-layer override. By default, preserve the "
            "inference layer stored in the checkpoint configuration."
        ),
    )
    parser.add_argument(
        "--diagnostic-residual-prototypes",
        help="Train-only JSON produced by audit_motion_prototypes",
    )
    return parser.parse_args()


def _apply_forecast_layer_override(forecast_head, requested_layer: str) -> dict:
    """Apply an explicit diagnostic override without changing the default model."""

    has_layer = hasattr(forecast_head, "inference_layer")
    checkpoint_layer = (
        str(forecast_head.inference_layer) if has_layer else "refined"
    )
    overridden = requested_layer != "checkpoint"
    if overridden:
        if not has_layer and requested_layer != "refined":
            raise ValueError("checkpoint forecast head has no proposal layer")
        if has_layer:
            forecast_head.inference_layer = requested_layer
    effective_layer = (
        str(forecast_head.inference_layer) if has_layer else "refined"
    )
    return {
        "checkpoint_inference_layer": checkpoint_layer,
        "effective_inference_layer": effective_layer,
        "diagnostic_forecast_layer_override": overridden,
    }


def main():
    args = _arguments()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda:%d" % local_rank if torch.cuda.is_available() else "cpu")
    requested_hypes = load_yaml(args.hypes_yaml)
    payload = torch.load(args.checkpoint, map_location="cpu")
    # Non-parameter runtime semantics (thresholds, clocks, TTL) are part of
    # the checkpoint contract and must not silently revert to another YAML.
    hypes = payload.get("config", requested_hypes)
    receiver_ids = apply_receiver_override(hypes, args.receiver_ids)
    forecast_config = hypes["model"]["args"]["forecast"]
    if args.diagnostic_min_forecast_history is not None:
        forecast_config["min_forecast_history_steps"] = int(
            args.diagnostic_min_forecast_history
        )
    if args.diagnostic_min_context_history is not None:
        forecast_config["min_context_history_steps"] = int(
            args.diagnostic_min_context_history
        )
    if args.diagnostic_heading_smoothing_steps is not None:
        forecast_config["heading_smoothing_steps"] = int(
            args.diagnostic_heading_smoothing_steps
        )
    _, loader = build_colif_evaluation_loader(
        hypes, split=args.split, rank=rank, world_size=world_size,
        workers=args.workers,
    )
    model = train_utils.create_model(hypes).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    if args.diagnostic_disable_actor_cooperative_fusion:
        model.actor_cooperative_fusion_enabled = False
    if args.diagnostic_disable_temporal_history:
        if not hasattr(model.forecast_head, "temporal_history_enabled"):
            raise ValueError("checkpoint forecast head has no temporal-history branch")
        model.forecast_head.temporal_history_enabled = False
    forecast_layer_report = _apply_forecast_layer_override(
        model.forecast_head, args.diagnostic_forecast_layer
    )
    mtr_reload_report = None
    if args.diagnostic_reload_mtr_checkpoint:
        model.forecast_head.enabled = True
        mtr_reload_report = model.forecast_head.load_cmp_checkpoint(
            args.diagnostic_reload_mtr_checkpoint
        )
    if args.force_measurement_gain is not None:
        gain = float(args.force_measurement_gain)
        if not 0.0 <= gain <= 1.0:
            raise ValueError("--force-measurement-gain must lie in [0,1]")
        model.measurement_state_update.min_gain = gain
        model.measurement_state_update.max_gain = gain
    if args.disable_future_feedback:
        model.future_adapter.mode = "none"
    if args.future_prior_mode != "checkpoint":
        model.forecast_motion_prior.mode = args.future_prior_mode
    diagnostic_residual_prototypes = None
    if args.diagnostic_residual_prototypes:
        prototype_payload = json.loads(
            Path(args.diagnostic_residual_prototypes).read_text(
                encoding="utf-8"
            )
        )
        if prototype_payload.get("split") != "train":
            raise ValueError("Residual prototypes must be fitted on train")
        diagnostic_residual_prototypes = torch.as_tensor(
            prototype_payload["centers_longitudinal_lateral_m"],
            dtype=torch.float32,
            device=device,
        )
    metrics = evaluate_colif_strict_online(
        model,
        loader,
        device,
        match_distance=float(
            hypes["loss"]["args"].get("matcher", {}).get(
                "max_center_distance", 12.0
            )
        ),
        max_frames=args.max_frames_per_rank,
        diagnostic_residual_prototypes=diagnostic_residual_prototypes,
        forecast_metric_protocol=args.forecast_metric_protocol,
    )
    metrics.update({
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_global_step": payload.get("global_step"),
        "forecast_bev_representation": getattr(
            model, "forecast_bev_representation", "per_cav"
        ),
        "source_aggregation": getattr(
            model, "source_aggregation", "attention"
        ),
        "source_geometry": (
            "on"
            if getattr(model, "source_geometry_conditioning", False)
            else "off"
        ),
        "trajectory_parameterization": getattr(
            model.forecast_head,
            "trajectory_parameterization",
            None,
        ),
        "world_size": world_size,
        "receiver_ids": list(receiver_ids),
        "max_frames_per_rank": args.max_frames_per_rank,
        "diagnostic_force_measurement_gain": args.force_measurement_gain,
        "diagnostic_future_feedback_disabled": bool(
            args.disable_future_feedback
        ),
        "future_prior_mode": model.forecast_motion_prior.mode,
        "diagnostic_future_prior_override": (
            None if args.future_prior_mode == "checkpoint"
            else args.future_prior_mode
        ),
        "diagnostic_reloaded_mtr_checkpoint": (
            str(Path(args.diagnostic_reload_mtr_checkpoint).resolve())
            if args.diagnostic_reload_mtr_checkpoint else None
        ),
        "diagnostic_mtr_reload_report": mtr_reload_report,
        "diagnostic_min_forecast_history": (
            args.diagnostic_min_forecast_history
        ),
        "diagnostic_min_context_history": (
            args.diagnostic_min_context_history
        ),
        "diagnostic_heading_smoothing_steps": (
            args.diagnostic_heading_smoothing_steps
        ),
        "diagnostic_actor_cooperative_fusion_disabled": bool(
            args.diagnostic_disable_actor_cooperative_fusion
        ),
        "diagnostic_temporal_history_disabled": bool(
            args.diagnostic_disable_temporal_history
        ),
        "diagnostic_forecast_layer": (
            None
            if args.diagnostic_forecast_layer == "checkpoint"
            else args.diagnostic_forecast_layer
        ),
        **forecast_layer_report,
        "diagnostic_residual_prototypes": (
            str(Path(args.diagnostic_residual_prototypes).resolve())
            if args.diagnostic_residual_prototypes else None
        ),
    })
    if rank == 0:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        print(json.dumps(metrics, sort_keys=True))
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
