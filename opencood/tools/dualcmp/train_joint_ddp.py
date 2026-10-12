"""Two-GPU Phase-C joint fine-tuning from an accepted Phase-B checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

from opencood.data_utils.dualcmp_dataloader import build_dataloader
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import (
    DualCMPSequenceDataset,
)
from opencood.loss.dualcmp_loss import DualCmpLoss
from opencood.loss.dualcmp_gmm_loss import FORECAST_SUPERVISIONS, PARTIAL_FORECAST_SUPERVISION
from opencood.loss.forecast_association import (
    LEGACY_ASSOCIATION, VALIDATED_ASSOCIATION, association_contract, associate_forecast_clip,
    forecast_metric_version,
)
from opencood.models.dualcmp.model import DualCmpClipModel
from opencood.tools.dualcmp.protocol import (
    FORECAST_SAMPLES,
    FORECAST_WINDOW,
    TRAIN_LABEL_WINDOW,
    checkpoint_detector_compression,
    normalize_train_delays_ms,
)
from opencood.tools.dualcmp.temporal_protocol import (
    resolve_forecast_ablation, validate_training_delay_contract,
)
from opencood.tools.dualcmp.train_tracking_ddp import _assert_sync, _sha256_file, _sync_buffers
from opencood.tools.dualcmp.train_tracking_ddp import _append_csv, _atomic_save, _trim_csv_after_checkpoint
from opencood.tools.dualcmp.eval_tracking_ddp import evaluate_tracking_model_ddp, evaluate_tracking_test_model_ddp
from opencood.tools.dualcmp.eval_forecast_mainstream_ddp import evaluate_forecast_training_ddp
from opencood.tools.dualcmp.cmp_adapted_forecast_metrics import CONTRACT as EVALUATION_CONTRACT, PRIMARY_METRICS, PROTOCOL as EVALUATION_PROTOCOL
from opencood.tools.dualcmp.train_forecast_ddp import (
    _ensure_forecast_ddp_backward, _snapshot_training_normalization, _restore_training_normalization,
    _configure_deterministic_runtime,
)
from opencood.tools.train_utils import to_device
from opencood.tools.dualcmp.detector_config import (
    DEFAULT_HYPES, load_checkpoint_hypes,
)
from opencood.tools.dualcmp.forecast_bn_refresh import refresh_forecast_bn_ddp
from opencood.tools.dualcmp.checkpoint_handoff import save_external_checkpoint, prepare_training_output


SOURCES = (
    "opencood/tools/dualcmp/train_tracking_ddp.py",
    "opencood/tools/dualcmp/checkpoint_handoff.py",
    "opencood/data_utils/v2v4real/datasets/dualcmp_sequence_dataset.py",
    "opencood/data_utils/v2v4real/datasets/dualcmp_target_builder.py",
    "opencood/loss/dualcmp_loss.py",
    "opencood/loss/forecast_association.py",
    "opencood/loss/dualcmp_gmm_loss.py",
    "opencood/loss/point_pillar_loss.py",
    "opencood/models/dualcmp/model.py",
    "opencood/models/dualcmp/box_refinement.py",
    "opencood/models/dualcmp/kalman.py",
    "opencood/models/dualcmp/state.py",
    "opencood/models/dualcmp/tracker.py",
    "opencood/models/dualcmp/actor_source_tokenizer.py",
    "opencood/models/dualcmp/source_temporal_encoder.py",
    "opencood/models/dualcmp/source_motion_encoder.py",
    "opencood/models/dualcmp/source_evidence_policy.py",
    "opencood/models/dualcmp/forecast_normalization.py",
    "opencood/tools/dualcmp/forecast_bn_refresh.py",
    "opencood/models/dualcmp/forecast_frames.py",
    "opencood/data_utils/v2v4real/datasets/dualcmp_sequence_index.py",
    "opencood/data_utils/v2v4real/datasets/dualcmp_sequence_boundaries.py",
    "opencood/hypes_yaml/dualcmp_sequence_boundaries_v1.json",
    "opencood/tools/dualcmp/aligned_protocol.py",
    "opencood/tools/dualcmp/build_aligned_labels.py",
    "opencood/models/dualcmp/capture_time_evidence.py",
    "opencood/models/dualcmp/mtr_actor_adapter.py",
    "opencood/models/dualcmp/mtr_vendor.py",
    "opencood/models/dualcmp/mtr_attention.py",
    "opencood/models/dualcmp/intention_query_forecast.py",
    "opencood/models/point_pillar_fax.py",
    "opencood/tools/dualcmp/ab3dmot_metrics.py",
    "opencood/tools/dualcmp/eval_forecast_ddp.py",
    "opencood/tools/dualcmp/online.py",
    "opencood/tools/dualcmp/eval_forecast_mainstream_ddp.py",
    "opencood/tools/dualcmp/mainstream_forecast_metrics.py",
    "opencood/tools/dualcmp/cmp_adapted_forecast_metrics.py",
    "opencood/tools/dualcmp/eval_tracking_ddp.py",
    "opencood/tools/dualcmp/detector_config.py",
    "opencood/tools/dualcmp/forecast_metrics.py",
    "opencood/tools/dualcmp/protocol.py",
    "opencood/tools/dualcmp/temporal_protocol.py",
    "opencood/tools/dualcmp/train_joint_ddp.py",
    "opencood/tools/dualcmp/train_forecast_ddp.py",
    "opencood/hypes_yaml/point_pillar_v2v4real.yaml",
    "MTR/mtr/models_v2v4real/context_encoder/mtr_encoder.py",
    "MTR/mtr/models_v2v4real/motion_decoder/mtr_decoder.py",
    "MTR/mtr/models_v2v4real/utils/transformer/position_encoding_utils.py",
    "MTR/mtr/models_v2v4real/utils/transformer/transformer_encoder_layer.py",
    "MTR/mtr/models_v2v4real/utils/transformer/multi_head_attention.py",
)
LEGACY_FORECAST_METRICS = ("minADE_1s", "minADE_3s", "minADE_5s",
                    "minFDE_1s", "minFDE_3s", "minFDE_5s",
                    "MR_3.6", "count")
FORECAST_METRICS = PRIMARY_METRICS
TRACKING_METRICS = ("MOTA", "AMOTA", "sAMOTA", "tracking_recall",
                    "tracking_precision", "identity_switches", "fragments")
TRAIN_LABEL_FORECAST_SAMPLES = 10
CSV_FIELDS = (("kind", "split", "epoch", "global_step", "global_samples",
               "lr_detector", "lr_tracking", "lr_forecast",
               "L_det", "L_track", "L_pred", "L_total")
              + tuple("test_track_" + key for key in TRACKING_METRICS)
              + ("test_track_timelines", "test_track_frames",)
              + ("evaluation_protocol",)
              + tuple("test_forecast_" + key for key in FORECAST_METRICS))
LEGACY_CSV_FIELDS = tuple(key for key in CSV_FIELDS if not key.startswith("test_forecast_")) + tuple(
    "test_forecast_" + key for key in LEGACY_FORECAST_METRICS)


def _provenance():
    root = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root)).decode().strip()
    code_sha = os.environ.get("DUALCMP_CODE_SHA")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *SOURCES], cwd=str(root)
    ).decode().strip()
    if code_sha:
        if not re.match(r"^[0-9a-fA-F]{40}$", code_sha) or code_sha.lower() != head.lower() or dirty:
            raise RuntimeError("DUALCMP_CODE_SHA requires matching clean Phase-C sources")
        return {"source_state": "clean", "source_sha": code_sha.lower(),
                "protocol_sha256": _sha256_file(Path(__file__).with_name("protocol.py"))}
    digest = hashlib.sha256()
    for relative in sorted(SOURCES):
        digest.update(relative.encode() + b"\0")
        digest.update((root / relative).read_bytes())
    return {"source_state": "dirty-smoke", "base_sha": head,
            "source_tree_sha256": digest.hexdigest(),
            "protocol_sha256": _sha256_file(Path(__file__).with_name("protocol.py"))}


def _hypes(path, run_args):
    split = "train" if run_args.train_label_smoke else run_args.data_split
    root = os.environ.get("V2V4REAL_ROOT")
    weights = os.environ.get("DUALCMP_INIT_WEIGHTS")
    labels = os.environ.get("DUALCMP_TRACKING_LABEL_ROOT")
    train_labels = os.environ.get("DUALCMP_TRAIN_TRACKING_LABEL_ROOT")
    if not root or not weights or not labels:
        raise EnvironmentError(
            "V2V4REAL_ROOT, DUALCMP_INIT_WEIGHTS, and "
            "DUALCMP_TRACKING_LABEL_ROOT are required"
        )
    payload = torch.load(str(weights), map_location="cpu")
    detector_compression = checkpoint_detector_compression(payload)
    h, detector_init = load_checkpoint_hypes(path, payload)
    h["root_dir"] = str(Path(root) / split)
    h["validate_dir"] = h["root_dir"]
    h["model"]["args"]["compression"] = detector_compression
    fusion_args = {
        "stage": "forecast", "tracking_frames": 3,
        "forecast_history_frames": 10, "future_frames": 50,
        "runtime_only": False, "runtime_forbid_gt": True,
        "receiver_ids": [0, 1], "max_actors": 128, "max_cav": 2,
        "partner_delay_ms": float(
            getattr(run_args, "partner_delay_ms", 0.0)),
        "partner_delay_choices_ms": list(run_args.train_delays_ms),
        "partner_delay_seed": int(run_args.delay_seed),
        "split_name": split, "shuffle_training_points": False,
    }
    if (run_args.full_data or run_args.train_label_smoke) and split == "train":
        if not train_labels:
            raise EnvironmentError("formal train split requires DUALCMP_TRAIN_TRACKING_LABEL_ROOT")
        fusion_args.update({"tracking_label_root": train_labels,
                            "tracking_label_layout": "dmstrack_train"})
    elif labels:
        fusion_args["tracking_label_root"] = labels
    if run_args.train_label_smoke:
        from opencood.tools.dualcmp.aligned_protocol import ALIGNED_FORECAST_WINDOW
        fusion_args["window"] = list(ALIGNED_FORECAST_WINDOW if run_args.forecast_representation == "source_temporal_aligned"
                                     else TRAIN_LABEL_WINDOW)
    elif not run_args.full_data:
        if split != "test":
            raise ValueError("the frozen Phase-C acceptance subset belongs to test")
        fusion_args["window"] = list(FORECAST_WINDOW)
    h["fusion"] = {"core_method": "DualCMPSequenceDataset", "args": fusion_args}
    return h, Path(weights), payload, detector_compression, detector_init


def _label_contract(args):
    if args.train_label_smoke:
        from opencood.tools.dualcmp.aligned_protocol import ALIGNED_FORECAST_WINDOW
        return {"layout": "dmstrack_train", "split": "train",
                "window": list(ALIGNED_FORECAST_WINDOW if args.forecast_representation == "source_temporal_aligned"
                               else TRAIN_LABEL_WINDOW)}
    if args.full_data:
        return {"layout": "dmstrack_train" if args.data_split == "train" else "legacy",
                "split": args.data_split, "window": None}
    return {"layout": "legacy", "split": "test", "window": list(FORECAST_WINDOW)}


def _build_optimizer(model, base_lr, detector_scale, forecast_scale, weight_decay):
    groups = {"detector": [], "tracking": [], "forecast": []}
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise AssertionError("optimizer parameter appears twice: %s" % name)
        seen.add(id(parameter))
        if name.startswith("detector."):
            group = "detector"
        elif name.startswith(("forecast_head.", "actor_source_tokenizer.",
                              "source_temporal_encoder.")):
            group = "forecast"
        else:
            group = "tracking"
        groups[group].append(parameter)
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if seen != expected or not all(groups.values()):
        raise AssertionError("joint optimizer groups do not cover the trainable model")
    parameter_groups = [
        {"params": groups["detector"], "lr": base_lr * detector_scale,
         "group_name": "detector"},
        {"params": groups["tracking"], "lr": base_lr,
         "group_name": "tracking"},
        {"params": groups["forecast"], "lr": base_lr * forecast_scale,
         "group_name": "forecast"},
    ]
    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay), {
        name: sum(p.numel() for p in parameters) for name, parameters in groups.items()
    }


def main():
    parser = argparse.ArgumentParser(description="two-process DualCMP Phase-C trainer")
    parser.add_argument("--hypes", default=DEFAULT_HYPES)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--full-data", action="store_true")
    parser.add_argument("--train-label-smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=6,
                        help="independent clips per GPU")
    parser.add_argument("--workers", type=int, default=8,
                        help="DataLoader workers per GPU process")
    parser.add_argument("--base-lr", type=float, default=1e-5)
    parser.add_argument("--detector-lr-scale", type=float, default=0.1)
    parser.add_argument("--forecast-lr-scale", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--partner-delay-ms", type=float, default=0.0,
                        help="fixed epoch-evaluation delay and fixed-training fallback")
    parser.add_argument(
        "--train-delays-ms", type=float, nargs="+",
        help="balanced discrete partner delays used for training")
    parser.add_argument("--delay-seed", type=int, default=20260901,
                        help="deterministic clip-to-delay assignment seed")
    parser.add_argument(
        "--forecast-representation",
        choices=("history", "fused", "source", "source_temporal", "source_temporal_aligned"),
        default="source")
    parser.add_argument("--disable-capture-alignment", action="store_true")
    parser.add_argument("--disable-age-encoding", action="store_true")
    parser.add_argument("--mean-source-aggregation", action="store_true",
                        help="mask-average source tokens before intention attention")
    parser.add_argument("--disable-capture-time-evidence", action="store_true",
                        help="matched stale-direct ablation for delayed source BEV")
    parser.add_argument("--lambda-det", type=float, default=1.0)
    parser.add_argument("--lambda-track", type=float, default=1.0)
    parser.add_argument("--lambda-pred", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--min-lr-ratio", type=float, default=1e-3)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--no-validation", action="store_true",
                        help="publish immutable checkpoints for external validation")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true",
                        help="require deterministic PyTorch algorithms before CUDA initialization")
    parser.add_argument("--refresh-forecast-bn", action="store_true",
                        help="refresh head BN from GT-free train observations before saving/evaluation")
    args = parser.parse_args()
    if args.no_validation and args.refresh_forecast_bn:
        raise ValueError("external validation mode cannot refresh forecast BN on the remote trainer")
    ablation = resolve_forecast_ablation(
        args.forecast_representation, args.disable_capture_alignment,
        args.disable_age_encoding, args.disable_capture_time_evidence,
        args.mean_source_aggregation)
    args.forecast_representation = ablation["forecast_representation"]
    args.capture_alignment = ablation["capture_alignment"]
    args.age_encoding = ablation["age_encoding"]
    args.mean_source_aggregation = ablation["mean_source_aggregation"]
    if (args.forecast_representation not in ("source", "source_temporal", "source_temporal_aligned")
            and (args.disable_capture_alignment
                 or args.disable_age_encoding
                 or args.disable_capture_time_evidence
                 or args.mean_source_aggregation)):
        raise ValueError(
            "capture alignment and age ablations require source representation")
    args.train_delays_ms = list(normalize_train_delays_ms(
        args.train_delays_ms, args.partner_delay_ms))
    if args.max_steps < 0:
        raise ValueError("--max-steps must be non-negative")
    if args.train_label_smoke and args.full_data:
        raise ValueError("--train-label-smoke cannot be combined with --full-data")
    if args.log_every < 1 or args.save_every < 1 or args.eval_every < 1:
        raise ValueError("Phase C logging, saving, and evaluation intervals must be positive")
    if args.full_data and (args.save_every != 1 or (not args.no_validation and args.eval_every != 1)):
        raise ValueError("formal Phase C requires save_every=1 and eval_every=1")
    if args.resume and Path(args.resume).resolve().parent != Path(args.output_dir).resolve():
        raise ValueError("--resume checkpoint must belong to --output-dir")
    if args.full_data and not os.environ.get("DUALCMP_CODE_SHA"):
        raise EnvironmentError("formal full-data Phase C requires DUALCMP_CODE_SHA")
    if min(args.base_lr, args.detector_lr_scale, args.forecast_lr_scale,
           args.lambda_det, args.lambda_track, args.lambda_pred) <= 0:
        raise ValueError("learning rates, scales, and loss weights must be positive")
    if (not all(np.isfinite(value) for value in (
            args.base_lr, args.detector_lr_scale, args.forecast_lr_scale,
            args.lambda_det, args.lambda_track, args.lambda_pred, args.weight_decay, args.min_lr_ratio))
            or args.weight_decay < 0 or not 0 < args.min_lr_ratio <= 1
            or args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.warmup_steps < 0):
        raise ValueError("invalid finite Phase-C optimizer/budget configuration")
    if not np.isfinite(args.partner_delay_ms) or args.partner_delay_ms < 0:
        raise ValueError("partner delay must be finite and non-negative")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Phase C requires torchrun")
    deterministic_runtime = _configure_deterministic_runtime(args.deterministic)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dist.init_process_group("nccl")
    rank, local_rank = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    h, init_weights, payload, detector_compression, detector_init = _hypes(
        args.hypes, args
    )
    forecast_association = payload.get("provenance", {}).get("forecast_association", association_contract(LEGACY_ASSOCIATION))
    association_policy = forecast_association["policy"]
    forecast_supervision = payload.get("provenance", {}).get("forecast_supervision", PARTIAL_FORECAST_SUPERVISION)
    if forecast_supervision not in FORECAST_SUPERVISIONS:
        raise RuntimeError("Phase C requires a supported Phase-B future-supervision contract")
    if forecast_association != association_contract(association_policy):
        raise RuntimeError("Phase C requires the exact supported Phase-B forecast association contract")
    metric_version = payload.get("provenance", {}).get("forecast_metric_version", forecast_metric_version(LEGACY_ASSOCIATION))
    if metric_version != forecast_metric_version(association_policy):
        raise RuntimeError("Phase C requires a consistent Phase-B association/metric contract")
    h["fusion"]["args"]["include_forecast_association_history"] = association_policy == VALIDATED_ASSOCIATION
    if args.forecast_representation == "source_temporal_aligned":
        h["fusion"]["args"]["forecast_coordinate_frame"] = "current_receiver"
        h["fusion"]["args"]["excluded_scene_ids"] = payload.get("provenance", {}).get("excluded_scene_ids", [])
        point_file = payload.get("provenance", {}).get("intention_points_file")
        point_sha = payload.get("provenance", {}).get("intention_points_sha256")
        if not point_file or not point_sha or _sha256_file(point_file) != point_sha:
            raise RuntimeError("Phase C requires the exact Phase-B aligned intention prior")
        if payload["provenance"].get("decoder_query_contract") != "persistent_static_intention_position_v1":
            raise RuntimeError("Phase C requires persistent-intention Phase-B weights")
    dataset = DualCMPSequenceDataset(h, train=True)
    expected_samples = (TRAIN_LABEL_FORECAST_SAMPLES if args.train_label_smoke
                        else FORECAST_SAMPLES)
    if args.train_label_smoke and args.forecast_representation == "source_temporal_aligned":
        expected_samples = 4
    if not args.full_data and len(dataset) != expected_samples:
        raise RuntimeError("fixed Phase-C protocol requires %d samples" % expected_samples)
    _, loader, sampler = build_dataloader(
        dataset, args.batch_size, True, workers=args.workers, rank=rank,
        world_size=world_size, seed=args.seed, training=True,
        collate_fn=dataset.collate_batch_train, pin_memory=True,
    )
    if len(loader) < 1:
        raise RuntimeError(
            "Phase-C loader has no complete distributed batch; reduce "
            "--batch-size for this sample set")
    forecast_args = {
        "dropout": float(payload.get("provenance", {}).get("forecast_dropout", .1)),
        "history_frames": 11, "future_frames": 50, "modes": 6,
        "intention_points_file": (payload.get("provenance", {}).get("intention_points_file")
                                  if args.forecast_representation == "source_temporal_aligned" else
                                  "preprocessed_data/v2v4real/v2v4real_cluster_64_center_dict.pkl"),
        "representation": args.forecast_representation,
        "capture_alignment": args.capture_alignment,
        "age_encoding": args.age_encoding,
        "mean_source_aggregation": args.mean_source_aggregation,
    }
    aligned_choices = {}
    if args.forecast_representation == "source_temporal_aligned":
        # Reuse online checkpoint reconstruction defaults. A Phase-B policy
        # is not silently promoted to the full Source model in Phase C.
        aligned_choices = {key: payload.get("provenance", {}).get(key, default)
                           for key, default in (("aligned_evidence_policy", "source_temporal"),
                                                ("source_feature_encoder", "legacy_additive_v1"),
                                                ("forecast_variance", "log_std_clamp_v1"),
                                                ("forecast_normalization", "batch_norm_v1"))}
        forecast_args.update(aligned_choices)
    if args.refresh_forecast_bn and (
            forecast_args.get("forecast_normalization", "batch_norm_v1") != "batch_norm_v1"
            or not (args.train_label_smoke or (args.full_data and args.data_split == "train"))):
        raise ValueError("BN refresh requires batch_norm_v1 and actual training observations")
    model = DualCmpClipModel(
        h["model"]["args"], dataset.anchor_box,
        h["preprocess"]["cav_lidar_range"], forecast_args=forecast_args,
    ).to(device)
    state = payload.get("model", payload.get("model_state_dict", payload))
    state = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    optimizer, parameter_counts = _build_optimizer(
        model, args.base_lr, args.detector_lr_scale,
        args.forecast_lr_scale, args.weight_decay,
    )
    model.train()
    model.forecast_association_policy = association_policy
    model.forecast_supervision = forecast_supervision
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    forecast_mean_weight = float(
        payload.get("provenance", {}).get("forecast_mean_weight", 0.0))
    if (args.forecast_representation in ("source_temporal", "source_temporal_aligned")
            and forecast_mean_weight <= 0):
        raise RuntimeError(
            "source_temporal Phase C requires Phase-B mean supervision")
    criterion = DualCmpLoss({
        "detector": h["loss"]["args"],
        "forecast_mean_weight": forecast_mean_weight,
        "forecast_association_policy": association_policy,
        "forecast_supervision": forecast_supervision,
    }).to(device)
    run_provenance = _provenance()
    aligned_evaluation = args.forecast_representation == "source_temporal_aligned"
    evaluation_protocol = EVALUATION_PROTOCOL if aligned_evaluation else metric_version
    forecast_metrics = FORECAST_METRICS if aligned_evaluation else LEGACY_FORECAST_METRICS
    csv_fields = CSV_FIELDS if aligned_evaluation else LEGACY_CSV_FIELDS
    run_provenance["forecast_evaluation"] = dict(EVALUATION_CONTRACT) if aligned_evaluation else {"protocol": evaluation_protocol}
    run_provenance["validation_mode"] = "external" if args.no_validation else "inline"
    run_provenance["batch_contract"] = {
        "per_rank_batch": args.batch_size,
        "world_size": world_size,
        "accumulation_steps": 1,
        "global_batch": args.batch_size * world_size,
    }
    run_provenance["deterministic_runtime"] = deterministic_runtime
    run_provenance["forecast_association"] = forecast_association
    run_provenance["forecast_supervision"] = forecast_supervision
    run_provenance["forecast_metric_version"] = metric_version
    run_provenance["joint_forecast_empty_contract"] = (
        "skip_forecast_params_norm_keep_perception_update_v1" if association_policy == VALIDATED_ASSOCIATION else "legacy_joint_update_v1")
    run_provenance["training_hyperparameters"] = {key: getattr(args, key) for key in (
        "base_lr", "detector_lr_scale", "forecast_lr_scale", "weight_decay",
        "lambda_det", "lambda_track", "lambda_pred", "warmup_steps", "min_lr_ratio", "seed", "amp")}
    if args.forecast_representation == "source_temporal_aligned":
        run_provenance.update(intention_points_file=point_file, intention_points_sha256=point_sha,
                              forecast_coordinate_frame="current_receiver")
        run_provenance["motion_input_contract"] = payload["provenance"].get("motion_input_contract")
        run_provenance["decoder_query_contract"] = payload["provenance"]["decoder_query_contract"]
        if payload["provenance"].get("sequence_policy", {}).get("sha256") != dataset.sequence_policy["sha256"]:
            raise RuntimeError("Phase C acquisition boundaries differ from Phase B")
        if payload["provenance"].get("sequence_policy", {}).get("label_identity") != dataset.sequence_policy.get("label_identity"):
            raise RuntimeError("Phase C GT identities differ from Phase B")
        run_provenance["sequence_policy"] = dataset.sequence_policy
        run_provenance.update(aligned_choices)
        if aligned_choices["source_feature_encoder"] == "motion_correlation_v1":
            from opencood.models.dualcmp.source_evidence_policy import MOTION_EVIDENCE_CONTRACT
            if payload["provenance"].get("source_motion_contract") != MOTION_EVIDENCE_CONTRACT:
                raise RuntimeError("Phase C requires the current Source feature-injection contract")
            run_provenance["source_motion_contract"] = MOTION_EVIDENCE_CONTRACT
        run_provenance["excluded_scene_ids"] = list(dataset.excluded_scene_ids)
    run_provenance["tracking_label_contract"] = (
        "dmstrack_train" if (args.full_data and args.data_split == "train")
        or args.train_label_smoke else "legacy")
    run_provenance["tracking_label_scope"] = _label_contract(args)
    run_provenance["partner_delay_ms"] = args.partner_delay_ms
    run_provenance["train_delays_ms"] = list(args.train_delays_ms)
    run_provenance["delay_seed"] = args.delay_seed
    run_provenance["train_delay_histogram"] = list(
        dataset.partner_delay_histogram)
    run_provenance["forecast_representation"] = args.forecast_representation
    run_provenance["forecast_mean_weight"] = forecast_mean_weight
    run_provenance["forecast_dropout"] = forecast_args["dropout"]
    run_provenance["refresh_forecast_bn"] = args.refresh_forecast_bn
    run_provenance["capture_alignment"] = args.capture_alignment
    run_provenance["age_encoding"] = args.age_encoding
    run_provenance["mean_source_aggregation"] = (
        args.mean_source_aggregation)
    run_provenance["capture_time_evidence"] = (
        args.capture_alignment and args.age_encoding)
    if payload.get("provenance", {}).get("partial_epoch"):
        raise RuntimeError(
            "Phase C requires a completed Phase-B epoch checkpoint")
    validate_training_delay_contract(
        payload.get("provenance"), args.train_delays_ms, args.delay_seed,
        context="Phase C initialization",
        forecast_representation=args.forecast_representation,
        capture_alignment=args.capture_alignment,
        age_encoding=args.age_encoding,
        mean_source_aggregation=args.mean_source_aggregation)
    if args.train_label_smoke and payload.get("provenance", {}).get("tracking_label_contract") != "dmstrack_train":
        raise RuntimeError("Phase-C train-label smoke requires a native-label Phase-B checkpoint")
    total_steps = (
        args.epochs * len(loader)
        if args.full_data else (args.max_steps or args.epochs * len(loader))
    )
    warmup_steps = min(args.warmup_steps, max(total_steps - 1, 0))

    def multiplier(step):
        if warmup_steps and step < warmup_steps:
            return 1.0 / 3.0 + (2.0 / 3.0) * step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    output = Path(args.output_dir)
    start_epoch = step = global_samples = 0
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu")
        if not isinstance(resume_payload, dict) or any(key not in resume_payload for key in ("model", "optimizer", "scheduler", "epoch", "step", "provenance")):
            raise RuntimeError("resume checkpoint must contain model, optimizer, scheduler, epoch, step, and provenance")
        if args.max_steps and int(resume_payload["step"]) >= args.max_steps:
            raise RuntimeError(
                "Phase C resume checkpoint already reached --max-steps")
        if int(resume_payload["provenance"].get("total_steps", -1)) != total_steps:
            raise RuntimeError("resume scheduler budget disagrees with requested 20-epoch schedule")
        if int(resume_payload["scheduler"].get("last_epoch", -1)) != int(resume_payload["step"]):
            raise RuntimeError("resume scheduler step does not match checkpoint step")
        previous = resume_payload["provenance"]
        if "evaluation_migration" in previous:
            run_provenance["evaluation_migration"] = previous["evaluation_migration"]
        if previous.get("forecast_evaluation", {"protocol": metric_version}) != run_provenance["forecast_evaluation"]:
            raise RuntimeError("resume evaluation protocol changed; keep historical outputs and use an explicit migration")
        if previous.get("validation_mode", "inline") != run_provenance["validation_mode"]:
            raise RuntimeError("Phase C resume must retain its validation mode")
        if (("deterministic_runtime" in previous and previous["deterministic_runtime"] != deterministic_runtime)
                or ("deterministic_runtime" not in previous and args.deterministic)):
            raise RuntimeError("Phase C resume must retain its deterministic runtime contract")
        if previous.get("forecast_association", association_contract(LEGACY_ASSOCIATION)) != forecast_association:
            raise RuntimeError("Phase C resume must retain its forecast association/supervision contract")
        if previous.get("forecast_supervision", PARTIAL_FORECAST_SUPERVISION) != forecast_supervision:
            raise RuntimeError("Phase C resume must retain its original future-supervision contract")
        if previous.get("forecast_metric_version", forecast_metric_version(LEGACY_ASSOCIATION)) != metric_version:
            raise RuntimeError("Phase C resume must retain its forecast metric version")
        if previous.get("joint_forecast_empty_contract", "legacy_joint_update_v1") != run_provenance["joint_forecast_empty_contract"]:
            raise RuntimeError("Phase C resume must retain its unsupervised forecast-update contract")
        if previous.get("training_hyperparameters") != run_provenance["training_hyperparameters"]:
            raise RuntimeError("Phase C resume must retain optimizer, loss weights and precision configuration")
        if previous.get("batch_contract", run_provenance["batch_contract"]) != run_provenance["batch_contract"]:
            raise RuntimeError("Phase C resume must retain its per-rank and global batch contract")
        if previous.get("refresh_forecast_bn", False) != args.refresh_forecast_bn:
            raise RuntimeError("Phase C resume must retain its BN refresh contract")
        if previous.get("forecast_dropout", .1) != run_provenance["forecast_dropout"]:
            raise RuntimeError("Phase C resume must retain its dropout contract")
        if args.forecast_representation == "source_temporal_aligned":
            for key, value in dict(aligned_choices, excluded_scene_ids=list(dataset.excluded_scene_ids)).items():
                if previous.get(key, value) != value:
                    raise RuntimeError("Phase C resume must retain its evidence/feature/split contract")
        if previous.get("partial_epoch"):
            raise RuntimeError(
                "partial-epoch checkpoints are evaluation-only and cannot resume")
        if previous.get("tracking_label_contract") != run_provenance["tracking_label_contract"]:
            raise RuntimeError("resume tracking label contract disagrees with requested training split")
        if previous.get("tracking_label_scope") != run_provenance["tracking_label_scope"]:
            raise RuntimeError("resume tracking label scope disagrees with requested training window")
        if previous.get("partner_delay_ms") != args.partner_delay_ms:
            raise RuntimeError("Phase C resume requires the original evaluation delay")
        validate_training_delay_contract(
            previous, args.train_delays_ms, args.delay_seed,
            context="Phase C resume",
            forecast_representation=args.forecast_representation,
            capture_alignment=args.capture_alignment,
            age_encoding=args.age_encoding,
            mean_source_aggregation=args.mean_source_aggregation)
        source_key = "source_sha" if "source_sha" in run_provenance else "source_tree_sha256"
        if previous.get(source_key) != run_provenance.get(source_key):
            raise RuntimeError("Phase C resume requires the original source state")
        if previous.get("init_checkpoint_sha256") != _sha256_file(init_weights):
            raise RuntimeError("Phase C resume requires the original Phase-B checkpoint")
        model.module.load_state_dict(resume_payload["model"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"]); scheduler.load_state_dict(resume_payload["scheduler"])
        start_epoch, step, global_samples = int(resume_payload["epoch"]), int(resume_payload["step"]), int(resume_payload.get("global_samples", 0))
    if rank == 0:
        prepare_training_output(output, resume=bool(args.resume), external=args.no_validation)
        if args.resume:
            _trim_csv_after_checkpoint(output / "metrics.csv", step, csv_fields)
    dist.barrier()
    epoch = start_epoch
    started = time.monotonic()
    last = None
    last_forecast_components = None
    forecast_to_evidence_grad = None
    joint_detector_grad = None
    stopped_partial = False
    while epoch < args.epochs:
        sampler.set_epoch(epoch)
        batches_seen = 0
        epoch_totals = np.zeros(4, dtype=np.float64); epoch_count = 0
        log_totals = np.zeros(4, dtype=np.float64); log_count = 0
        for batch in loader:
            batches_seen += 1
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            forecast_normalization_snapshot = []
            if association_policy == VALIDATED_ASSOCIATION:
                for name in ("forecast_head", "actor_source_tokenizer", "source_temporal_encoder"):
                    module = getattr(model.module, name, None)
                    if module is not None:
                        forecast_normalization_snapshot.extend(_snapshot_training_normalization(module))
            outputs, _ = model(
                batch["runtime_frames"], detector_autocast=args.amp,
                run_forecast=True, current_frame_only_grad=True)
            if association_policy == VALIDATED_ASSOCIATION:
                association_targets = batch["association_target_frames"]
                if not torch.equal(association_targets[-1]["actor_ids"], batch["target_frames"][-1]["actor_ids"]):
                    raise RuntimeError("Phase C causal association and final GT identities disagree")
                assigned, records = associate_forecast_clip(outputs, association_targets)
                loss = criterion([outputs[-1]], batch["target_frames"], forecast_assignments=[assigned])
                outputs[-1]["forecast_association"] = dict(records[-1], assigned=assigned)
            else:
                loss = criterion([outputs[-1]], batch["target_frames"])
            counts = torch.stack((
                loss["C_gmm"], loss["C_mode"], loss["C_mean_reg"],
                loss["C_dense_gmm"],
            )).detach()
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            pred = dist.get_world_size() * (
                loss["S_gmm"] / counts[0].clamp_min(1.0)
                + loss["S_mode"] / counts[1].clamp_min(1.0)
                + forecast_mean_weight * loss["S_mean_reg"]
                / counts[2].clamp_min(1.0)
                + loss["S_dense_gmm"] / counts[3].clamp_min(1.0)
            )
            forecast_sums = torch.stack([loss["S_" + field].detach()
                for field in ("gmm", "mode", "mean_reg", "dense_gmm")])
            dist.all_reduce(forecast_sums, op=dist.ReduceOp.SUM)
            last_forecast_components = {
                "values": dict(zip(("gmm", "mode", "mean_reg", "dense_gmm"),
                                   (forecast_sums / counts.clamp_min(1.0)).cpu().tolist())),
                "valid_counts": dict(zip(("gmm", "mode", "mean_reg", "dense_gmm"), counts.cpu().tolist())),
                "weights": {"gmm": 1., "mode": 1., "mean_reg": forecast_mean_weight, "dense_gmm": 1.},
                "scope": "last global optimizer batch; not an epoch average",
            }
            pred = _ensure_forecast_ddp_backward(model, pred, loss, outputs[-1])
            track = loss["L_exist"] + loss["L_box"] + loss["L_velocity"]
            total = (args.lambda_det * loss["L_det"]
                     + args.lambda_track * track + args.lambda_pred * pred)
            if (forecast_to_evidence_grad is None and bool(counts[0] > 0)
                    and args.forecast_representation != "history"
                    and aligned_choices.get("aligned_evidence_policy") not in ("history", "geometry_temporal")):
                evidence_bev = outputs[-1]["forecast_evidence_bev"]
                if args.forecast_representation in ("source_temporal", "source_temporal_aligned"):
                    evidence_bev = outputs[-1]["fused_bev"] if aligned_choices.get("aligned_evidence_policy") == "fused_temporal" else outputs[-1]["source_bev"]
                source_grad = torch.autograd.grad(
                    pred, evidence_bev, retain_graph=True, allow_unused=True
                )[0]
                source_value = (float(source_grad.detach().float().norm()) if source_grad is not None else 0.0)
                # Ignore legitimately unsupervised ranks in this diagnostic;
                # they still participate in the normal joint backward.
                if not bool(loss["C_gmm"] > 0):
                    source_value = float("inf")
                source_min = torch.tensor(source_value, device=device)
                dist.all_reduce(source_min, op=dist.ReduceOp.MIN)
                forecast_to_evidence_grad = float(source_min)
                if not forecast_to_evidence_grad > 0:
                    raise RuntimeError(
                        "forecast loss gradient to evidence BEV is zero: %.8g"
                        % forecast_to_evidence_grad
                    )
            total.backward()
            if association_policy == VALIDATED_ASSOCIATION and not bool((counts > 0).any()):
                # Perception/track losses remain supervised. Only the wholly
                # unsupervised forecast group must avoid AdamW decay/momentum
                # and normalization learning from DDP's synthetic zero edges.
                for group in optimizer.param_groups:
                    if group["group_name"] == "forecast":
                        for parameter in group["params"]:
                            parameter.grad = None
                _restore_training_normalization(forecast_normalization_snapshot)
            if step == 0:
                detector_value = sum(
                    float(parameter.grad.detach().float().norm())
                    for parameter in model.module.detector.parameters()
                    if parameter.grad is not None
                )
                detector_min = torch.tensor(detector_value, device=device)
                dist.all_reduce(detector_min, op=dist.ReduceOp.MIN)
                joint_detector_grad = float(detector_min)
                if not joint_detector_grad > 0:
                    raise RuntimeError(
                        "joint loss gradient to detector parameters is zero: %.8g"
                        % joint_detector_grad
                    )
            grad_norm = clip_grad_norm_(model.parameters(), 1000.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite Phase-C gradient; optimizer step rejected")
            optimizer.step(); scheduler.step(); step += 1
            values = torch.stack((loss["L_det"].detach(), track.detach(), pred.detach(), total.detach())).float()
            if not torch.isfinite(values).all():
                raise FloatingPointError("non-finite Phase-C loss")
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            last = (values / dist.get_world_size()).cpu().tolist()
            batch_count = len(batch["scene_receiver_keys"]) * dist.get_world_size()
            epoch_totals += np.asarray(last) * batch_count; epoch_count += batch_count
            log_totals += np.asarray(last) * batch_count; log_count += batch_count
            global_samples += batch_count
            if step % args.log_every == 0 and rank == 0:
                row = dict(zip(("L_det", "L_track", "L_pred", "L_total"), (log_totals / log_count).tolist()))
                row.update({"kind": "step", "split": "train", "epoch": epoch + 1, "global_step": step,
                            "global_samples": global_samples, "lr_detector": optimizer.param_groups[0]["lr"],
                            "lr_tracking": optimizer.param_groups[1]["lr"], "lr_forecast": optimizer.param_groups[2]["lr"]})
                row["evaluation_protocol"] = evaluation_protocol
                _append_csv(output / "metrics.csv", row, csv_fields); print(json.dumps(row), flush=True)
                log_totals[:] = 0; log_count = 0
            if args.max_steps and step >= args.max_steps:
                break
        limit_reached = bool(args.max_steps and step >= args.max_steps)
        partial_epoch = bool(limit_reached and batches_seen < len(loader))
        stopped_partial = partial_epoch
        _sync_buffers(model)
        if args.refresh_forecast_bn and (limit_reached or (epoch + 1) % args.save_every == 0
                                         or (epoch + 1) % args.eval_every == 0):
            run_provenance["forecast_bn_calibration"] = refresh_forecast_bn_ddp(
                model, h, device, args.batch_size * world_size, args.workers)
        provenance = dict(run_provenance); provenance.update(detector_init); provenance.update({"detector_compression": detector_compression,
            "init_checkpoint_sha256": _sha256_file(init_weights), "total_steps": total_steps,
            "partial_epoch": partial_epoch})
        checkpoint_epoch = epoch if partial_epoch else epoch + 1
        checkpoint = {"model": model.module.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "epoch": checkpoint_epoch, "step": step, "global_samples": global_samples, "provenance": provenance}
        if rank == 0:
            if args.no_validation:
                if partial_epoch or (epoch + 1) % args.save_every == 0:
                    save_external_checkpoint(checkpoint, output)
            elif partial_epoch:
                _atomic_save(
                    checkpoint, output / ("checkpoint_step_%06d.pt" % step))
                _atomic_save(checkpoint, output / "checkpoint.pt")
            elif (epoch + 1) % args.save_every == 0:
                _atomic_save(checkpoint, output / ("checkpoint_epoch_%03d.pt" % (epoch + 1)))
                _atomic_save(checkpoint, output / "checkpoint.pt")
        dist.barrier()
        if partial_epoch:
            break
        tracking = forecast = None
        if not args.no_validation and (epoch + 1) % args.eval_every == 0:
            tracking = (evaluate_tracking_test_model_ddp if args.full_data else evaluate_tracking_model_ddp)(
                model, h, device, os.environ["V2V4REAL_ROOT"],
                os.environ["DUALCMP_TRACKING_LABEL_ROOT"],
                partner_delay_ms=args.partner_delay_ms)
            forecast = evaluate_forecast_training_ddp(
                model, h, device, os.environ["V2V4REAL_ROOT"],
                os.environ["DUALCMP_TRACKING_LABEL_ROOT"],
                batch_size=args.batch_size, workers=args.workers,
                full_test=args.full_data,
                partner_delay_ms=args.partner_delay_ms,
            )
        if rank == 0:
            row = dict(zip(("L_det", "L_track", "L_pred", "L_total"), (epoch_totals / epoch_count).tolist()))
            row.update({"kind": "epoch", "split": "train" if args.no_validation else "train+test", "epoch": epoch + 1, "global_step": step,
                        "global_samples": global_samples, "lr_detector": optimizer.param_groups[0]["lr"],
                        "lr_tracking": optimizer.param_groups[1]["lr"], "lr_forecast": optimizer.param_groups[2]["lr"]})
            row["evaluation_protocol"] = evaluation_protocol
            if tracking is not None:
                row.update({"test_track_" + key: tracking["metrics"][key]
                            for key in TRACKING_METRICS})
                row.update({"test_track_timelines": tracking.get("timeline_count", ""),
                            "test_track_frames": tracking.get("frame_count", "")})
                row.update({"test_forecast_" + key: forecast["model"][key]
                            for key in forecast_metrics})
                (output / ("test_tracking_epoch_%03d.json" % (epoch + 1))).write_text(json.dumps(tracking, indent=2), encoding="utf-8")
                (output / ("test_forecast_epoch_%03d.json" % (epoch + 1))).write_text(json.dumps(forecast, indent=2), encoding="utf-8")
                print(json.dumps({"kind": "forecast_evaluation", "epoch": epoch + 1, "report": forecast}), flush=True)
            _append_csv(output / "metrics.csv", row, csv_fields); print(json.dumps(row), flush=True)
        epoch += 1
        if limit_reached:
            break
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            break
    _assert_sync(model.module)
    elapsed = max(time.monotonic() - started, 1e-9)
    provenance = dict(run_provenance)
    provenance.update(detector_init)
    provenance.update({"detector_compression": detector_compression,
                       "init_checkpoint_sha256": _sha256_file(init_weights),
                       "partial_epoch": stopped_partial,
                       "init_provenance": payload.get("provenance") if isinstance(payload, dict) else None})
    report = {
        "steps": step, "epochs_completed": epoch, "dataset_samples": len(dataset),
        "validation_mode": "external" if args.no_validation else "inline",
        "loss": dict(zip(("L_det", "L_track", "L_pred", "L_total"), last or [])),
        "forecast_loss_components_last_update": last_forecast_components,
        "per_rank_batch": args.batch_size,
        "global_batch": args.batch_size * world_size,
        "workers_per_rank": args.workers, "parameter_counts": parameter_counts,
        "optimizer": "AdamW", "base_lr": args.base_lr,
        "detector_compression": detector_compression,
        "group_lrs_initial": {"detector": args.base_lr * args.detector_lr_scale,
                              "tracking": args.base_lr,
                              "forecast": args.base_lr * args.forecast_lr_scale},
        "loss_weights": {"det": args.lambda_det, "track": args.lambda_track,
                         "pred": args.lambda_pred,
                         "forecast_mean": forecast_mean_weight},
        "weight_decay": args.weight_decay,
        "partner_delay_ms": args.partner_delay_ms,
        "train_delays_ms": list(args.train_delays_ms),
        "delay_seed": args.delay_seed,
        "train_delay_histogram": list(dataset.partner_delay_histogram),
        "partial_epoch": stopped_partial,
        "forecast_representation": args.forecast_representation,
        "capture_alignment": args.capture_alignment,
        "age_encoding": args.age_encoding,
        "mean_source_aggregation": args.mean_source_aggregation,
        "capture_time_evidence": (
            args.capture_alignment and args.age_encoding),
        "scheduler": "linear-warmup-cosine",
        "warmup_steps": warmup_steps, "min_lr_ratio": args.min_lr_ratio,
        "last_group_lrs": {group["group_name"]: group["lr"] for group in optimizer.param_groups},
        "forecast_to_evidence_bev_grad_norm_min_rank": forecast_to_evidence_grad,
        "forecast_to_source_bev_grad_norm_min_rank": (
            forecast_to_evidence_grad
            if args.forecast_representation in (
                "source", "source_temporal", "source_temporal_aligned") else None),
        "joint_detector_grad_norm_min_rank": joint_detector_grad,
        "last_grad_norm": float(grad_norm), "ddp_sync": True,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
        "clips_per_second_per_rank": step * args.batch_size / elapsed,
        "provenance": provenance,
    }
    if rank == 0:
        checkpoint = output / "checkpoint.pt"
        report["checkpoint_sha256"] = _sha256_file(checkpoint)
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
