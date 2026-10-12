"""Two-process Phase-B source-preserved DualCMP forecast trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time, random, re
import numpy as np
import subprocess
import sys
from contextlib import nullcontext
from itertools import islice

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

from opencood.data_utils.dualcmp_dataloader import build_dataloader
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import (
    DualCMPSequenceDataset,
)
from opencood.loss.dualcmp_loss import DualCmpLoss
from opencood.loss.dualcmp_gmm_loss import (
    FORECAST_SUPERVISIONS, PARTIAL_FORECAST_SUPERVISION, COMPLETE_FORECAST_SUPERVISION,
)
from opencood.loss.forecast_association import (
    LEGACY_ASSOCIATION, VALIDATED_ASSOCIATION, ASSOCIATION_POLICIES,
    association_contract, associate_forecast_clip,
    forecast_metric_version,
)
from opencood.models.dualcmp.model import DualCmpClipModel
from opencood.models.dualcmp.source_evidence_policy import EVIDENCE_POLICIES, FEATURE_ENCODERS, MOTION_EVIDENCE_CONTRACT
from opencood.tools.train_utils import to_device
from opencood.tools.dualcmp.protocol import (
    DELAY_FORECAST_SAMPLES,
    DELAY_FORECAST_WINDOW,
    FORECAST_WINDOW,
    FORECAST_SAMPLES,
    TRAIN_LABEL_WINDOW,
    checkpoint_detector_compression,
    normalize_train_delays_ms,
)
from opencood.tools.dualcmp.temporal_protocol import (
    resolve_forecast_ablation, validate_training_delay_contract,
)
from opencood.tools.dualcmp.train_tracking_ddp import _sha256_file, _assert_sync, _sync_buffers
from opencood.tools.dualcmp.train_tracking_ddp import _append_csv, _atomic_save, _trim_csv_after_checkpoint
from opencood.tools.dualcmp.eval_forecast_mainstream_ddp import evaluate_forecast_training_ddp
from opencood.tools.dualcmp.cmp_adapted_forecast_metrics import CONTRACT as EVALUATION_CONTRACT, PRIMARY_METRICS, PROTOCOL as EVALUATION_PROTOCOL
from opencood.tools.dualcmp.detector_config import (
    DEFAULT_HYPES, load_checkpoint_hypes,
)
from opencood.tools.dualcmp.aligned_protocol import ALIGNED_FORECAST_WINDOW
from opencood.models.dualcmp.forecast_normalization import FORECAST_NORMALIZATIONS
from opencood.models.dualcmp.intention_query_forecast import GMM_VARIANCES
from opencood.tools.dualcmp.forecast_bn_refresh import refresh_forecast_bn_ddp
from opencood.tools.dualcmp.checkpoint_handoff import save_external_checkpoint, prepare_training_output

LEGACY_FORECAST_METRICS = ("minADE_1s", "minADE_3s", "minADE_5s",
                    "minFDE_1s", "minFDE_3s", "minFDE_5s",
                    "MR_3.6", "count")
FORECAST_METRICS = PRIMARY_METRICS
TRAIN_LABEL_FORECAST_SAMPLES = 10
CSV_FIELDS = (("kind", "split", "epoch", "global_step", "global_samples", "lr",
               "L_gmm", "L_mode", "L_mean_reg", "L_dense_gmm", "L_total",
               "forecast_association_policy", "forecast_metric_version", "forecast_supervision", "evaluation_protocol")
              + tuple("test_" + key for key in FORECAST_METRICS))
LEGACY_CSV_FIELDS = tuple(key for key in CSV_FIELDS if not key.startswith("test_")) + tuple(
    "test_" + key for key in LEGACY_FORECAST_METRICS)

PHASE_B_SOURCES = (
    "opencood/tools/dualcmp/train_tracking_ddp.py",
    "opencood/tools/dualcmp/checkpoint_handoff.py",
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
    "opencood/models/dualcmp/capture_time_evidence.py",
    "opencood/models/dualcmp/mtr_actor_adapter.py",
    "opencood/models/dualcmp/mtr_vendor.py",
    "opencood/models/dualcmp/mtr_attention.py",
    "opencood/models/dualcmp/intention_query_forecast.py",
    "opencood/loss/dualcmp_loss.py",
    "opencood/loss/forecast_association.py",
    "opencood/loss/point_pillar_loss.py",
    "opencood/models/point_pillar_fax.py",
    "opencood/loss/dualcmp_gmm_loss.py",
    "opencood/data_utils/v2v4real/datasets/dualcmp_sequence_dataset.py",
    "opencood/data_utils/v2v4real/datasets/dualcmp_sequence_index.py",
    "opencood/data_utils/v2v4real/datasets/dualcmp_sequence_boundaries.py",
    "opencood/hypes_yaml/dualcmp_sequence_boundaries_v1.json",
    "opencood/tools/dualcmp/aligned_protocol.py",
    "opencood/data_utils/v2v4real/datasets/dualcmp_target_builder.py",
    "opencood/data_utils/dualcmp_dataloader.py",
    "opencood/tools/dualcmp/protocol.py",
    "opencood/tools/dualcmp/temporal_protocol.py",
    "opencood/tools/dualcmp/eval_forecast_ddp.py",
    "opencood/tools/dualcmp/online.py",
    "opencood/tools/dualcmp/eval_forecast_mainstream_ddp.py",
    "opencood/tools/dualcmp/mainstream_forecast_metrics.py",
    "opencood/tools/dualcmp/cmp_adapted_forecast_metrics.py",
    "opencood/tools/dualcmp/detector_config.py",
    "opencood/tools/dualcmp/forecast_metrics.py",
    "opencood/tools/dualcmp/build_aligned_intentions.py",
    "opencood/tools/dualcmp/build_aligned_labels.py",
    "opencood/tools/dualcmp/train_forecast_ddp.py",
    "opencood/hypes_yaml/point_pillar_v2v4real.yaml",
    "MTR/mtr/models_v2v4real/context_encoder/mtr_encoder.py",
    "MTR/mtr/models_v2v4real/motion_decoder/mtr_decoder.py",
    "MTR/mtr/utils/common_utils.py",
    "MTR/mtr/models_v2v4real/utils/transformer/position_encoding_utils.py",
    "MTR/mtr/models_v2v4real/utils/transformer/transformer_encoder_layer.py",
    "MTR/mtr/models_v2v4real/utils/transformer/multi_head_attention.py",
)


def _provenance():
    root = Path(__file__).resolve().parents[3]
    code_sha = os.environ.get("DUALCMP_CODE_SHA")
    if code_sha:
        if not re.match(r"^[0-9a-fA-F]{40}$", code_sha):
            raise EnvironmentError("DUALCMP_CODE_SHA must be 40 hexadecimal characters")
        head = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root))
            .decode("utf-8")
            .strip()
        )
        dirty = (
            subprocess.check_output(
                ["git", "status", "--porcelain", "--", *PHASE_B_SOURCES], cwd=str(root)
            )
            .decode("utf-8")
            .strip()
        )
        if code_sha.lower() != head.lower() or dirty:
            raise RuntimeError(
                "clean source SHA requires HEAD match and no Phase-B source changes"
            )
        return {
            "source_state": "clean",
            "source_sha": code_sha.lower(),
            "protocol_sha256": _sha256_file(Path(__file__).with_name("protocol.py")),
        }
    base_sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root))
        .decode("utf-8")
        .strip()
    )
    digest = hashlib.sha256()
    for relative in sorted(PHASE_B_SOURCES):
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update((root / relative).read_bytes())
    return {
        "source_state": "dirty-smoke",
        "base_sha": base_sha,
        "source_tree_sha256": digest.hexdigest(),
        "protocol_sha256": _sha256_file(Path(__file__).with_name("protocol.py")),
    }


def _hypes(path, args=None):
    root, labels, tracker_weights = (
        os.environ.get(key)
        for key in (
            "V2V4REAL_ROOT",
            "DUALCMP_TRACKING_LABEL_ROOT",
            "DUALCMP_TRACKER_WEIGHTS",
        )
    )
    train_labels = os.environ.get("DUALCMP_TRAIN_TRACKING_LABEL_ROOT")
    if os.environ.get("DUALCMP_MTR_WEIGHTS"):
        raise RuntimeError("formal Phase B rejects pretrained MTR weights")
    if not root or not labels or not tracker_weights:
        raise EnvironmentError(
            "V2V4REAL_ROOT, DUALCMP_TRACKING_LABEL_ROOT, "
            "and DUALCMP_TRACKER_WEIGHTS are required"
        )
    full_data = bool(args is not None and args.full_data)
    train_label_smoke = bool(args is not None and args.train_label_smoke)
    delay_smoke = bool(args is not None and getattr(args, "delay_smoke", False))
    train_subset = train_label_smoke or delay_smoke
    split = "train" if train_subset else (args.data_split if full_data else "test")
    if (full_data or train_subset) and split == "train" and not train_labels:
        raise EnvironmentError("formal train split requires DUALCMP_TRAIN_TRACKING_LABEL_ROOT")
    tracker_payload = torch.load(tracker_weights, map_location="cpu")
    detector_compression = checkpoint_detector_compression(tracker_payload)
    h, detector_init = load_checkpoint_hypes(path, tracker_payload)
    h["root_dir"] = str(Path(root) / split)
    h["validate_dir"] = h["root_dir"]
    h["model"]["args"]["compression"] = detector_compression
    h["fusion"] = {
        "args": {
            "stage": "forecast",
            "tracking_frames": 3,
            "forecast_history_frames": 10,
            "future_frames": 50,
            "runtime_only": False,
            "runtime_forbid_gt": True,
            "receiver_ids": [0, 1],
            "max_actors": 128,
            "max_cav": 2,
            "partner_delay_ms": float(
                getattr(args, "partner_delay_ms", 0.0))
                if args is not None else 0.0,
            "partner_delay_choices_ms": list(
                getattr(args, "train_delays_ms", (0.0,)))
                if args is not None else [0.0],
            "partner_delay_seed": int(
                getattr(args, "delay_seed", 20260901))
                if args is not None else 20260901,
            "shuffle_training_points": False,
        }
    }
    if args is not None and getattr(args, "exclude_scenes_manifest", None):
        h["fusion"]["args"]["excluded_scene_ids"] = json.loads(Path(args.exclude_scenes_manifest).read_text())["shared_scene_names"]
    if full_data or train_subset:
        h["fusion"]["args"]["split_name"] = split
        if split == "train":
            h["fusion"]["args"].update({"tracking_label_root": train_labels,
                                           "tracking_label_layout": "dmstrack_train"})
            if train_subset:
                h["fusion"]["args"]["window"] = list(
                    (ALIGNED_FORECAST_WINDOW if args.forecast_representation == "source_temporal_aligned"
                     else DELAY_FORECAST_WINDOW) if delay_smoke
                    else TRAIN_LABEL_WINDOW)
        elif split == "test":
            h["fusion"]["args"]["tracking_label_root"] = labels
    else:
        h["fusion"]["args"].update({
            "tracking_label_root": labels, "window": list(FORECAST_WINDOW),
        })
    return h, tracker_weights, tracker_payload, detector_compression, detector_init


def _label_contract(args):
    if getattr(args, "delay_smoke", False):
        return {"layout": "dmstrack_train", "split": "train",
                "window": list(ALIGNED_FORECAST_WINDOW if args.forecast_representation == "source_temporal_aligned"
                               else DELAY_FORECAST_WINDOW)}
    if args.train_label_smoke:
        return {"layout": "dmstrack_train", "split": "train",
                "window": list(TRAIN_LABEL_WINDOW)}
    if args.full_data:
        return {"layout": "dmstrack_train" if args.data_split == "train" else "legacy",
                "split": args.data_split, "window": None}
    return {"layout": "legacy", "split": "test", "window": list(FORECAST_WINDOW)}


def _ensure_forecast_ddp_backward(model, objective, loss, output):
    """Connect absent branches only; preserve normal supervised autograd."""
    if (not bool(loss["C_gmm"] > 0)
            or output.get("forecast", {}).get("needs_forecast_ddp_anchor", False)):
        objective = objective + sum(parameter.reshape(-1)[0] * 0.0
                                    for parameter in model.parameters() if parameter.requires_grad)
    return objective


def _forecast_batch_windows(loader, accumulation_steps):
    """Drop an incomplete epoch tail so every update has the declared batch."""
    iterator = iter(loader)
    while True:
        window = list(islice(iterator, accumulation_steps))
        if len(window) != accumulation_steps:
            return
        yield window


def _forecast_window_loss(model, criterion, window, device):
    """Keep loss graphs until valid counts over the entire window are known.

    This uses memory for all microbatch graphs, avoiding a second forward and
    avoiding separately normalized microbatch means. Only the final forward
    prepares the DDP reducer; one combined backward follows in the caller.
    """
    fields = ("gmm", "mode", "mean_reg", "dense_gmm")
    loss, sample_count, needs_anchor = None, 0, False
    for index, host_batch in enumerate(window):
        batch = to_device(host_batch, device)
        with model.no_sync() if index + 1 < len(window) else nullcontext():
            outputs, _ = model(batch["runtime_frames"], run_forecast=True)
            if getattr(criterion, "forecast_association_policy", LEGACY_ASSOCIATION) == VALIDATED_ASSOCIATION:
                association_targets = batch["association_target_frames"]
                if not torch.equal(association_targets[-1]["actor_ids"], batch["target_frames"][-1]["actor_ids"]):
                    raise RuntimeError("causal association and final forecast GT identities disagree")
                assigned, records = associate_forecast_clip(outputs, association_targets)
                current = criterion([outputs[-1]], batch["target_frames"], forecast_assignments=[assigned])
                outputs[-1]["forecast_association"] = dict(records[-1], assigned=assigned)
            else:
                current = criterion([outputs[-1]], batch["target_frames"])
        output = outputs[-1]
        needs_anchor = needs_anchor or output.get("forecast", {}).get("needs_forecast_ddp_anchor", False)
        sample_count += len(batch["scene_receiver_keys"])
        loss = current if loss is None else {
            prefix + field: loss[prefix + field] + current[prefix + field]
            for prefix in ("S_", "C_") for field in fields}
    if needs_anchor and not output.get("forecast", {}).get("needs_forecast_ddp_anchor", False):
        output = dict(output, forecast=dict(output.get("forecast", {}), needs_forecast_ddp_anchor=True))
    return loss, output, sample_count


@torch.no_grad()
def _snapshot_training_normalization(model):
    return [(buffer, buffer.detach().clone()) for module in model.modules()
            if module.training and isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            for buffer in module.buffers(recurse=False)]


@torch.no_grad()
def _restore_training_normalization(snapshot):
    for buffer, previous in snapshot:
        buffer.copy_(previous)


def _configure_deterministic_runtime(enabled):
    if enabled:
        if torch.cuda.is_initialized():
            raise RuntimeError("configure deterministic runtime before initializing CUDA")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "mtr_local_attention": ("pytorch_indexed_dot_sum_v1" if enabled else "native_atomic_v2"),
        "scope": "same-runtime reproducibility; custom CUDA kernels and GPU-count parity are not guaranteed",
    }


def _persistent_query_position_gradients(head, device):
    return torch.stack([
        sum((parameter.grad.detach().float().norm()
             for parameter in layer.ca_qpos_proj.parameters()
             if parameter.grad is not None), torch.zeros((), device=device))
        for layer in list(head.decoder.obj_decoder_layers) + list(head.evidence_decoder_layers)
        if any(parameter.requires_grad for parameter in layer.ca_qpos_proj.parameters())
    ])


def _validate_source_motion_gradients(gradients, policy, full_data):
    # These ablations deliberately remove appearance matching or recurrence.
    required = gradients.new_tensor((policy not in ("source_current", "geometry_temporal"),
                                     policy != "source_current"), dtype=torch.bool)
    if (not torch.isfinite(gradients).all()
            or (not full_data and (gradients[required] <= 0).any())):
        raise RuntimeError("active learning-subset correspondence/recurrent-state gradients must be finite positive")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hypes",
        default=DEFAULT_HYPES,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=1,
                        help="microbatches per update; normalize by whole-window valid counts")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-subset", action="store_true",
                        help="cache <=16 deterministic real smoke samples, never full data")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", choices=("cmp_step_v1", "constant_v1"), default="cmp_step_v1",
                        help="constant_v1 is an explicit small-subset learning diagnostic")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--forecast-mean-weight", type=float)
    parser.add_argument("--forecast-association-policy", choices=ASSOCIATION_POLICIES,
                        help="versioned target-only forecast supervision, independent of evidence ablations")
    parser.add_argument("--forecast-supervision", choices=FORECAST_SUPERVISIONS,
                        help="complete fifty-frame main forecasting targets; explicit partial_future_v1 retains old training")
    parser.add_argument("--forecast-variance", choices=GMM_VARIANCES,
                        help="versioned positive GMM scale parameterization; old checkpoints retain log_std_clamp_v1")
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
    parser.add_argument("--intention-points-file",
                        help="train-only radian/current-receiver prior; required for source_temporal_aligned")
    parser.add_argument("--aligned-evidence-policy", choices=EVIDENCE_POLICIES, default="source_temporal")
    parser.add_argument("--source-feature-encoder", choices=FEATURE_ENCODERS,
                        help="aligned new training defaults to motion_correlation_v1; resume uses the saved encoder")
    parser.add_argument("--forecast-normalization", choices=FORECAST_NORMALIZATIONS,
                        help="retain MTR batch_norm_v1 by default; resume uses the saved contract")
    parser.add_argument("--refresh-forecast-bn", action="store_true",
                        help="refresh head BN from GT-free train observations before saving/evaluation")
    parser.add_argument("--forecast-dropout", type=float,
                        help="forecast attention dropout; default .1, resume retains its recorded value")
    parser.add_argument("--exclude-scenes-manifest", help="opt-in audited overlap exclusion for aligned training")
    parser.add_argument("--disable-capture-alignment", action="store_true")
    parser.add_argument("--disable-age-encoding", action="store_true")
    parser.add_argument("--mean-source-aggregation", action="store_true",
                        help="mask-average source tokens before intention attention")
    parser.add_argument("--disable-capture-time-evidence", action="store_true",
                        help="matched stale-direct ablation for delayed source BEV")
    parser.add_argument("--full-data", action="store_true")
    parser.add_argument("--dirty-smoke", action="store_true",
                        help="Record an exact working-tree manifest; does not limit the dataset")
    parser.add_argument("--train-label-smoke", action="store_true")
    parser.add_argument(
        "--delay-smoke", action="store_true",
        help="use the frozen moving train subset for delayed-evidence acceptance")
    parser.add_argument("--data-split", choices=("train", "val", "test"),
                        default="train")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--no-validation", action="store_true",
                        help="publish immutable checkpoints for external validation")
    parser.add_argument("--resume")
    parser.add_argument("--deterministic", action="store_true",
                        help="require deterministic PyTorch algorithms; configure before CUDA initialization")
    parser.add_argument("--stop-after-epoch", type=int, default=0)
    args = parser.parse_args()
    if args.no_validation and args.refresh_forecast_bn:
        raise ValueError("external validation mode cannot refresh forecast BN on the remote trainer")
    if (not np.isfinite(args.lr) or args.lr <= 0
            or not np.isfinite(args.weight_decay) or args.weight_decay < 0):
        raise ValueError("learning rate must be finite and positive; weight decay finite and non-negative")
    if (args.epochs < 1 or args.max_steps < 0 or args.stop_after_epoch < 0
            or min(args.batch_size, args.accumulation_steps, args.save_every, args.eval_every, args.log_every) < 1
            or args.workers < 0):
        raise ValueError("invalid Phase-B budget, batch size, worker count or reporting interval")
    ablation = resolve_forecast_ablation(
        args.forecast_representation, args.disable_capture_alignment,
        args.disable_age_encoding, args.disable_capture_time_evidence,
        args.mean_source_aggregation)
    args.forecast_representation = ablation["forecast_representation"]
    args.capture_alignment = ablation["capture_alignment"]
    args.age_encoding = ablation["age_encoding"]
    args.mean_source_aggregation = ablation["mean_source_aggregation"]
    resume_payload = torch.load(args.resume, map_location="cpu") if args.resume else None
    if resume_payload is not None and (not isinstance(resume_payload, dict)
            or not isinstance(resume_payload.get("provenance"), dict)):
        raise RuntimeError("resume checkpoint requires model and provenance metadata")
    if args.forecast_dropout is None:
        args.forecast_dropout = (float(resume_payload["provenance"].get("forecast_dropout", .1))
                                 if resume_payload is not None else .1)
    if not np.isfinite(args.forecast_dropout) or not 0 <= args.forecast_dropout < 1:
        raise ValueError("forecast dropout must be finite and in [0,1)")
    if args.source_feature_encoder is None:
        args.source_feature_encoder = (resume_payload.get("provenance", {}).get("source_feature_encoder", "legacy_additive_v1")
            if resume_payload is not None else ("motion_correlation_v1"
                if args.forecast_representation == "source_temporal_aligned" else "legacy_additive_v1"))
    if args.forecast_normalization is None:
        args.forecast_normalization = (resume_payload.get("provenance", {}).get("forecast_normalization", "batch_norm_v1")
            if resume_payload is not None else "batch_norm_v1")
    if args.refresh_forecast_bn and (args.forecast_normalization != "batch_norm_v1"
            or not (args.delay_smoke or args.train_label_smoke or (args.full_data and args.data_split == "train"))):
        raise ValueError("BN refresh requires batch_norm_v1 and actual training observations")
    if args.forecast_representation != "source_temporal_aligned" and (
            args.aligned_evidence_policy != "source_temporal"
            or args.source_feature_encoder != "legacy_additive_v1"
            or args.forecast_normalization != "batch_norm_v1"
            or args.exclude_scenes_manifest):
        raise ValueError("new feature/policy/split parameters require source_temporal_aligned")
    if args.forecast_association_policy is None:
        args.forecast_association_policy = (VALIDATED_ASSOCIATION
            if args.forecast_representation == "source_temporal_aligned" else LEGACY_ASSOCIATION)
    if args.forecast_supervision is None:
        args.forecast_supervision = (COMPLETE_FORECAST_SUPERVISION
            if args.forecast_representation == "source_temporal_aligned" and args.forecast_association_policy == VALIDATED_ASSOCIATION
            else PARTIAL_FORECAST_SUPERVISION)
    if args.forecast_representation != "source_temporal_aligned" and args.forecast_supervision != PARTIAL_FORECAST_SUPERVISION:
        raise ValueError("legacy evidence ablations must retain their original future supervision")
    if args.forecast_variance is None:
        args.forecast_variance = (resume_payload.get("provenance", {}).get("forecast_variance", "log_std_clamp_v1")
            if resume_payload is not None else "log_std_clamp_v1")
    if args.forecast_representation != "source_temporal_aligned" and args.forecast_variance != "log_std_clamp_v1":
        raise ValueError("legacy evidence ablations must retain their original GMM parameterization")
    if args.forecast_mean_weight is None:
        args.forecast_mean_weight = (
            5.0 if args.forecast_representation in ("source_temporal", "source_temporal_aligned") else 0.0)
    if (not np.isfinite(args.forecast_mean_weight)
            or args.forecast_mean_weight < 0):
        raise ValueError("--forecast-mean-weight must be finite and non-negative")
    if (args.forecast_representation not in ("source_temporal", "source_temporal_aligned")
            and args.forecast_mean_weight != 0):
        raise ValueError(
            "--forecast-mean-weight is exclusive to source_temporal")
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
    if args.full_data and args.lr_schedule != "cmp_step_v1":
        raise ValueError("constant LR is a small-subset diagnostic, not the formal full-data schedule")
    if not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("lr must be finite and positive; weight_decay must be finite and nonnegative")
    if not np.isfinite(args.partner_delay_ms) or args.partner_delay_ms < 0:
        raise ValueError("partner delay must be finite and non-negative")
    subset_modes = int(args.train_label_smoke) + int(args.delay_smoke)
    if subset_modes > 1 or (subset_modes and args.full_data):
        raise ValueError(
            "--train-label-smoke, --delay-smoke, and --full-data are mutually exclusive")
    if args.log_every < 1 or args.save_every < 1 or args.eval_every < 1:
        raise ValueError("Phase B logging, saving, and evaluation intervals must be positive")
    if args.full_data and (args.save_every != 1 or (not args.no_validation and args.eval_every != 1)):
        raise ValueError("formal Phase B requires save_every=1 and eval_every=1")
    if args.resume and Path(args.resume).resolve().parent != Path(args.output_dir).resolve():
        raise ValueError("--resume checkpoint must belong to --output-dir")
    if args.dirty_smoke and os.environ.get("DUALCMP_CODE_SHA"):
        raise EnvironmentError("--dirty-smoke requires unsetting DUALCMP_CODE_SHA")
    if args.full_data and not args.dirty_smoke and not os.environ.get("DUALCMP_CODE_SHA"):
        raise EnvironmentError("full-data Phase B requires clean DUALCMP_CODE_SHA or explicit --dirty-smoke")
    run_provenance = _provenance()
    if args.forecast_representation == "source_temporal_aligned":
        if not args.intention_points_file:
            raise ValueError("aligned Source requires --intention-points-file built on train labels")
        points_path = Path(args.intention_points_file)
        metadata = json.loads(points_path.with_suffix(".json").read_text())
        points_sha = _sha256_file(points_path)
        if (metadata.get("coordinate_frame") != "current_receiver" or metadata.get("heading_unit") != "radians"
                or metadata.get("split") != "train" or metadata.get("sha256") != points_sha):
            raise RuntimeError("aligned intention point provenance disagrees with the requested contract")
        run_provenance.update(intention_points_file=args.intention_points_file,
                              intention_points_sha256=points_sha,
                              forecast_coordinate_frame="current_receiver")
        excluded = []
        if args.exclude_scenes_manifest:
            excluded = json.loads(Path(args.exclude_scenes_manifest).read_text())["shared_scene_names"]
        if sorted(metadata.get("excluded_scenes", [])) != sorted(excluded):
            raise RuntimeError("intention prior and requested training scene exclusions disagree")
        run_provenance.update(aligned_evidence_policy=args.aligned_evidence_policy,
                              source_feature_encoder=args.source_feature_encoder,
                              forecast_normalization=args.forecast_normalization,
                              excluded_scene_ids=sorted(excluded))
        if args.exclude_scenes_manifest:
            run_provenance["excluded_scenes_manifest_sha256"] = _sha256_file(args.exclude_scenes_manifest)
    elif args.intention_points_file is not None:
        raise ValueError("legacy ablations must retain their original intention points")
    aligned_evaluation = args.forecast_representation == "source_temporal_aligned"
    evaluation_protocol = EVALUATION_PROTOCOL if aligned_evaluation else forecast_metric_version(args.forecast_association_policy)
    forecast_metrics = FORECAST_METRICS if aligned_evaluation else LEGACY_FORECAST_METRICS
    csv_fields = CSV_FIELDS if aligned_evaluation else LEGACY_CSV_FIELDS
    run_provenance["forecast_evaluation"] = dict(EVALUATION_CONTRACT) if aligned_evaluation else {"protocol": evaluation_protocol}
    run_provenance["validation_mode"] = "external" if args.no_validation else "inline"
    run_provenance.update({"initial_lr": args.lr, "weight_decay": args.weight_decay,
                           "refresh_forecast_bn": args.refresh_forecast_bn,
                           "forecast_dropout": args.forecast_dropout,
                           "lr_schedule": args.lr_schedule,
                           "forecast_association": association_contract(args.forecast_association_policy),
                           "forecast_supervision": args.forecast_supervision,
                           "forecast_variance": args.forecast_variance,
                           "forecast_metric_version": forecast_metric_version(args.forecast_association_policy),
                           "unsupervised_batch_contract": ("conditional_ddp_zero_backward_skip_update_restore_norm_v3"
                               if args.forecast_association_policy == VALIDATED_ASSOCIATION
                               else "conditional_ddp_zero_backward_skip_global_empty_update_v2"),
                           "input_subset_cached": bool(args.cache_subset),
                           "partner_delay_ms": args.partner_delay_ms,
                           "train_delays_ms": list(args.train_delays_ms),
                           "delay_seed": args.delay_seed,
                           "forecast_representation": args.forecast_representation,
                           "capture_alignment": args.capture_alignment,
                           "age_encoding": args.age_encoding,
                           "mean_source_aggregation": (
                               args.mean_source_aggregation),
                           "capture_time_evidence": (
                               args.capture_alignment and args.age_encoding),
                           "argv": list(sys.argv)})
    run_provenance["deterministic_runtime"] = _configure_deterministic_runtime(args.deterministic)
    random.seed(20260901)
    np.random.seed(20260901)
    torch.manual_seed(20260901)
    torch.cuda.manual_seed_all(20260901)
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Phase B requires torchrun")
    dist.init_process_group("nccl")
    rank, local_rank = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    global_batch = args.batch_size * world_size * args.accumulation_steps
    run_provenance["batch_contract"] = {
        "per_rank_batch": args.batch_size, "world_size": world_size,
        "accumulation_steps": args.accumulation_steps, "global_batch": global_batch,
        "loss_normalization": "global_valid_counts_over_whole_accumulation_window",
        "tail_policy": "drop_incomplete_accumulation_window"}
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    h, tracker_weights, payload, detector_compression, detector_init = _hypes(
        args.hypes, args
    )
    h["fusion"]["args"]["include_forecast_association_history"] = (
        args.forecast_association_policy == VALIDATED_ASSOCIATION)
    if args.forecast_representation == "source_temporal_aligned":
        h["fusion"]["args"]["forecast_coordinate_frame"] = "current_receiver"
        run_provenance["motion_input_contract"] = "causal_velocity_acceleration_and_predicted_dense_velocity_v1"
        run_provenance["decoder_query_contract"] = "persistent_static_intention_position_v1"
    dataset = DualCMPSequenceDataset(h, train=True)
    if dataset.sequence_policy is not None:
        run_provenance["sequence_policy"] = dataset.sequence_policy
        if metadata.get("sequence_policy", {}).get("sha256") != dataset.sequence_policy["sha256"]:
            raise RuntimeError("aligned intention prior uses a different sequence-boundary protocol")
        if metadata.get("sequence_policy", {}).get("label_identity") != dataset.sequence_policy.get("label_identity"):
            raise RuntimeError("aligned intention prior uses different GT identities")
    run_provenance["train_delay_histogram"] = list(
        dataset.partner_delay_histogram)
    expected_samples = (
        DELAY_FORECAST_SAMPLES if args.delay_smoke
        else (TRAIN_LABEL_FORECAST_SAMPLES if args.train_label_smoke
              else FORECAST_SAMPLES)
    )
    if not args.full_data and len(dataset) != expected_samples:
        raise RuntimeError("forecast protocol requires exactly %d samples" % expected_samples)
    if args.cache_subset:
        if args.full_data or not (args.delay_smoke or args.train_label_smoke):
            raise ValueError("--cache-subset is restricted to the real train smoke window")
        dataset.cache_small_subset()
    _, loader, sampler = build_dataloader(
        dataset,
        args.batch_size,
        True,
        workers=args.workers,
        rank=rank,
        world_size=world_size,
        seed=20260901,
        training=True,
        collate_fn=dataset.collate_batch_train,
        pin_memory=True,
    )
    updates_per_epoch = len(loader) // args.accumulation_steps
    if updates_per_epoch < 1:
        raise RuntimeError(
            "Phase-B loader has no complete distributed batch; reduce "
            "--batch-size for this sample set")
    forecast_args = {
        "dropout": args.forecast_dropout,
        "history_frames": 11,
        "future_frames": 50,
        "modes": 6,
        "intention_points_file": args.intention_points_file or "preprocessed_data/v2v4real/v2v4real_cluster_64_center_dict.pkl",
        "representation": args.forecast_representation,
        "capture_alignment": args.capture_alignment,
        "age_encoding": args.age_encoding,
        "mean_source_aggregation": args.mean_source_aggregation,
    }
    if args.forecast_representation == "source_temporal_aligned":
        forecast_args.update(aligned_evidence_policy=args.aligned_evidence_policy,
                             source_feature_encoder=args.source_feature_encoder,
                             forecast_normalization=args.forecast_normalization,
                             forecast_variance=args.forecast_variance)
    model = DualCmpClipModel(
        h["model"]["args"],
        dataset.anchor_box,
        h["preprocess"]["cav_lidar_range"],
        forecast_args=forecast_args,
    ).to(device)
    model.forecast_association_policy = args.forecast_association_policy
    model.forecast_supervision = args.forecast_supervision
    state = payload.get("model", payload.get("model_state_dict", payload))
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    forbidden = tuple(
        key for key in state
        if key.startswith(("forecast_head.", "actor_source_tokenizer.",
                           "source_temporal_encoder."))
    )
    if forbidden:
        raise RuntimeError("Phase-A checkpoint contains forbidden forecast weights")
    loaded = model.load_state_dict(state, strict=False)
    allowed = ("forecast_head.", "actor_source_tokenizer.",
               "source_temporal_encoder.")
    invalid = [key for key in loaded.missing_keys if not key.startswith(allowed)]
    if loaded.unexpected_keys or invalid:
        raise RuntimeError(
            "Phase-A tracker checkpoint mismatch: missing=%s unexpected=%s"
            % (invalid, loaded.unexpected_keys)
        )
    for parameter in (
        list(model.detector.parameters())
        + list(model.tracker.parameters())
        + list(model.temporal_attention.parameters())
        + list(model.track_update.parameters())
        + list(model.box_encoder.parameters())
        + list(model.box_refinement.parameters())
    ):
        parameter.requires_grad = False
    frozen_temporal_attention = {
        key: value.detach().cpu().clone()
        for key, value in model.temporal_attention.state_dict().items()
    }
    model.train()
    for module in (
        model.detector,
        model.tracker,
        model.roi,
        model.temporal_attention,
        model.track_update,
        model.box_encoder,
        model.box_refinement,
    ):
        module.eval()
    model.forecast_head.train()
    model.actor_source_tokenizer.train()
    if model.source_temporal_encoder is not None:
        model.source_temporal_encoder.train()
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_budget_steps = (
        args.epochs * updates_per_epoch
        if args.full_data else (args.max_steps or args.epochs * updates_per_epoch)
    )
    decay_steps = [
        round(total_budget_steps * epoch / 30.0) for epoch in (22, 24, 26, 28)
    ]

    def lr_multiplier(current_step):
        if args.lr_schedule == "constant_v1":
            return 1.0
        decay = 1.0
        for decay_step in decay_steps:
            if current_step >= decay_step:
                decay *= 0.5
        return max(decay, 1e-6 / args.lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    forecast_mean_weight = args.forecast_mean_weight
    criterion = DualCmpLoss({
        "detector": h["loss"]["args"],
        "forecast_mean_weight": forecast_mean_weight,
        "forecast_association_policy": args.forecast_association_policy,
        "forecast_supervision": args.forecast_supervision,
    }).to(device)
    run_provenance["forecast_mean_weight"] = forecast_mean_weight
    if args.source_feature_encoder == "motion_correlation_v1":
        run_provenance["source_motion_contract"] = MOTION_EVIDENCE_CONTRACT
    run_provenance["tracking_label_contract"] = (
        "dmstrack_train" if (args.full_data and args.data_split == "train")
        or args.train_label_smoke or args.delay_smoke else "legacy")
    run_provenance["tracking_label_scope"] = _label_contract(args)
    if ((args.train_label_smoke or args.delay_smoke)
            and payload.get("provenance", {}).get(
                "tracking_label_contract") != "dmstrack_train"):
        raise RuntimeError(
            "Phase-B train subset requires a native-label Phase-A checkpoint")
    output = Path(args.output_dir)
    start_epoch = step = global_samples = 0
    empty_forecast_batches = 0
    processed_batches = 0
    if args.resume:
        if not isinstance(resume_payload, dict) or any(key not in resume_payload for key in ("model", "optimizer", "scheduler", "epoch", "step")):
            raise RuntimeError("resume checkpoint must contain model, optimizer, scheduler, epoch, and step")
        if args.max_steps and int(resume_payload["step"]) >= args.max_steps:
            raise RuntimeError(
                "Phase B resume checkpoint already reached --max-steps")
        if int(resume_payload["scheduler"].get("last_epoch", -1)) != int(resume_payload["step"]):
            raise RuntimeError("resume scheduler step does not match checkpoint step")
        if int(resume_payload["provenance"].get("total_budget_steps", -1)) != total_budget_steps:
            raise RuntimeError("resume scheduler budget disagrees with requested 30-epoch schedule")
        previous = resume_payload["provenance"]
        if "evaluation_migration" in previous:
            run_provenance["evaluation_migration"] = previous["evaluation_migration"]
        if previous.get("forecast_evaluation", {"protocol": forecast_metric_version(args.forecast_association_policy)}) != run_provenance["forecast_evaluation"]:
            raise RuntimeError("resume evaluation protocol changed; keep historical outputs and use an explicit migration")
        if previous.get("validation_mode", "inline") != run_provenance["validation_mode"]:
            raise RuntimeError("Phase B resume must retain its validation mode")
        if (("deterministic_runtime" in previous and previous["deterministic_runtime"] != run_provenance["deterministic_runtime"])
                or ("deterministic_runtime" not in previous and args.deterministic)):
            raise RuntimeError("Phase B resume must retain its deterministic runtime contract")
        if previous.get("forecast_association", association_contract(LEGACY_ASSOCIATION)) != run_provenance["forecast_association"]:
            raise RuntimeError("Phase B resume requires the original forecast supervision/association contract")
        if previous.get("forecast_supervision", PARTIAL_FORECAST_SUPERVISION) != args.forecast_supervision:
            raise RuntimeError("Phase B resume must retain its original future-supervision contract")
        if previous.get("forecast_variance", "log_std_clamp_v1") != args.forecast_variance:
            raise RuntimeError("Phase B resume must retain its original GMM parameterization")
        if previous.get("lr_schedule", "cmp_step_v1") != args.lr_schedule:
            raise RuntimeError("Phase B resume must retain its original learning-rate schedule")
        if previous.get("refresh_forecast_bn", False) != args.refresh_forecast_bn:
            raise RuntimeError("Phase B resume must retain its BN refresh contract")
        if previous.get("forecast_dropout", .1) != args.forecast_dropout:
            raise RuntimeError("Phase B resume must retain its dropout contract")
        if previous.get("forecast_metric_version", forecast_metric_version(LEGACY_ASSOCIATION)) != run_provenance["forecast_metric_version"]:
            raise RuntimeError("Phase B resume requires the original forecast metric version")
        if previous.get("batch_contract", run_provenance["batch_contract"]) != run_provenance["batch_contract"]:
            raise RuntimeError("Phase B resume requires the original batch/accumulation contract")
        for key, default in (("aligned_evidence_policy", "source_temporal"),
                             ("source_feature_encoder", "legacy_additive_v1"),
                             ("source_motion_contract", None),
                             ("forecast_normalization", "batch_norm_v1"),
                             ("excluded_scene_ids", [])):
            if previous.get(key, default) != run_provenance.get(key, default):
                raise RuntimeError("resume must retain its evidence/feature/split contract")
        if previous.get("sequence_policy") != run_provenance.get("sequence_policy"):
            raise RuntimeError("resume acquisition-boundary contract must not change")
        if (previous.get("intention_points_sha256") != run_provenance.get("intention_points_sha256")
                or previous.get("forecast_coordinate_frame", "legacy_capture") != run_provenance.get("forecast_coordinate_frame", "legacy_capture")):
            raise RuntimeError("resume forecast coordinates and intention points must not change")
        if previous.get("partial_epoch"):
            raise RuntimeError(
                "partial-epoch checkpoints are evaluation-only and cannot resume")
        if previous.get("initial_lr") != args.lr or previous.get("weight_decay") != args.weight_decay:
            raise RuntimeError("Phase B resume requires the original learning rate and weight decay")
        if previous.get("partner_delay_ms") != args.partner_delay_ms:
            raise RuntimeError("Phase B resume requires the original evaluation delay")
        if previous.get("forecast_mean_weight", 0.0) != forecast_mean_weight:
            raise RuntimeError(
                "Phase B resume requires the original forecast mean weight")
        validate_training_delay_contract(
            previous, args.train_delays_ms, args.delay_seed,
            context="Phase B resume",
            forecast_representation=args.forecast_representation,
            capture_alignment=args.capture_alignment,
            age_encoding=args.age_encoding,
            mean_source_aggregation=args.mean_source_aggregation)
        if previous.get("tracking_label_contract") != run_provenance["tracking_label_contract"]:
            raise RuntimeError("resume tracking label contract disagrees with requested training split")
        if previous.get("tracking_label_scope") != run_provenance["tracking_label_scope"]:
            raise RuntimeError("resume tracking label scope disagrees with requested training window")
        source_key = "source_sha" if "source_sha" in run_provenance else "source_tree_sha256"
        if previous.get(source_key) != run_provenance.get(source_key):
            raise RuntimeError("Phase B resume requires the original source state")
        if previous.get("phase_a_checkpoint_sha256") != _sha256_file(tracker_weights):
            raise RuntimeError("Phase B resume requires the original Phase-A checkpoint")
        model.module.load_state_dict(resume_payload["model"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"]); scheduler.load_state_dict(resume_payload["scheduler"])
        start_epoch, step = int(resume_payload["epoch"]), int(resume_payload["step"])
        global_samples = int(resume_payload.get("global_samples", 0))
        empty_forecast_batches = int(resume_payload.get("empty_forecast_batches", 0))
        processed_batches = int(resume_payload.get("processed_batches", step))
    if rank == 0:
        prepare_training_output(output, resume=bool(args.resume), external=args.no_validation)
        if args.resume:
            _trim_csv_after_checkpoint(output / "metrics.csv", step, csv_fields)
    dist.barrier()
    started = time.monotonic()
    initial_step, initial_global_samples = step, global_samples
    last = None
    last_local_forecast_counts = None
    last_source_temporal_grad = None
    last_source_motion_grads = None
    last_source_query_fusion_grad = None
    last_source_actor_grad = None
    last_source_actor_normalization_grad = None
    last_persistent_query_grad = None
    last_persistent_query_layer_grads = None
    stopped_partial = False
    grad_norm = torch.zeros((), device=device)
    epoch = start_epoch
    while epoch < args.epochs:
        sampler.set_epoch(epoch)
        batches_seen = 0
        epoch_totals = np.zeros(4, dtype=np.float64); epoch_counts = np.zeros(4, dtype=np.float64)
        log_totals = np.zeros(4, dtype=np.float64); log_counts = np.zeros(4, dtype=np.float64)
        for window in _forecast_batch_windows(loader, args.accumulation_steps):
            batches_seen += len(window)
            optimizer.zero_grad(set_to_none=True)
            normalization_snapshot = (_snapshot_training_normalization(model)
                if args.forecast_association_policy == VALIDATED_ASSOCIATION else [])
            loss, forecast_output, window_samples = _forecast_window_loss(model, criterion, window, device)
            global_counts = torch.stack(
                (
                    loss["C_gmm"],
                    loss["C_mode"],
                    loss["C_mean_reg"],
                    loss["C_dense_gmm"],
                )
            ).detach()
            last_local_forecast_counts = global_counts.cpu().tolist()
            dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
            forecast_loss = dist.get_world_size() * (
                loss["S_gmm"] / global_counts[0].clamp_min(1.0)
                + loss["S_mode"] / global_counts[1].clamp_min(1.0)
                + forecast_mean_weight * loss["S_mean_reg"]
                / global_counts[2].clamp_min(1.0)
                + loss["S_dense_gmm"] / global_counts[3].clamp_min(1.0)
            )
            forecast_loss = _ensure_forecast_ddp_backward(model, forecast_loss, loss, forecast_output)
            forecast_loss.backward()
            processed_batches += len(window)
            global_samples += window_samples * world_size
            if not bool((global_counts > 0).any()):
                # All ranks finish backward, including locally empty ranks.
                # A wholly empty batch is not an AdamW/momentum/LR update.
                # `step` remains the optimizer-update count; samples still
                # count all consumed clips. Never skip on local counts alone.
                empty_forecast_batches += len(window)
                _restore_training_normalization(normalization_snapshot)
                continue
            if args.forecast_representation == "source_temporal_aligned":
                head = model.module.forecast_head
                position_grads = _persistent_query_position_gradients(head, device)
                # A flat/saturated attention layer can legitimately have zero
                # gradient; connectivity concerns the active branch as a whole.
                position_grad = position_grads.sum()
                dist.all_reduce(position_grad, op=dist.ReduceOp.MIN)
                dist.all_reduce(position_grads, op=dist.ReduceOp.MIN)
                last_persistent_query_layer_grads = position_grads.cpu().tolist()
                last_persistent_query_grad = float(position_grad)
                if not np.isfinite(last_persistent_query_grad):
                    raise RuntimeError("non-finite persistent intention projection gradient")
                # A one-key object attention has constant softmax and exactly
                # zero query-position gradient. Zero supervision can likewise
                # yield legitimate zero gradients. Non-zero checks decide the
                # frozen learning subset, not every arbitrary full-data batch.
                if not args.full_data and last_persistent_query_grad <= 0:
                    raise RuntimeError("learning subset requires non-zero persistent intention gradients")
            if args.forecast_representation in ("source_temporal", "source_temporal_aligned") and model.module.forecast_head.source_evidence_enabled:
                temporal_grad = sum(
                    parameter.grad.detach().float().norm()
                    for parameter in model.module.source_temporal_encoder.parameters()
                    if parameter.grad is not None
                )
                fusion_parameters = ((model.module.forecast_head.temporal_source_fusion_gates,)
                                     if args.forecast_representation == "source_temporal" else
                                     tuple(model.module.forecast_head.evidence_decoder_layers.parameters()))
                fusion_grad = sum(
                    parameter.grad.detach().float().norm()
                    for parameter in fusion_parameters
                    if parameter.grad is not None
                )
                temporal_grad = torch.as_tensor(
                    temporal_grad, device=device, dtype=torch.float32)
                fusion_grad = torch.as_tensor(
                    fusion_grad, device=device, dtype=torch.float32)
                dist.all_reduce(temporal_grad, op=dist.ReduceOp.MIN)
                dist.all_reduce(fusion_grad, op=dist.ReduceOp.MIN)
                last_source_temporal_grad = float(temporal_grad)
                last_source_query_fusion_grad = float(fusion_grad)
                reader = model.module.forecast_head.source_actor_readout
                if reader is not None:
                    normalization_grad = sum(parameter.grad.detach().float().norm()
                        for parameter in reader.output_normalization.parameters() if parameter.grad is not None)
                    normalization_grad = torch.as_tensor(normalization_grad, device=device, dtype=torch.float32)
                    dist.all_reduce(normalization_grad, op=dist.ReduceOp.MIN)
                    last_source_actor_normalization_grad = float(normalization_grad)
                    if not np.isfinite(last_source_actor_normalization_grad) or (
                            not args.full_data and last_source_actor_normalization_grad <= 0):
                        raise RuntimeError("Source attention-output normalization has invalid learning gradient")
                    actor_grad = sum(parameter.grad.detach().float().norm()
                                     for parameter in reader.parameters() if parameter.grad is not None)
                    actor_grad = torch.as_tensor(actor_grad, device=device, dtype=torch.float32)
                    dist.all_reduce(actor_grad, op=dist.ReduceOp.MIN)
                    last_source_actor_grad = float(actor_grad)
                    if not torch.isfinite(actor_grad) or (not args.full_data and actor_grad <= 0):
                        raise RuntimeError("learning subset requires finite positive pre-scene Source readout gradients")
                if args.source_feature_encoder == "motion_correlation_v1":
                    motion = model.module.source_temporal_encoder
                    motion_grads = torch.stack((
                        motion.correlation_encoder[0].weight.grad[:, :motion.pooled_tokens].float().norm(),
                        motion.temporal_encoder[0].convq.weight.grad[:, :motion.feature_dim].float().norm()))
                    dist.all_reduce(motion_grads, op=dist.ReduceOp.MIN)
                    last_source_motion_grads = motion_grads.tolist()
                    _validate_source_motion_gradients(motion_grads, args.aligned_evidence_policy, args.full_data)
                if (not np.isfinite(last_source_temporal_grad)
                        or not np.isfinite(last_source_query_fusion_grad)):
                    raise RuntimeError("non-finite Source temporal gradient")
                if ((not args.full_data or args.forecast_representation != "source_temporal_aligned")
                        and min(last_source_temporal_grad, last_source_query_fusion_grad) <= 0):
                    raise RuntimeError(
                        "learning subset requires non-zero Source temporal gradients on every rank")
            grad_norm = clip_grad_norm_(model.parameters(), 1000.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite Phase-B gradient; optimizer step rejected")
            optimizer.step()
            scheduler.step()
            sums_and_counts = torch.stack(
                (
                    loss["S_gmm"].detach(), loss["C_gmm"],
                    loss["S_mode"].detach(), loss["C_mode"],
                    loss["S_mean_reg"].detach(), loss["C_mean_reg"],
                    loss["S_dense_gmm"].detach(), loss["C_dense_gmm"],
                )
            ).float()
            dist.all_reduce(sums_and_counts, op=dist.ReduceOp.SUM)
            last = {
                "L_gmm": float(sums_and_counts[0] / sums_and_counts[1].clamp_min(1.0)),
                "L_mode": float(sums_and_counts[2] / sums_and_counts[3].clamp_min(1.0)),
                "L_mean_reg": float(sums_and_counts[4] / sums_and_counts[5].clamp_min(1.0)),
                "L_dense_gmm": float(sums_and_counts[6] / sums_and_counts[7].clamp_min(1.0)),
            }
            step += 1
            if step == initial_step + 1 and rank == 0:
                # Caller registers a FIFO before launch. A blocking writer
                # latches successful real forward/loss/backward/optimizer on
                # both ranks, without a polling or log-watching process.
                startup = {"status": "first_optimizer_step_complete", "step": step,
                           "resumed_from_step": initial_step,
                           "current_lr": optimizer.param_groups[0]["lr"],
                           "forecast_association_policy": args.forecast_association_policy,
                           "forecast_supervision": args.forecast_supervision,
                           "forecast_metric_version": forecast_metric_version(args.forecast_association_policy),
                           "world_size": world_size, "global_batch": global_batch,
                           "per_rank_batch": args.batch_size, "accumulation_steps": args.accumulation_steps,
                           "initial_lr": args.lr, "forecast_mean_weight": forecast_mean_weight,
                           "aligned_evidence_policy": args.aligned_evidence_policy,
                           "dataset_samples": len(dataset), "forecast_representation": args.forecast_representation,
                           "loss": last, "gradient_norm": float(grad_norm),
                           "source_temporal_grad": last_source_temporal_grad,
                           "source_query_grad": last_source_query_fusion_grad,
                           "deterministic_runtime": run_provenance["deterministic_runtime"],
                           "source_tree_sha256": run_provenance.get("source_tree_sha256"),
                           "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2 ** 20}
                print(json.dumps(startup), flush=True)
                if os.environ.get("DUALCMP_STARTUP_FIFO"):
                    with open(os.environ["DUALCMP_STARTUP_FIFO"], "w") as event:
                        event.write(json.dumps(startup) + "\n")
            sums = sums_and_counts.detach().cpu().numpy()
            epoch_totals += sums[[0, 2, 4, 6]]; epoch_counts += sums[[1, 3, 5, 7]]
            log_totals += sums[[0, 2, 4, 6]]; log_counts += sums[[1, 3, 5, 7]]
            if step % args.log_every == 0 and rank == 0:
                row = {"kind": "step", "split": "train", "epoch": epoch + 1, "global_step": step,
                       "global_samples": global_samples, "lr": optimizer.param_groups[0]["lr"],
                      "L_gmm": float(log_totals[0] / max(log_counts[0], 1)),
                      "L_mode": float(log_totals[1] / max(log_counts[1], 1)),
                      "L_mean_reg": float(log_totals[2] / max(log_counts[2], 1)),
                      "L_dense_gmm": float(log_totals[3] / max(log_counts[3], 1))}
                row["L_total"] = (row["L_gmm"] + row["L_mode"]
                                  + forecast_mean_weight * row["L_mean_reg"]
                                  + row["L_dense_gmm"])
                row.update(forecast_association_policy=args.forecast_association_policy,
                           forecast_supervision=args.forecast_supervision,
                           forecast_metric_version=forecast_metric_version(args.forecast_association_policy))
                row["evaluation_protocol"] = evaluation_protocol
                _append_csv(output / "metrics.csv", row, csv_fields); print(json.dumps(row), flush=True)
                log_totals[:] = 0; log_counts[:] = 0
            if args.max_steps and step >= args.max_steps:
                break
        limit_reached = bool(args.max_steps and step >= args.max_steps)
        partial_epoch = bool(limit_reached and batches_seen < updates_per_epoch * args.accumulation_steps)
        stopped_partial = partial_epoch
        _sync_buffers(model)
        if args.refresh_forecast_bn and (limit_reached or (epoch + 1) % args.save_every == 0
                                         or (epoch + 1) % args.eval_every == 0):
            run_provenance["forecast_bn_calibration"] = refresh_forecast_bn_ddp(
                model, h, device, global_batch, args.workers)
        provenance = dict(run_provenance); provenance.update(detector_init); provenance.update({"detector_compression": detector_compression,
            "phase_a_checkpoint_sha256": _sha256_file(tracker_weights), "total_budget_steps": total_budget_steps,
            "forecast_initialization": "random", "evidence_initialization": "random",
            "partial_epoch": partial_epoch})
        checkpoint_epoch = epoch if partial_epoch else epoch + 1
        checkpoint = {"model": model.module.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "epoch": checkpoint_epoch, "step": step, "global_samples": global_samples,
                      "empty_forecast_batches": empty_forecast_batches,
                      "processed_batches": processed_batches, "provenance": provenance}
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
        test_report = None
        if not args.no_validation and (epoch + 1) % args.eval_every == 0:
            test_report = evaluate_forecast_training_ddp(
                model, h, device, os.environ["V2V4REAL_ROOT"],
                os.environ["DUALCMP_TRACKING_LABEL_ROOT"],
                batch_size=args.batch_size, workers=args.workers,
                full_test=args.full_data,
                partner_delay_ms=args.partner_delay_ms,
            )
        if rank == 0:
            row = {"kind": "epoch", "split": "train" if args.no_validation else "train+test", "epoch": epoch + 1, "global_step": step,
                   "global_samples": global_samples, "lr": optimizer.param_groups[0]["lr"],
                   "L_gmm": float(epoch_totals[0] / max(epoch_counts[0], 1)), "L_mode": float(epoch_totals[1] / max(epoch_counts[1], 1)),
                   "L_mean_reg": float(epoch_totals[2] / max(epoch_counts[2], 1)),
                   "L_dense_gmm": float(epoch_totals[3] / max(epoch_counts[3], 1))}
            row["L_total"] = (row["L_gmm"] + row["L_mode"]
                              + forecast_mean_weight * row["L_mean_reg"]
                              + row["L_dense_gmm"])
            row.update(forecast_association_policy=args.forecast_association_policy,
                       forecast_supervision=args.forecast_supervision,
                       forecast_metric_version=forecast_metric_version(args.forecast_association_policy))
            row["evaluation_protocol"] = evaluation_protocol
            if test_report is not None:
                row.update({"test_" + key: test_report["model"][key]
                            for key in forecast_metrics})
                (output / ("test_forecast_epoch_%03d.json" % (epoch + 1))).write_text(json.dumps(test_report, indent=2), encoding="utf-8")
                print(json.dumps({"kind": "forecast_evaluation", "epoch": epoch + 1, "report": test_report}), flush=True)
            _append_csv(output / "metrics.csv", row, csv_fields); print(json.dumps(row), flush=True)
        epoch += 1
        if limit_reached:
            break
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            break
    if step == int(resume_payload["step"] if resume_payload is not None else 0):
        raise RuntimeError("Phase B completed without any supervised optimizer update")
    _assert_sync(model.module)
    gathered_local_counts = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_local_counts, last_local_forecast_counts)
    if any(
        not torch.equal(
            value.detach().cpu(), frozen_temporal_attention[key]
        )
        for key, value in model.module.temporal_attention.state_dict().items()
    ):
        raise RuntimeError("Phase-B temporal attention changed while frozen")
    elapsed = max(time.monotonic() - started, 1e-9)
    report = {
        "steps": step,
        "processed_batches": processed_batches,
        "empty_forecast_batches": empty_forecast_batches,
        "epochs_completed": epoch,
        "validation_mode": "external" if args.no_validation else "inline",
        "dataset_samples": len(dataset),
        "loss": last,
        "per_rank_batch": args.batch_size,
        "global_batch": global_batch,
        "world_size": world_size,
        "accumulation_steps": args.accumulation_steps,
        "updates_per_epoch": updates_per_epoch,
        "dropped_training_samples_per_epoch": len(dataset) - updates_per_epoch * global_batch,
        "workers": args.workers,
        "input_subset_cached": bool(args.cache_subset),
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "clips_per_second_per_rank": (global_samples - initial_global_samples) / world_size / elapsed,
        "steps_per_second_per_rank": (step - initial_step) / elapsed,
        "ddp_sync": True,
        "optimizer": "AdamW",
        "forecast_mean_weight": forecast_mean_weight,
        "weight_decay": args.weight_decay,
        "grad_norm_clip": 1000.0,
        "last_grad_norm": float(grad_norm),
        "source_temporal_grad_norm_min_rank": last_source_temporal_grad,
        "source_query_fusion_grad_norm_min_rank": (
            last_source_query_fusion_grad),
        "source_fusion_location": ("actor_pre_scene_and_decoder" if model.module.forecast_head.source_actor_readout is not None else "decoder_query"),
        "source_actor_readout_grad_norm_min_rank": last_source_actor_grad,
        "source_actor_normalization_grad_norm_min_rank": last_source_actor_normalization_grad,
        "source_motion_gradient_min_rank": last_source_motion_grads,
        "persistent_query_position_grad_norm_min_rank": last_persistent_query_grad,
        "persistent_query_position_layer_grad_norms_min_rank": last_persistent_query_layer_grads,
        "last_lr": float(optimizer.param_groups[0]["lr"]),
        "lr_decay_steps": decay_steps,
        "last_local_forecast_counts": gathered_local_counts,
        "forecast_initialization": "random",
        "forecast_initialization_seed": 20260901,
        "evidence_initialization": "random",
        "temporal_attention_unchanged": True,
        "detector_compression": detector_compression,
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
    }
    if args.forecast_representation == "source_temporal_aligned":
        report.update(aligned_evidence_policy=args.aligned_evidence_policy,
                      source_feature_encoder=args.source_feature_encoder,
                      forecast_normalization=args.forecast_normalization)
    provenance = dict(run_provenance)
    provenance.update(detector_init)
    provenance["detector_compression"] = detector_compression
    provenance["partial_epoch"] = stopped_partial
    provenance["phase_a_checkpoint_sha256"] = _sha256_file(tracker_weights)
    provenance["phase_a_source_provenance"] = (
        payload.get("provenance", {}).get("source_sha")
        if isinstance(payload, dict)
        else None
    )
    report["provenance"] = provenance
    if rank == 0:
        checkpoint = output / "checkpoint.pt"
        report["checkpoint_sha256"] = _sha256_file(checkpoint)
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
