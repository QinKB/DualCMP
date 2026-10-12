"""Frozen DualCMP tracking-window protocol shared by train and evaluation."""
from __future__ import annotations

import math


WINDOW = (
    {"scene_id": "testoutput_CAV_data_2022-03-17-11-02-23_2", "receiver_id": 0,
     "start_frame": 237, "end_frame": 260},
    {"scene_id": "testoutput_CAV_data_2022-03-17-11-02-23_2", "receiver_id": 1,
     "start_frame": 273, "end_frame": 296},
)
WINDOW_FRAMES = 24
TRACKING_FRAMES = 3
DETECTOR_COMPRESSION = 32
CLIPS_PER_RECEIVER = WINDOW_FRAMES - TRACKING_FRAMES + 1
TOTAL_CLIPS = CLIPS_PER_RECEIVER * len(WINDOW)
TRAIN_LABEL_WINDOW = (
    {"scene_id": "testoutput_CAV_data_2022-03-17-10-43-13", "receiver_id": 0,
     "start_frame": 2, "end_frame": 25},
    {"scene_id": "testoutput_CAV_data_2022-03-17-10-43-13", "receiver_id": 1,
     "start_frame": 2, "end_frame": 25},
)
# Capture-time alignment acceptance window.  Keep this independent from the
# historical native-label smoke window: these four clips contain twelve
# complete, meaningfully moving trajectories and are used only for matched
# delayed-evidence learning/evaluation.
DELAY_FORECAST_WINDOW = (
    {"scene_id": "testoutput_CAV_data_2022-03-17-10-43-13", "receiver_id": 0,
     "start_frame": 171, "end_frame": 172},
    {"scene_id": "testoutput_CAV_data_2022-03-17-10-43-13", "receiver_id": 1,
     "start_frame": 75, "end_frame": 76},
)
DELAY_FORECAST_SAMPLES = 4
DELAY_FORECAST_COMPLETE_TARGETS = 12
# Phase-B forecast target-only protocol: eleven capture-native observations
# (ten history plus current) and fifty moving-receiver future states.
FORECAST_WINDOW = (
    {"scene_id": "testoutput_CAV_data_2022-03-17-11-02-23_2", "receiver_id": 0,
     "start_frame": 237, "end_frame": 244},
    {"scene_id": "testoutput_CAV_data_2022-03-17-11-02-23_2", "receiver_id": 1,
     "start_frame": 237, "end_frame": 244},
)
FORECAST_HISTORY_FRAMES = 11
FORECAST_FUTURE_FRAMES = 50
FORECAST_SAMPLES = 16
FORECAST_COMPLETE_TARGETS = 94
AUDIT_DENOMINATORS = {"states": 948, "first_frame_states": 38,
                      "birth": 35, "keep": 875, "delete": 26,
                      "duplicate": 948}
AUDIT_THRESHOLDS = {"birth_tp": 29, "keep_same_runtime": 832,
                    "delete_tp": 21, "duplicate_max": 9, "iou": 0.25}
METRIC_THRESHOLDS = {"sAMOTA": 0.70, "AMOTA": 0.70, "MOTA": 0.70,
                     "tracking_recall": 0.80, "tracking_precision": 0.80,
                     "identity_switches_max": 8}
FORECAST_REPRESENTATIONS = ("history", "fused", "source")


def normalize_forecast_representation(value):
    value = str(value)
    if value not in FORECAST_REPRESENTATIONS:
        raise ValueError(
            "forecast representation must be one of %s"
            % (FORECAST_REPRESENTATIONS,))
    return value


def resolve_forecast_ablation(forecast_representation="source",
                              disable_capture_alignment=False,
                              disable_age_encoding=False,
                              disable_capture_time_evidence=False,
                              mean_source_aggregation=False):
    """Normalize CLI ablation switches; the legacy switch disables both parts."""
    representation = normalize_forecast_representation(
        forecast_representation)
    if representation != "source":
        if mean_source_aggregation:
            raise ValueError(
                "mean source aggregation requires source representation")
        return {
            "forecast_representation": representation,
            "capture_alignment": False,
            "age_encoding": False,
            "mean_source_aggregation": False,
        }
    legacy = bool(disable_capture_time_evidence)
    return {
        "forecast_representation": representation,
        "capture_alignment": not (legacy or bool(disable_capture_alignment)),
        "age_encoding": not (legacy or bool(disable_age_encoding)),
        "mean_source_aggregation": bool(mean_source_aggregation),
    }


def checkpoint_forecast_contract(provenance):
    """Return representation/alignment/age, including pre-ablation checkpoints."""
    if not isinstance(provenance, dict):
        raise RuntimeError("checkpoint provenance is required")
    representation = normalize_forecast_representation(
        provenance.get("forecast_representation", "source"))
    if representation != "source":
        return {
            "forecast_representation": representation,
            "capture_alignment": False,
            "age_encoding": False,
            "mean_source_aggregation": False,
        }
    legacy = bool(provenance.get("capture_time_evidence", True))
    return {
        "forecast_representation": representation,
        "capture_alignment": bool(
            provenance.get("capture_alignment", legacy)),
        "age_encoding": bool(provenance.get("age_encoding", legacy)),
        "mean_source_aggregation": bool(
            provenance.get("mean_source_aggregation", False)),
    }


def validate_forecast_contract(provenance, forecast_representation,
                               capture_alignment, age_encoding,
                               context="checkpoint",
                               mean_source_aggregation=False):
    checkpoint = checkpoint_forecast_contract(provenance)
    requested = {
        "forecast_representation": normalize_forecast_representation(
            forecast_representation),
        "capture_alignment": bool(capture_alignment),
        "age_encoding": bool(age_encoding),
        "mean_source_aggregation": bool(mean_source_aggregation),
    }
    if checkpoint != requested:
        raise RuntimeError(
            "%s forecast ablation must match checkpoint provenance" % context)
    return checkpoint


def normalize_train_delays_ms(delays_ms=None, fixed_delay_ms=0.0):
    """Normalize a fixed or discrete mixed-delay training contract."""
    values = (fixed_delay_ms,) if delays_ms is None else tuple(delays_ms)
    if not values:
        raise ValueError("training delays must not be empty")
    normalized = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in normalized):
        raise ValueError("training delays must be finite and non-negative")
    if len(set(normalized)) != len(normalized):
        raise ValueError("training delays must be unique")
    return normalized


def checkpoint_train_delays_ms(provenance):
    """Read the training-delay distribution, including fixed-delay checkpoints."""
    if not isinstance(provenance, dict):
        raise RuntimeError("checkpoint provenance is required")
    values = provenance.get("train_delays_ms")
    if values is None:
        if "partner_delay_ms" not in provenance:
            raise RuntimeError("checkpoint training-delay provenance is required")
        values = (provenance["partner_delay_ms"],)
    try:
        return normalize_train_delays_ms(values)
    except (TypeError, ValueError) as error:
        raise RuntimeError("checkpoint training-delay provenance is invalid") from error


def validate_training_delay_contract(provenance, train_delays_ms, delay_seed,
                                     capture_time_evidence=None,
                                     context="checkpoint",
                                     forecast_representation=None,
                                     capture_alignment=None,
                                     age_encoding=None,
                                     mean_source_aggregation=False):
    """Require resume or Phase-C training to reuse the checkpoint distribution."""
    requested = normalize_train_delays_ms(train_delays_ms)
    checkpoint = checkpoint_train_delays_ms(provenance)
    if checkpoint != requested:
        raise RuntimeError(
            "%s training delays must match checkpoint provenance" % context)
    if (len(requested) > 1
            and provenance.get("delay_seed") != int(delay_seed)):
        raise RuntimeError(
            "%s delay seed must match checkpoint provenance" % context)
    if capture_time_evidence is not None:
        if capture_alignment is None:
            capture_alignment = capture_time_evidence
        if age_encoding is None:
            age_encoding = capture_time_evidence
    if forecast_representation is not None:
        validate_forecast_contract(
            provenance, forecast_representation, capture_alignment,
            age_encoding, context=context,
            mean_source_aggregation=mean_source_aggregation)
    elif capture_alignment is not None or age_encoding is not None:
        current = checkpoint_forecast_contract(provenance)
        if (capture_alignment is not None
                and current["capture_alignment"] != bool(capture_alignment)):
            raise RuntimeError(
                "%s evidence alignment must match checkpoint provenance" % context)
        if (age_encoding is not None
                and current["age_encoding"] != bool(age_encoding)):
            raise RuntimeError(
                "%s age encoding must match checkpoint provenance" % context)
    return checkpoint


def validate_evaluation_delay_contract(
        provenance, partner_delay_ms, capture_time_evidence=None,
        context="checkpoint", forecast_representation=None,
        capture_alignment=None, age_encoding=None,
        mean_source_aggregation=False):
    """Validate an evaluation delay without pretending it was a train setting."""
    checkpoint_delays = checkpoint_train_delays_ms(provenance)
    delay = float(partner_delay_ms)
    if not math.isfinite(delay) or delay < 0.0:
        raise ValueError("evaluation partner delay must be finite and non-negative")
    if capture_time_evidence is not None:
        if capture_alignment is None:
            capture_alignment = capture_time_evidence
        if age_encoding is None:
            age_encoding = capture_time_evidence
    if forecast_representation is not None:
        validate_forecast_contract(
            provenance, forecast_representation, capture_alignment,
            age_encoding, context=context,
            mean_source_aggregation=mean_source_aggregation)
    elif capture_alignment is not None or age_encoding is not None:
        current = checkpoint_forecast_contract(provenance)
        if (capture_alignment is not None
                and current["capture_alignment"] != bool(capture_alignment)):
            raise RuntimeError(
                "%s evidence alignment must match checkpoint provenance" % context)
        if (age_encoding is not None
                and current["age_encoding"] != bool(age_encoding)):
            raise RuntimeError(
                "%s age encoding must match checkpoint provenance" % context)
    return checkpoint_delays


def checkpoint_detector_compression(payload):
    """Return the detector compression frozen in a staged checkpoint."""
    provenance = payload.get("provenance", {}) if isinstance(payload, dict) else {}
    value = provenance.get("detector_compression")
    if value is None:
        state = payload.get("model", payload.get("model_state_dict", payload))
        if not isinstance(state, dict):
            raise RuntimeError("checkpoint has no detector state dictionary")
        keys = (
            "detector.naive_compressor.encoder.0.weight",
            "module.detector.naive_compressor.encoder.0.weight",
        )
        tensor = next((state[key] for key in keys if key in state), None)
        if tensor is None or tensor.ndim != 4 or tensor.shape[1] != 256:
            raise RuntimeError("checkpoint does not declare an inferable detector compression")
        value = 256 // int(tensor.shape[0])
    value = int(value)
    if value < 1 or 256 % value:
        raise RuntimeError("checkpoint detector compression is invalid")
    return value


def window_for_receiver(receiver_id):
    matches = [entry for entry in WINDOW if entry["receiver_id"] == int(receiver_id)]
    if len(matches) != 1:
        raise ValueError("frozen protocol has no unique receiver window")
    return matches[0]
