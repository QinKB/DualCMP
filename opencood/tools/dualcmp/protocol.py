"""Frozen DualCMP tracking-window protocol shared by train and evaluation."""
from __future__ import annotations


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
