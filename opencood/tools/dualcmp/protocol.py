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
AUDIT_DENOMINATORS = {"states": 948, "first_frame_states": 38,
                      "birth": 35, "keep": 875, "delete": 26,
                      "duplicate": 948}
AUDIT_THRESHOLDS = {"birth_tp": 29, "keep_same_runtime": 832,
                    "delete_tp": 21, "duplicate_max": 9, "iou": 0.25}
METRIC_THRESHOLDS = {"sAMOTA": 0.70, "AMOTA": 0.70, "MOTA": 0.70,
                     "tracking_recall": 0.80, "tracking_precision": 0.80,
                     "identity_switches_max": 8}


def window_for_receiver(receiver_id):
    matches = [entry for entry in WINDOW if entry["receiver_id"] == int(receiver_id)]
    if len(matches) != 1:
        raise ValueError("frozen protocol has no unique receiver window")
    return matches[0]
