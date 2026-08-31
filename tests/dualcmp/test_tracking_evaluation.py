"""Environment-gated lifecycle-oracle regression for frozen DualCMP windows."""
import os
from pathlib import Path

import pytest
import numpy as np

from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
from opencood.data_utils.v2v4real.datasets.dualcmp_target_builder import load_tracking_label_file, transform_tracking_boxes
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools.dualcmp.ab3dmot_metrics import compute_ab3dmot_metrics
from opencood.tools.dualcmp.online import _gt_conditioned_audit, _target_lifecycle_audit
from opencood.tools.dualcmp.protocol import AUDIT_DENOMINATORS, AUDIT_THRESHOLDS, METRIC_THRESHOLDS, WINDOW


def _hypes():
    hypes = load_yaml("opencood/hypes_yaml/point_pillar_cobevt_multiego_1x_v2v4real.yaml")
    hypes["root_dir"] = str(Path(os.environ["V2V4REAL_ROOT"]) / "test")
    hypes["validate_dir"] = hypes["root_dir"]
    hypes["fusion"] = {"args": {"tracking_label_root": os.environ["DUALCMP_TRACKING_LABEL_ROOT"]}}
    return hypes


def _oracle(hypes):
    result = {}
    for entry in WINDOW:
        labels = load_tracking_label_file(str(Path(os.environ["DUALCMP_TRACKING_LABEL_ROOT"]) /
                                             (entry["scene_id"] + "-" + str(entry["receiver_id"]) + ".txt")))
        frames = [{"frame_index": frame, "runtime_ids": list(labels[frame]),
                   "boxes": [labels[frame][track_id].tolist() for track_id in labels[frame]],
                   "scores": [1.0] * len(labels[frame])}
                  for frame in range(entry["start_frame"], entry["end_frame"] + 1)]
        result[(entry["scene_id"], entry["receiver_id"])] = frames
    return result


@pytest.mark.integration
def test_frozen_lifecycle_oracle_and_empty():
    if "V2V4REAL_ROOT" not in os.environ or "DUALCMP_TRACKING_LABEL_ROOT" not in os.environ:
        pytest.skip("V2V4REAL_ROOT and DUALCMP_TRACKING_LABEL_ROOT are required")
    hypes = _hypes(); oracle = _oracle(hypes)
    values = _target_lifecycle_audit(hypes, oracle, ".")
    assert tuple(values[key] for key in ("states", "first_frame_states", "birth", "keep", "delete")) == tuple(AUDIT_DENOMINATORS[key] for key in ("states", "first_frame_states", "birth", "keep", "delete"))
    assert (values["birth_tp"], values["keep_tp"], values["delete_tp"]) == (35, 875, 26)
    assert values["birth_tp"] >= AUDIT_THRESHOLDS["birth_tp"]
    assert values["keep_tp"] >= AUDIT_THRESHOLDS["keep_same_runtime"]
    assert values["delete_tp"] >= AUDIT_THRESHOLDS["delete_tp"]
    assert METRIC_THRESHOLDS["identity_switches_max"] == 8
    metrics = compute_ab3dmot_metrics(oracle, repository_root=".", ground_truth_root=os.environ["DUALCMP_TRACKING_LABEL_ROOT"])
    assert all(metrics[key] == 1.0 for key in ("sAMOTA", "AMOTA", "MOTA", "tracking_recall", "tracking_precision"))
    assert metrics["identity_switches"] == 0
    assert _gt_conditioned_audit(".", oracle)["duplicate"] == "0/948"
    empty = {key: [{"frame_index": frame["frame_index"], "runtime_ids": [], "boxes": [], "scores": []} for frame in frames]
             for key, frames in oracle.items()}
    values = _target_lifecycle_audit(hypes, empty, ".")
    assert (values["birth_tp"], values["keep_tp"], values["delete_tp"]) == (0, 0, 0)


@pytest.mark.integration
def test_label_targets_cover_frozen_windows_and_receiver_one_roundtrip():
    if "V2V4REAL_ROOT" not in os.environ or "DUALCMP_TRACKING_LABEL_ROOT" not in os.environ:
        pytest.skip("V2V4REAL_ROOT and DUALCMP_TRACKING_LABEL_ROOT are required")
    hypes = _hypes()
    hypes["fusion"] = {"args": {"stage": "tracking", "tracking_frames": 3,
        "runtime_only": False, "runtime_forbid_gt": True, "receiver_ids": [0, 1],
        "max_actors": 128, "max_cav": 2, "tracking_label_root": os.environ["DUALCMP_TRACKING_LABEL_ROOT"],
        "window": list(WINDOW)}}
    dataset = DualCMPSequenceDataset(hypes, train=True)
    assert len(dataset) == 44
    states = 0
    for entry in WINDOW:
        keys = [key for key in dataset.index.receiver_frame_keys(entry["scene_id"], str(entry["receiver_id"]))
                if entry["start_frame"] <= int(key) <= entry["end_frame"]]
        states += sum(int(dataset._target_frame(entry["scene_id"], str(entry["receiver_id"]), key, None, False)["current_mask"].sum()) for key in keys)
    assert states == 948
    entry, key = WINDOW[1], "000273"
    reference_pose = dataset._pose(dataset.index.receiver_record(entry["scene_id"], "0", key))
    receiver_pose = dataset._pose(dataset.index.receiver_record(entry["scene_id"], "1", key))
    labels = load_tracking_label_file(str(Path(os.environ["DUALCMP_TRACKING_LABEL_ROOT"]) / (entry["scene_id"] + "-1.txt")))
    reference = np.stack([labels[int(key)][track_id] for track_id in sorted(labels[int(key)])])
    local = transform_tracking_boxes(reference, reference_pose, receiver_pose)
    roundtrip = transform_tracking_boxes(local, receiver_pose, reference_pose)
    assert np.max(np.abs(roundtrip - reference)) < 0.005
    sample = dataset[0]
    from opencood.data_utils.v2v4real.datasets.dualcmp_schema import assert_runtime_is_supervision_free
    assert_runtime_is_supervision_free(sample["runtime_frames"])
