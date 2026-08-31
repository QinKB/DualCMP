"""Environment-gated lifecycle-oracle regression for frozen DualCMP windows."""
import os
from pathlib import Path

import pytest

from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools.dualcmp.online import _target_lifecycle_audit
from opencood.tools.dualcmp.protocol import WINDOW
from opencood.tools.dualcmp.protocol import AUDIT_THRESHOLDS, METRIC_THRESHOLDS


def _hypes():
    hypes = load_yaml("opencood/hypes_yaml/point_pillar_cobevt_multiego_1x_v2v4real.yaml")
    hypes["root_dir"] = str(Path(os.environ["V2V4REAL_ROOT"]) / "test")
    hypes["validate_dir"] = hypes["root_dir"]
    return hypes


def _oracle(hypes):
    result = {}
    for entry in WINDOW:
        local = dict(hypes)
        local["fusion"] = {"args": {"stage": "tracking", "tracking_frames": 3,
            "runtime_only": False, "runtime_forbid_gt": True,
            "receiver_ids": [entry["receiver_id"]], "max_actors": 128, "max_cav": 2}}
        dataset = DualCMPSequenceDataset(local, train=True)
        keys = [key for key in dataset.index.receiver_frame_keys(entry["scene_id"], str(entry["receiver_id"]))
                if entry["start_frame"] <= int(key) <= entry["end_frame"]]
        frames = []
        for key in keys:
            target = dataset._target_frame(entry["scene_id"], str(entry["receiver_id"]), key, None, False)
            mask = target["actor_ids"] >= 0
            frames.append({"frame_index": int(key), "runtime_ids": target["actor_ids"][mask].tolist(),
                           "boxes": target["current_boxes"][mask].tolist(),
                           "scores": [1.0] * int(mask.sum())})
        result[(entry["scene_id"], entry["receiver_id"])] = frames
    return result


@pytest.mark.integration
def test_frozen_lifecycle_oracle_and_empty():
    if "V2V4REAL_ROOT" not in os.environ:
        pytest.skip("V2V4REAL_ROOT is required")
    hypes = _hypes(); oracle = _oracle(hypes)
    values = _target_lifecycle_audit(hypes, oracle, ".")
    assert (values["birth"], values["keep"], values["delete"]) == (29, 732, 22)
    assert (values["birth_tp"], values["keep_tp"], values["delete_tp"]) == (29, 732, 22)
    assert values["birth_tp"] >= AUDIT_THRESHOLDS["birth_tp"]
    assert values["keep_tp"] >= AUDIT_THRESHOLDS["keep_same_runtime"]
    assert values["delete_tp"] >= AUDIT_THRESHOLDS["delete_tp"]
    assert METRIC_THRESHOLDS["identity_switches_max"] == 8
    empty = {key: [{"frame_index": frame["frame_index"], "runtime_ids": [], "boxes": [], "scores": []} for frame in frames]
             for key, frames in oracle.items()}
    values = _target_lifecycle_audit(hypes, empty, ".")
    assert (values["birth_tp"], values["keep_tp"], values["delete_tp"]) == (0, 0, 0)
