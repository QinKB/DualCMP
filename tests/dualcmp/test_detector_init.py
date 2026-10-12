import argparse
import os

import pytest
import torch

from opencood.data_utils.v2v4real.post_processor import build_postprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.dualcmp.model import DualCmpClipModel
from opencood.models.point_pillar_fax import PointPillarFax
from opencood.tools.dualcmp.detector_config import load_checkpoint_hypes
from opencood.tools.dualcmp.online import _checkpoint_provenance
from opencood.tools.dualcmp.train_tracking_ddp import _dirty_provenance, _hypes, _load_official_detector
from opencood.tools.dualcmp import train_tracking_ddp


def test_v2v4real_detector_config_is_official_32x_async_false():
    hypes = load_yaml("opencood/hypes_yaml/point_pillar_v2v4real.yaml")
    assert hypes["model"]["core_method"] == "point_pillar_fax"
    assert hypes["model"]["args"]["compression"] == 32
    assert hypes["wild_setting"]["async"] is False


def test_new_phase_a_rejects_non32_compression(monkeypatch):
    monkeypatch.setenv("V2V4REAL_ROOT", "data/V2V4Real")
    monkeypatch.setenv("DUALCMP_TRACKING_LABEL_ROOT", "labels")
    monkeypatch.setenv("DUALCMP_DETECTOR_WEIGHTS", "weights.pt")
    args = argparse.Namespace(full_data=False, detector_compression=256, data_split="train",
                              resume=None)
    with pytest.raises(RuntimeError, match="compression 32"):
        _hypes("opencood/hypes_yaml/point_pillar_v2v4real.yaml", args)


def test_official_detector_strict_load(monkeypatch):
    weights = os.environ.get("DUALCMP_DETECTOR_WEIGHTS")
    if not weights:
        pytest.skip("DUALCMP_DETECTOR_WEIGHTS is required for official-weight validation")
    hypes = load_yaml("opencood/hypes_yaml/point_pillar_v2v4real.yaml")
    postprocessor = build_postprocessor(hypes["postprocess"], train=False)
    model = DualCmpClipModel(hypes["model"]["args"], postprocessor.generate_anchor_box(),
                             hypes["preprocess"]["cav_lidar_range"])
    assert _load_official_detector(model.detector, weights) == 243
    assert tuple(model.detector.naive_compressor.encoder[0].weight.shape) == (8, 256, 3, 3)


def test_official_detector_rejects_wrong_weight(tmp_path):
    hypes = load_yaml("opencood/hypes_yaml/point_pillar_v2v4real.yaml")
    bad = tmp_path / "wrong.pt"
    bad.write_bytes(b"wrong detector")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        _load_official_detector(PointPillarFax(hypes["model"]["args"]), bad)


def test_partial_new_detector_provenance_does_not_fall_back_to_legacy():
    with pytest.raises(RuntimeError, match="incomplete"):
        load_checkpoint_hypes("opencood/hypes_yaml/point_pillar_v2v4real.yaml",
                              {"provenance": {"detector_source": "wrong"}})


@pytest.mark.parametrize("full_data", [False, True])
def test_phase_a_rejects_full_model_initialization(monkeypatch, full_data):
    monkeypatch.setenv("DUALCMP_PHASE_A_INIT_WEIGHTS", "trained_tracker.pt")
    args = argparse.Namespace(full_data=full_data, detector_compression=32,
                              resume=None)
    with pytest.raises(RuntimeError, match="rejects DUALCMP_PHASE_A_INIT_WEIGHTS"):
        _hypes("opencood/hypes_yaml/point_pillar_v2v4real.yaml", args)


def test_tracking_dirty_checkpoint_uses_tracking_manifest(tmp_path):
    provenance = _dirty_provenance(argparse.Namespace(seed=0), 2, ["a", "b"])
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": {}, "provenance": provenance}, checkpoint)
    assert _checkpoint_provenance(checkpoint, allow_dirty_smoke=True)["stage"] == "tracking"


def test_epoch_buffer_sync_broadcasts_unwrapped_buffers(monkeypatch):
    module = torch.nn.BatchNorm1d(2)
    calls = []
    monkeypatch.setattr(train_tracking_ddp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(train_tracking_ddp.dist, "broadcast",
                        lambda buffer, src: calls.append((buffer, src)))
    train_tracking_ddp._sync_buffers(type("DDP", (), {"module": module})())
    assert len(calls) == len(list(module.buffers()))
    assert all(src == 0 for _, src in calls)
