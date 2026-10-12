import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencood.tools.dualcmp.checkpoint_handoff import atomic_json, file_sha256
from opencood.tools.dualcmp.cmp_adapted_forecast_metrics import CONTRACT
from opencood.tools.dualcmp.eval_forecast_mainstream_ddp import _training_bn_hypes
from opencood.tools.dualcmp.watch_checkpoints import FileEvents, ready_checkpoint, write_summary


def test_event_latches_before_wait_and_hash_blocks_incomplete(tmp_path):
    events = tmp_path / "checkpoint_events"
    events.mkdir()
    watcher = FileEvents([tmp_path, events])
    try:
        name = "checkpoint_epoch_001_step_000000001.pt"
        path = tmp_path / name
        path.write_bytes(b"complete immutable checkpoint")
        event = dict(version="dualcmp_external_checkpoint_v1", epoch=1, step=1,
                     checkpoint=name, forecast_evaluation=CONTRACT,
                     size_bytes=path.stat().st_size, sha256=file_sha256(path))
        atomic_json(events / "checkpoint_step_000000001.json", event)
        watcher.wait(.2)
        assert ready_checkpoint(tmp_path, event) == path
        path.write_bytes(b"partial")
        assert ready_checkpoint(tmp_path, event) is None
        path.write_bytes(b"wrong hash".ljust(event["size_bytes"], b" "))
        assert ready_checkpoint(tmp_path, event) is None
        event["checkpoint"] = "../checkpoint.pt"
        with pytest.raises(ValueError):
            ready_checkpoint(tmp_path, event)
    finally:
        watcher.close()


def test_calibration_reconstructs_train_only_and_preserves_input():
    hypes = {"root_dir": "test", "fusion": {"args": {"split_name": "test"}}}
    payload = {"provenance": {"tracking_label_scope": {
        "split": "train", "layout": "dmstrack_train", "window": [["scene", 0, 50]]},
        "partner_delay_ms": 100., "excluded_scene_ids": ["shared_scene"]}}
    result = _training_bn_hypes(hypes, payload, "/datasets", "/train_labels")
    assert result["root_dir"] == "/datasets/train"
    args = result["fusion"]["args"]
    assert args["runtime_forbid_gt"] and args["split_name"] == "train"
    assert args["window"] == payload["provenance"]["tracking_label_scope"]["window"]
    assert args["excluded_scene_ids"] == ["shared_scene"]
    assert hypes["fusion"]["args"]["split_name"] == "test"
    payload["provenance"]["tracking_label_scope"]["split"] = "test"
    with pytest.raises(ValueError):
        _training_bn_hypes(hypes, payload, "/datasets", "/train_labels")


def test_summary_contains_only_primary_metrics(tmp_path):
    from opencood.tools.dualcmp.cmp_adapted_forecast_metrics import PRIMARY_METRICS
    record = dict(epoch=1, step=1, metrics={key: .1 for key in PRIMARY_METRICS})
    write_summary(tmp_path, [record])
    first = (tmp_path / "metrics.csv").read_text().splitlines()[0].split(",")
    assert first == ["evaluation_protocol", "checkpoint_epoch", "checkpoint_step"] + list(PRIMARY_METRICS)


def test_successful_training_without_checkpoint_is_not_validation_success(tmp_path, monkeypatch):
    from opencood.tools.dualcmp import watch_checkpoints as watcher
    root, output = tmp_path / "train", tmp_path / "validation"
    root.mkdir()
    atomic_json(root / "training_finished.json", {"exit_code": 0, "expected_events": []})
    monkeypatch.setattr(watcher, "verify_sync", lambda path: None)
    args = SimpleNamespace(run_root=str(root), output=str(output), sync_proof="unused",
                           train_window=True, ready_fifo=None, remote="", batch_size=2, workers=0)
    with pytest.raises(RuntimeError, match="at least one checkpoint"):
        watcher.consume(args)
    assert not (output / "completed.json").exists()
