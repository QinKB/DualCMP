"""Fail closed on source/state drift while preserving every training tensor."""
import hashlib
from pathlib import Path

import pytest
import torch

from opencood.tools.dualcmp.migrate_forecast_evaluation_checkpoint import migrate_payload, state_digest


def test_preserved_original_manifest_and_training_state():
    snapshot = Path("/data/DualCMP-runs/source-before-mainstream-20261008")
    if not snapshot.exists():
        pytest.skip("the intentionally preserved migration snapshot is not present")
    payload = {"model": {"weight": torch.arange(8.)}, "optimizer": {"state": {0: {"exp_avg": torch.ones(8)}}},
        "scheduler": {"last_epoch": 12}, "epoch": 3, "step": 12, "global_samples": 48,
        "provenance": {"source_state": "dirty-smoke", "base_sha": "9fd656a84196fdb0d8ca625655b73be2b2e241d2",
            "source_tree_sha256": "4786248c86273dc879d3ed9194df351d8a337ba03f3e14762a94fbba8c0c8cf8",
            "protocol_sha256": "1f61b51a39dff945cf3550c53d4550b7107c89a809d1a49b014e98009c920b56",
            "forecast_representation": "source_temporal_aligned", "forecast_metric_version": "exact_10hz_v1"}}
    migrated, checksum = migrate_payload(payload, snapshot)
    assert migrated["model"] is payload["model"] and migrated["optimizer"] is payload["optimizer"]
    assert checksum == state_digest({key: value for key, value in payload.items() if key != "provenance"})
    from opencood.tools.dualcmp.cmp_adapted_forecast_metrics import PROTOCOL
    assert migrated["provenance"]["forecast_evaluation"]["protocol"] == PROTOCOL
    assert payload["provenance"]["source_tree_sha256"] == "4786248c86273dc879d3ed9194df351d8a337ba03f3e14762a94fbba8c0c8cf8"
    bad = dict(payload, provenance=dict(payload["provenance"], source_tree_sha256="0" * 64))
    with pytest.raises(RuntimeError, match="does not prove"):
        migrate_payload(bad, snapshot)
