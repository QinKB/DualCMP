"""An evaluator transition must not change any learned or optimizer state."""
import ast
import hashlib
from pathlib import Path

import pytest
import torch

from opencood.tools.dualcmp.migrate_forecast_evaluation_checkpoint import (
    _cmp_trainer_ast, migrate_payload, state_digest,
)
from opencood.tools.dualcmp.cmp_adapted_forecast_metrics import PROTOCOL


def test_cmp_transition_preserves_training_state_and_original():
    snapshot = Path("runs/cmp-adapted-eval-20261008-r1")
    if not snapshot.exists():
        pytest.skip("intentionally preserved original source snapshot is absent")
    trainer = snapshot / "opencood/tools/dualcmp/train_forecast_ddp.py"
    tree = ast.parse(trainer.read_text())
    paths = next(ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
        and target.id == "PHASE_B_SOURCES" for target in node.targets))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.encode() + b"\0" + (snapshot / path).read_bytes())
    payload = {"model": {"weight": torch.arange(8.)},
        "optimizer": {"state": {0: {"exp_avg": torch.ones(8)}}},
        "scheduler": {"last_epoch": 12}, "epoch": 10, "step": 25551,
        "global_samples": 102440,
        "provenance": {"source_state": "dirty-smoke",
            "base_sha": "9fd656a84196fdb0d8ca625655b73be2b2e241d2",
            "source_tree_sha256": digest.hexdigest(),
            "protocol_sha256": hashlib.sha256((snapshot /
                "opencood/tools/dualcmp/protocol.py").read_bytes()).hexdigest(),
            "forecast_representation": "source_temporal_aligned",
            "forecast_evaluation": {"protocol": "uniad_distance_v1"}}}
    migrated, checksum = migrate_payload(payload, snapshot)
    assert checksum == state_digest({key: value for key, value in payload.items()
                                    if key != "provenance"})
    for key in payload:
        if key != "provenance":
            assert migrated[key] is payload[key]
    assert migrated["provenance"]["forecast_evaluation"]["protocol"] == PROTOCOL
    assert payload["provenance"]["forecast_evaluation"]["protocol"] == "uniad_distance_v1"
    bad = dict(payload, provenance=dict(payload["provenance"], source_tree_sha256="0" * 64))
    with pytest.raises(RuntimeError, match="does not prove"):
        migrate_payload(bad, snapshot)


def test_trainer_normalization_permits_no_learning_changes():
    before = 'from opencood.tools.dualcmp.mainstream_forecast_metrics import CONTRACT\nPHASE_B_SOURCES = ("a",)\nlearning_rate = 0.001\n'
    after = 'from opencood.tools.dualcmp.cmp_adapted_forecast_metrics import CONTRACT\nPHASE_B_SOURCES = ("a", "opencood/tools/dualcmp/cmp_adapted_forecast_metrics.py")\nlearning_rate = 0.001\n'
    assert _cmp_trainer_ast(before) == _cmp_trainer_ast(after)
    assert _cmp_trainer_ast(before) != _cmp_trainer_ast(after.replace("0.001", "0.002"))
