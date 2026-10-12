import json
from pathlib import Path
import tempfile
import unittest

import torch

from opencood.tools.dualcmp.checkpoint_handoff import save_external_checkpoint
from opencood.tools.dualcmp.checkpoint_retention import (
    PLAN_CONTRACT, prune_checkpoints,
)
from opencood.tools.dualcmp.cmp_adapted_forecast_metrics import PRIMARY_METRICS


def _publish(root, step):
    return save_external_checkpoint({
        "model": {"weight": torch.tensor([float(step)])},
        "optimizer": {"step": torch.tensor(step)},
        "epoch": step,
        "step": step,
        "provenance": {"forecast_evaluation": {"protocol": "cmp_gtfree_adapter_iou05_mr2_v1"}},
    }, root)


def _plan(manifests, ranked_keep):
    return {
        "contract": PLAN_CONTRACT,
        "primary_metric": "minADE_5s",
        "user_confirmed_keep_top_k": 3,
        "ranked_keep": ranked_keep,
        "evaluated": [{key: item[key] for key in ("checkpoint", "sha256", "step")}
                      for item in manifests],
    }


class CheckpointRetentionTest(unittest.TestCase):
    def test_evaluated_top3_latest_and_pending_are_preserved_idempotently(self):
        self.assertIn("minADE_5s", PRIMARY_METRICS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending = _publish(root, 1)
            evaluated = [_publish(root, step) for step in range(11, 16)]
            ranked_keep = [item["checkpoint"] for item in evaluated[:3]]
            plan = _plan(evaluated, ranked_keep)

            result = prune_checkpoints(root, plan)
            self.assertEqual(result["removed"], [evaluated[3]["checkpoint"]])
            self.assertEqual(result["ranked_keep"], ranked_keep)
            self.assertEqual(result["latest"]["published_checkpoint"], evaluated[-1]["checkpoint"])
            self.assertEqual(result["latest"]["event"], "checkpoint_step_000000015.json")
            self.assertTrue((root / "checkpoint.pt").is_file())
            self.assertTrue((root / pending["checkpoint"]).is_file())
            self.assertTrue((root / evaluated[-1]["checkpoint"]).is_file())
            self.assertFalse((root / evaluated[3]["checkpoint"]).exists())
            for item in [pending] + evaluated:
                self.assertTrue((root / "checkpoint_events" / ("checkpoint_step_%09d.json" % item["step"])).is_file())
            self.assertEqual(json.loads((root / "retention.json").read_text(encoding="utf-8")), result)

            retry = prune_checkpoints(root, plan)
            self.assertEqual(retry["removed"], [])
            self.assertEqual(retry["already_missing"], [evaluated[3]["checkpoint"]])
            self.assertTrue((root / pending["checkpoint"]).is_file())

    def test_hash_mismatch_fails_before_any_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = [_publish(root, step) for step in range(1, 6)]
            candidate = root / manifests[3]["checkpoint"]
            with candidate.open("ab") as stream:
                stream.write(b"changed")
            plan = _plan(manifests, [item["checkpoint"] for item in manifests[:3]])
            with self.assertRaises(RuntimeError):
                prune_checkpoints(root, plan)
            self.assertTrue(all((root / item["checkpoint"]).exists() for item in manifests))
            self.assertFalse((root / "retention.json").exists())

    def test_symlink_path_traversal_and_unexpected_root_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir()
            manifests = [_publish(root, step) for step in range(1, 6)]
            victim = root / manifests[3]["checkpoint"]
            victim.unlink()
            victim.symlink_to(root / manifests[0]["checkpoint"])
            plan = _plan(manifests, [item["checkpoint"] for item in manifests[:3]])
            with self.assertRaises(RuntimeError):
                prune_checkpoints(root, plan)
            self.assertTrue(victim.is_symlink())
            self.assertTrue((root / manifests[0]["checkpoint"]).is_file())

            unsafe = _plan(manifests[:1], [])
            unsafe["evaluated"][0]["checkpoint"] = "../outside.pt"
            with self.assertRaises(ValueError):
                prune_checkpoints(root, unsafe)
            self.assertTrue(victim.is_symlink())

            without_events = Path(directory) / "unexpected"
            without_events.mkdir()
            with self.assertRaises(RuntimeError):
                prune_checkpoints(without_events, _plan([], []))


if __name__ == "__main__":
    unittest.main()
