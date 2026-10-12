import ast
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

import opencood.tools.dualcmp.checkpoint_handoff as checkpoint_handoff
from opencood.tools.dualcmp.checkpoint_handoff import (
    file_sha256, prepare_training_output, save_external_checkpoint,
)


ROOT = Path(__file__).resolve().parents[2]


def _checkpoint(epoch, step):
    return {
        "model": {"weight": torch.tensor([float(step), 2.0])},
        "optimizer": {"step": torch.tensor(step)},
        "epoch": epoch,
        "step": step,
        "provenance": {
            "source_tree_sha256": "tree-digest",
            "base_sha": "base-commit",
            "forecast_evaluation": {"protocol": "cmp_iou05_mr2_v1"},
            "partial_epoch": False,
        },
    }


class CheckpointHandoffTest(unittest.TestCase):
    def test_real_checkpoint_blob_event_and_latest_hardlink(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            old_latest = output / "checkpoint.pt"
            old_latest.write_bytes(b"previous checkpoint")

            atomic_json = checkpoint_handoff.atomic_json

            def observe_completed_blob(path, payload):
                blob = output / payload["checkpoint"]
                self.assertFalse(Path(path).exists())
                self.assertTrue(blob.is_file())
                self.assertEqual(payload["size_bytes"], blob.stat().st_size)
                self.assertEqual(payload["sha256"], file_sha256(blob))
                torch.load(blob, map_location="cpu")
                atomic_json(path, payload)

            checkpoint_handoff.atomic_json = observe_completed_blob
            try:
                first = save_external_checkpoint(_checkpoint(1, 3), output)
            finally:
                checkpoint_handoff.atomic_json = atomic_json
            first_blob = output / first["checkpoint"]
            first_event = output / "checkpoint_events" / "checkpoint_step_000000003.json"
            latest = output / "checkpoint.pt"
            self.assertTrue(first_event.is_file())
            self.assertEqual(first["version"], "dualcmp_external_checkpoint_v1")
            self.assertEqual(first["checkpoint"], first_blob.name)
            self.assertEqual(first["forecast_evaluation"], {"protocol": "cmp_iou05_mr2_v1"})
            self.assertEqual(first["size_bytes"], first_blob.stat().st_size)
            self.assertEqual(first["sha256"], file_sha256(first_blob))
            self.assertEqual(os.stat(first_blob).st_ino, os.stat(latest).st_ino)
            self.assertEqual(json.loads(first_event.read_text(encoding="utf-8")), first)
            loaded = torch.load(first_blob, map_location="cpu")
            self.assertTrue(torch.equal(loaded["model"]["weight"], torch.tensor([3.0, 2.0])))

            first_event_bytes = first_event.read_bytes()
            second = save_external_checkpoint(_checkpoint(2, 6), output)
            second_blob = output / second["checkpoint"]
            self.assertTrue(first_blob.is_file() and first_event.is_file())
            self.assertEqual(first_event.read_bytes(), first_event_bytes)
            self.assertEqual(os.stat(second_blob).st_ino, os.stat(latest).st_ino)
            self.assertNotEqual(os.stat(first_blob).st_ino, os.stat(latest).st_ino)
            with self.assertRaises(FileExistsError):
                save_external_checkpoint(_checkpoint(2, 6), output)
            self.assertEqual(file_sha256(first_blob), first["sha256"])
            self.assertEqual(file_sha256(second_blob), second["sha256"])

    def test_checkpoint_path_fields_cannot_escape_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(ValueError):
                save_external_checkpoint(_checkpoint("../../outside", 1), output)
            self.assertEqual([path.name for path in output.iterdir()], [".checkpoint.lock"])

    def test_fresh_external_output_only_accepts_an_empty_event_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-run"
            prepare_training_output(output, external=True)
            self.assertEqual([path.name for path in output.iterdir()], ["checkpoint_events"])
            prepare_training_output(output, external=True)

            (output / "metrics.csv").write_text("old run", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_training_output(output, external=True)

            symlink_root = Path(directory) / "linked-run"
            symlink_root.mkdir()
            event_target = Path(directory) / "events-target"
            event_target.mkdir()
            (symlink_root / "checkpoint_events").symlink_to(event_target, target_is_directory=True)
            with self.assertRaises(FileExistsError):
                prepare_training_output(symlink_root, external=True)

            inline = Path(directory) / "inline-run"
            prepare_training_output(inline)
            with self.assertRaises(FileExistsError):
                prepare_training_output(inline)
            (inline / "checkpoint.pt").write_bytes(b"resume state")
            self.assertEqual(prepare_training_output(inline, resume=True), inline)

    def test_trainers_publish_externally_and_guard_inline_evaluation(self):
        cases = (
            ("train_forecast_ddp.py", ("evaluate_forecast_training_ddp",)),
            ("train_joint_ddp.py", (
                "evaluate_forecast_training_ddp", "evaluate_tracking_model_ddp",
                "evaluate_tracking_test_model_ddp")),
        )
        for filename, evaluator_names in cases:
            path = ROOT / "opencood" / "tools" / "dualcmp" / filename
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            guarded = set()
            published = False
            for node in ast.walk(tree):
                if not isinstance(node, ast.If):
                    continue
                calls = {child.id for statement in node.body for child in ast.walk(statement)
                         if isinstance(child, ast.Name)}
                test = ast.dump(node.test)
                if "no_validation" in test and "save_external_checkpoint" in calls:
                    published = True
                if "no_validation" in test and "Not()" in test:
                    guarded.update(calls.intersection(evaluator_names))
            self.assertTrue(published, filename)
            self.assertEqual(guarded, set(evaluator_names), filename)
            self.assertIn("checkpoint_handoff.py", source, filename)
            if filename == "train_joint_ddp.py":
                for field in ("batch_contract", "per_rank_batch", "world_size",
                              "accumulation_steps", "global_batch"):
                    self.assertIn(field, source)


if __name__ == "__main__":
    unittest.main()
