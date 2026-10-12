"""Explicit metadata-only migration; preserve original checkpoint and state."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch

from opencood.tools.dualcmp.mainstream_forecast_metrics import CONTRACT
from opencood.tools.dualcmp.cmp_adapted_forecast_metrics import CONTRACT as CMP_CONTRACT

# These exact trainer changes were reviewed as evaluation/logging-only.
APPROVED_TRAINERS = {
    "opencood/tools/dualcmp/train_forecast_ddp.py": "d0f4684b444341e219a884176295c91395b1cd31c02864a0f8e47d76a294f5d5",
    "opencood/tools/dualcmp/train_joint_ddp.py": "3e89bedf4d5d243e5985bd12de5217a08be263bc9aaa0668cb02ee1f72cef4ef",
}
NEW_EVALUATION_FILES = {"opencood/tools/dualcmp/mainstream_forecast_metrics.py",
                        "opencood/tools/dualcmp/eval_forecast_mainstream_ddp.py"}
CMP_METRIC_FILE = "opencood/tools/dualcmp/cmp_adapted_forecast_metrics.py"
CMP_APPROVED_TRAINERS = {
    "opencood/tools/dualcmp/train_forecast_ddp.py": "70a7acf0706e3bca2908c993ac136da4c30340b94d1200197ba80f3163434519",
    "opencood/tools/dualcmp/train_joint_ddp.py": "ed273cb05fb34b026e2268d5164dfbeca6de2a7f85f61cc7418ef75832a80f13",
}
CMP_APPROVED_EVALUATION = {
    "opencood/tools/dualcmp/eval_forecast_mainstream_ddp.py": "700dfbc2b2735e0e36e8d0247012cc95165bd466de8462b0015dbfc3d34aafcd",
    CMP_METRIC_FILE: "ced926479e1dabe95931335270f1c815e25c2e3c8c15d309b1b681617347caa9",
}


def state_digest(value):
    """Hash tensors and scalar training state independently of serialization."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        elif isinstance(item, (str, int, float, bool, type(None))):
            digest.update(repr(item).encode())
        else:
            raise TypeError("unsupported checkpoint state value: " + type(item).__name__)
    visit(value)
    return digest.hexdigest()


def migrate_payload(payload, original_source):
    repository = Path(__file__).resolve().parents[3]
    original_source = Path(original_source).resolve()
    if original_source == repository or not original_source.is_dir():
        raise ValueError("migration needs a separate preserved original source snapshot")
    previous = payload["provenance"]
    if previous.get("source_state") != "dirty-smoke" or previous.get("forecast_representation") != "source_temporal_aligned":
        raise ValueError("this migration accepts only the preserved aligned native checkpoints")
    if previous.get("forecast_evaluation", {}).get("protocol") == "uniad_distance_v1":
        return _migrate_cmp_payload(payload, original_source, repository)
    joint = "init_checkpoint_sha256" in previous
    module, symbol = (("train_joint_ddp.py", "SOURCES") if joint else ("train_forecast_ddp.py", "PHASE_B_SOURCES"))
    relative = "opencood/tools/dualcmp/" + module
    tree = ast.parse((original_source / relative).read_text())
    paths = next(ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == symbol for target in node.targets))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.encode() + b"\0")
        original = (original_source / path).read_bytes()
        digest.update(original)
        current = (repository / path).read_bytes()
        if path in APPROVED_TRAINERS:
            if hashlib.sha256(current).hexdigest() not in (
                    APPROVED_TRAINERS[path], CMP_APPROVED_TRAINERS[path]):
                raise RuntimeError("trainer has unreviewed changes: " + path)
        elif current != original:
            raise RuntimeError("model/data/loss/runtime code changed: " + path)
    if digest.hexdigest() != previous.get("source_tree_sha256"):
        raise RuntimeError("original source snapshot does not prove checkpoint provenance")
    if joint:
        from opencood.tools.dualcmp.train_joint_ddp import SOURCES as current_paths, _provenance
    else:
        from opencood.tools.dualcmp.train_forecast_ddp import PHASE_B_SOURCES as current_paths, _provenance
    if set(current_paths) not in (set(paths) | NEW_EVALUATION_FILES,
                                 set(paths) | NEW_EVALUATION_FILES | {CMP_METRIC_FILE}):
        raise RuntimeError("source manifest changed beyond approved evaluation files")
    if CMP_METRIC_FILE in current_paths:
        _check_cmp_evaluation_sources(repository)
    current = _provenance()
    if current["protocol_sha256"] != previous["protocol_sha256"]:
        raise RuntimeError("training temporal protocol changed")
    state = {key: value for key, value in payload.items() if key != "provenance"}
    checksum = state_digest(state)
    result = dict(payload)
    result["provenance"] = dict(previous, **current)
    target_contract = CMP_CONTRACT if CMP_METRIC_FILE in current_paths else CONTRACT
    result["provenance"]["forecast_evaluation"] = dict(target_contract)
    result["provenance"]["evaluation_migration"] = {
        "kind": "reviewed_evaluation_and_logging_only_v1",
        "original_source_tree_sha256": previous["source_tree_sha256"],
        "original_base_sha": previous["base_sha"],
        "original_evaluation_protocol": previous.get("forecast_evaluation", previous.get("forecast_metric_version")),
        "new_source_tree_sha256": current["source_tree_sha256"],
        "preserved_training_state_sha256": checksum,
        "approved_trainers_sha256": (CMP_APPROVED_TRAINERS
            if CMP_METRIC_FILE in current_paths else APPROVED_TRAINERS),
    }
    assert all(result[key] is value for key, value in state.items())
    return result, checksum


def _cmp_trainer_ast(source):
    """Permit only evaluation imports and the additional metric manifest member."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "opencood.tools.dualcmp.cmp_adapted_forecast_metrics":
            node.module = "opencood.tools.dualcmp.mainstream_forecast_metrics"
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id in ("SOURCES", "PHASE_B_SOURCES") for target in node.targets):
            if not isinstance(node.value, ast.Tuple):
                raise RuntimeError("source manifest must remain a literal tuple")
            node.value.elts = [item for item in node.value.elts
                               if ast.literal_eval(item) != CMP_METRIC_FILE]
    return ast.dump(tree, include_attributes=False)


def _migrate_cmp_payload(payload, original_source, repository):
    """Metadata-only UniAD -> CMP-adapter transition; learned code is immutable."""
    _check_cmp_evaluation_sources(repository)
    previous = payload["provenance"]
    joint = "init_checkpoint_sha256" in previous
    relative, symbol = (("opencood/tools/dualcmp/train_joint_ddp.py", "SOURCES")
                        if joint else ("opencood/tools/dualcmp/train_forecast_ddp.py", "PHASE_B_SOURCES"))
    tree = ast.parse((original_source / relative).read_text())
    paths = next(ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
        and target.id == symbol for target in node.targets))
    digest = hashlib.sha256()
    changed = {}
    for path in sorted(paths):
        original = (original_source / path).read_bytes()
        current = (repository / path).read_bytes()
        digest.update(path.encode() + b"\0" + original)
        if current == original:
            continue
        if path in APPROVED_TRAINERS:
            if _cmp_trainer_ast(original.decode()) != _cmp_trainer_ast(current.decode()):
                raise RuntimeError("trainer changed beyond CMP evaluation import/manifest: " + path)
        elif path != "opencood/tools/dualcmp/eval_forecast_mainstream_ddp.py":
            raise RuntimeError("model/data/loss/runtime code changed: " + path)
        changed[path] = hashlib.sha256(current).hexdigest()
    if digest.hexdigest() != previous["source_tree_sha256"]:
        raise RuntimeError("preserved source snapshot does not prove checkpoint provenance")
    if joint:
        from opencood.tools.dualcmp.train_joint_ddp import SOURCES as current_paths, _provenance
    else:
        from opencood.tools.dualcmp.train_forecast_ddp import PHASE_B_SOURCES as current_paths, _provenance
    if set(current_paths) != set(paths) | {CMP_METRIC_FILE}:
        raise RuntimeError("source manifest changed beyond the CMP metric module")
    current = _provenance()
    if current["protocol_sha256"] != previous["protocol_sha256"]:
        raise RuntimeError("training temporal protocol changed")
    checksum = state_digest({key: value for key, value in payload.items() if key != "provenance"})
    result = dict(payload)
    result["provenance"] = dict(previous, **current)
    result["provenance"].update(forecast_evaluation=dict(CMP_CONTRACT),
        evaluation_migration={
            "kind": "cmp_gtfree_evaluation_only_v1",
            "original_source_tree_sha256": previous["source_tree_sha256"],
            "original_base_sha": previous["base_sha"],
            "original_evaluation_protocol": previous["forecast_evaluation"],
            "new_source_tree_sha256": current["source_tree_sha256"],
            "preserved_training_state_sha256": checksum,
            "changed_evaluation_files_sha256": dict(changed, **{CMP_METRIC_FILE:
                hashlib.sha256((repository / CMP_METRIC_FILE).read_bytes()).hexdigest()}),
            "previous_migration": previous.get("evaluation_migration"),
        })
    return result, checksum


def _check_cmp_evaluation_sources(repository):
    for path, expected in CMP_APPROVED_EVALUATION.items():
        if hashlib.sha256((repository / path).read_bytes()).hexdigest() != expected:
            raise RuntimeError("CMP evaluator has unreviewed changes: " + path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--original-source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists() or destination.resolve() == Path(args.checkpoint).resolve():
        raise FileExistsError("migration must create a separate checkpoint")
    original_hash = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    migrated, checksum = migrate_payload(torch.load(args.checkpoint, map_location="cpu"), args.original_source)
    migrated["provenance"]["evaluation_migration"]["original_checkpoint_sha256"] = original_hash
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError("previous partial migration exists; retain it for diagnosis")
    torch.save(migrated, str(temporary))
    restored = torch.load(str(temporary), map_location="cpu")
    if state_digest({key: value for key, value in restored.items() if key != "provenance"}) != checksum:
        raise RuntimeError("migration changed serialized model/optimizer/scheduler/progress")
    if hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() != original_hash:
        raise RuntimeError("original checkpoint changed during migration")
    temporary.replace(destination)
    proof = dict(restored["provenance"]["evaluation_migration"], output=str(destination),
        epoch=restored["epoch"], step=restored["step"], original_preserved=True, verified_state_preserved=True)
    destination.with_suffix(".migration.json").write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
