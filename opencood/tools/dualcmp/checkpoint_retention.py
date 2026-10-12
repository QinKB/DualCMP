"""Prune only checkpoint files named by an authoritative evaluated plan."""
import argparse
import json
import os
from pathlib import Path
import re
import sys

from opencood.tools.dualcmp.checkpoint_handoff import (
    atomic_json, checkpoint_lock, file_sha256,
)
from opencood.tools.dualcmp.cmp_adapted_forecast_metrics import PRIMARY_METRICS


PLAN_CONTRACT = "dualcmp_evaluated_top3_v1"
CHECKPOINT_PATTERN = re.compile(r"^checkpoint_epoch_(\d{3,})_step_(\d{9,})\.pt$")
EVENT_PATTERN = re.compile(r"^checkpoint_step_(\d{9,})\.json$")


def _checkpoint_identity(name):
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("checkpoint entries must be plain basenames")
    match = CHECKPOINT_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError("checkpoint entry is not a canonical epoch/step filename")
    epoch, step = int(match.group(1)), int(match.group(2))
    if name != "checkpoint_epoch_%03d_step_%09d.pt" % (epoch, step):
        raise ValueError("checkpoint filename is not canonical")
    return epoch, step


def _validate_plan(plan):
    if not isinstance(plan, dict) or plan.get("contract") != PLAN_CONTRACT:
        raise ValueError("unsupported evaluated-checkpoint retention plan")
    if plan.get("primary_metric") != "minADE_5s" or plan["primary_metric"] not in PRIMARY_METRICS:
        raise ValueError("retention plan must use the primary minADE_5s metric")
    if plan.get("user_confirmed_keep_top_k", 3) != 3:
        raise ValueError("retention plan keep count must remain three")
    ranked_keep = plan.get("ranked_keep")
    evaluated = plan.get("evaluated")
    if not isinstance(ranked_keep, list) or len(ranked_keep) > 3:
        raise ValueError("ranked_keep must contain at most three checkpoint basenames")
    if any(not isinstance(name, str) for name in ranked_keep):
        raise ValueError("ranked_keep entries must be checkpoint basenames")
    if len(set(ranked_keep)) != len(ranked_keep):
        raise ValueError("ranked_keep contains duplicates")
    if not isinstance(evaluated, list):
        raise ValueError("evaluated must be a list")

    records, steps = {}, set()
    for item in evaluated:
        if not isinstance(item, dict):
            raise ValueError("evaluated entries must be objects")
        name = item.get("checkpoint")
        _, filename_step = _checkpoint_identity(name)
        step = item.get("step")
        digest = item.get("sha256")
        if isinstance(step, bool) or not isinstance(step, int) or step != filename_step:
            raise ValueError("evaluated step disagrees with checkpoint filename")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("evaluated checkpoint SHA256 is invalid")
        if name in records or step in steps:
            raise ValueError("evaluated checkpoint names and steps must be unique")
        records[name] = {"checkpoint": name, "sha256": digest, "step": step}
        steps.add(step)
    for name in ranked_keep:
        _checkpoint_identity(name)
        if name not in records:
            raise ValueError("ranked_keep must be a subset of evaluated checkpoints")
    return ranked_keep, records


def _event_path(events, record):
    path = events / ("checkpoint_step_%09d.json" % record["step"])
    if path.is_symlink() or not path.is_file() or path.resolve().parent != events.resolve():
        raise RuntimeError("checkpoint event manifest is missing or unsafe: " + path.name)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError("checkpoint event manifest is invalid: " + path.name) from error
    if (not isinstance(manifest, dict)
            or manifest.get("version") != "dualcmp_external_checkpoint_v1"
            or manifest.get("checkpoint") != record["checkpoint"]
            or manifest.get("sha256") != record["sha256"]
            or manifest.get("step") != record["step"]):
        raise RuntimeError("plan identity disagrees with checkpoint event: " + path.name)
    size = manifest.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise RuntimeError("checkpoint event size is invalid: " + path.name)
    return path, manifest


def _latest_event(root, events, latest, latest_sha):
    matches = []
    for path in events.iterdir():
        match = EVENT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("unsafe file in checkpoint_events: " + path.name)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(manifest, dict)
                or manifest.get("version") != "dualcmp_external_checkpoint_v1"
                or manifest.get("sha256") != latest_sha
                or manifest.get("size_bytes") != latest.stat().st_size):
            continue
        name = manifest.get("checkpoint")
        _, step = _checkpoint_identity(name)
        event_step = manifest.get("step")
        if (isinstance(event_step, bool) or not isinstance(event_step, int)
                or step != event_step or step != int(match.group(1))):
            raise RuntimeError("latest event identity is inconsistent: " + path.name)
        candidate = root / name
        if candidate.is_symlink() or not candidate.is_file() or candidate.resolve().parent != root:
            raise RuntimeError("latest event checkpoint is missing or unsafe: " + name)
        if os.path.samefile(latest, candidate) or file_sha256(candidate) == latest_sha:
            matches.append((step, name, path))
    if not matches:
        raise RuntimeError("checkpoint.pt has no matching immutable event; refusing to prune")
    _, name, event = max(matches)
    return name, event


def prune_checkpoints(run_root, plan):
    root = Path(run_root).resolve(strict=True)
    with checkpoint_lock(root):
        return _prune_checkpoints_locked(root, plan)


def _prune_checkpoints_locked(root, plan):
    ranked_keep, records = _validate_plan(plan)
    if not root.is_dir():
        raise ValueError("run root must be a directory")
    events = root / "checkpoint_events"
    if events.is_symlink() or not events.is_dir():
        raise RuntimeError("run root has no safe checkpoint_events directory")
    latest = root / "checkpoint.pt"
    if latest.is_symlink() and latest.resolve(strict=True).parent != root:
        raise RuntimeError("checkpoint.pt resolves outside run root")
    if not latest.is_file():
        raise RuntimeError("checkpoint.pt is missing; refusing to prune")
    latest_sha = file_sha256(latest)
    latest_name, latest_event = _latest_event(root, events, latest, latest_sha)

    keep = set(ranked_keep)
    keep.add(latest_name)
    deletions = []
    already_missing = []
    for name, record in records.items():
        event_path, event = _event_path(events, record)
        candidate = root / name
        if candidate.is_symlink():
            raise RuntimeError("refusing to prune or keep a symlink checkpoint: " + name)
        if not candidate.exists():
            if name in ranked_keep:
                raise RuntimeError("ranked checkpoint is missing: " + name)
            already_missing.append(name)
            continue
        if not candidate.is_file() or candidate.resolve().parent != root:
            raise RuntimeError("checkpoint is not a regular file under run root: " + name)
        size = candidate.stat().st_size
        if size != event["size_bytes"] or file_sha256(candidate) != record["sha256"]:
            raise RuntimeError("checkpoint size or SHA256 disagrees with its event: " + name)
        if name in ranked_keep:
            continue
        if name == latest_name or record["sha256"] == latest_sha or os.path.samefile(candidate, latest):
            keep.add(name)
            continue
        deletions.append((candidate, record))

    # Validate every candidate first so a later mismatch cannot follow earlier deletions.
    removed = []
    for candidate, record in deletions:
        candidate.unlink()
        removed.append(record["checkpoint"])
    result = {
        "contract": PLAN_CONTRACT,
        "primary_metric": plan["primary_metric"],
        "ranked_keep": list(ranked_keep),
        "removed": removed,
        "already_missing": already_missing,
        "kept": sorted(keep),
        "latest": {
            "checkpoint": "checkpoint.pt",
            "published_checkpoint": latest_name,
            "sha256": latest_sha,
            "event": latest_event.name,
        },
    }
    atomic_json(root / "retention.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    plans = parser.add_mutually_exclusive_group(required=True)
    plans.add_argument("--plan")
    plans.add_argument("--plan-stdin", action="store_true")
    args = parser.parse_args()
    plan = json.load(sys.stdin) if args.plan_stdin else json.loads(Path(args.plan).read_text(encoding="utf-8"))
    result = prune_checkpoints(args.run_root, plan)
    print(json.dumps({"removed": len(result["removed"]), "kept": len(result["kept"]),
                      "latest_event": result["latest"]["event"]}))


if __name__ == "__main__":
    main()
