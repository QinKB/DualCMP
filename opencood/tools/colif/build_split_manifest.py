"""Build a deterministic scenario-disjoint V2V4Real split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


def _scene_entries(root: str) -> List[dict]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)
    entries = []
    for scene_path in sorted(path for path in root_path.iterdir() if path.is_dir()):
        source_dirs = sorted(path for path in scene_path.iterdir() if path.is_dir())
        frame_counts = {}
        for source_path in source_dirs:
            frame_counts[source_path.name] = len(
                [
                    path
                    for path in source_path.glob("*.yaml")
                    if "additional" not in path.name and "camera_gt" not in path.name
                ]
            )
        entries.append(
            {
                "scene_id": scene_path.name,
                "source_root": str(root_path),
                "source_count": len(source_dirs),
                "frame_counts": frame_counts,
            }
        )
    return entries


def build_disjoint_manifest(
    split_roots: Mapping[str, str],
    priority: Sequence[str] = ("test", "val", "train"),
) -> dict:
    """Assign duplicate scene basenames to exactly one split.

    Priority defaults to preserving official test and validation membership;
    a duplicate copy in a lower-priority split is recorded and excluded.
    """

    unknown = set(split_roots) - set(priority)
    if unknown:
        raise ValueError("priority is missing splits: %s" % sorted(unknown))

    discovered = {name: _scene_entries(root) for name, root in split_roots.items()}
    owner = {}
    excluded = []
    assigned: Dict[str, List[dict]] = {name: [] for name in split_roots}
    for split_name in priority:
        for entry in discovered.get(split_name, []):
            scene_id = entry["scene_id"]
            if scene_id in owner:
                excluded.append(
                    {
                        "scene_id": scene_id,
                        "excluded_split": split_name,
                        "kept_split": owner[scene_id],
                        "source_root": entry["source_root"],
                    }
                )
                continue
            owner[scene_id] = split_name
            assigned[split_name].append(entry)

    scene_sets = {name: {entry["scene_id"] for entry in entries}
                  for name, entries in assigned.items()}
    split_names = sorted(scene_sets)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = scene_sets[left] & scene_sets[right]
            if overlap:
                raise AssertionError("Non-disjoint output: %s" % sorted(overlap))

    provenance = {
        "policy": "scene_basename_disjoint_priority",
        "priority": list(priority),
        "input_roots": {name: str(Path(root).expanduser().resolve())
                        for name, root in split_roots.items()},
    }
    digest_input = json.dumps(
        {"splits": assigned, "excluded_duplicates": excluded, "provenance": provenance},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "version": 1,
        "splits": assigned,
        "excluded_duplicates": excluded,
        "provenance": provenance,
        "content_sha256": hashlib.sha256(digest_input).hexdigest(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_disjoint_manifest(
        {"train": args.train_root, "val": args.val_root, "test": args.test_root}
    )
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({
        "output": str(output_path),
        "split_sizes": {name: len(entries) for name, entries in manifest["splits"].items()},
        "excluded_duplicate_count": len(manifest["excluded_duplicates"]),
        "content_sha256": manifest["content_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
