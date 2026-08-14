from pathlib import Path

from opencood.tools.colif.build_split_manifest import build_disjoint_manifest


def _make_scene(root: Path, scene_id: str) -> None:
    source = root / scene_id / "0"
    source.mkdir(parents=True)
    (source / "000000.yaml").write_text("lidar_pose: []\n", encoding="utf-8")


def test_duplicate_scene_is_kept_only_in_highest_priority_split(tmp_path: Path) -> None:
    roots = {name: tmp_path / name for name in ("train", "val", "test")}
    for root in roots.values():
        root.mkdir()
    _make_scene(roots["train"], "duplicate")
    _make_scene(roots["train"], "train_only")
    _make_scene(roots["val"], "val_only")
    _make_scene(roots["test"], "duplicate")

    manifest = build_disjoint_manifest({name: str(root) for name, root in roots.items()})

    train_ids = {entry["scene_id"] for entry in manifest["splits"]["train"]}
    test_ids = {entry["scene_id"] for entry in manifest["splits"]["test"]}
    assert train_ids == {"train_only"}
    assert test_ids == {"duplicate"}
    assert manifest["excluded_duplicates"] == [
        {
            "scene_id": "duplicate",
            "excluded_split": "train",
            "kept_split": "test",
            "source_root": str(roots["train"].resolve()),
        }
    ]
    assert len(manifest["content_sha256"]) == 64
