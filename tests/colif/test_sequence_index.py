import json
from pathlib import Path

import pytest

from opencood.data_utils.v2v4real.datasets.colif_sequence_index import (
    SceneSequenceIndex,
)


def _frame(source_dir: Path, key: str, with_lidar: bool = True) -> None:
    (source_dir / (key + ".yaml")).write_text("lidar_pose: []\n", encoding="utf-8")
    if with_lidar:
        (source_dir / (key + ".pcd")).write_text("pcd\n", encoding="utf-8")


def _scene(root: Path, name: str = "scene_a") -> Path:
    scene = root / name
    (scene / "0").mkdir(parents=True)
    (scene / "1").mkdir()
    return scene


def test_receiver_timeline_is_not_cross_source_intersection(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    for key in ("000000", "000001", "000002"):
        _frame(scene / "0", key)
    for key in ("000000", "000002"):
        _frame(scene / "1", key)

    index = SceneSequenceIndex(str(tmp_path), frame_period_s=0.1)

    assert index.receiver_frame_keys("scene_a", "0") == (
        "000000", "000001", "000002"
    )
    assert index.receiver_frame_keys("scene_a", "1") == ("000000", "000002")
    assert index.source_record("scene_a", "1", "000001") is None


def test_arrival_causality_uses_latest_available_capture(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    for key in ("000000", "000001", "000002"):
        _frame(scene / "0", key)
        _frame(scene / "1", key)

    index = SceneSequenceIndex(str(tmp_path), frame_period_s=0.1)

    delayed = index.latest_arrived_source_record(
        "scene_a", "0", "1", "000002", delay_s=0.1
    )
    assert delayed is not None
    assert delayed.frame_key == "000001"
    assert index.latest_arrived_source_record(
        "scene_a", "0", "1", "000000", delay_s=0.1
    ) is None


def test_incomplete_lidar_is_reported_and_excluded(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    _frame(scene / "0", "000000")
    _frame(scene / "0", "000001", with_lidar=False)
    _frame(scene / "1", "000000")

    index = SceneSequenceIndex(str(tmp_path))

    assert index.receiver_frame_keys("scene_a", "0") == ("000000",)
    assert index.incomplete_files == (
        ("scene_a", "0", "000001", "missing_lidar"),
    )


def test_manifest_filters_by_split_and_source_root(tmp_path: Path) -> None:
    root = tmp_path / "train"
    root.mkdir()
    for name in ("keep", "drop"):
        scene = _scene(root, name)
        _frame(scene / "0", "000000")
        _frame(scene / "1", "000000")

    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "splits": {
                    "train": [
                        {"scene_id": "keep", "source_root": str(root.resolve())},
                        {"scene_id": "from_elsewhere", "source_root": str(tmp_path)},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    index = SceneSequenceIndex(
        str(root), split_manifest=str(manifest), split_name="train"
    )
    assert index.scene_ids() == ("keep",)


def test_rejects_negative_delay(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    _frame(scene / "0", "000000")
    _frame(scene / "1", "000000")
    index = SceneSequenceIndex(str(tmp_path))

    with pytest.raises(ValueError, match="non-negative"):
        index.latest_arrived_source_record(
            "scene_a", "0", "1", "000000", delay_s=-0.1
        )
