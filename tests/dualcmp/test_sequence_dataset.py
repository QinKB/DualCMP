"""CPU-only contracts for the causal DualCMP V2V4Real data path."""

from __future__ import annotations

import builtins
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import TensorDataset

from opencood.data_utils.dualcmp_dataloader import build_dataloader
from opencood.data_utils.v2v4real.datasets import build_v2v4real_dataset
from opencood.data_utils.v2v4real.datasets.dualcmp_schema import assert_runtime_is_supervision_free
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_index import SceneSequenceIndex
from opencood.data_utils.v2v4real.datasets.dualcmp_target_builder import CausalTargetBuilder, source_to_receiver_transform
import opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset as sequence_module


class _RandomDataset:
    def __len__(self):
        return 6

    def __getitem__(self, index):
        return torch.tensor(
            [index, torch.rand(()).item(), np.random.random(), random.random()],
            dtype=torch.float64,
        )


class _Preprocessor:
    def preprocess(self, points):
        return {"voxel_features": np.ones((1, 1, 7), dtype=np.float32), "voxel_coords": np.zeros((1, 3), dtype=np.int32)}

    def collate_batch(self, entries):
        return {"count": len(entries)}


class _Postprocessor:
    def generate_anchor_box(self):
        return np.zeros((1, 1, 1, 7), dtype=np.float32)

    def generate_label(self, **kwargs):
        return {"mask": kwargs["mask"]}

    def collate_batch(self, labels):
        return {"count": len(labels)}


def _pose(x):
    result = np.eye(4, dtype=np.float32)
    result[0, 3] = x
    return result.tolist()


def _frame(root: Path, scene: str, source: str, key: str, ego_x: float, actor_x: float,
           vehicles=None) -> None:
    folder = root / scene / source
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (key + ".pcd")).write_text("fixture", encoding="utf-8")
    with (folder / (key + ".yaml")).open("w", encoding="utf-8") as stream:
        yaml.safe_dump({"lidar_pose": _pose(ego_x), "vehicles": vehicles or {}}, stream)


@pytest.fixture
def v2v4real_root(tmp_path):
    # Actor moves 1 m per 0.1 s in world, while the receiver moves 1 m too.
    # Thus its previous local box must be reprojected into the current ego frame.
    for key, ego_x, actor_x in (("0000", 0, 5), ("0001", 1, 5), ("0002", 2, 5), ("0003", 3, 5), ("0004", 4, 5)):
        _frame(tmp_path, "scene_a", "0", key, ego_x, actor_x,
               {"7": {"ass_id": -1, "location": [actor_x, 0], "local_x": actor_x}})
    for key, ego_x, actor_x in (("0000", 0, 8),):
        _frame(tmp_path, "scene_a", "1", key, ego_x, actor_x)
    _frame(tmp_path, "scene_b", "0", "0000", 0, 5)
    return tmp_path


def _params(root, stage="tracking", **args):
    fusion_args = {"stage": stage, "tracking_frames": 3, "forecast_history_frames": 2,
                   "future_frames": 2, "max_actors": 4, "max_cav": 2,
                   "receiver_ids": [0], "runtime_forbid_gt": True,
                   "runtime_only": False, "frame_period_s": 0.1, **args}
    return {"root_dir": str(root), "validate_dir": str(root),
            "fusion": {"core_method": "DualCMPSequenceDataset", "args": fusion_args},
            "preprocess": {"cav_lidar_range": [-20, -20, -5, 20, 20, 5]},
            "postprocess": {"order": "hwl"}, "train_params": {"max_cav": 2}}


@pytest.fixture(autouse=True)
def fake_runtime(monkeypatch):
    monkeypatch.setattr(sequence_module, "build_preprocessor", lambda *args, **kwargs: _Preprocessor())
    monkeypatch.setattr(sequence_module, "build_postprocessor", lambda *args, **kwargs: _Postprocessor())
    monkeypatch.setattr(sequence_module, "pcd_to_np", lambda path: np.array([[2, 0, 0, 1]], dtype=np.float32))
    monkeypatch.setattr(sequence_module, "mask_ego_points", lambda points: points)
    monkeypatch.setattr(sequence_module, "mask_points_by_range", lambda points, _: points)
    monkeypatch.setattr(sequence_module.box_utils_v2v4real, "project_points_by_matrix_torch", lambda points, _: points)

    def project(objects, output, transform, *_args, **_kwargs):
        for object_key, content in objects.items():
            local = np.array([content["local_x"], 0, 0, 1], dtype=np.float32)
            point = transform @ local
            output[object_key] = {"ass_id": content.get("ass_id", -1),
                                  "coord": np.array([[point[0], point[1], 0, 1, 1, 1, 0]], dtype=np.float32)}

    monkeypatch.setattr(sequence_module.box_utils_v2v4real, "project_world_objects", project)
    sequence_module._load_yaml_cached.cache_clear()


def test_stage_contract_and_scene_boundary(v2v4real_root):
    with pytest.raises(ValueError, match="tracking_frames == 3"):
        sequence_module.DualCMPSequenceDataset(_params(v2v4real_root, tracking_frames=2))
    detection = sequence_module.DualCMPSequenceDataset(_params(v2v4real_root, "detection"))
    assert detection.clip_length == 1
    tracking = sequence_module.DualCMPSequenceDataset(_params(v2v4real_root))
    assert all(record.scene_id == "scene_a" and record.receiver_id == "0" for record in tracking.clip_records)
    assert all(record.start_position >= 1 for record in tracking.clip_records)
    assert isinstance(build_v2v4real_dataset(_params(v2v4real_root)), sequence_module.DualCMPSequenceDataset)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_clock_parameters_reject_nonfinite_values(v2v4real_root, invalid):
    with pytest.raises(ValueError, match="finite"):
        SceneSequenceIndex(str(v2v4real_root), frame_period_s=invalid)
    with pytest.raises(ValueError, match="finite"):
        SceneSequenceIndex(str(v2v4real_root), timeline_tolerance_s=invalid)
    index = SceneSequenceIndex(str(v2v4real_root))
    with pytest.raises(ValueError, match="finite"):
        index.latest_arrived_source_record("scene_a", "0", "1", "0001", invalid)


def test_sparse_source_slots_fail_before_fax_regroup(v2v4real_root):
    dataset = sequence_module.DualCMPSequenceDataset(_params(v2v4real_root, max_cav=3))
    frame = dataset._runtime_frame("scene_a", "0", "0000")
    frame["source_captures"][1]["source_slot"] = 2
    with pytest.raises(ValueError, match="contiguous prefix"):
        dataset._collate_runtime_frame([frame])


def test_partial_future_and_velocity_stop_at_acquisition_cut(v2v4real_root, monkeypatch):
    def boundaries(index):
        index.set_sequence_boundaries({"scene_a": [3]})
        return {"sha256": "fixture"}
    monkeypatch.setattr(sequence_module, "configure_aligned_boundaries", boundaries)
    dataset = sequence_module.DualCMPSequenceDataset(_params(v2v4real_root, "forecast",
        forecast_history_frames=1, forecast_coordinate_frame="current_receiver",
        require_full_future=False))
    before = dataset._target_frame("scene_a", "0", "0002", "0001", True)
    assert not before["future_mask"].any()
    after = dataset._target_frame("scene_a", "0", "0003", "0002", True)
    assert not after["velocity_mask"].any()
    assert after["future_mask"].any()  # Later observations in this segment remain valid.
    legacy = sequence_module.DualCMPSequenceDataset(_params(v2v4real_root, "forecast"))
    assert legacy._target_frame("scene_a", "0", "0002", "0001", True)["future_mask"].all(axis=1).any()


def test_small_cache_retains_real_samples_and_rejects_randomized_io(v2v4real_root):
    dataset = sequence_module.DualCMPSequenceDataset(_params(v2v4real_root, shuffle_training_points=False))
    sample = dataset[0]
    dataset.cache_small_subset()
    assert dataset[0]["scene_receiver_key"] == sample["scene_receiver_key"]
    for cached, original in zip(dataset[0]["runtime_frames"], sample["runtime_frames"]):
        assert cached["receiver_frame_key"] == original["receiver_frame_key"]
        assert np.array_equal(cached["receiver_pose"], original["receiver_pose"])
    randomized = sequence_module.DualCMPSequenceDataset(_params(v2v4real_root, shuffle_training_points=True))
    with pytest.raises(ValueError, match="deterministic"):
        randomized.cache_small_subset()


def test_timeline_gap_is_rejected(tmp_path):
    _frame(tmp_path, "scene", "0", "0000", 0, 5)
    _frame(tmp_path, "scene", "0", "0002", 2, 5)
    with pytest.raises(ValueError, match="timeline interval"):
        SceneSequenceIndex(str(tmp_path), frame_period_s=0.1)


def test_tracking_is_causal_and_velocity_uses_current_receiver_frame(v2v4real_root, monkeypatch):
    opened = []
    pcd_paths = []
    original_open = builtins.open

    def spy_open(path, *args, **kwargs):
        opened.append(str(path))
        return original_open(path, *args, **kwargs)

    dataset = sequence_module.DualCMPSequenceDataset(_params(v2v4real_root))
    # A tracking item must not parse a payload after its current clip boundary.
    sequence_module._load_yaml_cached.cache_clear()
    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(
        sequence_module, "pcd_to_np",
        lambda path: (pcd_paths.append(str(path)) or np.array([[2, 0, 0, 1]], dtype=np.float32)),
    )
    sample = dataset[0]
    velocity = sample["target_frames"][0]["current_velocity"][0]
    assert sample["target_frames"][0]["current_mask"][0]
    assert sample["target_frames"][0]["actor_ids"][0] == sample["target_frames"][1]["actor_ids"][0] == 7
    assert sample["target_frames"][0]["velocity_mask"][0]
    np.testing.assert_allclose(velocity, [10, 0], atol=1e-5)
    assert not any("0004.yaml" in path for path in opened)
    assert not any("0004.pcd" in path for path in pcd_paths)
    for frame in sample["runtime_frames"]:
        assert frame["source_captures"][0]["source_slot"] == 0
        assert frame["source_captures"][0]["arrival_time_s"] <= frame["receiver_time_s"]


def test_receiver_local_identity_ignores_association_id_and_future_yaml(tmp_path, monkeypatch):
    for key, x in (("0000", 0), ("0001", 1), ("0002", 2), ("0003", 3), ("0004", 4)):
        _frame(tmp_path, "scene", "0", key, x, 5)
        association_id = -1 if key in ("0000", "0001") else 17
        _frame(tmp_path, "scene", "1", key, x, 5, {
            "7": {"ass_id": association_id, "location": [5, 0], "local_x": 5},
        })
    opened = []
    original_open = builtins.open

    def spy_open(path, *args, **kwargs):
        opened.append(str(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)
    dataset = sequence_module.DualCMPSequenceDataset(_params(tmp_path, receiver_ids=[1]))
    assert not any(path.endswith("scene/1/0004.yaml") for path in opened)
    future_path = tmp_path / "scene" / "1" / "0004.yaml"
    future = yaml.safe_load(future_path.read_text(encoding="utf-8"))
    future["vehicles"]["7"]["ass_id"] = 99
    with future_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(future, stream)
    sample = dataset[0]
    actor_ids = [target["actor_ids"][0] for target in sample["target_frames"]]
    assert actor_ids == [7, 7, 7]
    assert not any(path.endswith("scene/1/0004.yaml") for path in opened)


@pytest.mark.parametrize("object_key", ["not-an-id", "-1"])
def test_receiver_local_object_key_fails_closed(object_key):
    builder = CausalTargetBuilder([-1, -1, -1, 1, 1, 1], 1)
    with pytest.raises(ValueError, match="receiver-local vehicle object key"):
        builder.actors_in_receiver({
            "lidar_pose": _pose(0),
            "vehicles": {object_key: {"ass_id": 17, "local_x": 0}},
        }, np.eye(4, dtype=np.float32))


def test_runtime_target_isolation_collate_and_missing_partner(v2v4real_root):
    dataset = sequence_module.DualCMPSequenceDataset(_params(v2v4real_root))
    result = dataset.collate_batch_train([dataset[0], dataset[1]])
    runtime = result["runtime_frames"][0]
    assert_runtime_is_supervision_free(runtime)
    assert runtime["source_to_batch"].tolist() == [0, 1]
    assert runtime["source_slot"].tolist() == [0, 0]
    assert runtime["source_mask"].tolist() == [[True, False], [True, False]]
    assert runtime["record_len"].tolist() == [1, 1]
    assert result["partner_delay_ms"].tolist() == [0.0, 0.0]
    with pytest.raises(ValueError, match="nested.target_frames"):
        assert_runtime_is_supervision_free({"nested": {"target_frames": {"future_xy": []}}})


def test_partner_delay_provenance(tmp_path):
    for key, x in (("0000", 0), ("0001", 1), ("0002", 2)):
        _frame(tmp_path, "scene", "0", key, x, 5)
        _frame(tmp_path, "scene", "1", key, x, 5)
    current = sequence_module.DualCMPSequenceDataset(
        _params(tmp_path, "detection", runtime_only=True),
    )
    current_partner = current[1]["runtime_frames"][0]["source_captures"][1]
    assert current_partner["delay_source"] == "partner_current"

    delayed_params = _params(tmp_path, "detection", runtime_only=True, async_overhead_ms=100)
    delayed_params["wild_setting"] = {"async": True}
    delayed = sequence_module.DualCMPSequenceDataset(delayed_params)
    delayed_partner = delayed[1]["runtime_frames"][0]["source_captures"][1]
    assert delayed_partner["delay_source"] == "synthetic_fixed"
    assert delayed_partner["capture_frame_idx"] == 0
    assert delayed_partner["arrival_time_s"] == pytest.approx(0.1)
    assert delayed_partner["delta_t_s"] == pytest.approx(0.1)

    explicit = sequence_module.DualCMPSequenceDataset(
        _params(tmp_path, "detection", runtime_only=True,
                partner_delay_ms=200),
    )
    explicit_partner = explicit[2]["runtime_frames"][0]["source_captures"][1]
    assert explicit_partner["capture_frame_idx"] == 0
    assert explicit_partner["arrival_time_s"] == pytest.approx(0.2)
    assert explicit_partner["delta_t_s"] == pytest.approx(0.2)

    mixed_args = dict(
        runtime_only=True,
        partner_delay_ms=100,
        partner_delay_choices_ms=[0, 100, 200],
        partner_delay_seed=23,
    )
    mixed_a = sequence_module.DualCMPSequenceDataset(
        _params(tmp_path, "detection", **mixed_args))
    mixed_b = sequence_module.DualCMPSequenceDataset(
        _params(tmp_path, "detection", **mixed_args))
    assigned_a = [mixed_a[index]["partner_delay_ms"]
                  for index in range(len(mixed_a))]
    assigned_b = [mixed_b[index]["partner_delay_ms"]
                  for index in range(len(mixed_b))]
    assert assigned_a == assigned_b
    assert set(assigned_a) == {0.0, 100.0, 200.0}
    counts = [entry["clips"] for entry in mixed_a.partner_delay_histogram]
    assert max(counts) - min(counts) <= 1


def test_forecast_only_constructs_future_and_masks_tail(v2v4real_root):
    forecast = sequence_module.DualCMPSequenceDataset(_params(v2v4real_root, "forecast"))
    sample = forecast[len(forecast) - 1]
    assert len(sample["runtime_frames"]) == 3
    assert len(sample["target_frames"]) == 1
    assert not sample["target_frames"][0]["future_mask"].any()


def test_positive_backward_dt_and_distributed_per_gpu_batch():
    builder = CausalTargetBuilder([-1, -1, -1, 1, 1, 1], 1)
    with pytest.raises(ValueError, match="positive backward"):
        builder.build_current({1: np.zeros(7)}, {1: np.zeros(7)}, 0)
    dataset, loader, sampler = build_dataloader(TensorDataset(torch.arange(8)), 2, True,
                                                workers=0, rank=1, world_size=2, training=True)
    assert sampler is not None and loader.batch_size == 2 and len(dataset) == 8


@pytest.mark.parametrize("world_size", [1, 2, 3])
def test_distributed_loader_uses_runtime_world_size(world_size):
    dataset = TensorDataset(torch.arange(12))
    samplers = [build_dataloader(
        dataset, 2, True, workers=0, rank=rank, world_size=world_size,
        training=True, pin_memory=False,
    )[2] for rank in range(world_size)]
    assert all(sampler.num_replicas == world_size for sampler in samplers)
    assert len(set().union(*(set(sampler) for sampler in samplers))) == 12


def test_training_distributed_odd_dataset_drops_padding_and_tail_batches():
    dataset = TensorDataset(torch.arange(9))
    _, rank_zero_loader, rank_zero_sampler = build_dataloader(
        dataset, 2, True, workers=0, rank=0, world_size=2, training=True,
        pin_memory=False,
    )
    _, rank_one_loader, rank_one_sampler = build_dataloader(
        dataset, 2, True, workers=0, rank=1, world_size=2, training=True,
        pin_memory=False,
    )
    rank_zero_indices = list(rank_zero_sampler)
    rank_one_indices = list(rank_one_sampler)
    assert rank_zero_sampler.drop_last and rank_one_sampler.drop_last
    assert rank_zero_loader.drop_last and rank_one_loader.drop_last
    assert set(rank_zero_indices).isdisjoint(rank_one_indices)
    assert len(rank_zero_indices) == len(rank_one_indices) == 4
    assert len(rank_zero_loader) == len(rank_one_loader) == 2
    assert all(batch[0].shape[0] == 2 for batch in rank_zero_loader)
    assert all(batch[0].shape[0] == 2 for batch in rank_one_loader)


def _random_loader_values(seed, rank):
    _, loader, _ = build_dataloader(
        _RandomDataset(), 2, False, workers=1, rank=rank, seed=seed,
        pin_memory=False,
    )
    iterator = iter(loader)
    try:
        return [batch.clone() for batch in iterator]
    finally:
        iterator._shutdown_workers()


def test_dataloader_worker_rng_and_generator_are_rank_seeded():
    baseline = _random_loader_values(seed=17, rank=0)
    repeated = _random_loader_values(seed=17, rank=0)
    other_seed = _random_loader_values(seed=18, rank=0)
    other_rank = _random_loader_values(seed=17, rank=1)
    assert all(torch.equal(first, second) for first, second in zip(baseline, repeated))
    assert any(not torch.equal(first, second) for first, second in zip(baseline, other_seed))
    assert any(not torch.equal(first, second) for first, second in zip(baseline, other_rank))


def _main_process_rng_values(seed, rank):
    build_dataloader(
        TensorDataset(torch.arange(1)), 1, False, workers=0, rank=rank,
        seed=seed, training=False, pin_memory=False,
    )
    return torch.rand(()).item(), np.random.random(), random.random()


def test_dataloader_main_process_rng_is_rank_seeded_without_workers():
    baseline = _main_process_rng_values(seed=29, rank=0)
    repeated = _main_process_rng_values(seed=29, rank=0)
    other_rank = _main_process_rng_values(seed=29, rank=1)
    assert baseline == repeated
    assert baseline != other_rank
