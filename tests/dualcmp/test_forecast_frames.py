"""Focused coordinate and horizon checks, not training acceptance."""
import math

import numpy as np
import pytest
import torch

from opencood.models.dualcmp.forecast_frames import align_actor_history, align_source_record
from opencood.models.dualcmp.mtr_actor_adapter import pack_mtr_actor_polylines
from opencood.tools.dualcmp.forecast_metrics import cmp_metrics, horizon_metrics
from opencood.tools.dualcmp.temporal_protocol import checkpoint_forecast_contract


def test_history_reexpresses_moving_receiver():
    boxes = torch.tensor([[[[3., 0., 0., 1., 2., 4., 0.]]]])
    mask = torch.ones(1, 1, 1, dtype=torch.bool)
    capture = torch.eye(4).reshape(1, 1, 1, 4, 4)
    current = torch.eye(4).unsqueeze(0)
    current[:, 0, 3] = 2.
    aligned = align_actor_history(boxes, mask, capture, current)
    assert torch.allclose(aligned[..., :3], torch.tensor([[[[1., 0., 0.]]]]))
    assert torch.equal(boxes[..., 0], torch.tensor([[[3.]]]))


def test_source_positions_include_motion_and_ego_transform():
    from types import SimpleNamespace
    reference = torch.tensor([[[3., 0., 0., 1., 2., 4., 0.]]])
    current_box = reference.clone()
    current_box[..., 0] = 2.
    current_pose = torch.eye(4).unsqueeze(0)
    current_pose[:, 0, 3] = 2.
    record = {"reference_boxes": reference, "receiver_pose": torch.eye(4).unsqueeze(0),
              "source_to_receiver": torch.eye(4).reshape(1, 1, 4, 4),
              "positions": torch.zeros(1, 1, 1, 1, 3)}
    positions, pose = align_source_record(record, SimpleNamespace(box=current_box, receiver_pose=current_pose))
    assert torch.allclose(positions[0, 0, 0, 0], torch.tensor([-1., 0., 0.]))
    assert pose[0, 0, 0, 3] == -2.


def test_radian_actor_frame_is_periodic():
    boxes = torch.tensor([[[0., 0., 0., 1., 2., 4., math.pi]]])
    history = boxes[:, :, None].expand(1, 1, 11, 7).clone()
    history[..., 0] = 1.
    mask = torch.ones(1, 1, 11, dtype=torch.bool)
    times = torch.arange(11).float()[None, None] * .1
    kwargs = dict(min_forecast_history_steps=2, min_context_history_steps=2,
                  frame_period_s=.1, legacy_heading=False)
    left = pack_mtr_actor_polylines(boxes, mask.any(-1), history, mask, times, torch.tensor([1.]), **kwargs)
    boxes[..., 6] = -math.pi
    history[..., 6] = -math.pi
    right = pack_mtr_actor_polylines(boxes, mask.any(-1), history, mask, times, torch.tensor([1.]), **kwargs)
    assert torch.allclose(left.batch_dict["input_dict"]["obj_trajs"], right.batch_dict["input_dict"]["obj_trajs"], atol=1e-5)


def test_exact_horizon_includes_endpoint_without_changing_cmp():
    pred = np.zeros((6, 50, 2), np.float32)
    pred[:, 49, 0] = 50.
    target = np.zeros((50, 2), np.float32)
    mask = np.ones(50, bool)
    assert cmp_metrics([pred], [target], [mask])["minADE_5s"] == 0.
    assert horizon_metrics([pred], [target], [mask])["minADE_5s"] == 1.
    assert horizon_metrics([pred], [target], [mask])["minFDE_5s"] == 50.


@pytest.mark.parametrize("metric", [cmp_metrics, horizon_metrics])
def test_metric_rejects_population_truncation(metric):
    pred = np.zeros((6, 50, 2), np.float32)
    target = np.zeros((50, 2), np.float32)
    mask = np.ones(50, bool)
    with pytest.raises(ValueError, match="population lengths"):
        metric([pred], [target], [mask], prediction_centers=[])
    with pytest.raises(ValueError, match="population lengths"):
        metric([pred], [target], [], target_centers=[np.zeros(2)])


def test_cmp_rejects_nonfinite_prediction():
    pred = np.zeros((6, 50, 2), np.float32)
    pred[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cmp_metrics([pred], [np.zeros((50, 2))], [np.ones(50, bool)])


def test_aligned_contract_is_explicit_and_legacy_stays_legacy():
    assert checkpoint_forecast_contract({"forecast_representation": "source_temporal"})["forecast_representation"] == "source_temporal"
    assert checkpoint_forecast_contract({"forecast_representation": "source_temporal_aligned"})["forecast_representation"] == "source_temporal_aligned"


def test_future_target_not_cropped_to_current_sensor_range(tmp_path):
    from types import SimpleNamespace
    from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
    dataset = object.__new__(DualCMPSequenceDataset)
    dataset.tracking_label_layout = "dmstrack_train"
    dataset.dmstrack_train_labels = {"scene": (tmp_path / "unused", ["0"])}
    dataset.index = SimpleNamespace(receiver_record=lambda *args: None)
    dataset._pose = lambda record: np.eye(4)
    boxes = {5: np.array([120., 0., 0., 1., 2., 4., 0.], np.float32)}
    dataset.target_builder = SimpleNamespace(actors_from_tracking_label_file=lambda *args: boxes)
    dataset.lidar_range = [-20., -20., -5., 20., 20., 5.]
    dataset.params = {"postprocess": {"order": "hwl"}}
    assert not dataset._actors("scene", "0", "0", np.eye(4))
    assert 5 in dataset._actors("scene", "0", "0", np.eye(4), range_crop=False)
