"""CPU preflight for B/C native DMSTrack forecast-label smoke selection."""
from __future__ import print_function

from collections import Counter
from types import SimpleNamespace

from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
from opencood.data_utils.v2v4real.datasets.dualcmp_schema import assert_runtime_is_supervision_free
from opencood.tools.dualcmp.protocol import TRAIN_LABEL_WINDOW
from opencood.tools.dualcmp import train_forecast_ddp, train_joint_ddp


EXPECTED_FORECAST_CLIPS = 10


def _args():
    return SimpleNamespace(full_data=False, train_label_smoke=True,
                           data_split="train")


def _check(name, hypes):
    args = hypes["fusion"]["args"]
    assert hypes["root_dir"].endswith("/train")
    assert args["split_name"] == "train"
    assert args["tracking_label_layout"] == "dmstrack_train"
    assert args["window"] == list(TRAIN_LABEL_WINDOW)
    dataset = DualCMPSequenceDataset(hypes, train=True)
    assert len(dataset) == EXPECTED_FORECAST_CLIPS
    counts = Counter((record.scene_id, record.receiver_id)
                     for record in dataset.clip_records)
    assert set(counts.values()) == {EXPECTED_FORECAST_CLIPS // 2}
    for record in dataset.clip_records:
        keys = dataset.index.receiver_frame_keys(record.scene_id, record.receiver_id)
        assert int(keys[record.start_position]) >= 2
        assert int(keys[record.start_position + dataset.clip_length - 1]) <= 25
    for receiver_id in ("0", "1"):
        item_index = next(index for index, record in enumerate(dataset.clip_records)
                          if record.receiver_id == receiver_id)
        item = dataset[item_index]
        assert_runtime_is_supervision_free(item["runtime_frames"])
        target = item["target_frames"][-1]
        current_mask = target["current_mask"]
        future_mask = target["future_mask"]
        assert current_mask.any()
        assert (target["actor_ids"][current_mask] >= 0).all()
        assert target["current_boxes"].shape == (dataset.max_actors, 7)
        assert future_mask.shape == (dataset.max_actors, dataset.future_frames)
        assert future_mask[current_mask].any()
    print("%s native forecast labels: %d clips %r" % (name, len(dataset), dict(counts)))


def main():
    args = _args()
    forecast = train_forecast_ddp._hypes("opencood/hypes_yaml/point_pillar_v2v4real.yaml", args)[0]
    joint = train_joint_ddp._hypes("opencood/hypes_yaml/point_pillar_v2v4real.yaml", args)[0]
    _check("phase_b", forecast)
    _check("phase_c", joint)


if __name__ == "__main__":
    main()
