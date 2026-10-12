"""Focused acquisition-boundary contracts; not learning acceptance."""
from pathlib import Path

from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_index import SceneSequenceIndex, SourceFrameRecord


def index_fixture():
    index = object.__new__(SceneSequenceIndex)
    records = {str(n): SourceFrameRecord("scene", "0", str(n), n, n * .1,
                                       Path("unused"), Path("unused")) for n in range(5)}
    index._records = {"scene": {"0": records}}
    index._frame_keys = {"scene": {"0": tuple(records)}}
    index._sequence_boundaries = {}
    return index


def test_delayed_capture_cannot_cross_acquisition_boundary():
    index = index_fixture()
    assert index.latest_arrived_source_record("scene", "0", "0", "2", .1).frame_key == "1"
    index.set_sequence_boundaries({"scene": (2,)})
    assert index.latest_arrived_source_record("scene", "0", "0", "2", .1) is None
    assert index.latest_arrived_source_record("scene", "0", "0", "3", .1).frame_key == "2"
    assert index.segment("scene", 1) == 0 and index.segment("scene", 2) == 1
    assert index.is_segment_start("scene", 2) and not index.is_segment_start("scene", 3)


def test_legacy_index_still_uses_same_capture_clock():
    index = index_fixture()
    for n in range(1, 5):
        assert index.latest_arrived_source_record("scene", "0", "0", str(n), .1).frame_index == n - 1
