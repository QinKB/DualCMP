import pytest

from opencood.data_utils.v2v4real.datasets.colif_schema import (
    assert_runtime_is_supervision_free,
    find_forbidden_runtime_keys,
)


def test_runtime_schema_accepts_sensor_and_timing_metadata() -> None:
    runtime = {
        "scene_id": ["scene_a"],
        "receiver_id": [0],
        "processed_lidar": {"voxel_features": "placeholder"},
        "sources": [{"source_id": 1, "capture_time_s": 0.1}],
    }
    assert find_forbidden_runtime_keys(runtime) == ()
    assert_runtime_is_supervision_free(runtime)


def test_runtime_schema_rejects_nested_gt_and_future() -> None:
    runtime = {
        "sources": [{"source_id": 1, "vehicles": {7: {"ass_id": 42}}}],
        "metadata": {"future_xy": [[0.0, 0.0]]},
    }
    violations = find_forbidden_runtime_keys(runtime)
    assert ("sources", "0", "vehicles") in violations
    assert ("sources", "0", "vehicles", "7", "ass_id") in violations
    assert ("metadata", "future_xy") in violations
    with pytest.raises(ValueError, match="supervision-only"):
        assert_runtime_is_supervision_free(runtime)
