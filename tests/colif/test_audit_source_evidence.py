import numpy as np

from opencood.tools.colif.audit_source_evidence import (
    _canonical_id,
    _point_support_by_object,
)


def test_canonical_map_recovers_reference_and_partner_identity() -> None:
    mapping = {
        ("scene", "0", "23"): 23,
        ("scene", "1", "7"): 23,
    }

    assert _canonical_id(
        "scene", "0", "23", {"ass_id": -1}, mapping
    ) == 23
    assert _canonical_id(
        "scene", "1", "7", {"ass_id": 23}, mapping
    ) == 23


def test_canonical_lookup_does_not_guess_unmapped_local_identity() -> None:
    assert _canonical_id(
        "scene", "1", "7", {"ass_id": -1}, {}
    ) is None


def test_point_support_uses_source_native_object_coordinates() -> None:
    yaml_params = {
        "lidar_pose": np.array(
            [
                [1.0, 0.0, 0.0, 100.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        "vehicles": {
            23: {
                "location": [5.0, 0.0, 0.0],
                "center": [0.0, 0.0, 0.0],
                "extent": [2.0, 1.0, 1.0],
                "angle": [0.0, 0.0, 0.0],
                "ass_id": -1,
                "obj_type": "Car",
            }
        },
    }
    lidar = np.array(
        [
            [5.0, 0.0, 0.0, 1.0],
            [5.2, 0.1, 0.0, 1.0],
            [4.8, -0.1, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    support = _point_support_by_object(
        "scene",
        "0",
        yaml_params,
        lidar,
        [-10, -10, -3, 10, 10, 3],
        {("scene", "0", "23"): 23},
    )

    assert support[23]["canonical_id"] == 23
    assert support[23]["point_count"] == 3
