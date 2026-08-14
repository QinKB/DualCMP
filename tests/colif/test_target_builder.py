import numpy as np

from opencood.data_utils.v2v4real.datasets.colif_target_builder import (
    CanonicalTargetBuilder,
)


def test_point_support_controls_local_positive_and_unknown_identity_is_ignored() -> None:
    builder = CanonicalTargetBuilder(
        lidar_range=[-10, -10, -3, 10, 10, 3], min_points=3, max_objects=4
    )
    yaml_params = {
        "lidar_pose": np.eye(4),
        "vehicles": {
            7: {
                "location": [5.0, 0.0, 0.0],
                "center": [0.0, 0.0, 0.0],
                "extent": [2.0, 1.0, 1.0],
                "angle": [0.0, 0.0, 0.0],
                "ass_id": -1,
                "obj_type": "Car",
            },
            8: {
                "location": [-5.0, 0.0, 0.0],
                "center": [0.0, 0.0, 0.0],
                "extent": [2.0, 1.0, 1.0],
                "angle": [0.0, 0.0, 0.0],
                "ass_id": 42,
                "obj_type": "Car",
            },
        },
    }
    lidar = np.array(
        [[5.0, 0.0, 0.0, 1.0], [5.2, 0.1, 0.0, 1.0], [4.8, -0.1, 0.0, 1.0]],
        dtype=np.float32,
    )
    target = builder.build_source_target(yaml_params, lidar)

    assert target["source_box_mask"].sum() == 1
    assert target["source_actor_ids"][0] == -1
    assert target["association_ignore_mask"][0]
    assert target["source_point_counts"][0] == 3
