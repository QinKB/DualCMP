import numpy as np

from opencood.data_utils.v2v4real.datasets.colif_sequence_dataset import (
    points_for_split,
    source_to_receiver_transform,
)


def test_test_split_preserves_point_order_while_train_split_shuffles() -> None:
    points = np.arange(40, dtype=np.float32).reshape(10, 4)
    np.random.seed(7)
    evaluation = points_for_split(points.copy(), train=False)
    np.random.seed(7)
    training = points_for_split(points.copy(), train=True)

    np.testing.assert_array_equal(evaluation, points)
    assert not np.array_equal(training, points)
    np.testing.assert_array_equal(
        np.sort(training[:, 0]), np.sort(points[:, 0])
    )


def test_source_to_receiver_transform_identity() -> None:
    identity = np.eye(4, dtype=np.float32)
    np.testing.assert_allclose(
        source_to_receiver_transform(identity, identity), identity
    )


def test_source_to_receiver_transform_respects_lidar_to_world_convention() -> None:
    source_lidar_to_world = np.eye(4, dtype=np.float32)
    receiver_lidar_to_world = np.eye(4, dtype=np.float32)
    receiver_lidar_to_world[0, 3] = 5.0

    transform = source_to_receiver_transform(
        source_lidar_to_world, receiver_lidar_to_world
    )
    source_point = np.array([6.0, 0.0, 0.0, 1.0], dtype=np.float32)
    receiver_point = transform @ source_point
    np.testing.assert_allclose(receiver_point[:3], [1.0, 0.0, 0.0])


def test_transform_rotation_and_translation_are_receiver_centric() -> None:
    source = np.eye(4, dtype=np.float32)
    source[0, 3] = 2.0
    receiver = np.eye(4, dtype=np.float32)
    receiver[1, 3] = 3.0
    expected = np.linalg.inv(receiver) @ source
    np.testing.assert_allclose(source_to_receiver_transform(source, receiver), expected)
