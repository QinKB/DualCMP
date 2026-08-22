import numpy as np
import pytest
import torch

from opencood.tools.plot_dualcmp_attention import (
    cluster_bootstrap_binned_mean,
    source_attention_mass,
    support_index,
)


@pytest.mark.parametrize(
    "bits, expected",
    ((0, None), (0b01, 0), (0b10, 1), (0b11, 2), (0b101, 2)),
)
def test_support_index(bits, expected):
    assert support_index(bits) == expected


def test_source_attention_mass_sums_tokens_and_masks_invalid_sources():
    weights = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.7, 0.8]])
    mass = source_attention_mass(
        weights, torch.tensor([True, True, False])
    )
    torch.testing.assert_close(mass, torch.tensor([0.3, 0.7, 0.0]))


def test_source_attention_mass_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        source_attention_mass(torch.ones(2, 3), torch.ones(3, dtype=torch.bool))


def test_cluster_bootstrap_binned_mean_is_deterministic():
    x = np.asarray([5.0, 10.0, 25.0, 30.0, 45.0, 50.0])
    values = np.asarray([1.0, 3.0, 2.0, 4.0, 3.0, 5.0])
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])
    edges = np.asarray([0.0, 20.0, 40.0, 60.0])
    first = cluster_bootstrap_binned_mean(
        x, values, groups, edges, repeats=50, seed=7
    )
    second = cluster_bootstrap_binned_mean(
        x, values, groups, edges, repeats=50, seed=7
    )
    np.testing.assert_allclose(first["mean"], np.asarray([2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(first["count"], np.asarray([2, 2, 2]))
    np.testing.assert_allclose(first["centers"], np.asarray([7.5, 27.5, 47.5]))
    np.testing.assert_allclose(first["lower"], second["lower"], equal_nan=True)
    np.testing.assert_allclose(first["upper"], second["upper"], equal_nan=True)
