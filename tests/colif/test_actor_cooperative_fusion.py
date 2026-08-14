import torch


def _module(**kwargs):
    from opencood.models.colif.actor_cooperative_fusion import (
        ActorConditionedCooperativeFusion,
    )

    return ActorConditionedCooperativeFusion(
        input_dim=8,
        query_dim=8,
        heads=2,
        max_cav=2,
        lidar_range=[-8.0, -8.0, -3.0, 8.0, 8.0, 3.0],
        patch_radius_m=2.0,
        patch_size=3,
        dropout=0.0,
        **kwargs,
    )


def test_masked_partner_cannot_change_actor_context() -> None:
    torch.manual_seed(3)
    module = _module().eval()
    source = torch.randn(1, 2, 8, 8, 8)
    query = torch.randn(1, 1, 8)
    boxes = torch.zeros(1, 1, 7)
    valid = torch.ones(1, 1, dtype=torch.bool)
    receiver_only = torch.tensor([[True, False]])

    first = module(source, receiver_only, query, boxes, valid)["context"]
    changed = source.clone()
    changed[:, 1] = 10_000.0
    second = module(changed, receiver_only, query, boxes, valid)["context"]

    torch.testing.assert_close(first, second)


def test_valid_partner_changes_context_and_receives_forecast_gradient() -> None:
    torch.manual_seed(5)
    module = _module().train()
    source = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
    query = torch.randn(1, 1, 8, requires_grad=True)
    boxes = torch.zeros(1, 1, 7)
    valid = torch.ones(1, 1, dtype=torch.bool)
    both = torch.tensor([[True, True]])

    context = module(source, both, query, boxes, valid)["context"]
    changed = source.detach().clone()
    changed[:, 1] += 20.0
    changed_context = module(changed, both, query.detach(), boxes, valid)[
        "context"
    ]

    assert not torch.allclose(context.detach(), changed_context)
    context.square().mean().backward()
    assert source.grad is not None
    assert torch.count_nonzero(source.grad[:, 1]) > 0
    assert query.grad is not None
    assert module.source_projection.weight.grad is not None


def test_inactive_actor_has_exactly_zero_cooperative_context() -> None:
    module = _module().eval()
    output = module(
        torch.randn(1, 2, 8, 8, 8),
        torch.tensor([[True, True]]),
        torch.randn(1, 1, 8),
        torch.zeros(1, 1, 7),
        torch.zeros(1, 1, dtype=torch.bool),
    )

    assert torch.count_nonzero(output["context"]) == 0
    assert torch.count_nonzero(output["attention_weights"]) == 0


def test_source_geometry_conditions_actor_context_without_source_index() -> None:
    torch.manual_seed(9)
    module = _module(
        geometry_conditioning=True,
        source_index_embedding=False,
    ).eval()
    source = torch.randn(1, 2, 8, 8, 8)
    query = torch.randn(1, 1, 8)
    boxes = torch.tensor([[[1.0, -2.0, 0.0, 1.8, 4.2, 1.6, 0.3]]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    both = torch.tensor([[True, True]])
    transforms = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    transforms[:, 1, 0, 3] = 8.0

    first = module(
        source,
        both,
        query,
        boxes,
        valid,
        source_to_receiver=transforms,
    )["context"]
    moved = transforms.clone()
    moved[:, 1, 0, 3] = -8.0
    second = module(
        source,
        both,
        query,
        boxes,
        valid,
        source_to_receiver=moved,
    )["context"]

    assert not torch.allclose(first, second)


def test_geometry_conditioned_fusion_is_source_permutation_invariant() -> None:
    torch.manual_seed(11)
    module = _module(
        geometry_conditioning=True,
        source_index_embedding=False,
    ).eval()
    source = torch.randn(1, 2, 8, 8, 8)
    query = torch.randn(1, 2, 8)
    boxes = torch.zeros(1, 2, 7)
    valid = torch.ones(1, 2, dtype=torch.bool)
    mask = torch.tensor([[True, True]])
    transforms = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    transforms[:, 1, 1, 3] = 5.0

    original = module(
        source,
        mask,
        query,
        boxes,
        valid,
        source_to_receiver=transforms,
    )["context"]
    swapped = module(
        source.flip(1),
        mask.flip(1),
        query,
        boxes,
        valid,
        source_to_receiver=transforms.flip(1),
    )["context"]

    torch.testing.assert_close(original, swapped)


def test_geometry_ablation_preserves_downstream_rng_and_freezes_inactive_branch() -> None:
    torch.manual_seed(31)
    geometry = _module(
        geometry_conditioning=True,
        source_index_embedding=False,
    )
    geometry_next = torch.nn.Linear(8, 8)
    geometry.freeze_inactive_parameters()

    torch.manual_seed(31)
    source_index = _module(
        geometry_conditioning=False,
        source_index_embedding=True,
    )
    source_index_next = torch.nn.Linear(8, 8)
    source_index.freeze_inactive_parameters()

    torch.testing.assert_close(geometry_next.weight, source_index_next.weight)
    assert not any(
        parameter.requires_grad
        for parameter in geometry.source_embedding.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in source_index.geometry_projection.parameters()
    )
