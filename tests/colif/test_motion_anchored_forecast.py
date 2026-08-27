import torch

from opencood.models.colif.motion_anchored_forecast import (
    MotionAnchoredForecastHead,
)
from opencood.models.colif.actor_source_tokenizer import (
    ActorAlignedSourceTokenizer,
)


def _head(
    temporal_history=True,
    trajectory_parameterization="anchored",
):
    return MotionAnchoredForecastHead(
        16,
        {
            "history_frames": 11,
            "future_frames": 50,
            "frame_period_s": 0.1,
            "min_forecast_history_steps": 2,
            "motion_context_heads": 4,
            "motion_context_layers": 1,
            "motion_context_ffn": 32,
            "source_attention_heads": 4,
            "source_attention_ffn": 32,
            "dropout": 0.0,
            "temporal_history": temporal_history,
            "history_encoder_layers": 1,
            "history_encoder_heads": 4,
            "trajectory_parameterization": trajectory_parameterization,
        },
    )


def _inputs(batch=1, queries=1):
    query = torch.randn(batch, queries, 16)
    boxes = torch.zeros(batch, queries, 7)
    boxes[..., 3:6] = torch.tensor([1.6, 1.8, 4.0])
    history = torch.zeros(batch, queries, 11, 7)
    history[..., -3:, 0] = torch.tensor([0.0, 0.1, 0.2])
    history[..., -3:, 3:6] = torch.tensor([1.6, 1.8, 4.0])
    mask = torch.zeros(batch, queries, 11, dtype=torch.bool)
    mask[..., -3:] = True
    valid = torch.ones(batch, queries, dtype=torch.bool)
    return query, boxes, valid, history, mask, torch.zeros(batch)


def test_first_observation_creates_no_long_horizon_forecast() -> None:
    head = _head().eval()
    query, boxes, valid, history, mask, time = _inputs()
    mask.zero_()
    mask[..., -1] = True
    output = head(query, boxes, valid, history, mask, time)

    assert not output["forecast_valid_mask"].any()
    assert torch.count_nonzero(output["pred_trajs"]) == 0


def test_zero_residual_initialization_equals_physics_anchors() -> None:
    head = _head().eval()
    query, boxes, valid, history, mask, time = _inputs()
    history[0, 0, -3:, 0] = torch.tensor([0.0, 0.2, 0.4])
    boxes[0, 0, 0] = 0.4
    output = head(query, boxes, valid, history, mask, time)

    expected_velocity = torch.tensor([2.0, 0.0])
    expected = (
        expected_velocity[None]
        * head.anchor_scales[:, None]
        * head.future_times_s[-1]
    )
    torch.testing.assert_close(output["pred_trajs"][0, 0, :, -1, :2], expected)
    assert output["forecast_loss_in_receiver_frame"] is True


def test_direct_parameterization_starts_from_zero_and_is_unbounded() -> None:
    direct = _head(trajectory_parameterization="direct").eval()
    inputs = _inputs()
    output = direct(*inputs)
    assert torch.count_nonzero(output["pred_trajs"][..., :2]) == 0

    with torch.no_grad():
        direct.residual_head.bias.fill_(20.0)
    displaced = direct(*inputs)["pred_trajs"][..., :2]
    torch.testing.assert_close(displaced, torch.full_like(displaced, 20.0))


def test_source_tokens_are_consumed_after_mode_construction() -> None:
    torch.manual_seed(7)
    head = _head().eval()
    inputs = _inputs()
    tokens = torch.randn(1, 1, 2, 5, 16)
    positions = torch.randn_like(tokens)
    mask = torch.ones(1, 1, 2, 5, dtype=torch.bool)

    first = head(
        *inputs,
        source_tokens=tokens,
        source_positions=positions,
        source_token_mask=mask,
    )
    changed = tokens.clone()
    changed[:, :, 1, :, 0] += 20.0
    second = head(
        *inputs,
        source_tokens=changed,
        source_positions=positions,
        source_token_mask=mask,
    )

    assert not torch.allclose(first["loss_pred_scores"], second["loss_pred_scores"])
    assert first["mode_source_attention_weights"].shape == (1, 1, 6, 2, 5)
    assert first["cooperative_attention_weights"].shape == (1, 1, 2, 5)
    assert first["actor_source_attention_weights"].shape == (1, 1, 1, 2)


def test_masked_source_cannot_change_prediction() -> None:
    torch.manual_seed(9)
    head = _head().eval()
    inputs = _inputs()
    tokens = torch.randn(1, 1, 2, 5, 16)
    positions = torch.randn_like(tokens)
    mask = torch.ones(1, 1, 2, 5, dtype=torch.bool)
    mask[:, :, 1] = False

    first = head(
        *inputs,
        source_tokens=tokens,
        source_positions=positions,
        source_token_mask=mask,
    )
    changed = tokens.clone()
    changed[:, :, 1] += 10_000.0
    second = head(
        *inputs,
        source_tokens=changed,
        source_positions=positions,
        source_token_mask=mask,
    )

    torch.testing.assert_close(first["pred_scores"], second["pred_scores"])
    torch.testing.assert_close(first["pred_trajs"], second["pred_trajs"])


def test_position_is_preserved_in_attention_values() -> None:
    torch.manual_seed(10)
    head = _head().eval()
    inputs = _inputs()
    tokens = torch.zeros(1, 1, 2, 5, 16)
    positions = torch.randn_like(tokens) * 100.0
    mask = torch.ones(1, 1, 2, 5, dtype=torch.bool)

    output = head(
        *inputs,
        source_tokens=tokens,
        source_positions=positions,
        source_token_mask=mask,
    )

    assert torch.count_nonzero(output["cooperative_context"]) > 0


def test_neighbour_actor_source_tokens_change_target_prediction() -> None:
    torch.manual_seed(12)
    head = _head().eval()
    inputs = _inputs(1, 2)
    tokens = torch.randn(1, 2, 2, 5, 16)
    positions = torch.randn_like(tokens)
    mask = torch.ones(1, 2, 2, 5, dtype=torch.bool)

    first = head(
        *inputs,
        source_tokens=tokens,
        source_positions=positions,
        source_token_mask=mask,
    )
    changed = tokens.clone()
    changed[:, 1, 1, :, 0] += 50.0
    second = head(
        *inputs,
        source_tokens=changed,
        source_positions=positions,
        source_token_mask=mask,
    )

    assert not torch.allclose(
        first["loss_pred_scores"][:, 0], second["loss_pred_scores"][:, 0]
    )
    assert torch.count_nonzero(
        first["actor_source_attention_weights"][:, 0, 0]
    ) == 0
    assert torch.count_nonzero(
        first["actor_source_attention_weights"][:, 0, 1]
    ) > 0

    masked = mask.clone()
    masked[:, 1, 1] = False
    masked_first = head(
        *inputs,
        source_tokens=tokens,
        source_positions=positions,
        source_token_mask=masked,
    )
    masked_second = head(
        *inputs,
        source_tokens=changed,
        source_positions=positions,
        source_token_mask=masked,
    )
    torch.testing.assert_close(
        masked_first["pred_scores"][:, 0], masked_second["pred_scores"][:, 0]
    )


def test_source_permutation_preserves_prediction() -> None:
    torch.manual_seed(11)
    head = _head().eval()
    inputs = _inputs()
    tokens = torch.randn(1, 1, 2, 5, 16)
    positions = torch.randn(1, 1, 1, 5, 16).expand_as(tokens).clone()
    mask = torch.ones(1, 1, 2, 5, dtype=torch.bool)

    original = head(
        *inputs,
        source_tokens=tokens,
        source_positions=positions,
        source_token_mask=mask,
    )
    swapped = head(
        *inputs,
        source_tokens=tokens.flip(2),
        source_positions=positions.flip(2),
        source_token_mask=mask.flip(2),
    )

    torch.testing.assert_close(original["pred_scores"], swapped["pred_scores"])
    torch.testing.assert_close(original["pred_trajs"], swapped["pred_trajs"])
    torch.testing.assert_close(
        original["cooperative_attention_weights"].flip(2),
        swapped["cooperative_attention_weights"],
    )


def test_forecast_loss_reaches_source_decoder_and_tokens() -> None:
    head = _head().train()
    query, boxes, valid, history, mask, time = _inputs(1, 2)
    query.requires_grad_()
    tokens = torch.randn(1, 2, 2, 5, 16, requires_grad=True)
    positions = torch.randn_like(tokens)
    token_mask = torch.ones(1, 2, 2, 5, dtype=torch.bool)
    output = head(
        query,
        boxes,
        valid,
        history,
        mask,
        time,
        source_tokens=tokens,
        source_positions=positions,
        source_token_mask=token_mask,
    )
    loss = (
        output["loss_pred_trajs_local"][..., :2].square().mean()
        + output["loss_pred_scores"].square().mean()
    )
    loss.backward()

    assert tokens.grad is not None
    assert torch.count_nonzero(tokens.grad) > 0
    assert head.source_attention.in_proj_weight.grad is not None
    assert head.source_ffn[0].weight.grad is not None
    assert head.residual_head.weight.grad is not None
    assert head.context_encoder.layers[0].linear1.weight.grad is not None
    assert query.grad is not None


def test_modes_read_source_memory_differently() -> None:
    torch.manual_seed(19)
    head = _head().eval()
    tokens = torch.randn(1, 1, 2, 5, 16)
    positions = torch.randn_like(tokens)
    output = head(
        *_inputs(),
        source_tokens=tokens,
        source_positions=positions,
        source_token_mask=torch.ones(1, 1, 2, 5, dtype=torch.bool),
    )

    attention = output["mode_source_attention_weights"][0, 0]
    assert not torch.allclose(attention[0], attention[1])


def test_all_invalid_source_memory_has_finite_backward() -> None:
    torch.manual_seed(23)
    head = _head().train()
    tokens = torch.randn(1, 2, 2, 5, 16, requires_grad=True)
    positions = torch.randn_like(tokens, requires_grad=True)
    output = head(
        *_inputs(1, 2),
        source_tokens=tokens,
        source_positions=positions,
        source_token_mask=torch.zeros(1, 2, 2, 5, dtype=torch.bool),
    )
    loss = output["loss_pred_scores"].square().mean()
    loss.backward()

    assert torch.isfinite(output["pred_scores"]).all()
    assert torch.isfinite(tokens.grad).all()
    assert torch.count_nonzero(tokens.grad) == 0
    assert torch.isfinite(positions.grad).all()


def test_forecast_loss_reaches_actor_source_tokenizer() -> None:
    torch.manual_seed(29)
    tokenizer = ActorAlignedSourceTokenizer(
        input_dim=8,
        query_dim=16,
        max_cav=2,
        lidar_range=[-8.0, -8.0, -3.0, 8.0, 8.0, 3.0],
        roi_grid_size=5,
        roi_scale=2.0,
    ).train()
    head = _head().train()
    query, boxes, valid, history, history_mask, time = _inputs(1, 2)
    boxes[0, 1, 0] = 2.0
    source = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
    evidence = tokenizer(
        source,
        torch.ones(1, 2, dtype=torch.bool),
        boxes,
        valid,
    )
    output = head(
        query,
        boxes,
        valid,
        history,
        history_mask,
        time,
        source_tokens=evidence["tokens"],
        source_positions=evidence["positions"],
        source_actor_positions=evidence["actor_positions"],
        source_token_mask=evidence["token_mask"],
    )
    output["loss_pred_scores"].square().mean().backward()

    assert torch.count_nonzero(source.grad) > 0
    assert tokenizer.content_projection.weight.grad is not None
    assert torch.count_nonzero(tokenizer.content_projection.weight.grad) > 0
    assert tokenizer.offset_projection[0].weight.grad is not None


def test_anchor_aware_queries_change_with_motion_state() -> None:
    torch.manual_seed(13)
    head = _head().eval()
    slow = list(_inputs())
    fast = list(_inputs())
    fast[3][..., -3:, 0] = torch.tensor([0.0, 0.4, 0.8])

    slow_scores = head(*slow)["pred_scores"]
    fast_scores = head(*fast)["pred_scores"]

    assert not torch.allclose(slow_scores, fast_scores)


def test_temporal_history_preserves_intermediate_motion_structure() -> None:
    torch.manual_seed(17)
    head = _head(temporal_history=True).eval()
    straight = torch.zeros(1, 1, 11, 7)
    curved = straight.clone()
    mask = torch.zeros(1, 1, 11, dtype=torch.bool)
    mask[..., -5:] = True
    straight[0, 0, -5:, 0] = torch.linspace(0.0, 4.0, 5)
    curved[0, 0, -5:, 0] = torch.linspace(0.0, 4.0, 5)
    curved[0, 0, -4:-1, 1] = torch.tensor([0.8, 1.2, 0.8])

    straight_feature = head._encode_temporal_history(straight, mask)
    curved_feature = head._encode_temporal_history(curved, mask)

    assert not torch.allclose(straight_feature, curved_feature)


def test_unique_prediction_head_exposes_one_supervised_layer() -> None:
    head = _head().eval()
    output = head(*_inputs())

    assert head.loss_layers == 1
    assert output["loss_pred_scores_layers"].shape[2] == 1
    assert output["loss_pred_trajs_local_layers"].shape[2] == 1
    torch.testing.assert_close(
        output["pred_trajs"], output["loss_pred_trajs_local_layers"][:, :, 0]
    )


def test_no_source_ablation_freezes_source_decoder() -> None:
    head = _head()
    head.freeze_inactive_parameters(actor_cooperative_fusion_enabled=False)

    for module in (
        head.source_query_norm,
        head.source_token_norm,
        head.source_attention,
        head.source_ffn_norm,
        head.source_ffn,
    ):
        assert not any(parameter.requires_grad for parameter in module.parameters())
    assert all(parameter.requires_grad for parameter in head.residual_head.parameters())
