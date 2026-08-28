import math

import torch

from opencood.models.colif.actor_source_tokenizer import (
    ActorAlignedSourceTokenizer,
)
from opencood.models.colif.intention_query_forecast import (
    IntentionQueryForecastHead,
)


def _head(temporal_history=True, decoder_layers=2):
    return IntentionQueryForecastHead(
        16,
        {
            "history_frames": 11,
            "future_frames": 50,
            "frame_period_s": 0.1,
            "min_forecast_history_steps": 2,
            "modes": 6,
            "d_model": 32,
            "map_d_model": 16,
            "decoder_layers": decoder_layers,
            "attention_heads": 4,
            "source_attention_heads": 4,
            "source_attention_ffn": 64,
            "dropout": 0.0,
            "temporal_history": temporal_history,
            "history_encoder_layers": 1,
            "history_encoder_heads": 4,
            "history_encoder_ffn": 32,
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


def _evidence(batch=1, queries=1, sources=2, patches=5):
    tokens = torch.randn(batch, queries, sources, patches, 16)
    positions = torch.randn_like(tokens)
    mask = torch.ones(batch, queries, sources, patches, dtype=torch.bool)
    return tokens, positions, mask


def _forward(head, inputs, source=None, consensus=None):
    source = source or (None, None, None)
    consensus = consensus or (None, None, None)
    return head(
        *inputs,
        consensus_tokens=consensus[0],
        consensus_positions=consensus[1],
        consensus_token_mask=consensus[2],
        source_tokens=source[0],
        source_positions=source[1],
        source_token_mask=source[2],
    )


def test_first_observation_creates_no_long_horizon_forecast():
    head = _head().eval()
    query, boxes, valid, history, mask, time = _inputs()
    mask.zero_()
    mask[..., -1] = True

    output = head(query, boxes, valid, history, mask, time)

    assert not output["forecast_valid_mask"].any()
    assert torch.count_nonzero(output["pred_trajs"]) == 0
    assert output["loss_pred_scores"].shape == (1, 1, 64)


def test_fixed_64_intentions_train_and_nms_six_modes():
    torch.manual_seed(3)
    head = _head().eval()
    output = head(*_inputs())
    intention_buffer = getattr(
        head.decoder,
        head.decoder.intention_points["TYPE_VEHICLE"],
    )

    assert head.raw_modes == 64
    assert output["pred_trajs"].shape == (1, 1, 6, 50, 5)
    assert output["loss_pred_scores_layers"].shape == (1, 1, 2, 64)
    assert output["loss_pred_trajs_local_layers"].shape == (
        1,
        1,
        2,
        64,
        50,
        5,
    )
    torch.testing.assert_close(
        output["loss_intention_points"][0, 0],
        intention_buffer,
    )
    torch.testing.assert_close(
        output["loss_intention_points_layers"][0, 0, 0],
        intention_buffer,
    )
    torch.testing.assert_close(
        output["loss_intention_points_layers"][0, 0, 1],
        intention_buffer,
    )


def test_each_decoder_layer_updates_dynamic_query_center():
    torch.manual_seed(5)
    head = _head(decoder_layers=2).eval()
    observed_centers = []
    original = head.decoder.apply_cross_attention

    def record_center(*args, **kwargs):
        observed_centers.append(kwargs["dynamic_query_center"].detach().clone())
        return original(*args, **kwargs)

    head.decoder.apply_cross_attention = record_center
    output = head(*_inputs())

    assert len(observed_centers) == 2
    torch.testing.assert_close(
        observed_centers[0].permute(1, 0, 2),
        output["loss_intention_points"].reshape(1, 64, 2),
    )
    torch.testing.assert_close(
        observed_centers[1].permute(1, 0, 2),
        output["loss_pred_trajs_local_layers"][0, 0, 0, :, -1, :2].unsqueeze(0),
    )


def test_source_tokens_change_intention_query_predictions():
    torch.manual_seed(7)
    head = _head().eval()
    inputs = _inputs()
    source = _evidence()
    first = _forward(head, inputs, source=source)
    changed = tuple(item.clone() for item in source)
    changed[0][:, :, 1, :, 0] += 20.0
    second = _forward(head, inputs, source=changed)

    assert not torch.allclose(
        first["loss_pred_scores"],
        second["loss_pred_scores"],
    )
    assert first["mode_source_attention_weights"].shape == (1, 1, 6, 2, 5)
    assert first["source_attention_weights"].shape == (1, 1, 2, 5)


def test_masked_source_cannot_change_prediction():
    torch.manual_seed(9)
    head = _head().eval()
    inputs = _inputs()
    source = list(_evidence())
    source[2][:, :, 1] = False
    first = _forward(head, inputs, source=tuple(source))
    changed = tuple(item.clone() for item in source)
    changed[0][:, :, 1] += 10_000.0
    second = _forward(head, inputs, source=changed)

    torch.testing.assert_close(first["pred_scores"], second["pred_scores"])
    torch.testing.assert_close(first["pred_trajs"], second["pred_trajs"])


def test_source_permutation_preserves_prediction():
    torch.manual_seed(11)
    head = _head().eval()
    inputs = _inputs()
    source = _evidence()
    original = _forward(head, inputs, source=source)
    swapped_source = (
        source[0].flip(2),
        source[1].flip(2),
        source[2].flip(2),
    )
    swapped = _forward(head, inputs, source=swapped_source)

    torch.testing.assert_close(
        original["pred_scores"],
        swapped["pred_scores"],
        atol=5e-6,
        rtol=5e-6,
    )
    torch.testing.assert_close(
        original["pred_trajs"],
        swapped["pred_trajs"],
        atol=5e-6,
        rtol=5e-6,
    )
    torch.testing.assert_close(
        original["source_attention_weights"].flip(2),
        swapped["source_attention_weights"],
        atol=5e-6,
        rtol=5e-6,
    )


def test_consensus_and_source_are_independent_motion_conditioned_inputs():
    torch.manual_seed(13)
    head = _head().eval()
    inputs = _inputs()
    consensus = _evidence(sources=1)
    source = _evidence(sources=2)
    baseline = _forward(head, inputs, source=source, consensus=consensus)
    changed_consensus = tuple(item.clone() for item in consensus)
    changed_consensus[0][..., 0] += 20.0
    consensus_output = _forward(
        head,
        inputs,
        source=source,
        consensus=changed_consensus,
    )
    changed_source = tuple(item.clone() for item in source)
    changed_source[0][:, :, 1, :, 0] += 20.0
    source_output = _forward(
        head,
        inputs,
        source=changed_source,
        consensus=consensus,
    )

    assert not torch.allclose(
        baseline["loss_pred_scores"],
        consensus_output["loss_pred_scores"],
    )
    assert not torch.allclose(
        baseline["loss_pred_scores"],
        source_output["loss_pred_scores"],
    )
    assert baseline["mode_consensus_attention_weights"].shape == (
        1,
        1,
        6,
        5,
    )
    assert baseline["mode_source_attention_weights"].shape == (
        1,
        1,
        6,
        2,
        5,
    )


def test_dual_path_count_uses_variance_preserving_pre_norm_scaling():
    torch.manual_seed(15)
    head = _head().eval()
    inputs = _inputs()
    evidence = _evidence(sources=1)
    captured = []

    def capture_input(_module, args):
        captured.append(args[0].detach().clone())

    hook = head.attention_output_norms[0].register_forward_pre_hook(
        capture_input
    )
    _forward(head, inputs)
    base = captured.pop()
    _forward(head, inputs, consensus=evidence)
    single = captured.pop()
    _forward(head, inputs, consensus=evidence, source=evidence)
    dual = captured.pop()
    hook.remove()

    torch.testing.assert_close(
        dual - base,
        (single - base) * math.sqrt(2.0),
        atol=1e-5,
        rtol=1e-5,
    )


def test_forecast_loss_reaches_source_decoder_and_tokens():
    torch.manual_seed(17)
    head = _head().train()
    inputs = _inputs(1, 2)
    source = list(_evidence(1, 2))
    consensus = list(_evidence(1, 2, sources=1))
    source[0].requires_grad_()
    consensus[0].requires_grad_()
    output = _forward(
        head,
        inputs,
        source=tuple(source),
        consensus=tuple(consensus),
    )
    loss = (
        output["loss_pred_trajs_local_layers"][..., :2].square().mean()
        + output["loss_pred_scores_layers"].square().mean()
    )
    loss.backward()

    assert source[0].grad is not None
    assert torch.count_nonzero(source[0].grad) > 0
    assert consensus[0].grad is not None
    assert torch.count_nonzero(consensus[0].grad) > 0
    assert head.shared_attentions[0].in_proj_weight.grad is not None
    assert head.dual_ffns[0][0].weight.grad is not None
    assert head.decoder.motion_reg_heads[0][0].weight.grad is not None
    assert head.history_encoder.layers[0].linear1.weight.grad is not None


def test_all_invalid_source_memory_is_finite_and_has_zero_source_gradient():
    torch.manual_seed(19)
    head = _head().train()
    source = list(_evidence(1, 2))
    source[0].requires_grad_()
    source[2].zero_()
    output = _forward(head, _inputs(1, 2), source=tuple(source))
    output["loss_pred_scores"].square().mean().backward()

    assert torch.isfinite(output["pred_scores"]).all()
    assert torch.isfinite(source[0].grad).all()
    assert torch.count_nonzero(source[0].grad) == 0


def test_forecast_loss_reaches_actor_source_tokenizer():
    torch.manual_seed(23)
    tokenizer = ActorAlignedSourceTokenizer(
        input_dim=8,
        query_dim=16,
        max_cav=2,
        lidar_range=[-8.0, -8.0, -3.0, 8.0, 8.0, 3.0],
        roi_grid_size=5,
        roi_scale=2.0,
    ).train()
    head = _head().train()
    inputs = list(_inputs(1, 2))
    inputs[1][0, 1, 0] = 2.0
    source_bev = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
    evidence = tokenizer(
        source_bev,
        torch.ones(1, 2, dtype=torch.bool),
        inputs[1],
        inputs[2],
    )
    output = head(
        *inputs,
        source_tokens=evidence["tokens"],
        source_positions=evidence["positions"],
        source_token_mask=evidence["token_mask"],
    )
    output["loss_pred_scores_layers"].square().mean().backward()

    assert torch.count_nonzero(source_bev.grad) > 0
    assert tokenizer.content_projection.weight.grad is not None
    assert tokenizer.offset_projection[0].weight.grad is not None


def test_temporal_history_preserves_intermediate_motion_structure():
    torch.manual_seed(29)
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


def test_no_source_ablation_freezes_source_decoder():
    head = _head()
    head.freeze_inactive_parameters(actor_cooperative_fusion_enabled=False)

    for module in (
        head.attention_memory_projection,
        head.attention_query_norms,
        head.shared_attentions,
        head.attention_output_norms,
        head.dual_ffn_norms,
        head.dual_ffns,
    ):
        assert not any(parameter.requires_grad for parameter in module.parameters())
    assert all(
        parameter.requires_grad
        for parameter in head.decoder.motion_reg_heads.parameters()
    )
