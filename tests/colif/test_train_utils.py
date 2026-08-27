import torch

from opencood.tools.colif_train_utils import (
    build_colif_optimizer,
    configure_trainable_phase,
    load_colif_weights,
    set_colif_train_mode,
)


class _CoBEVT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pillar_vfe = torch.nn.Linear(2, 2)
        self.backbone = torch.nn.Linear(2, 2)
        self.fusion_net = torch.nn.Linear(2, 2)
        self.cls_head = torch.nn.Linear(2, 1)
        self.reg_head = torch.nn.Linear(2, 1)
        self.shrink_flag = False
        self.compression = False


class _Forecast(torch.nn.Linear):
    def __init__(self):
        super().__init__(2, 2)
        self.enabled = False

    def freeze_unused_map_parameters(self):
        return None

    def freeze_inactive_parameters(
        self, actor_cooperative_fusion_enabled=True
    ):
        self.actor_cooperative_fusion_enabled = bool(
            actor_cooperative_fusion_enabled
        )


class _Prior(torch.nn.Module):
    MODES = (
        "none",
        "association",
        "propagation_detached",
        "propagation_bptt",
        "detached",
        "recurrent_bptt",
    )

    def __init__(self):
        super().__init__()
        self.mode = "none"


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cobevt = _CoBEVT()
        self.query_decoder = torch.nn.Linear(2, 2)
        self.temporal_memory = torch.nn.Linear(2, 2)
        self.memory_scale = torch.nn.Parameter(torch.tensor(0.1))
        self.forecast_head = _Forecast()
        self.actor_source_tokenizer = torch.nn.Linear(2, 2)
        self.actor_cooperative_fusion_enabled = True
        self.future_adapter = torch.nn.Linear(2, 2)
        self.future_adapter.mode = "none"
        self.phase2_feedback_mode = "none"
        self.forecast_motion_prior = _Prior()
        self.phase2_future_prior_mode = "none"


def test_optimizer_groups_cover_each_parameter_once():
    model = _Model()
    optimizer = build_colif_optimizer(
        {
            "optimizer": {
                "core_method": "AdamW",
                "lr": 1e-3,
                "trunk_lr_scale": 0.1,
                "forecast_lr_scale": 0.5,
            }
        },
        model,
    )
    assert [group["group_name"] for group in optimizer.param_groups] == [
        "cobevt", "forecast", "colif_heads"
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 5e-4, 1e-3]
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    assert {id(p) for p in optimized} == {id(p) for p in model.parameters()}


def test_two_phase_configuration_uses_spatial_prior_and_disables_latent_feedback():
    model = _Model()
    phase1 = configure_trainable_phase(model, 1)
    assert phase1["future_adapter_mode"] == "none"
    assert phase1["future_prior_mode"] == "none"
    assert not model.forecast_head.enabled
    assert not any(p.requires_grad for p in model.forecast_head.parameters())

    phase2 = configure_trainable_phase(model, 2)
    assert phase2["forecast_enabled"]
    assert phase2["future_adapter_mode"] == "none"
    assert phase2["future_prior_mode"] == "none"
    assert all(p.requires_grad for p in model.forecast_head.parameters())
    assert all(
        p.requires_grad for p in model.actor_source_tokenizer.parameters()
    )
    assert not any(p.requires_grad for p in model.future_adapter.parameters())
    assert not any(p.requires_grad for p in model.query_decoder.parameters())
    set_colif_train_mode(model, 2)
    assert not model.cobevt.training
    assert not model.query_decoder.training
    assert not model.temporal_memory.training
    assert not any(p.requires_grad for p in model.temporal_memory.parameters())
    assert model.forecast_head.training
    assert model.actor_source_tokenizer.training
    assert not model.future_adapter.training


def test_phase2_does_not_train_disabled_actor_cooperative_fusion():
    model = _Model()
    model.actor_cooperative_fusion_enabled = False
    configure_trainable_phase(model, 2)

    assert not any(
        parameter.requires_grad
        for parameter in model.actor_source_tokenizer.parameters()
    )
    assert model.forecast_head.actor_cooperative_fusion_enabled is False


def test_cross_phase_load_is_strict_and_complete(tmp_path):
    source = _Model()
    for parameter in source.parameters():
        torch.nn.init.constant_(parameter, 7.0)
    checkpoint = tmp_path / "phase1.pth"
    torch.save({"model_state": source.state_dict(), "phase": 1, "epoch": 0}, checkpoint)
    target = _Model()
    report = load_colif_weights(target, str(checkpoint))
    assert all(torch.all(parameter == 7.0) for parameter in target.parameters())
    assert report["source_phase"] == 1
    assert not report["retained_current_initialization"]


def test_phase2_keeps_fresh_forecast_and_future_adapter_initialization(tmp_path):
    source = _Model()
    for parameter in source.parameters():
        torch.nn.init.constant_(parameter, 7.0)
    checkpoint = tmp_path / "phase1.pth"
    torch.save({"model_state": source.state_dict(), "phase": 1, "epoch": 0}, checkpoint)
    target = _Model()
    configure_trainable_phase(target, 2)
    forecast_before = target.forecast_head.weight.detach().clone()
    adapter_before = target.future_adapter.weight.detach().clone()
    cooperative_before = target.actor_source_tokenizer.weight.detach().clone()
    report = load_colif_weights(target, str(checkpoint))
    torch.testing.assert_close(target.forecast_head.weight, forecast_before)
    torch.testing.assert_close(target.future_adapter.weight, adapter_before)
    torch.testing.assert_close(
        target.actor_source_tokenizer.weight, cooperative_before
    )
    assert torch.all(target.query_decoder.weight == 7.0)
    assert report["retained_current_initialization"]
