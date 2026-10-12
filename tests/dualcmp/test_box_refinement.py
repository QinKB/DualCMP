"""Sparse4D@249ffbb box-code contract."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch
from mmcv.cnn import Scale
from mmengine.model import bias_init_with_prob
from scipy.optimize import linear_sum_assignment

from opencood.models.dualcmp.box_refinement import decode_box, encode_box
from opencood.models.dualcmp import box_refinement as local
from opencood.loss.dualcmp_loss import DualCmpLoss, update_identity_bindings


def _upstream():
    """Execute exact upstream symbols, isolating only registry/package imports."""
    root = Path(__file__).resolve().parents[3] / "Reference/Code/Sparse4D/projects/mmdet3d_plugin"
    namespace = dict(torch=torch, nn=torch.nn, Linear=torch.nn.Linear,
                     Sequential=torch.nn.Sequential, BaseModule=torch.nn.Module,
                     Scale=Scale, bias_init_with_prob=bias_init_with_prob)
    constants = root / "core/box3d.py"
    exec(compile(constants.read_text(), str(constants), "exec"), namespace)
    for relative, names in (
        ("models/blocks.py", {"linear_relu_ln"}),
        ("models/detection3d/detection3d_blocks.py", {"SparseBox3DEncoder", "SparseBox3DRefinementModule"}),
    ):
        path = root / relative
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        for node in tree.body:
            node.decorator_list = []
        exec(compile(tree, str(path), "exec"), namespace)
    path = root / "models/detection3d/target.py"
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "SparseBox3DTarget")
    tree = ast.parse("")
    tree.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in ("encode_reg_target", "_cls_cost", "_box_cost")]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def test_recurrent_update_matches_upstream_and_bounds_cache():
    source = _upstream()
    update = torch.nn.Sequential(*local.linear_relu_ln(256, 1, 1))
    reference = torch.nn.Sequential(*source['linear_relu_ln'](256, 1, 1))
    reference.load_state_dict(update.state_dict(), strict=True)
    feature = torch.randn(2, 4, 256, requires_grad=True)
    expected_input = feature.detach().clone().requires_grad_()
    actual, expected = update(feature), reference(expected_input)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(feature.grad, expected_input.grad, rtol=0, atol=0)
    # Explicit amplifying recurrence: short clips alone would miss this bug.
    with torch.no_grad():
        cached = feature.detach()
        for _ in range(400):
            cached = update(cached * 3 + .1)
            assert torch.isfinite(cached).all()
            assert cached.abs().max() <= 16.01


def test_upstream_encoder_refinement_outputs_and_gradients():
    source = _upstream()
    torch.manual_seed(17)
    encoder = local.SparseBox3DEncoder()
    reference_encoder = source["SparseBox3DEncoder"](
        [128, 32, 32, 64], vel_dims=2, mode="cat", output_fc=False,
        in_loops=1, out_loops=4)
    reference_encoder.load_state_dict(encoder.state_dict(), strict=True)
    head = local.SparseBox3DRefinementModule()
    reference_head = source["SparseBox3DRefinementModule"](
        output_dim=10, num_cls=1, refine_yaw=True)
    reference_head.load_state_dict({("layers.11.scale" if key == "scale" else key): value
                                    for key, value in head.state_dict().items()}, strict=True)
    box = torch.tensor([[[3., 2., 0., 1.5, 2., 4., 3.13]]]).expand(2, 3, -1)
    anchor = encode_box(box, torch.ones(2, 3, 2) * 10)
    feature = torch.randn(2, 3, 256, requires_grad=True)
    reference_feature = feature.detach().clone().requires_grad_()
    interval = torch.tensor([.1, .15])
    code, logits = head(feature, anchor, encoder(anchor), interval)
    expected_code, expected_logits, _ = reference_head(
        reference_feature, anchor, reference_encoder(anchor), interval)
    torch.testing.assert_close(code, expected_code, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(logits, expected_logits.squeeze(-1), atol=1e-6, rtol=1e-6)
    (code.square().mean() + logits.square().mean()).backward()
    (expected_code.square().mean() + expected_logits.square().mean()).backward()
    torch.testing.assert_close(feature.grad, reference_feature.grad, atol=1e-6, rtol=1e-5)
    for (_, value), (_, reference_value) in zip(encoder.named_parameters(), reference_encoder.named_parameters()):
        torch.testing.assert_close(value.grad, reference_value.grad, atol=1e-6, rtol=1e-5)


def test_upstream_encoding_assignment_and_masked_l1():
    source = _upstream()
    box = torch.tensor([[[0., 0., 0., 1.5, 2., 4., 3.13],
                         [8., 2., 0., 1.5, 2., 4., -3.13]]])
    velocity = torch.tensor([[[10., -3.], [2., 1.]]])
    raw = torch.cat((box[..., [0, 1, 2, 4, 5, 3, 6]], velocity), -1)
    expected = source["encode_reg_target"](None, list(raw))[0]
    torch.testing.assert_close(encode_box(box, velocity)[0], expected)
    code = (encode_box(box, velocity)[:, [1, 0]] + .1).requires_grad_()
    logits = torch.tensor([[2., -.7]], requires_grad=True)
    cfg = SimpleNamespace(alpha=.25, gamma=2., eps=1e-12, cls_weight=2., box_weight=.25,
                          reg_weights=[2., 2., 2., .5, .5, .5, 0., 0., 0., 0.])
    cls_cost = source["_cls_cost"](cfg, logits[..., None], [torch.zeros(2, dtype=torch.long)])[0]
    box_cost = source["_box_cost"](cfg, code, [expected], [torch.ones_like(expected)])[0]
    left, right = linear_sum_assignment((cls_cost + box_cost).detach().numpy())
    output = {"psm": torch.zeros(1, requires_grad=True), "state": SimpleNamespace(
        box=box, runtime_id=torch.tensor([[1, 2]]), valid_mask=torch.ones(1, 2, dtype=torch.bool), score=torch.ones(1, 2)),
        "track_box_code": code, "track_exist_logits": logits}
    target = {"current_mask": torch.ones(1, 2, dtype=torch.bool), "current_boxes": box,
              "actor_ids": torch.tensor([[11, 12]]), "current_velocity": velocity,
              "velocity_mask": torch.tensor([[True, False]]), "fused_label_dict": {}}
    assigned, _ = update_identity_bindings(output, target, [{}])
    assert assigned[0, left].tolist() == right.tolist()
    criterion = DualCmpLoss({"detector": {"cls_weight": 1., "reg": 1.}})
    criterion.detector_loss.forward = lambda prediction, labels: prediction["psm"].sum() * 0
    losses = criterion([output], [target])
    paired = expected[assigned[0]]
    weights = code.new_tensor([2., 2., 2., 1., 1., 1., 1., 1., 1., 1.]).expand(2, -1).clone()
    weights[:, 8:] *= target["velocity_mask"][0, assigned[0], None]
    expected_loss = .25 * ((code[0] - paired).abs() * weights).sum() / 2
    torch.testing.assert_close(losses["L_box"] + losses["L_velocity"], expected_loss)


def test_sparse4d_encode_decode_and_velocity_gradient():
    box = torch.tensor([[[1., 2., 3., 1.5, 2., 4., .4]]], requires_grad=True)
    velocity = torch.tensor([[[3., -2.]]], requires_grad=True)
    code = encode_box(box, velocity)
    decoded, decoded_velocity = decode_box(code)
    torch.testing.assert_close(decoded, box, atol=1e-6, rtol=0)
    torch.testing.assert_close(decoded_velocity, velocity, atol=1e-6, rtol=0)
    (code.square().sum()).backward()
    assert torch.isfinite(box.grad).all() and torch.isfinite(velocity.grad).all()
