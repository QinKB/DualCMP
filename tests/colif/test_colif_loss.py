import torch

from opencood.loss.colif_loss import ColifLoss


def _frame(actor_x, actor_id=7):
    outputs = {
        "exist_logits": torch.zeros(1, 2, requires_grad=True),
        "boxes": torch.tensor(
            [[[actor_x, 0, 0, 1.5, 1.6, 3.9, 0], [20, 0, 0, 1.5, 1.6, 3.9, 0]]],
            dtype=torch.float32,
            requires_grad=True,
        ),
        "velocity": torch.zeros(1, 2, 2, requires_grad=True),
        "aux_exist_logits": torch.zeros(1, 1, 2, requires_grad=True),
        "aux_boxes": torch.tensor(
            [[[[actor_x, 0, 0, 1.5, 1.6, 3.9, 0], [20, 0, 0, 1.5, 1.6, 3.9, 0]]]],
            dtype=torch.float32,
            requires_grad=True,
        ),
        "auxiliary_psm": torch.zeros(1, 2, 1, 1, requires_grad=True),
        "auxiliary_rm": torch.zeros(1, 14, 1, 1, requires_grad=True),
        "forecast_valid_mask": torch.zeros(1, 2, dtype=torch.bool),
    }
    targets = {
        "current_mask": torch.tensor([[True]]),
        "actor_ids": torch.tensor([[actor_id]]),
        "current_boxes": torch.tensor(
            [[[actor_x, 0, 0, 1.5, 1.6, 3.9, 0]]], dtype=torch.float32
        ),
        "current_velocity": torch.zeros(1, 1, 2),
        "future_xy": torch.zeros(1, 1, 50, 2),
        "future_mask": torch.ones(1, 1, 50, dtype=torch.bool),
        "fused_label_dict": {
            "pos_equal_one": torch.tensor([[[[1.0, 0.0]]]]),
            "neg_equal_one": torch.tensor([[[[0.0, 1.0]]]]),
            "targets": torch.zeros(1, 1, 1, 14),
        },
    }
    return outputs, targets


def test_sequential_match_retains_slot_and_backpropagates():
    criterion = ColifLoss({})
    outputs1, targets1 = _frame(0.0)
    loss1, count1, state = criterion.forward_frame(outputs1, {}, targets1, None)
    assigned_slot = int(torch.where(state["actor_ids"][0] == 7)[0][0])
    outputs2, targets2 = _frame(20.0)
    # Move the other query onto the target; persistent matching must still
    # retain the actor's original slot rather than globally rematching it.
    outputs2["boxes"].data[0, 1 - assigned_slot, 0] = 20.0
    aux = {}
    loss2, count2, state2 = criterion.forward_frame(outputs2, aux, targets2, state)
    assert aux["matched_query_target_pairs"][0][1] == assigned_slot
    assert int(state2["actor_ids"][0, assigned_slot]) == 7
    total = criterion.weighted_sum(loss1, count1, {"phase": 1})
    total = total + criterion.weighted_sum(loss2, count2, {"phase": 1})
    total.backward()
    assert torch.isfinite(total)
    assert outputs1["boxes"].grad is not None


def test_supervision_identity_is_released_when_runtime_slot_dies():
    criterion = ColifLoss({})
    outputs1, targets1 = _frame(0.0)
    _, _, state = criterion.forward_frame(outputs1, {}, targets1, None)
    old_slot = int(torch.where(state["actor_ids"][0] == 7)[0][0])
    new_slot = 1 - old_slot

    outputs2, targets2 = _frame(20.0)
    outputs2["boxes"].data[0, old_slot, 0] = 0.0
    outputs2["boxes"].data[0, new_slot, 0] = 20.0
    outputs2["active_mask"] = torch.zeros(1, 2, dtype=torch.bool)
    aux = {}
    _, _, state2 = criterion.forward_frame(outputs2, aux, targets2, state)
    assert aux["matched_query_target_pairs"][0][1] == new_slot
    assert int(state2["actor_ids"][0, new_slot]) == 7
    assert int(state2["actor_ids"][0, old_slot]) == -1


def test_phase2_weighted_sum_does_not_require_frozen_aux_detection_losses():
    criterion = ColifLoss({})
    names = (
        "existence", "box", "velocity", "aux_existence", "aux_box",
        "gmm", "mode", "dense_gmm",
    )
    loss_dict = {name: torch.ones((), requires_grad=True) for name in names}
    count_dict = {name: torch.ones(()) for name in names}
    total = criterion.weighted_sum(loss_dict, count_dict, {"phase": 2})
    total.backward()
    assert torch.isfinite(total)
    assert all(
        loss_dict[name].grad is not None
        for name in ("gmm", "mode", "dense_gmm")
    )
    assert all(
        loss_dict[name].grad is None
        for name in ("existence", "box", "velocity", "aux_existence", "aux_box")
    )


def test_newborn_match_does_not_force_distant_local_reference_to_target():
    criterion = ColifLoss({"matcher": {"max_center_distance": 2.0}})
    outputs, targets = _frame(0.0)
    outputs["boxes"].data[..., 0] = 20.0
    outputs["birth_candidate_mask"] = torch.ones(1, 2, dtype=torch.bool)
    aux = {}
    _, counts, state = criterion.forward_frame(outputs, aux, targets, None)
    assert aux["matched_query_target_pairs"] == []
    assert counts["matched"] == 0
    assert (state["actor_ids"] < 0).all()


def test_first_supervised_frame_can_bind_query_created_during_burnin():
    criterion = ColifLoss({"matcher": {"max_center_distance": 2.0}})
    outputs, targets = _frame(0.0)
    outputs["active_mask"] = torch.tensor([[True, False]])
    outputs["birth_candidate_mask"] = torch.zeros(1, 2, dtype=torch.bool)
    aux = {}
    _, counts, state = criterion.forward_frame(outputs, aux, targets, None)
    assert counts["matched"] == 1
    assert aux["matched_query_target_pairs"][0][1] == 0
    assert state["actor_ids"][0, 0] == 7


def test_dense_context_future_loss_reaches_packed_mtr_prediction():
    criterion = ColifLoss({})
    outputs, targets = _frame(0.0)
    outputs["track_boxes"] = outputs["boxes"]
    dense = torch.zeros(1, 1, 50, 5, requires_grad=True)
    outputs.update(
        {
            "packed_entries": ((0, 0),),
            "packed_context_slots": torch.tensor([[0]]),
            "packed_dense_trajs_local": dense,
            "packed_center_heading": torch.zeros(1),
        }
    )
    # Make the future nonzero so the dense GMM receives a real regression
    # signal.  This target remains criterion-only.
    targets["future_xy"][0, 0, :, 0] = torch.arange(1, 51) * 0.1
    losses, counts, _ = criterion.forward_frame(outputs, {}, targets, None)
    assert counts["dense_gmm"] == 1
    assert torch.isfinite(losses["dense_gmm"])
    losses["dense_gmm"].backward()
    assert dense.grad is not None
    assert dense.grad.abs().sum() > 0
