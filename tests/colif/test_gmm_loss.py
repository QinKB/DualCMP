import torch

from opencood.loss.colif_gmm_loss import nearest_mode_gmm_loss


def test_gmm_loss_handles_partial_and_all_masked_future() -> None:
    scores = torch.zeros(2, 3, requires_grad=True)
    trajectories = torch.zeros(2, 3, 4, 5, requires_grad=True)
    target = torch.zeros(2, 4, 2)
    mask = torch.tensor([[True, True, False, False], [False, False, False, False]])
    result = nearest_mode_gmm_loss(scores, trajectories, target, mask)
    loss = result["gmm"] + result["mode"]
    loss.backward()
    assert torch.isfinite(loss)
    assert trajectories.grad is not None
    assert result["mode_count"] == 1


def test_cmp_intention_goal_selects_mode_before_gmm_regression() -> None:
    scores = torch.zeros(1, 2, requires_grad=True)
    trajectories = torch.zeros(1, 2, 3, 5, requires_grad=True)
    target = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [9.0, 0.0]]])
    mask = torch.tensor([[True, True, False]])
    intentions = torch.tensor([[[2.0, 0.0], [20.0, 0.0]]])
    result = nearest_mode_gmm_loss(
        scores, trajectories, target, mask, intention_points=intentions
    )
    assert result["selected_mode"].tolist() == [0]
    # Match CMP/MTR: GMM NLL is summed over valid future timestamps and
    # normalized once per actor, not once per timestamp.
    assert float(result["gmm_count"]) == 1.0
    assert float(result["mode_count"]) == 1.0


def test_gmm_normalization_matches_cmp_actor_average() -> None:
    scores = torch.zeros(2, 1)
    trajectories = torch.zeros(2, 1, 3, 5)
    target = torch.ones(2, 3, 2)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    result = nearest_mode_gmm_loss(scores, trajectories, target, mask)
    assert float(result["gmm_count"]) == 2.0
    assert float(result["mode_count"]) == 2.0
    # Unit-variance, rho=0 gives NLL=1 for target (1,1) at each valid step.
    assert float(result["gmm"]) == 4.0
