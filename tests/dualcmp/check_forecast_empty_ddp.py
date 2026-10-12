"""Real Phase-B entry regression: global-empty -> rank-empty -> supervised.

Use the four-clip aligned delay window, batch=2/GPU, workers=0, epochs=4,
stop-after-epoch=3. Resume the same budget without injecting empty labels.
Only this test changes supervision masks; production datasets stay unchanged.
"""
import copy
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

from opencood.tools.dualcmp.eval_forecast_ddp import _forecast_report, _rank_zero_evaluation


def main():
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("this real-model regression requires two GPUs")
    joint = "--joint" in sys.argv
    if joint:
        sys.argv.remove("--joint")
        from opencood.tools.dualcmp import train_joint_ddp as trainer
    else:
        from opencood.tools.dualcmp import train_forecast_ddp as trainer
    joint_global_empty = "--joint-global-empty" in sys.argv
    if joint_global_empty:
        sys.argv.remove("--joint-global-empty")
        if not joint:
            raise ValueError("--joint-global-empty requires --joint")
    resume = "--resume" in sys.argv
    partial_only_empty = "--partial-only-empty" in sys.argv
    if partial_only_empty:
        sys.argv.remove("--partial-only-empty")
    calls, records, optimizer_calls = [0], [], [0]
    original_dataset = trainer.DualCMPSequenceDataset
    original_loss = trainer.DualCmpLoss.forward
    original_step = torch.optim.AdamW.step
    original_init = torch.optim.AdamW.__init__
    original_restore = trainer._restore_training_normalization
    initial_weights = []
    restored_normalizations = [0]
    from opencood.loss import forecast_association
    original_iou = forecast_association.forecast_bev_iou
    iou_parity = []

    def iou_with_cpu_cuda_parity(predicted, ground_truth):
        overlap = original_iou(predicted, ground_truth)
        if not iou_parity and overlap.numel():
            reference = original_iou(predicted.cpu(), ground_truth.cpu())
            torch.testing.assert_close(overlap.cpu(), reference, atol=1e-5, rtol=1e-5)
            iou_parity.append(float((overlap.cpu() - reference).abs().max()))
        return overlap

    class BoundaryDataset(original_dataset):
        def collate_batch_train(self, batch):
            data = super().collate_batch_train(batch)
            iteration = calls[0]
            calls[0] += 1
            empty = ((iteration == 1) if joint_global_empty else
                     ((dist.get_rank() == 0) if joint else
                      (iteration == 0 or (iteration == 1 and dist.get_rank() == 0))))
            if not resume and empty:
                for target in data["target_frames"]:
                    # Mutate only this newly collated test batch.
                    if partial_only_empty:
                        target["future_mask"][..., -1].zero_()
                        continue
                    if not joint_global_empty:
                        target["current_mask"].zero_()
                    target["future_mask"].zero_()
                    if not joint_global_empty:
                        target["velocity_mask"].zero_()
                if "association_target_frames" in data and not joint_global_empty and not partial_only_empty:
                    data["association_target_frames"][-1]["current_mask"].zero_()
            return data

    def loss_with_checks(self, outputs, targets, **kwargs):
        loss = original_loss(self, outputs, targets, **kwargs)
        counts = [float(loss[key]) for key in ("C_gmm", "C_mode", "C_mean_reg", "C_dense_gmm")]
        iteration = len(records)
        if not resume:
            empty = ((iteration == 1) if joint_global_empty else
                     ((dist.get_rank() == 0) if joint else
                      (iteration == 0 or (iteration == 1 and dist.get_rank() == 0))))
            assert (max(counts) == 0) if empty else (counts[0] > 0), counts
            # Empty loss sums may be disconnected; the production trainer
            # now adds the DDP anchor conditionally, after matching.
        if iteration == 0:
            try:
                _rank_zero_evaluation(lambda: _forecast_report([], []))
                raise AssertionError("empty evaluation must fail, not fabricate metrics")
            except RuntimeError as exc:
                error = str(exc)
            errors = [None, None]
            dist.all_gather_object(errors, error)
            assert errors[0] == errors[1] and "complete target" in error, errors
            assert _rank_zero_evaluation(lambda: {"status": "valid"}) == {"status": "valid"}
        records.append(dict(iteration=iteration, counts=counts))
        return loss

    def optimizer_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not resume and not joint:
            initial_weights.extend(p.detach().cpu().clone() for group in self.param_groups for p in group["params"])

    def optimizer_step(self, *args, **kwargs):
        if not resume and not joint and optimizer_calls[0] == 0:
            assert calls[0] == 2, "global-empty batch must not update AdamW"
            parameters = [p for group in self.param_groups for p in group["params"]]
            assert all(torch.equal(p.detach().cpu(), old) for p, old in zip(parameters, initial_weights))
            initial_weights.clear()
        forecast_parameters = []
        if joint_global_empty and optimizer_calls[0] == 1:
            forecast_parameters = [p for group in self.param_groups
                                   if group["group_name"] == "forecast" for p in group["params"]]
            assert all(p.grad is None for p in forecast_parameters)
            weights = [p.detach().clone() for p in forecast_parameters]
            states = [copy.deepcopy(self.state[p]) for p in forecast_parameters]
            assert any(states), "first supervised update must initialize AdamW momentum"
            perception_parameters = [p for group in self.param_groups
                                    if group["group_name"] != "forecast" for p in group["params"] if p.grad is not None]
            perception_weights = [p.detach().clone() for p in perception_parameters]
        optimizer_calls[0] += 1
        result = original_step(self, *args, **kwargs)
        if forecast_parameters:
            assert all(torch.equal(p.detach(), old) for p, old in zip(forecast_parameters, weights))
            for p, old_state in zip(forecast_parameters, states):
                assert self.state[p].keys() == old_state.keys()
                assert all(torch.equal(self.state[p][key], value) if torch.is_tensor(value)
                           else self.state[p][key] == value for key, value in old_state.items())
            assert any(not torch.equal(p.detach(), old) for p, old in zip(perception_parameters, perception_weights)), "perception must still learn"
        return result

    def restore_normalization(snapshot):
        original_restore(snapshot)
        assert all(torch.equal(buffer, old) for buffer, old in snapshot)
        restored_normalizations[0] += 1

    trainer.DualCMPSequenceDataset = BoundaryDataset
    trainer.DualCmpLoss.forward = loss_with_checks
    torch.optim.AdamW.__init__ = optimizer_init
    torch.optim.AdamW.step = optimizer_step
    trainer._restore_training_normalization = restore_normalization
    forecast_association.forecast_bev_iou = iou_with_cpu_cuda_parity
    trainer.main()
    assert len(iou_parity) == 1
    assert calls[0] == (2 if joint_global_empty else (1 if resume or joint else 3))
    assert optimizer_calls[0] == (2 if joint_global_empty else (1 if resume or joint else 2))
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    report = json.loads((output / "report.json").read_text())
    assert report["steps"] == (2 if joint_global_empty else (1 if joint else (3 if resume else 2)))
    assert restored_normalizations[0] == (1 if joint_global_empty or (not resume and not joint) else 0)
    if not joint:
        assert report["processed_batches"] == (4 if resume else 3)
        assert report["empty_forecast_batches"] == 1
    rank = int(os.environ["RANK"])
    name = "joint_global_empty" if joint_global_empty else ("joint" if joint else ("resume" if resume else "boundary"))
    (output / ("%s_rank%d.json" % (name, rank))).write_text(json.dumps(dict(
        status="passed", optimizer_calls=optimizer_calls[0], records=records,
        normalization_rollbacks=restored_normalizations[0],
        cpu_cuda_iou_max_error=iou_parity[0],
        partial_only_empty_batch=partial_only_empty,
        global_empty_weights_unchanged=True if joint_global_empty or (not resume and not joint) else None,
        joint_empty_keeps_perception_update=True if joint_global_empty else None), indent=2))


if __name__ == "__main__":
    main()
