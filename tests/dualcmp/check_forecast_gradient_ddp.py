"""Paired real-batch proof that the DDP zero anchor preserves valid updates.

Use one real forward/dropout realization. Check that frozen psm has no anchor,
loss sums/counts are unchanged, and the conditional guard returns the exact
normal objective tensor. Replay that gradient with identical AdamW state;
this is graph-identity proof, not an independent old-model training comparison.
"""
import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from opencood.data_utils.dualcmp_dataloader import build_dataloader
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
from opencood.loss.dualcmp_loss import DualCmpLoss
from opencood.tools.dualcmp.online import _model_from_checkpoint
from opencood.tools.dualcmp.train_forecast_ddp import _hypes, _ensure_forecast_ddp_backward
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES
from opencood.tools.train_utils import to_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("real gradient proof requires two GPUs")
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    hypes, _, _, _, _ = _hypes(DEFAULT_HYPES, SimpleNamespace(
        full_data=False, train_label_smoke=False, delay_smoke=True,
        forecast_representation="source_temporal_aligned", partner_delay_ms=100.,
        train_delays_ms=[100.], delay_seed=20260901))
    hypes["fusion"]["args"]["forecast_coordinate_frame"] = "current_receiver"
    payload = torch.load(args.checkpoint, map_location="cpu")
    from opencood.loss.forecast_association import LEGACY_ASSOCIATION, VALIDATED_ASSOCIATION, association_contract, associate_forecast_clip
    policy = payload["provenance"].get("forecast_association", association_contract(LEGACY_ASSOCIATION))["policy"]
    hypes["fusion"]["args"]["include_forecast_association_history"] = policy == VALIDATED_ASSOCIATION
    dataset = DualCMPSequenceDataset(hypes, train=True)
    assert len(dataset) == 4
    _, loader, _ = build_dataloader(dataset, 2, True, workers=0,
        rank=rank, world_size=2, training=False, seed=0,
        collate_fn=dataset.collate_batch_train, pin_memory=True)
    batch = to_device(next(iter(loader)), device)
    model = _model_from_checkpoint(hypes, args.checkpoint, device, allow_dirty_smoke=True)
    for name, parameter in model.named_parameters():
        if not name.startswith(("forecast_head.", "source_temporal_encoder.")):
            parameter.requires_grad = False
    model.train()
    for module in (model.detector, model.tracker, model.roi, model.temporal_attention,
                   model.track_update, model.box_encoder, model.box_refinement):
        module.eval()
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    weight = payload["provenance"]["forecast_mean_weight"]
    criterion = DualCmpLoss({"detector": hypes["loss"]["args"], "forecast_mean_weight": weight,
                             "forecast_association_policy": policy,
                             "forecast_supervision": model.forecast_supervision}).to(device)
    wrapper = DDP(model, device_ids=[rank], broadcast_buffers=False)
    fields = ("gmm", "mode", "mean_reg", "dense_gmm")
    torch.cuda.reset_peak_memory_stats(device)
    with wrapper.no_sync():
        outputs, _ = wrapper(batch["runtime_frames"], run_forecast=True)
        kwargs = {}
        if policy == VALIDATED_ASSOCIATION:
            assigned, _ = associate_forecast_clip(outputs, batch["association_target_frames"])
            kwargs["forecast_assignments"] = [assigned]
        current = criterion([outputs[-1]], batch["target_frames"], **kwargs)
        reference_output = dict(outputs[-1], psm=outputs[-1]["psm"].detach())
        reference = criterion([reference_output], batch["target_frames"], **kwargs)
        counts = torch.stack([current["C_" + field] for field in fields]).detach()
        dist.all_reduce(counts)
        assert counts[0] > 0
        assert all(torch.equal(current[key + field], reference[key + field])
                   for field in fields for key in ("S_", "C_"))
        def objective(loss):
            return 2 * sum(loss["S_" + field] * scale / count.clamp_min(1.)
                           for field, scale, count in zip(fields, (1., 1., weight, 1.), counts))
        assert not outputs[-1]["psm"].requires_grad, "normal frozen-perception psm must not carry an unconditional anchor"
        base_objective = objective(current)
        protected_objective = _ensure_forecast_ddp_backward(wrapper, base_objective, current, outputs[-1])
        assert protected_objective is base_objective, "normal supervision must preserve the exact autograd tensor"
        protected_objective.backward()
        current_grad = [p.grad.detach().cpu().clone() if p.grad is not None else None for _, p in named]
        # Exact graph identity above is stronger than independently rebuilding
        # equivalent FP32 loss graphs (whose backward reduction order may
        # differ). Replay the one gradient with identical AdamW state below.
    max_gradient_error = 0.
    newly_connected = []
    reference_grad = []
    for (name, parameter), new in zip(named, current_grad):
        old = parameter.grad
        present = torch.tensor(int(old is not None), device=device)
        dist.all_reduce(present)
        if not present:
            newly_connected.append(name)
        pair = torch.stack((new.to(device) if new is not None else torch.zeros_like(parameter),
                            old if old is not None else torch.zeros_like(parameter)))
        dist.all_reduce(pair)
        pair /= 2
        max_gradient_error = max(max_gradient_error, float((pair[0] - pair[1]).abs().max()))
        reference_grad.append(pair[1].detach().cpu().clone() if present else None)
        parameter.grad = pair[0].clone()
    baseline = [p.detach().cpu().clone() for _, p in named]
    optimizer = torch.optim.AdamW([p for _, p in named])
    optimizer.load_state_dict(copy.deepcopy(payload["optimizer"]))
    torch.nn.utils.clip_grad_norm_([p for _, p in named], 1000.)
    optimizer.step()
    current_weights = [p.detach().cpu().clone() for _, p in named]
    with torch.no_grad():
        for (_, p), initial in zip(named, baseline):
            p.copy_(initial.to(device))
    optimizer.load_state_dict(copy.deepcopy(payload["optimizer"]))
    for (_, p), gradient in zip(named, reference_grad):
        p.grad = gradient.to(device) if gradient is not None else None
    torch.nn.utils.clip_grad_norm_([p for _, p in named], 1000.)
    optimizer.step()
    max_update_error = max(float((p.detach().cpu() - new).abs().max())
                           for (_, p), new in zip(named, current_weights))
    result = dict(rank=rank, loss_sums_counts_equal=True,
                  normal_objective_graph_identity=True,
                  max_gradient_error=max_gradient_error, max_update_error=max_update_error,
                  formerly_unused_parameters=newly_connected,
                  peak_memory_mib=torch.cuda.max_memory_allocated(device) / 2 ** 20)
    gathered = [None, None]
    dist.all_gather_object(gathered, result)
    passed = all(item["max_gradient_error"] <= 1e-5 and item["max_update_error"] <= 1e-7 for item in gathered)
    if rank == 0:
        Path(args.output).write_text(json.dumps(dict(status="passed" if passed else "failed",
            scope="one paired supervised update, not learning-quality acceptance", ranks=gathered), indent=2))
    dist.barrier()
    dist.destroy_process_group()
    if not passed:
        raise AssertionError("zero anchor changed a supervised gradient/update beyond the declared tolerance")


if __name__ == "__main__":
    main()
