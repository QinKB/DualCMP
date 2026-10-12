"""Training-observation head BN refresh, adapted from torch.optim.swa_utils.update_bn."""
import copy
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm

from opencood.data_utils.dualcmp_dataloader import build_dataloader
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
from opencood.tools.train_utils import to_device


@torch.no_grad()
def refresh_forecast_bn(model, runtime_batches):
    """Only head BN trains; dropout, detector, tracking and parameters stay fixed.

    Like PyTorch update_bn, use a cumulative average of observed batch moments.
    Unvisited/single-row BN keeps its old buffers rather than inventing statistics.
    """
    if model.forecast_head is None:
        raise ValueError("forecast BN refresh requires a forecasting head")
    modules = [module for module in model.forecast_head.modules() if isinstance(module, _BatchNorm)]
    if not modules or any(not module.track_running_stats for module in modules):
        raise ValueError("forecast BN refresh requires running-statistics BatchNorm")
    modes = [(module, module.training) for module in model.modules()]
    momenta = {module: module.momentum for module in modules}
    original_buffers = [(buffer, buffer.clone()) for buffer in model.buffers()]
    parameters = [(parameter, parameter.detach().cpu().clone()) for parameter in model.parameters()]
    calls = dict.fromkeys(modules, 0)
    skipped = dict.fromkeys(modules, 0)
    hooks = []

    def before(module, args):
        tensor = args[0]
        rows = tensor.numel() // module.num_features
        module.training = rows > 1
        if rows <= 1:
            skipped[module] += 1
            return
        if calls[module] == 0:
            module.reset_running_stats()
        module.momentum = None
        calls[module] += 1

    completed = False
    try:
        model.eval()
        for module in modules:
            hooks.append(module.register_forward_pre_hook(before))
        batches = 0
        for runtime in runtime_batches:
            model(runtime, run_forecast=True)
            batches += 1
        if not any(calls.values()):
            raise RuntimeError("no valid forecast BN observations; calibration rejected")
        if any(not torch.equal(parameter.detach().cpu(), original) for parameter, original in parameters):
            raise RuntimeError("forecast BN refresh changed a learned parameter")
        head_buffers = {id(buffer) for module in modules for buffer in module.buffers(recurse=False)}
        for buffer, original in original_buffers:
            if id(buffer) not in head_buffers and not torch.equal(buffer, original):
                raise RuntimeError("forecast BN refresh changed a non-head buffer")
            if not torch.isfinite(buffer).all():
                raise RuntimeError("forecast BN refresh produced non-finite buffers")
        completed = True
        return {"method": "pytorch_update_bn_batchwise_cma_head_only_v1", "batches": batches,
                "bn_layers": len(modules), "updated_layers": sum(count > 0 for count in calls.values()),
                "unvisited_active_layers": sum(calls[module] == 0 and any(
                    parameter.requires_grad for parameter in module.parameters()) for module in modules),
                "single_or_empty_row_calls_skipped": sum(skipped.values()),
                "future_gt_used": False, "optimizer_steps": 0,
                "parameters_bitwise_unchanged": True, "non_head_buffers_unchanged": True,
                "dropout_enabled": False}
    finally:
        for hook in hooks:
            hook.remove()
        for module, momentum in momenta.items():
            module.momentum = momentum
        for module, mode in modes:
            module.training = mode
        if not completed:
            for buffer, original in original_buffers:
                buffer.copy_(original)


def refresh_forecast_bn_ddp(model, hypes, device, global_batch, workers):
    """Calibrate once on the whole train split, then broadcast only head buffers.

    Rank zero uses the same global calibration batch on either host. No SyncBN
    collectives enter conditional actor forwards, so empty actors cannot deadlock.
    """
    if not dist.is_initialized():
        raise RuntimeError("forecast BN refresh requires initialized DDP")
    args = hypes["fusion"]["args"]
    if args.get("split_name") != "train" or not args.get("runtime_forbid_gt"):
        raise ValueError("forecast BN refresh only accepts GT-free training observations")
    implementation = model.module if hasattr(model, "module") else model
    result = [None]
    if dist.get_rank() == 0:
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[device]):
                dataset = DualCMPSequenceDataset(copy.deepcopy(hypes), train=False)
                _, loader, _ = build_dataloader(dataset, global_batch, False,
                    workers=workers, training=False, collate_fn=dataset.collate_batch_test,
                    pin_memory=True)
                # Targets are never transferred to CUDA or passed to the model.
                runtime = (to_device(batch["runtime_frames"], device) for batch in loader)
                result[0] = refresh_forecast_bn(implementation, runtime)
                result[0].update(samples=len(dataset), global_calibration_batch=global_batch,
                                 split="train", calibration_rank=0,
                                 tail_policy="include_incomplete_final_batch")
        except Exception as error:
            result[0] = {"error": "%s: %s" % (type(error).__name__, error)}
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
    dist.broadcast_object_list(result, src=0)
    if "error" in result[0]:
        raise RuntimeError("forecast BN refresh failed: " + result[0]["error"])
    for module in implementation.forecast_head.modules():
        if isinstance(module, _BatchNorm):
            for buffer in module.buffers(recurse=False):
                dist.broadcast(buffer, src=0)
    return result[0]
