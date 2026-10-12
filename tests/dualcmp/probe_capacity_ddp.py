"""Two-GPU capacity probe on four density-selected real train clips.

This decides batch capacity, not model learning or held-out quality. Selection
uses native label counts only, never model scores or trajectory errors.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset

from opencood.data_utils.dualcmp_dataloader import build_dataloader
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
from opencood.data_utils.v2v4real.datasets.dualcmp_target_builder import load_tracking_label_file
from opencood.loss.dualcmp_loss import DualCmpLoss
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES, load_checkpoint_hypes
from opencood.tools.dualcmp.eval_forecast_ddp import _forecast_rows, _forecast_report
from opencood.tools.dualcmp.online import _model_from_checkpoint, _sha256_file
from opencood.tools.train_utils import to_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, required=True, choices=(1, 2))
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("capacity probe requires both GPUs")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("capacity probe refuses artifact overwrite")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    payload = torch.load(args.checkpoint, map_location="cpu")
    h, _ = load_checkpoint_hypes(DEFAULT_HYPES, payload)
    h.update(root_dir=str(Path(os.environ["V2V4REAL_ROOT"]) / "train"),
             validate_dir=str(Path(os.environ["V2V4REAL_ROOT"]) / "train"))
    h["fusion"] = {"args": dict(stage="forecast", forecast_history_frames=10,
        future_frames=50, runtime_only=False, runtime_forbid_gt=True,
        receiver_ids=[0, 1], max_actors=128, max_cav=2, split_name="train",
        tracking_label_root=os.environ["DUALCMP_TRAIN_TRACKING_LABEL_ROOT"],
        tracking_label_layout="dmstrack_train", partner_delay_ms=100.,
        shuffle_training_points=False, forecast_coordinate_frame="current_receiver")}
    dataset = DualCMPSequenceDataset(h, train=True)
    labels = {scene: load_tracking_label_file(str(value[0]))
              for scene, value in dataset.dmstrack_train_labels.items()}
    maxima = {}
    for index, clip in enumerate(dataset.clip_records):
        # Aligned full-horizon candidates are already boundary-validated.
        key = dataset.index.receiver_frame_keys(clip.scene_id, clip.receiver_id)[clip.start_position + 10]
        reference_keys = dataset.dmstrack_train_labels[clip.scene_id][1]
        ordinal = reference_keys.index(key)
        count = len(labels[clip.scene_id].get(ordinal, {}))
        owner = (clip.scene_id, clip.receiver_id)
        if owner not in maxima or count > maxima[owner][0]:
            maxima[owner] = (count, index, key)
    selected = sorted(maxima.items(), key=lambda item: (-item[1][0], item[0]))[:4]
    selection = [{"scene": owner[0], "receiver": owner[1], "prediction_key": value[2],
                  "native_gt_count": value[0], "index": value[1]} for owner, value in selected]
    if rank == 0:
        output.mkdir(parents=True)
        (output / "selection.json").write_text(json.dumps({"selection_rule": "largest current native label count per scene/receiver; first four distinct owners",
            "selected_before_training": selection, "dataset_samples": len(dataset)}, indent=2))
    dist.barrier()
    subset = Subset(dataset, [item["index"] for item in selection])
    _, loader, _ = build_dataloader(subset, args.batch_size, True, workers=8,
        rank=rank, world_size=2, training=False, collate_fn=dataset.collate_batch_train,
        pin_memory=True, prefetch_factor=2)
    model = _model_from_checkpoint(h, args.checkpoint, device, allow_dirty_smoke=True)
    for name, parameter in model.named_parameters():
        if not name.startswith(("forecast_head.", "source_temporal_encoder.", "actor_source_tokenizer.")):
            parameter.requires_grad = False
    model.train()
    for module in (model.detector, model.tracker, model.roi, model.temporal_attention,
                   model.track_update, model.box_encoder, model.box_refinement):
        module.eval()
    model = DDP(model, device_ids=[rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    criterion = DualCmpLoss({"detector": h["loss"]["args"], "forecast_mean_weight": 5.}).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    stats = []
    for batch in loader:
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs, _ = model(batch["runtime_frames"], run_forecast=True)
        loss = criterion([outputs[-1]], batch["target_frames"])
        counts = torch.stack([loss[key] for key in ("C_gmm", "C_mode", "C_mean_reg", "C_dense_gmm")]).detach()
        dist.all_reduce(counts)
        objective = 2 * sum(loss[key] * weight / count.clamp_min(1.) for key, weight, count in zip(
            ("S_gmm", "S_mode", "S_mean_reg", "S_dense_gmm"), (1., 1., 5., 1.), counts))
        objective.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("capacity probe rejected non-finite optimizer step")
        optimizer.step()
        stats.append({"forecast_actors_per_clip": outputs[-1]["forecast"]["forecast_valid_mask"].sum(-1).cpu().tolist(),
                      "global_forecast_supervision_counts": counts.cpu().tolist(), "gradient_norm": float(grad_norm)})
    torch.cuda.synchronize(device)
    local = {"rank": rank, "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2 ** 20,
             "clips_per_second_per_rank": 2 / (time.monotonic() - started), "steps": stats}
    all_stats = [None, None]
    dist.all_gather_object(all_stats, local)
    if rank == 0:
        saved = dict(payload, model=model.module.state_dict(), optimizer=optimizer.state_dict())
        saved["provenance"] = dict(payload["provenance"], capacity_probe_only=True,
            partial_epoch=True, capacity_parent_sha256=_sha256_file(args.checkpoint), capacity_selection=selection)
        saved["step"], saved["epoch"] = len(stats), 0
        torch.save(saved, output / "checkpoint.pt")
    dist.barrier()
    del model, optimizer, outputs, loss, objective, payload, criterion, batch
    torch.cuda.empty_cache()
    restored = _model_from_checkpoint(h, output / "checkpoint.pt", device, allow_dirty_smoke=True)
    rows, coverage = [], []
    with torch.no_grad():
        for batch in loader:
            keys = [(str(scene), int(receiver), int(frame)) for (scene, receiver), frame in zip(
                batch["scene_receiver_keys"], batch["runtime_frames"][-1]["receiver_frame_idx"])]
            batch = to_device(batch, device)
            outputs, _ = restored(batch["runtime_frames"], run_forecast=True)
            values, local_coverage = _forecast_rows(outputs[-1], batch["target_frames"][-1], keys)
            rows.extend(values)
            coverage.extend(local_coverage)
    gathered = [None, None]
    dist.all_gather_object(gathered, (rows, coverage))
    if rank == 0:
        report = {"scope": "capacity only; two actual Phase-B updates for batch1 or one for batch2; no learning/quality claim",
                  "world_size": 2, "per_rank_batch": args.batch_size, "global_batch": 2 * args.batch_size,
                  "workers_per_rank": 8, "pin_memory": True, "persistent_workers": True, "prefetch_factor": 2,
                  "ranks": all_stats, "fresh_checkpoint_inference": _forecast_report(
                      [row for group in gathered for row in group[0]], [row for group in gathered for row in group[1]])}
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"global_batch": report["global_batch"], "ranks": all_stats,
                          "coverage": report["fresh_checkpoint_inference"]["coverage"]}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
