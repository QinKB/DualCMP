"""Real frozen-perception inputs for paired forecast learning experiments.

No target is passed into the runtime or cache-input construction. Targets are
stored separately for the unchanged repository loss/evaluator. Both GPUs are
required, even though this stage does not optimize parameters.
"""
import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Subset

from audit_task_fitness_ddp import capture_forward
from opencood.data_utils.dualcmp_dataloader import build_dataloader
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
from opencood.models.dualcmp.forecast_frames import align_source_record
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES, load_checkpoint_hypes
from opencood.tools.dualcmp.eval_forecast_ddp import _forecast_report, _forecast_rows
from opencood.tools.dualcmp.online import _model_from_checkpoint, _sha256_file
from opencood.tools.train_utils import to_device


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    if isinstance(value, SimpleNamespace):
        return SimpleNamespace(**cpu_tree(vars(value)))
    return value


def make_dataset(hypes, split):
    h = copy.deepcopy(hypes)
    root = str(Path(os.environ["V2V4REAL_ROOT"]) / split)
    h.update(root_dir=root, validate_dir=root, test_dir=root)
    train = split == "train"
    labels = os.environ["DUALCMP_TRAIN_TRACKING_LABEL_ROOT" if train else "DUALCMP_TRACKING_LABEL_ROOT"]
    args = dict(stage="forecast", forecast_history_frames=10, future_frames=50,
                runtime_only=False, runtime_forbid_gt=True, receiver_ids=[0, 1],
                max_actors=128, max_cav=2, split_name=split,
                tracking_label_root=labels, partner_delay_ms=100.,
                shuffle_training_points=False, forecast_coordinate_frame="current_receiver")
    if train:
        args["tracking_label_layout"] = "dmstrack_train"
    h["fusion"] = {"args": args}
    return DualCMPSequenceDataset(h, train=False), h


def select_indices(count, size, midpoint=False, excluded=()):
    if midpoint:
        values = np.floor((np.arange(size) + .5) * count / size).astype(int)
    else:
        values = np.linspace(0, count - 1, size, dtype=int)
    selected = []
    blocked = set(excluded)
    for value in values.tolist():
        distance = 0
        while True:
            candidates = (value,) if distance == 0 else (value - distance, value + distance)
            available = [x for x in candidates if 0 <= x < count and x not in blocked]
            if available:
                selected.append(available[0])
                blocked.add(available[0])
                break
            distance += 1
            if distance >= count:
                raise ValueError("not enough non-overlapping indices")
    if len(set(selected)) != size:
        raise ValueError("representative selection has duplicates")
    return selected


def fused_history_inputs(model, outputs, final_state, slots, source_inputs):
    records = []
    for output in outputs:
        state = output["state"]
        mask = torch.ones(state.box.shape[0], 1, dtype=torch.bool, device=state.box.device)
        patch = model.actor_source_tokenizer(output["fused_bev"][:, None].float(), mask,
                                            state.box, state.active_mask)
        record = dict(positions=patch["positions"], reference_boxes=state.box,
                      receiver_pose=state.receiver_pose,
                      source_to_receiver=torch.eye(4, device=state.box.device)[None, None].expand(state.box.shape[0], 1, -1, -1))
        positions, _ = align_source_record(record, final_state)
        same = (state.runtime_id == final_state.runtime_id) & (final_state.runtime_id >= 0)
        valid = patch["token_mask"] & same[:, :, None, None] & final_state.active_mask[:, :, None, None]
        records.append((patch["tokens"], positions, valid))
    values = [torch.stack([record[i] for record in records], dim=3)[:, slots] for i in range(3)]
    # Fused capture time is the receiver's runtime time, not the source LiDAR
    # stamp (which can differ by a millisecond). Match the actual fused branch.
    receiver_times = torch.stack([output["state"].receiver_time_s for output in outputs], dim=-1)
    age = (final_state.receiver_time_s[:, None] - receiver_times)[:, None, None]
    age = age.expand(-1, slots.numel(), 1, -1)
    # Fused tokens have one receiver-centric observer, not invented CAV origins.
    return tuple(values + [age, source_inputs[4][:, slots, :1]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude-scenes-manifest")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("information cache requires both GPUs")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite information cache")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    payload = torch.load(args.checkpoint, map_location="cpu")
    hypes, _ = load_checkpoint_hypes(DEFAULT_HYPES, payload)
    model = _model_from_checkpoint(hypes, args.checkpoint, device)
    if rank == 0:
        output_dir.mkdir(parents=True)
    dist.barrier()
    manifest = {"teacher_checkpoint": args.checkpoint,
                "teacher_sha256": _sha256_file(args.checkpoint),
                "teacher_provenance": payload["provenance"],
                "perception_frozen": True, "world_size": 2,
                "selection": {}, "cache_replay": {},
                "scope": "frozen perception inputs, not training acceptance"}
    excluded_scenes = set()
    if args.exclude_scenes_manifest:
        excluded_scenes = set(json.loads(Path(args.exclude_scenes_manifest).read_text())["shared_scene_names"])
        manifest.update(excluded_scenes=sorted(excluded_scenes),
                        excluded_scenes_manifest=args.exclude_scenes_manifest,
                        excluded_scenes_manifest_sha256=_sha256_file(args.exclude_scenes_manifest))
    with torch.no_grad():
        for name, split, count, midpoint in (("train", "train", 128, False),
                                             ("development", "test", 64, False),
                                             ("confirmation", "test", 64, True)):
            dataset, _ = make_dataset(hypes, split)
            eligible = [index for index, clip in enumerate(dataset.clip_records) if clip.scene_id not in excluded_scenes]
            excluded = select_indices(len(eligible), count) if midpoint else ()
            indices = [eligible[index] for index in select_indices(len(eligible), count, midpoint, excluded)]
            _, loader, _ = build_dataloader(Subset(dataset, indices), 2, True,
                workers=8, rank=rank, world_size=2, training=False,
                collate_fn=dataset.collate_batch_test, pin_memory=True)
            full_rows, cache_rows, coverage, selections, files = [], [], [], [], []
            largest_error = 0.
            for ordinal, batch in enumerate(loader):
                keys = [(str(scene), int(receiver), int(frame)) for (scene, receiver), frame in zip(
                    batch["scene_receiver_keys"], batch["runtime_frames"][-1]["receiver_frame_idx"])]
                batch = to_device(batch, device)
                head_input, source_input = [], []
                head_forward = capture_forward(model.forecast_head, head_input)
                source_forward = capture_forward(model.source_temporal_encoder, source_input)
                try:
                    outputs, state = model(batch["runtime_frames"], run_forecast=True)
                finally:
                    model.forecast_head.forward = head_forward
                    model.source_temporal_encoder.forward = source_forward
                original = outputs[-1]
                target = batch["target_frames"][-1]
                values, local_coverage = _forecast_rows(original, target, keys)
                full_rows.extend(values)
                coverage.extend(local_coverage)
                # Retain ALL tracker-valid slots so the Hungarian population
                # cannot change. No selection by GT, future or prediction error.
                slots = torch.nonzero(state.valid_mask.any(0), as_tuple=False).flatten()
                hargs = tuple(value[:, slots] if index != 6 else value
                              for index, value in enumerate(head_input[0]))
                skwargs = {key: value[:, slots] if isinstance(value, torch.Tensor) else value
                           for key, value in head_input[1].items()}
                raw_source = tuple(value[:, slots] for value in source_input[0])
                raw_fused = fused_history_inputs(model, outputs, state, slots, source_input[0])
                base = {key: original[key] for key in ("psm", "rm")}
                base.update({key: original[key][:, slots] for key in
                    ("track_box", "track_box_code", "track_exist_logits")})
                base["state"] = SimpleNamespace(box=state.box[:, slots],
                    runtime_id=state.runtime_id[:, slots], valid_mask=state.valid_mask[:, slots])
                replay = model.forecast_head(*hargs, **skwargs)
                error = float((replay["pred_trajs"] - original["forecast"]["pred_trajs"][:, slots]).abs().max()) if slots.numel() else 0.
                largest_error = max(largest_error, error)
                if error > 1e-3:
                    raise RuntimeError("cache crop changed real forecast: %g" % error)
                changed, _ = _forecast_rows(dict(base, forecast=replay), target, keys)
                cache_rows.extend(changed)
                path = output_dir / ("%s_rank%d_%03d.pt" % (name, rank, ordinal))
                torch.save(cpu_tree(dict(head_args=hargs, source_inputs=raw_source,
                    fused_inputs=raw_fused, output=base, target=target, keys=keys,
                    original_slots=slots)), path)
                files.append(path.name)
                selections.extend(keys)
                del batch, outputs, original, target, head_input, source_input, replay, base, raw_fused, raw_source
            gathered = [None, None]
            dist.all_gather_object(gathered, (full_rows, cache_rows, coverage, selections, files, largest_error))
            if rank == 0:
                cov = [x for group in gathered for x in group[2]]
                full = _forecast_report([x for group in gathered for x in group[0]], cov)
                replay = _forecast_report([x for group in gathered for x in group[1]], cov)
                if full["coverage"] != replay["coverage"] or any(abs(full["model"][key] - replay["model"][key]) > 1e-6 for key in full["model"]):
                    raise RuntimeError("cache replay changed evaluation metrics")
                manifest["selection"][name] = dict(split=split, dataset_size=len(dataset), eligible_clips=len(eligible),
                    indices=indices, keys=[x for group in gathered for x in group[3]],
                    files=[x for group in gathered for x in group[4]])
                manifest["cache_replay"][name] = dict(full=full, replay=replay,
                    max_forecast_error=max(group[5] for group in gathered))
                (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            dist.barrier()
        if rank == 0:
            dev = set(tuple(x) for x in manifest["selection"]["development"]["keys"])
            confirm = set(tuple(x) for x in manifest["selection"]["confirmation"]["keys"])
            if dev & confirm:
                raise RuntimeError("development and confirmation overlap")
            train_scenes = set(x[0] for x in manifest["selection"]["train"]["keys"])
            test_scenes = set(x[0] for name in ("development", "confirmation") for x in manifest["selection"][name]["keys"])
            if train_scenes & test_scenes:
                raise RuntimeError("research train/test scenes overlap")
            (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(json.dumps({"cache": str(output_dir), "status": "replay_verified"}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
