"""Frozen-checkpoint query diversity and evidence intervention on real inputs."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from opencood.data_utils.dualcmp_dataloader import build_dataloader
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
from opencood.loss.dualcmp_loss import update_identity_bindings
from opencood.models.dualcmp.mtr_actor_adapter import pack_mtr_actor_polylines
from opencood.tools.dualcmp.aligned_protocol import ALIGNED_FORECAST_WINDOW
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES, load_checkpoint_hypes
from opencood.tools.dualcmp.eval_forecast_ddp import _forecast_rows, _forecast_report
from opencood.tools.dualcmp.online import _model_from_checkpoint
from opencood.tools.train_utils import to_device
from audit_task_fitness_ddp import capture_forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("refusing to overwrite diagnostic evidence")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("diagnostic requires two GPU processes")
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
        shuffle_training_points=False, forecast_coordinate_frame="current_receiver",
        window=list(ALIGNED_FORECAST_WINDOW))}
    dataset = DualCMPSequenceDataset(h, train=False)
    _, loader, _ = build_dataloader(dataset, 2, True, workers=2, rank=rank,
        world_size=2, training=False, collate_fn=dataset.collate_batch_test)
    model = _model_from_checkpoint(h, args.checkpoint, device, allow_dirty_smoke=True)
    decoder = model.forecast_head.decoder
    variants = ("normal", "no_evidence", "unit_scale_object_memory",
                "persistent_query_position", "persistent_position_unit_memory")
    rows = {name: [] for name in variants}
    coverage, actors, inputs = [], [], []
    with torch.no_grad():
        for batch in loader:
            keys = [(str(scene), int(receiver), int(frame)) for (scene, receiver), frame in zip(
                batch["scene_receiver_keys"], batch["runtime_frames"][-1]["receiver_frame_idx"])]
            batch = to_device(batch, device)
            captured = []
            original_forward = capture_forward(model.forecast_head, captured)
            try:
                outputs, _ = model(batch["runtime_frames"], run_forecast=True)
            finally:
                model.forecast_head.forward = original_forward
            output, target = outputs[-1], batch["target_frames"][-1]
            assigned, _ = update_identity_bindings(output, target, [dict() for _ in keys])
            head = model.forecast_head
            packed = pack_mtr_actor_polylines(*captured[0][1:], legacy_heading=False,
                min_forecast_history_steps=head.min_forecast_history_steps,
                min_context_history_steps=head.min_context_history_steps,
                frame_period_s=head.frame_period_s, timestamp_scale=head.timestamp_scale)
            features = packed.batch_dict["input_dict"]["obj_trajs"]
            valid_features = features[packed.batch_dict["input_dict"]["obj_trajs_mask"]]
            inputs.append({"keys": keys, "valid_observations": len(valid_features),
                "attribute_absolute_max": valid_features.abs().amax(dim=0).cpu().tolist(),
                "attribute_absolute_mean": valid_features.abs().mean(dim=0).cpu().tolist()})
            for variant in variants:
                traces = {}
                original_attention = decoder.apply_cross_attention
                def trace_attention(**kwargs):
                    kind = "object" if kwargs["attention_layer"] in decoder.obj_decoder_layers else "evidence"
                    if variant in ("unit_scale_object_memory", "persistent_position_unit_memory") and kind == "object":
                        kwargs["kv_feature"] = torch.nn.functional.layer_norm(
                            kwargs["kv_feature"], (kwargs["kv_feature"].shape[-1],))
                    result = original_attention(**kwargs)
                    traces[(kind, kwargs["layer_idx"])] = {
                        "query_std": result.std(dim=0, unbiased=False).mean(dim=-1).cpu().tolist(),
                        "memory_absolute_max": float(kwargs["kv_feature"][kwargs["kv_mask"]].abs().max()),
                    }
                    return result
                decoder.apply_cross_attention = trace_attention
                attention_layers = list(decoder.obj_decoder_layers) + list(model.forecast_head.evidence_decoder_layers)
                old_query_position = [layer.keep_query_pos for layer in attention_layers]
                if variant in ("persistent_query_position", "persistent_position_unit_memory"):
                    for layer in attention_layers:
                        layer.keep_query_pos = True
                kwargs = dict(captured[1])
                if variant == "no_evidence":
                    for prefix in ("source", "consensus"):
                        for suffix in ("tokens", "positions", "token_mask"):
                            kwargs[prefix + "_" + suffix] = None
                    kwargs["source_age_s"] = None
                try:
                    forecast = model.forecast_head(*captured[0], **kwargs)
                finally:
                    decoder.apply_cross_attention = original_attention
                    for layer, previous in zip(attention_layers, old_query_position):
                        layer.keep_query_pos = previous
                local_rows, local_coverage = _forecast_rows(dict(output, forecast=forecast), target, keys)
                rows[variant].extend(local_rows)
                if variant == "normal":
                    coverage.extend(local_coverage)
                for packed_row, (row, slot) in enumerate(forecast["packed_entries"]):
                    gt_index = int(assigned[row, slot])
                    if gt_index < 0 or not bool(target["future_mask"][row, gt_index].all()):
                        continue
                    trajectories = forecast["loss_pred_trajs_local_layers"][row, slot]
                    physical = (target["future_xy"][row, gt_index]
                        + target["current_boxes"][row, gt_index, None, :2]
                        - output["track_box"][row, slot, None, :2])
                    heading = output["track_box"][row, slot, 6]
                    c, s = heading.cos(), heading.sin()
                    gt = torch.stack((c * physical[:, 0] + s * physical[:, 1],
                                      -s * physical[:, 0] + c * physical[:, 1]), dim=-1)
                    layers = []
                    for layer, predictions in enumerate(trajectories):
                        errors = (predictions[..., :2] - gt).norm(dim=-1)
                        points = forecast["loss_intention_points_layers"][row, slot, layer]
                        mode = int((points - gt[-1]).norm(dim=-1).argmin())
                        logs = predictions[mode, :, 2:4]
                        item = {"layer": layer,
                            "raw64_xy_std_m": float(predictions[..., :2].std(dim=0, unbiased=False).mean()),
                            "raw64_endpoint_max_span_m": float((predictions[:, -1, :2].amax(0) - predictions[:, -1, :2].amin(0)).norm()),
                            "raw64_class_logit_span": float(forecast["loss_pred_scores_layers"][row, slot, layer].amax()
                                                            - forecast["loss_pred_scores_layers"][row, slot, layer].amin()),
                            "best_raw64_ade_5s": float(errors.mean(-1).min()),
                            "supervised_mode": mode, "supervised_ade_5s": float(errors[mode].mean()),
                            "supervised_fde_5s": float(errors[mode, -1]),
                            "max_clamped_std_fraction": float((logs >= 5.).float().mean()),
                            "mean_supervised_std_m": float(logs.clamp(-1.609, 5.).exp().mean())}
                        for kind in ("object", "evidence"):
                            if (kind, layer) in traces:
                                value = traces[(kind, layer)]
                                item[kind + "_query_feature_std"] = value["query_std"][packed_row]
                                item[kind + "_memory_absolute_max"] = value["memory_absolute_max"]
                        layers.append(item)
                    actors.append({"variant": variant,
                        "key": keys[row] + (int(target["actor_ids"][row, gt_index]),),
                        "runtime_id": int(output["state"].runtime_id[row, slot]),
                        "current_center_error_m": float((output["track_box"][row, slot, :2]
                            - target["current_boxes"][row, gt_index, :2]).norm()),
                        "gt_endpoint_norm_m": float(gt[-1].norm()), "layers": layers})
    gathered = [None, None]
    dist.all_gather_object(gathered, (rows, coverage, actors, inputs))
    if rank == 0:
        all_coverage = [row for group in gathered for row in group[1]]
        result = {"scope": "same checkpoint and real inputs; no optimizer; inference interventions are not trained ablations",
                  "checkpoint": args.checkpoint, "world_size": 2, "per_rank_batch": 2,
                  "actors": [row for group in gathered for row in group[2]],
                  "inputs": [row for group in gathered for row in group[3]], "variants": {}}
        for name in variants:
            result["variants"][name] = _forecast_report(
                [row for group in gathered for row in group[0][name]], all_coverage)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({name: report["diagnostics"]["exact_horizon_model"]
                          for name, report in result["variants"].items()}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
