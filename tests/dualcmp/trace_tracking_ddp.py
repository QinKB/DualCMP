"""Read-only two-GPU trace of the first failing actor-state transition."""
import argparse
import csv
import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES, load_checkpoint_hypes
from opencood.tools.dualcmp.online import OnlineDualCMP, _model_from_checkpoint, _prediction_in_reference
from opencood.tools.dualcmp.ab3dmot_metrics import compute_ab3dmot_metrics
from opencood.utils import box_utils_v2v4real
from opencood.tools.dualcmp.protocol import checkpoint_detector_compression
from opencood.loss.dualcmp_loss import update_identity_bindings
from opencood.loss.point_pillar_loss import PointPillarLoss, WeightedSmoothL1Loss
from check_detector_ddp import official_modules


def magnitude(value):
    return float(value.detach().abs().max()) if value.numel() else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--allow-dirty-smoke', action='store_true')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--scene')
    p.add_argument('--limit', type=int, default=600)
    p.add_argument('--reset-feature-ownership', action='store_true')
    p.add_argument('--legacy-feature-ownership', action='store_true')
    p.add_argument('--metrics', action='store_true')
    p.add_argument('--compare-proposals', action='store_true')
    p.add_argument('--require-proposal-parity', action='store_true')
    p.add_argument('--compare-geometry', action='store_true')
    p.add_argument('--no-learned-kf-feedback', action='store_true')
    p.add_argument('--normalize-state', action='store_true',
                   help='Diagnostic only: upstream LN before refine/state cache, after strict checkpoint load')
    a = p.parse_args()
    assert int(os.environ.get('WORLD_SIZE', '1')) == 2
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    payload = torch.load(a.checkpoint, map_location='cpu')
    h, _ = load_checkpoint_hypes(DEFAULT_HYPES, payload)
    h['model']['args']['compression'] = checkpoint_detector_compression(payload)
    h['root_dir'] = str(Path(os.environ['V2V4REAL_ROOT']) / 'test')
    h['validate_dir'] = h['root_dir']
    h['fusion'] = {'args': {'stage': 'tracking', 'tracking_frames': 3,
        'runtime_only': True, 'runtime_forbid_gt': True, 'receiver_ids': [0, 1],
        'tracking_label_root': os.environ['DUALCMP_TRACKING_LABEL_ROOT']}}
    dataset = DualCMPSequenceDataset(h, train=False)
    scene = a.scene or max(dataset.index.scene_ids(),
        key=lambda s: len(dataset.index.receiver_frame_keys(s, str(rank))))
    model = _model_from_checkpoint(h, a.checkpoint, device,
                                   allow_dirty_smoke=a.allow_dirty_smoke)
    if a.normalize_state:
        # Sparse4D operation_order: ffn -> norm -> refine. No checkpoint
        # rewriting, source-guard bypass, clipping, or invalid-frame skipping.
        model.track_update.register_forward_hook(
            lambda module, inputs, output: torch.nn.functional.layer_norm(
                output, (output.shape[-1],)))
    row, old = {}, {}
    metric_path = Path('../Reference/Code/V2V4Real/opencood/utils/eval_utils.py')
    spec = importlib.util.spec_from_file_location('official_detection_metrics', metric_path)
    metrics = importlib.util.module_from_spec(spec); spec.loader.exec_module(metrics)
    stats = {stage: {t: {'fp': [], 'tp': [], 'gt': 0} for t in (.25, .5, .7)}
             for stage in ('detector_proposals', 'learned_tracks')}
    official_post = None
    if a.compare_proposals:
        if not a.metrics:
            raise ValueError('--compare-proposals requires --metrics')
        reference = official_modules(Path('../Reference/Code/V2V4Real'))
        official_post = reference[2].VoxelPostprocessor(h['postprocess'], train=False)
        stats['official_proposals'] = {t: {'fp': [], 'tp': [], 'gt': 0} for t in (.25, .5, .7)}
    elif a.require_proposal_parity:
        raise ValueError('--require-proposal-parity requires --compare-proposals')
    if a.compare_geometry:
        if not a.metrics:
            raise ValueError('--compare-geometry requires --metrics')
        for stage in ('same_ids_kf_boxes', 'same_ids_kf_yaw',
                      'same_ids_kf_xy', 'same_ids_kf_size'):
            stats[stage] = {t: {'fp': [], 'tp': [], 'gt': 0} for t in (.25, .5, .7)}
    predictions = []
    geometry_bindings = [dict()]
    geometry_reg = WeightedSmoothL1Loss()

    def detector_hook(module, inputs, output):
        for key in ('psm', 'rm', 'fused_bev'):
            row[key] = magnitude(output[key])

    def tracker_pre(module, inputs):
        old['ids'] = inputs[0].runtime_id.clone()
        old['feature'] = inputs[0].feature
        row['incoming_feature'] = magnitude(inputs[0].feature)
        row['incoming_box'] = magnitude(inputs[0].box)

    def tracker_hook(module, inputs, state):
        old['kf_boxes'] = state.box.detach().clone()
        old['kf_scores'] = state.score.detach().clone()
        reused = state.valid_mask & (state.runtime_id != old['ids'])
        row['births'] = int(reused.sum())
        row['newborn_old_feature'] = magnitude(state.feature[reused])
        row['kf_box'] = magnitude(state.box)
        row['valid_actors'] = int(state.valid_mask.sum())
        row['max_feature_slot'] = int(state.feature.abs().amax(-1).argmax())
        if a.legacy_feature_ownership:
            return replace(state, feature=old['feature'])
        if a.reset_feature_ownership:
            retained = state.valid_mask & ~reused
            return replace(state, feature=torch.where(retained[..., None],
                                                     state.feature, torch.zeros_like(state.feature)))

    def value_hook(name):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            row[name] = magnitude(output)
            row[name + '_finite'] = bool(torch.isfinite(output).all())
        return hook

    model.detector.register_forward_hook(detector_hook)
    model.tracker.register_forward_pre_hook(tracker_pre)
    model.tracker.register_forward_hook(tracker_hook)
    for name in ('roi', 'temporal_attention', 'track_update', 'box_refinement'):
        getattr(model, name).register_forward_hook(value_hook(name))
    if a.no_learned_kf_feedback:
        apply_learned_update = model.apply_learned_update

        def without_learned_kf_feedback(state, update, boxes, velocities,
                                        quality, threshold,
                                        observation_time_s,
                                        observation_pose):
            result = apply_learned_update(state, update, boxes, velocities,
                                          quality, threshold,
                                          observation_time_s,
                                          observation_pose)
            return replace(result, filter_mean=state.filter_mean,
                           filter_cov=state.filter_cov)

        model.apply_learned_update = without_learned_kf_feedback
    online = OnlineDualCMP(model, device)
    rows, failure = [], None
    for key in dataset.index.receiver_frame_keys(scene, str(rank))[:a.limit]:
        row.clear(); row['frame'] = int(key)
        try:
            runtime = dataset._runtime_frame(scene, str(rank), key)
            prediction = online.step(dataset._collate_runtime_frame([runtime]))
            row['emitted'] = len(prediction['runtime_ids'])
            if a.metrics:
                actors = dataset._actors(scene, str(rank), key, runtime['receiver_pose'])
                gt = torch.tensor(list(actors.values()), device=device).reshape(-1, 7)
                gt = box_utils_v2v4real.boxes_to_corners_3d(gt, order='hwl')
                proposals = online._last_output['proposals']
                selected = proposals['mask'][0]
                if official_post is not None:
                    with patch.dict(sys.modules, {'opencood.data_utils.datasets': reference[-1]}):
                        corners, scores = official_post.post_process(
                            {'ego': {'transformation_matrix': torch.eye(4, device=device),
                                     'anchor_box': model.anchors}},
                            {'ego': online._last_output})
                    row['official_proposals'] = 0 if corners is None else len(corners)
                    row['current_proposals'] = int(selected.sum())
                    proposal_limit = proposals['boxes'].shape[1]
                    official_count = row['official_proposals']
                    compared_count = min(official_count, proposal_limit)
                    actual_boxes = proposals['boxes'][0, selected]
                    actual_corners = box_utils_v2v4real.boxes_to_corners_3d(
                        actual_boxes, order='hwl')
                    expected_corners = (actual_corners[:0] if corners is None
                                        else corners[:compared_count])
                    expected_scores = (proposals['scores'][0, selected][:0] if scores is None
                                       else scores[:compared_count])
                    row['proposal_capacity'] = proposal_limit
                    row['official_capacity_excess'] = max(0, official_count - proposal_limit)
                    row['proposal_corner_error'] = (magnitude(actual_corners - expected_corners)
                                                    if actual_corners.shape == expected_corners.shape else None)
                    row['proposal_score_error'] = (magnitude(proposals['scores'][0, selected] - expected_scores)
                                                   if proposals['scores'][0, selected].shape == expected_scores.shape else None)
                    row['proposal_parity'] = bool(
                        actual_corners.shape == expected_corners.shape and
                        proposals['scores'][0, selected].shape == expected_scores.shape and
                        torch.allclose(actual_corners, expected_corners, rtol=0, atol=1e-5) and
                        torch.allclose(proposals['scores'][0, selected], expected_scores,
                                       rtol=0, atol=1e-7))
                    if a.require_proposal_parity and not row['proposal_parity']:
                        raise AssertionError('proposal parity mismatch')
                    row['anchors_above_threshold'] = int((online._last_output['psm'].sigmoid() >
                        h['postprocess']['target_args']['score_threshold']).sum())
                    for threshold in stats['official_proposals']:
                        metrics.caluclate_tp_fp(corners, scores, gt, stats['official_proposals'], threshold)
                for stage, boxes, scores in (
                    ('detector_proposals', proposals['boxes'][0, selected], proposals['scores'][0, selected]),
                    ('learned_tracks', torch.tensor(prediction['boxes'], device=device).reshape(-1, 7),
                     torch.tensor(prediction['scores'], device=device))):
                    corners = box_utils_v2v4real.boxes_to_corners_3d(boxes, order='hwl')
                    for threshold in stats[stage]:
                        metrics.caluclate_tp_fp(corners, scores, gt, stats[stage], threshold)
                if a.compare_geometry:
                    state = online._last_output['state']
                    active = state.active_mask[0]
                    assert state.runtime_id[0, active].tolist() == prediction['runtime_ids']
                    scores = torch.as_tensor(prediction['scores'], device=device,
                                             dtype=state.score.dtype)
                    assert torch.equal(scores, state.score[0, active])
                    assert torch.equal(scores, old['kf_scores'][0, active])
                    learned = state.box[0, active]
                    kf = old['kf_boxes'][0, active]
                    valid = state.valid_mask[0]
                    track_box = online._last_output['track_box'][0, valid]
                    kf_valid = old['kf_boxes'][0, valid]
                    yaw_innovation = track_box[:, 6] - kf_valid[:, 6]
                    row['yaw_innovation_over_pi'] = int((yaw_innovation.abs() > torch.pi).sum())
                    row['max_raw_yaw_innovation'] = magnitude(yaw_innovation)
                    row['max_learned_xy_correction'] = magnitude(track_box[:, :2] - kf_valid[:, :2])
                    row['max_learned_size_correction'] = magnitude(track_box[:, 3:6] - kf_valid[:, 3:6])
                    actor_ids = sorted(actors)
                    target_boxes = torch.tensor(
                        [actors[actor_id] for actor_id in actor_ids], device=device,
                        dtype=state.box.dtype).reshape(1, len(actor_ids), 7)
                    target = {'current_mask': torch.ones(
                        (1, len(actor_ids)), dtype=torch.bool, device=device),
                        'current_boxes': target_boxes,
                        'actor_ids': torch.tensor(actor_ids, device=device,
                                                  dtype=state.runtime_id.dtype).reshape(1, len(actor_ids))}
                    assigned, geometry_bindings = update_identity_bindings(
                        online._last_output, target, geometry_bindings)
                    matched = (assigned[0] >= 0) & state.valid_mask[0]
                    slots = torch.nonzero(matched, as_tuple=False).flatten()
                    gt_indices = assigned[0, slots]
                    matched_kf = old['kf_boxes'][0, slots]
                    matched_learned = online._last_output['track_box'][0, slots]
                    matched_gt = target_boxes[0, gt_indices]
                    kf_yaw_difference = matched_kf[:, 6] - matched_gt[:, 6]
                    learned_yaw_difference = matched_learned[:, 6] - matched_gt[:, 6]
                    row['geometry_match_count'] = int(slots.numel())
                    row['kf_yaw_raw_abs_sum'] = float(kf_yaw_difference.abs().sum())
                    row['kf_yaw_sin_abs_sum'] = float(torch.sin(kf_yaw_difference).abs().sum())
                    row['learned_yaw_raw_abs_sum'] = float(learned_yaw_difference.abs().sum())
                    row['learned_yaw_sin_abs_sum'] = float(torch.sin(learned_yaw_difference).abs().sum())
                    row['learned_yaw_raw_over_pi_half'] = int(
                        (learned_yaw_difference.abs() > torch.pi / 2).sum())
                    row['learned_yaw_raw_loss_sum'] = float(
                        geometry_reg(matched_learned, matched_gt)[..., 6].sum())
                    sin_learned, sin_gt = PointPillarLoss.add_sin_difference(
                        matched_learned, matched_gt)
                    row['learned_yaw_sin_loss_sum'] = float(
                        geometry_reg(sin_learned, sin_gt)[..., 6].sum())
                    for stage, boxes in (
                        ('same_ids_kf_boxes', kf),
                        ('same_ids_kf_yaw', torch.cat((learned[:, :6], kf[:, 6:]), dim=1)),
                        ('same_ids_kf_xy', torch.cat((kf[:, :2], learned[:, 2:]), dim=1)),
                        ('same_ids_kf_size', torch.cat((learned[:, :3], kf[:, 3:6], learned[:, 6:]), dim=1))):
                        corners = box_utils_v2v4real.boxes_to_corners_3d(boxes, order='hwl')
                        for threshold in stats[stage]:
                            metrics.caluclate_tp_fp(corners, scores, gt, stats[stage], threshold)
                    for stage in ('same_ids_kf_boxes', 'same_ids_kf_yaw',
                                  'same_ids_kf_xy', 'same_ids_kf_size'):
                        for threshold in stats[stage]:
                            assert stats[stage][threshold]['gt'] == stats['learned_tracks'][threshold]['gt']
                reference_pose = dataset._pose(dataset.index.receiver_record(scene, '0', key))
                predictions.append(_prediction_in_reference(prediction, runtime['receiver_pose'], reference_pose))
        except Exception as exc:
            failure = repr(exc)
            row['error'] = failure
        rows.append(dict(row))
        if failure:
            break
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for item in rows for k in item))
    with (output / ('rank%d.csv' % rank)).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    report = {'checkpoint': a.checkpoint, 'scene': scene, 'receiver': rank,
              'reset_feature_ownership': a.reset_feature_ownership,
              'legacy_feature_ownership': a.legacy_feature_ownership,
              'compare_proposals': a.compare_proposals,
              'compare_geometry': a.compare_geometry,
              'no_learned_kf_feedback': a.no_learned_kf_feedback,
              'normalize_state': a.normalize_state,
              'yaw_innovation_over_pi': sum(
                  item.get('yaw_innovation_over_pi', 0) for item in rows),
              'max_raw_yaw_innovation': max(
                  (item.get('max_raw_yaw_innovation', 0) for item in rows), default=0),
              'max_learned_xy_correction': max(
                  (item.get('max_learned_xy_correction', 0) for item in rows), default=0),
              'max_learned_size_correction': max(
                  (item.get('max_learned_size_correction', 0) for item in rows), default=0),
              'geometry_match_count': sum(
                  item.get('geometry_match_count', 0) for item in rows),
              'kf_yaw_raw_abs_sum': sum(
                  item.get('kf_yaw_raw_abs_sum', 0) for item in rows),
              'kf_yaw_sin_abs_sum': sum(
                  item.get('kf_yaw_sin_abs_sum', 0) for item in rows),
              'learned_yaw_raw_abs_sum': sum(
                  item.get('learned_yaw_raw_abs_sum', 0) for item in rows),
              'learned_yaw_sin_abs_sum': sum(
                  item.get('learned_yaw_sin_abs_sum', 0) for item in rows),
              'learned_yaw_raw_over_pi_half': sum(
                  item.get('learned_yaw_raw_over_pi_half', 0) for item in rows),
              'learned_yaw_raw_loss_sum': sum(
                  item.get('learned_yaw_raw_loss_sum', 0) for item in rows),
              'learned_yaw_sin_loss_sum': sum(
                  item.get('learned_yaw_sin_loss_sum', 0) for item in rows),
              'proposal_capacity_exceeded_frames': sum(
                  bool(item.get('official_capacity_excess', 0)) for item in rows),
              'max_official_capacity_excess': max(
                  (item.get('official_capacity_excess', 0) for item in rows), default=0),
              'frames_processed': len(rows), 'failure': failure, 'last': rows[-1]}
    (output / ('rank%d.json' % rank)).write_text(json.dumps(report, indent=2))
    all_reports = [None, None]; dist.all_gather_object(all_reports, report)
    if a.metrics:
        gathered = [None, None]; dist.all_gather_object(gathered, (scene, rank, predictions, stats))
        if rank == 0 and not any(r['failure'] for r in all_reports):
            metric_report = {'tracking': compute_ab3dmot_metrics(
                {(s, r): frames for s, r, frames, _ in gathered}, repository_root='.',
                ground_truth_root=os.environ['DUALCMP_TRACKING_LABEL_ROOT']), 'stages': {}}
            for stage in stats:
                metric_report['stages'][stage] = {}
                for t in stats[stage]:
                    merged = {k: sum((group[3][stage][t][k] for group in gathered),
                                    0 if k == 'gt' else []) for k in ('fp', 'tp', 'gt')}
                    tp, fp, gt_count = sum(merged['tp']), sum(merged['fp']), merged['gt']
                    ap = metrics.calculate_ap({t: merged}, t)[0]
                    metric_report['stages'][stage][str(t)] = {'tp': tp, 'fp': fp, 'gt': gt_count, 'AP': ap}
            (output / 'metrics.json').write_text(json.dumps(metric_report, indent=2))
            print(json.dumps(metric_report), flush=True)
    if rank == 0:
        print(json.dumps(all_reports), flush=True)
    dist.destroy_process_group()
    if any(report['failure'] for report in all_reports):
        raise RuntimeError('tracking trace failed; inspect saved rank reports')


if __name__ == '__main__':
    main()
