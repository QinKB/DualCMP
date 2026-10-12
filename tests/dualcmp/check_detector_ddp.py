"""Two-GPU real-data equivalence check against an unmodified V2V4Real clone.

Run with torchrun; imports the reference model/dependencies in an isolated
module namespace. No reference algorithm or evaluator is reimplemented here.
This gate proves the detector port, not learned-tracker quality.
"""
import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist

from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.loss.point_pillar_loss import PointPillarLoss
from opencood.models.dualcmp.model import DualCmpClipModel
from opencood.tools.dualcmp.protocol import WINDOW
from opencood.tools.dualcmp.train_tracking_ddp import _load_official_detector
from opencood.tools.dualcmp.eval_tracking_ddp import evaluate_tracking_model_ddp
from opencood.tools.train_utils import to_device


def official_modules(root):
    """Import actual upstream Python sources; share only the compiled IoU ABI."""
    import opencood.utils.box_overlaps as overlaps
    saved = {k: v for k, v in sys.modules.items()
             if k == 'opencood' or k.startswith('opencood.')}
    old_path = list(sys.path)
    try:
        for key in saved:
            del sys.modules[key]
        sys.path.insert(0, str(root.resolve()))
        sys.modules['opencood.utils.box_overlaps'] = overlaps
        names = ('models.point_pillar_fax', 'loss.point_pillar_loss',
                 'data_utils.post_processor.voxel_postprocessor',
                 'utils.eval_utils', 'utils.box_utils', 'data_utils.datasets')
        modules = [importlib.import_module('opencood.' + name) for name in names]
        for module in modules:
            assert root.resolve() in Path(module.__file__).resolve().parents
        return modules
    finally:
        for key in list(sys.modules):
            if key == 'opencood' or key.startswith('opencood.'):
                del sys.modules[key]
        sys.modules.update(saved)
        sys.path[:] = old_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 2:
        raise RuntimeError('this check requires two torchrun ranks')
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.manual_seed(20260831)
    root = Path(args.reference)
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD']).decode().strip()
    if revision != '5a821e13753bafc611f95c47bc1a306acdcb0f7c':
        raise RuntimeError('unexpected V2V4Real reference revision')
    detector_ref, loss_ref, post_ref, metric_ref, boxes_ref, datasets_ref = official_modules(root)
    h = load_yaml('opencood/hypes_yaml/point_pillar_v2v4real.yaml')
    h['root_dir'] = str(Path(os.environ['V2V4REAL_ROOT']) / 'test')
    h['validate_dir'] = h['root_dir']
    h['fusion'] = {'args': {'stage': 'tracking', 'tracking_frames': 3,
        'runtime_only': False, 'runtime_forbid_gt': True, 'receiver_ids': [rank],
        'max_actors': 128, 'max_cav': 2, 'shuffle_training_points': False,
        'tracking_label_root': os.environ['DUALCMP_TRACKING_LABEL_ROOT'],
        'window': [dict(WINDOW[rank])]}}
    dataset = DualCMPSequenceDataset(h, train=False)
    assert dataset.partner_delay_s == 0
    # Two independent real clips on each GPU; compare all three causal frames.
    collated = dataset.collate_batch_train([dataset[0], dataset[1]])
    batch = to_device({key: collated[key] for key in ('runtime_frames', 'target_frames')}, device)
    model = DualCmpClipModel(h['model']['args'], dataset.anchor_box,
                            h['preprocess']['cav_lidar_range']).to(device).eval()
    _load_official_detector(model.detector, os.environ['DUALCMP_DETECTOR_WEIGHTS'])
    reference = detector_ref.PointPillarFax(h['model']['args']).to(device).eval()
    reference.load_state_dict(torch.load(os.environ['DUALCMP_DETECTOR_WEIGHTS'], map_location='cpu'), strict=True)
    criterion = PointPillarLoss(h['loss']['args']).to(device)
    criterion_ref = loss_ref.PointPillarLoss(h['loss']['args']).to(device)
    post = post_ref.VoxelPostprocessor(h['postprocess'], train=False)
    assert np.array_equal(dataset.anchor_box, post.generate_anchor_box())
    stats = {t: {'fp': [], 'tp': [], 'gt': 0} for t in (.5, .7)}
    errors, neutral_count = [], 0
    for frame, target in zip(batch['runtime_frames'], batch['target_frames']):
        official_input = dict(frame)
        # Official code reads this field but never uses it (points are aligned).
        official_input['spatial_correction_matrix'] = frame['source_to_receiver']
        with torch.no_grad():
            expected = reference(official_input)
            actual = model.detector(frame)
        for key in ('psm', 'rm'):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=1e-6)
            errors.append(float((actual[key] - expected[key]).abs().max()))
        assert torch.equal(actual['source_mask'], frame['source_mask'])
        assert actual['source_bev'].shape[:2] == (2, 2)
        assert torch.isfinite(actual['source_bev']).all()
        label = target['fused_label_dict']
        neutral = (label['pos_equal_one'] == 0) & (label['neg_equal_one'] == 0)
        neutral_count += int(neutral.sum())
        pred = {key: actual[key].detach().clone().requires_grad_() for key in ('psm', 'rm')}
        pred_ref = {key: value.detach().clone().requires_grad_() for key, value in pred.items()}
        loss, loss_expected = criterion(pred, label), criterion_ref(pred_ref, label)
        torch.testing.assert_close(loss, loss_expected, rtol=0, atol=1e-6)
        loss.backward(); loss_expected.backward()
        for key in pred:
            torch.testing.assert_close(pred[key].grad, pred_ref[key].grad, rtol=0, atol=1e-6)
        assert torch.count_nonzero(pred['psm'].grad.permute(0, 2, 3, 1)[neutral]) == 0
        for i in range(2):
            payload = {'ego': {'transformation_matrix': torch.eye(4, device=device),
                                'anchor_box': model.anchors}}
            result = {'ego': {key: actual[key][i:i + 1] for key in ('psm', 'rm')}}
            # Upstream range filtering imports GT_RANGE lazily; keep its real
            # dataset module available just for that call (no mocked values).
            with patch.dict(sys.modules, {'opencood.data_utils.datasets': datasets_ref}):
                corners, scores = post.post_process(payload, result)
            gt = boxes_ref.boxes_to_corners_3d(target['current_boxes'][i][target['current_mask'][i]], order='hwl')
            for threshold in stats:
                metric_ref.caluclate_tp_fp(corners, scores, gt, stats, threshold)
    assert neutral_count > 0, 'real sample did not exercise neutral anchors'
    baseline = evaluate_tracking_model_ddp(model, h, device,
        os.environ['V2V4REAL_ROOT'], os.environ['DUALCMP_TRACKING_LABEL_ROOT'])
    gathered = [None, None]
    dist.all_gather_object(gathered, {'stats': stats, 'max_error': max(errors), 'neutral': neutral_count})
    if rank == 0:
        merged = copy.deepcopy(stats)
        for t in merged:
            merged[t] = {k: sum((r['stats'][t][k] for r in gathered), [] if k != 'gt' else 0)
                         for k in stats[t]}
        ap = {str(t): metric_ref.calculate_ap(merged, t)[0] for t in merged}
        report = {'reference_revision': revision, 'world_size': 2, 'clips': 4,
                  'frame_evaluations': 12, 'max_prediction_error': max(r['max_error'] for r in gathered),
                  'neutral_anchors': sum(r['neutral'] for r in gathered),
                  'official_loss_and_gradients_match': True, 'official_detector_subset_AP': ap,
                  'random_tracker_baseline': baseline['metrics'],
                  'metric_scope': 'overlapping real clips, evaluator-native tracking labels; not full-test benchmark'}
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
