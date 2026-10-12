"""Fresh-process forecast check using the exact epoch evaluator and batching.

The historical window CLI evaluates one clip at a time; use this check when
testing checkpoint/CSV equivalence with the training epoch's batched protocol.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES, load_checkpoint_hypes
from opencood.tools.dualcmp.eval_forecast_mainstream_ddp import evaluate_forecast_training_ddp
from opencood.tools.dualcmp.mainstream_forecast_metrics import PRIMARY_METRICS
from opencood.tools.dualcmp.online import _model_from_checkpoint
from opencood.tools.dualcmp.protocol import checkpoint_detector_compression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch-size', required=True, type=int)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--prefix', default='test_')
    args = parser.parse_args()
    assert int(os.environ.get('WORLD_SIZE', '1')) == 2
    dist.init_process_group('nccl')
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    payload = torch.load(args.checkpoint, map_location='cpu')
    h, _ = load_checkpoint_hypes(DEFAULT_HYPES, payload)
    h['model']['args']['compression'] = checkpoint_detector_compression(payload)
    forecast_args = {'history_frames': 11, 'future_frames': 50, 'modes': 6,
        'intention_points_file': 'preprocessed_data/v2v4real/v2v4real_cluster_64_center_dict.pkl'}
    model = _model_from_checkpoint(h, args.checkpoint, device,
                                  forecast_args=forecast_args, allow_dirty_smoke=True)
    report = evaluate_forecast_training_ddp(model, h, device,
        os.environ['V2V4REAL_ROOT'], os.environ['DUALCMP_TRACKING_LABEL_ROOT'],
        batch_size=args.batch_size, workers=args.workers, full_test=False)
    if dist.get_rank() == 0:
        directory = Path(args.checkpoint).parent
        with (directory / 'metrics.csv').open() as stream:
            rows = [row for row in csv.DictReader(stream) if row['kind'] == 'epoch']
        expected = rows[-1]
        differences = {key: abs(float(value) - float(expected[args.prefix + key]))
                       for key, value in report['model'].items()
                       if args.prefix + key in expected}
        expected_count = len(PRIMARY_METRICS) if model.forecast_head.forecast_coordinate_frame == 'current_receiver' else 8
        assert len(differences) == expected_count, differences
        report['epoch_metric_differences'] = differences
        report['batch_size_per_rank'] = args.batch_size
        (directory / 'fresh_epoch_forecast.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'differences': differences, 'max_difference': max(differences.values())}), flush=True)
        assert max(differences.values()) <= 1e-5
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
