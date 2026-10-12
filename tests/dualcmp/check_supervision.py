"""Compare real training anchors with the unmodified official dataset."""
import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from check_detector_ddp import official_modules
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=12)
    parser.add_argument('--augment', action='store_true')
    args = parser.parse_args()
    random.seed(20260831)
    np.random.seed(20260831)
    modules = official_modules(Path(args.reference))
    hypes = load_yaml(DEFAULT_HYPES)
    hypes['root_dir'] = str(Path(os.environ['V2V4REAL_ROOT']) / 'train')
    hypes['validate_dir'] = hypes['root_dir']
    if not args.augment:
        hypes['data_augment'] = []
    own_hypes = copy.deepcopy(hypes)
    own_hypes['fusion'] = {'args': {'stage': 'tracking', 'tracking_frames': 3,
                                  'receiver_ids': [0, 1], 'sequence_augmentation': args.augment}}
    own = DualCMPSequenceDataset(own_hypes, train=True)
    records = []
    with patch.dict(sys.modules, {'opencood.data_utils.datasets': modules[-1]}):
        reference = modules[-1].IntermediateFusionDataset(
            hypes, visualize=False, train=True, isSim=False)
        augmentation = {}
        get_car = reference.get_item_single_car

        def observe_augmentation(selected, ego_pose):
            for name in ('flip', 'noise_rotation', 'noise_scale'):
                augmentation[name] = selected[name]
            return get_car(selected, ego_pose)

        reference.get_item_single_car = observe_augmentation
        for scene_index, scene in reference.scenario_database.items():
            receiver = next(iter(scene))
            keys = [key for key, value in scene[receiver].items()
                    if isinstance(value, dict) and 'yaml' in value]
            start = 0 if scene_index == 0 else reference.len_record[scene_index - 1]
            for offset, key in enumerate(keys[:3]):
                scene_id = Path(scene[receiver][key]['yaml']).parents[1].name
                original = reference[start + offset]['ego']
                rng_state = np.random.get_state()
                if args.augment:
                    basis = np.asarray([[0, 0, 0, 0], [1, 0, 0, 0],
                                        [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
                    transformed, _ = own._augment_points_boxes(basis, np.zeros((0, 7)), augmentation)
                    augmentation['matrix'] = np.eye(4, dtype=np.float32)
                    augmentation['matrix'][:3, :3] = (transformed[1:, :3] - transformed[0, :3]).T
                target = own._target_frame(scene_id, str(receiver), key, None, False,
                                           augmentation if args.augment else None)
                np.random.set_state(rng_state)
                official_labels, local_labels = original['label_dict'], target['fused_label_dict']
                local_boxes = target['current_boxes'][target['current_mask']]
                _, range_mask = modules[-2].mask_boxes_outside_range_numpy(
                    local_boxes, hypes['preprocess']['cav_lidar_range'], hypes['postprocess']['order'])
                positive = official_labels['pos_equal_one'].astype(bool)
                local_positive = local_labels['pos_equal_one'].astype(bool)
                records.append({'scene': scene_id, 'receiver': receiver, 'frame': key,
                    'official_boxes': int(original['object_bbx_mask'].sum()),
                    'local_boxes': int(target['current_mask'].sum()),
                    'local_outside_final_range': int((~range_mask).sum()),
                    'negative_mask_differences': int(np.sum(official_labels['neg_equal_one'] != local_labels['neg_equal_one'])),
                    'regression_max_difference': float(np.max(np.abs(official_labels['targets'] - local_labels['targets']))),
                    'official_positive_local_negative': int(np.sum(positive & local_labels['neg_equal_one'].astype(bool))),
                    'local_positive_official_negative': int(np.sum(local_positive & official_labels['neg_equal_one'].astype(bool))),
                    'positive_mask_differences': int(np.sum(positive != local_positive))})
                if len(records) >= args.samples:
                    break
            if len(records) >= args.samples:
                break
    report = {'reference': str(Path(args.reference).resolve()),
              'official_dataset_source': modules[-1].__file__, 'augmentation': args.augment,
              'samples': records,
              'totals': {key: (max if key == 'regression_max_difference' else sum)(row[key] for row in records) for key in records[0]
                         if key not in ('scene', 'receiver', 'frame')}}
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report['totals']))


if __name__ == '__main__':
    main()
