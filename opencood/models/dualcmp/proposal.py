"""Detector proposal decoding using the repository VoxelPostprocessor."""
from __future__ import annotations

import torch

from opencood.data_utils.v2v4real.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.utils import box_utils_v2v4real


def decode_proposals(psm, rm, anchors, top_k, *, training,
                     score_threshold, nms_thresh):
    """Decode proposals in the V2V4Real post-processing order."""
    if psm.ndim != 4 or rm.ndim != 4 or psm.shape[0] != rm.shape[0]:
        raise ValueError("psm/rm must be matching [B,C,H,W] tensors")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    # The repository helper uses ``anchors.cuda()`` without an index.  Pin its
    # implicit device to the calling tensor so receiver inference can run on
    # either DDP-local GPU rather than silently decoding on cuda:0.
    if rm.is_cuda:
        with torch.cuda.device(rm.device):
            boxes = VoxelPostprocessor.delta_to_boxes3d(rm, anchors)
    else:
        boxes = VoxelPostprocessor.delta_to_boxes3d(rm, anchors)
    scores = psm.sigmoid().permute(0, 2, 3, 1).reshape(psm.shape[0], -1)
    count = min(int(top_k), scores.shape[1])
    del training
    selected_boxes, selected_scores, selected_mask = [], [], []
    for row in range(boxes.shape[0]):
        candidate = torch.nonzero(
            scores[row] > float(score_threshold), as_tuple=False).flatten()
        kept = candidate
        if candidate.numel():
            corners = box_utils_v2v4real.boxes_to_corners_3d(
                boxes[row, candidate], order="hwl")
            kept = candidate[torch.as_tensor(
                box_utils_v2v4real.nms_rotated(
                    corners, scores[row, candidate], float(nms_thresh)),
                device=candidate.device)]
            corners = box_utils_v2v4real.boxes_to_corners_3d(
                boxes[row, kept], order="hwl")
            kept = kept[box_utils_v2v4real.get_mask_for_boxes_within_range_torch(corners)]
        kept = kept[:count]
        row_boxes = torch.zeros((count, 7), dtype=boxes.dtype, device=boxes.device)
        row_scores = torch.zeros(count, dtype=scores.dtype, device=scores.device)
        row_mask = torch.zeros(count, dtype=torch.bool, device=scores.device)
        if kept.numel():
            row_boxes[:kept.numel()] = boxes[row, kept]
            row_scores[:kept.numel()] = scores[row, kept]
            row_mask[:kept.numel()] = True
        selected_boxes.append(row_boxes)
        selected_scores.append(row_scores)
        selected_mask.append(row_mask)
    return {"boxes": torch.stack(selected_boxes),
            "scores": torch.stack(selected_scores),
            "mask": torch.stack(selected_mask)}
