"""Detector proposal decoding using the repository VoxelPostprocessor."""
from __future__ import annotations

import torch

from opencood.data_utils.v2v4real.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.utils import box_utils_v2v4real


def decode_proposals(psm, rm, anchors, top_k, *, training,
                     score_threshold, nms_thresh):
    """Decode one shared score-thresholded and NMS-filtered candidate contract."""
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
    scores, indices = scores.topk(count, dim=1)
    boxes = boxes.gather(1, indices[..., None].expand(-1, -1, 7))
    del training
    keep = scores >= float(score_threshold)
    # NMS has no batch interface; retain indices in each row's fixed top-K
    # layout so training and inference propagate identical candidates.
    for row in range(boxes.shape[0]):
        candidate = torch.nonzero(keep[row], as_tuple=False).flatten()
        if candidate.numel() == 0:
            continue
        corners = box_utils_v2v4real.boxes_to_corners_3d(
            boxes[row, candidate], order="hwl")
        selected = box_utils_v2v4real.nms_rotated(
            corners, scores[row, candidate], float(nms_thresh))
        row_keep = torch.zeros_like(keep[row])
        if len(selected):
            row_keep[candidate[torch.as_tensor(selected, device=candidate.device)]] = True
        keep[row] = row_keep
    return {"boxes": boxes, "scores": scores, "mask": keep}
