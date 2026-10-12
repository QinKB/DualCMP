"""Rebatch frozen-perception clips without changing inputs or GT ownership."""
from types import SimpleNamespace

import torch


def slice_clip(value, row):
    if isinstance(value, torch.Tensor):
        return value[row:row + 1]
    if isinstance(value, dict):
        return {key: slice_clip(item, row) for key, item in value.items()}
    if isinstance(value, SimpleNamespace):
        return SimpleNamespace(**slice_clip(vars(value), row))
    if isinstance(value, (tuple, list)):
        return type(value)(slice_clip(item, row) for item in value)
    return value


def split_cached_chunk(chunk):
    return [dict(head_args=slice_clip(chunk["head_args"], row),
                 source_inputs=slice_clip(chunk["source_inputs"], row),
                 fused_inputs=slice_clip(chunk["fused_inputs"], row),
                 output=slice_clip(chunk["output"], row),
                 target=slice_clip(chunk["target"], row),
                 keys=[chunk["keys"][row]], original_slots=chunk["original_slots"])
            for row in range(len(chunk["keys"]))]


def concatenate(values):
    first = values[0]
    if isinstance(first, torch.Tensor):
        if any(value.shape[0] != 1 or value.shape[1:] != first.shape[1:]
               for value in values):
            raise ValueError("cached clip non-actor tensors must have identical shapes")
        return torch.cat(values, dim=0)
    if isinstance(first, dict):
        if any(value.keys() != first.keys() for value in values):
            raise ValueError("cached clip dictionaries disagree")
        return {key: concatenate([value[key] for value in values]) for key in first}
    if isinstance(first, (tuple, list)):
        return type(first)(concatenate([value[index] for value in values])
                           for index in range(len(first)))
    if any(value != first for value in values):
        raise ValueError("cached clips disagree on a non-tensor contract")
    return first


def collate_cached_clips(clips):
    if not clips:
        raise ValueError("cannot collate an empty cached batch")
    slots = torch.unique(torch.cat([clip["original_slots"] for clip in clips]), sorted=True)
    destinations = [torch.searchsorted(slots, clip["original_slots"]) for clip in clips]

    def actors(values, fill=0):
        rows = []
        for clip, value, indices in zip(clips, values, destinations):
            if value.shape[:2] != (1, len(clip["original_slots"])):
                raise ValueError("cached actor tensor is not keyed by original_slots")
            expanded = value.new_full((1, len(slots)) + tuple(value.shape[2:]), fill)
            expanded[:, indices] = value
            rows.append(expanded)
        return torch.cat(rows, dim=0)

    head = tuple(concatenate([clip["head_args"][index] for clip in clips]) if index == 6
                 else actors([clip["head_args"][index] for clip in clips])
                 for index in range(7))
    raw = {key: tuple(actors([clip[key][index] for clip in clips]) for index in range(5))
           for key in ("source_inputs", "fused_inputs")}
    output = {key: concatenate([clip["output"][key] for clip in clips])
              for key in ("psm", "rm")}
    output.update({key: actors([clip["output"][key] for clip in clips])
                   for key in ("track_box", "track_box_code", "track_exist_logits")})
    output["state"] = SimpleNamespace(**{key: actors(
        [getattr(clip["output"]["state"], key) for clip in clips],
        fill=-1 if key == "runtime_id" else 0)
        for key in ("box", "runtime_id", "valid_mask")})
    return dict(head_args=head, output=output, target=concatenate([clip["target"] for clip in clips]),
                keys=[clip["keys"][0] for clip in clips], original_slots=slots, **raw)
