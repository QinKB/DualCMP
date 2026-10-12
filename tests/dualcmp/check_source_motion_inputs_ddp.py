"""Check source-motion duplicate masking on the real trainer inputs."""
import argparse
import json
import os
from pathlib import Path

import torch

from opencood.models.dualcmp.source_motion_encoder import SourceMotionEncoder, SourceActorReadout
from opencood.models.dualcmp.mtr_vendor import MTREncoder
from opencood.tools.dualcmp import train_forecast_ddp


def _patch_forward():
    original = SourceMotionEncoder.forward
    stats = {"calls": 0, "input_valid_tokens": 0, "current_input_valid_tokens": 0,
             "current_output_valid_tokens": 0,
             "invalid_prefix_actor_source_lanes": 0, "shapes": []}

    def checked(self, tokens, positions, token_mask, age_s, pose, encode_age=True):
        output = original(self, tokens, positions, token_mask, age_s, pose, encode_age)
        assert output["tokens"].shape == (*tokens.shape[:-1], 2 * tokens.shape[-1])
        observed = torch.where(output["token_mask"][..., None], tokens, torch.zeros_like(tokens))
        assert torch.equal(output["tokens"][..., :tokens.shape[-1]], observed)
        current_only = token_mask.detach().clone()
        current_only[..., :-1, :] = False
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(tokens.device) if tokens.is_cuda else None
        versions = tuple(parameter._version for parameter in self.parameters())
        with torch.no_grad():
            probe = original(self, tokens, positions, current_only, age_s, pose, encode_age)
        assert torch.equal(probe["token_mask"], current_only)
        assert torch.isfinite(output["tokens"]).all() and torch.isfinite(probe["tokens"]).all()
        assert torch.equal(cpu_rng, torch.random.get_rng_state())
        if cuda_rng is not None:
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state(tokens.device))
        assert versions == tuple(parameter._version for parameter in self.parameters())
        prefix = current_only[..., :-1, :].any(-1).any(-1)
        last = current_only[..., -1, :].any(-1)
        stats["calls"] += 1
        stats["input_valid_tokens"] += int(token_mask.sum())
        stats["current_input_valid_tokens"] += int(current_only.sum())
        stats["current_output_valid_tokens"] += int(probe["token_mask"].sum())
        stats["invalid_prefix_actor_source_lanes"] += int((last & ~prefix).sum())
        shape = list(tokens.shape)
        if shape not in stats["shapes"]:
            stats["shapes"].append(shape)
        return output

    SourceMotionEncoder.forward = checked
    return original, stats


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", required=True)
    args, _ = parser.parse_known_args()
    original, stats = _patch_forward()
    original_readout = SourceActorReadout.forward
    scene_methods = {name: getattr(MTREncoder, name) for name in ("apply_local_attn", "apply_global_attn")}
    own_methods = set(MTREncoder.__dict__)
    pending = []
    stats.update({"pre_scene_readout_calls": 0, "neighbor_contexts": 0, "scene_inputs_exact_readout": 0})

    def readout(self, *args, **kwargs):
        output = original_readout(self, *args, **kwargs)
        assert not pending
        pending.append(output)
        stats["pre_scene_readout_calls"] += 1
        stats["neighbor_contexts"] += self.last_counts["neighbors"]
        return output

    def scene_method(original_scene):
        def checked_scene(self, x, *args, **kwargs):
            if pending:
                assert pending.pop() is x  # exact enriched tensor enters MTR attention
                stats["scene_inputs_exact_readout"] += 1
            return original_scene(self, x, *args, **kwargs)
        return checked_scene

    SourceActorReadout.forward = readout
    for name, method in scene_methods.items():
        setattr(MTREncoder, name, scene_method(method))
    try:
        train_forecast_ddp.main()
    finally:
        SourceMotionEncoder.forward = original
        SourceActorReadout.forward = original_readout
        for name, method in scene_methods.items():
            if name in own_methods:
                setattr(MTREncoder, name, method)
            else:
                delattr(MTREncoder, name)
    assert stats["calls"] > 0
    assert not pending and stats["neighbor_contexts"] > 0
    assert stats["pre_scene_readout_calls"] == stats["scene_inputs_exact_readout"] > 0
    rank = int(os.environ.get("RANK", 0))
    stats.update({"rank": rank, "policy": "current_time_only_with_duplicate_prefix_masked",
                  "rng_unchanged": True, "parameters_unchanged": True,
                  "original_observations_bitwise_preserved": True})
    output = Path(args.output_dir) / ("source_motion_input_check_rank%d.json" % rank)
    output.write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
