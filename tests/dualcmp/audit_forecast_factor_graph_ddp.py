"""Compare association losses on one identical native forward, not two models."""
import json
import os
from pathlib import Path
import sys

import torch
from opencood.loss.dualcmp_loss import DualCmpLoss, update_identity_bindings
from opencood.loss.forecast_association import LEGACY_ASSOCIATION, VALIDATED_ASSOCIATION
from opencood.tools.dualcmp import train_forecast_ddp as trainer


if "--deterministic-probe" in sys.argv:
    sys.argv.remove("--deterministic-probe")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

output_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
rank = int(os.environ.get("RANK", 0))


class AssociationAuditLoss(DualCmpLoss):
    def forward(self, outputs, targets, forecast_assignments=None):
        if self.forecast_association_policy != VALIDATED_ASSOCIATION:
            raise ValueError("association probe requires the native causal policy")
        result = super().forward(outputs, targets, forecast_assignments)
        try:
            self.forecast_association_policy = LEGACY_ASSOCIATION
            legacy = super().forward(outputs, targets)
        finally:
            self.forecast_association_policy = VALIDATED_ASSOCIATION
        assigned, _ = update_identity_bindings(
            outputs[-1], targets[-1], [dict() for _ in range(forecast_assignments[-1].shape[0])])
        causal, target = forecast_assignments[-1], targets[-1]
        future_pairs = []
        for row, slot in torch.nonzero(outputs[-1]["forecast"]["forecast_valid_mask"], as_tuple=False).tolist():
            old_gt, new_gt = int(assigned[row, slot]), int(causal[row, slot])
            old_valid = old_gt >= 0 and bool(target["future_mask"][row, old_gt].any())
            new_valid = new_gt >= 0 and bool(target["future_mask"][row, new_gt].any())
            if old_valid or new_valid:
                future_pairs.append({"row": row, "runtime_id": int(outputs[-1]["state"].runtime_id[row, slot]),
                    "legacy_gt": int(target["actor_ids"][row, old_gt]) if old_valid else None,
                    "causal_gt": int(target["actor_ids"][row, new_gt]) if new_valid else None})
        audit = {"rank": rank, "same_forward": True,
                 "future_pairs": future_pairs,
                 "different_future_pairs": sum(item["legacy_gt"] != item["causal_gt"] for item in future_pairs),
                 "legacy": {key: float(legacy[key].detach()) for key in legacy if key.startswith(("S_", "C_"))},
                 "causal": {key: float(result[key].detach()) for key in result if key.startswith(("S_", "C_"))}}
        audit["absolute_loss_differences"] = {key: abs(audit["legacy"][key] - value)
                                               for key, value in audit["causal"].items()}
        (output_dir / ("association_graph_rank%d.json" % rank)).write_text(json.dumps(audit, indent=2))
        return result


trainer.DualCmpLoss = AssociationAuditLoss
try:
    trainer.main()
except Exception as exc:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ("probe_error_rank%d.json" % rank)).write_text(json.dumps({
        "error": str(exc), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}, indent=2))
    raise
