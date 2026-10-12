"""Isolate complete main futures from unnecessary dense-center eligibility."""
import hashlib
from pathlib import Path

from opencood.loss.dualcmp_loss import DualCmpLoss
from opencood.loss.dualcmp_gmm_loss import PARTIAL_FORECAST_SUPERVISION
from opencood.tools.dualcmp import train_forecast_ddp as trainer


class DenseContextControlLoss(DualCmpLoss):
    def _dense_context_future_loss(self, *args, **kwargs):
        original = self.forecast_supervision
        try:
            self.forecast_supervision = PARTIAL_FORECAST_SUPERVISION
            return super()._dense_context_future_loss(*args, **kwargs)
        finally:
            self.forecast_supervision = original


original_provenance = trainer._provenance


def provenance():
    value = original_provenance()
    value["loss_diagnostic_variant"] = "complete_main_masked_dense_context_any_center_v1"
    value["loss_diagnostic_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return value


trainer.DualCmpLoss = DenseContextControlLoss
trainer._provenance = provenance
trainer.main()
