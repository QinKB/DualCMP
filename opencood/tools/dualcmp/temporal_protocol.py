"""Additive forecast protocol for temporal Source checkpoints.

The frozen ``protocol.py`` remains byte-compatible with historical Phase-B
checkpoints.  New readers delegate every legacy representation to that module
and handle only the additive ``source_temporal`` representation here.
"""
from __future__ import annotations

from opencood.tools.dualcmp import protocol as legacy


SOURCE_TEMPORAL = "source_temporal"
SOURCE_TEMPORAL_ALIGNED = "source_temporal_aligned"
TEMPORAL_SOURCES = (SOURCE_TEMPORAL, SOURCE_TEMPORAL_ALIGNED)


def resolve_forecast_ablation(forecast_representation="source",
                              disable_capture_alignment=False,
                              disable_age_encoding=False,
                              disable_capture_time_evidence=False,
                              mean_source_aggregation=False):
    if str(forecast_representation) not in TEMPORAL_SOURCES:
        return legacy.resolve_forecast_ablation(
            forecast_representation, disable_capture_alignment,
            disable_age_encoding, disable_capture_time_evidence,
            mean_source_aggregation)
    if mean_source_aggregation:
        raise ValueError(
            "mean source aggregation belongs to the source ablation")
    disabled = bool(disable_capture_time_evidence)
    return {
        "forecast_representation": str(forecast_representation),
        "capture_alignment": not (
            disabled or bool(disable_capture_alignment)),
        "age_encoding": not (disabled or bool(disable_age_encoding)),
        "mean_source_aggregation": False,
    }


def checkpoint_forecast_contract(provenance):
    if (isinstance(provenance, dict)
            and provenance.get("forecast_representation") in TEMPORAL_SOURCES):
        legacy_capture = bool(provenance.get("capture_time_evidence", True))
        return {
            "forecast_representation": provenance["forecast_representation"],
            "capture_alignment": bool(
                provenance.get("capture_alignment", legacy_capture)),
            "age_encoding": bool(
                provenance.get("age_encoding", legacy_capture)),
            "mean_source_aggregation": False,
        }
    return legacy.checkpoint_forecast_contract(provenance)


def validate_forecast_contract(provenance, forecast_representation,
                               capture_alignment, age_encoding,
                               context="checkpoint",
                               mean_source_aggregation=False):
    if (str(forecast_representation) not in TEMPORAL_SOURCES
            and (not isinstance(provenance, dict)
                 or provenance.get("forecast_representation")
                 not in TEMPORAL_SOURCES)):
        return legacy.validate_forecast_contract(
            provenance, forecast_representation, capture_alignment,
            age_encoding, context=context,
            mean_source_aggregation=mean_source_aggregation)
    requested = {
        "forecast_representation": str(forecast_representation),
        "capture_alignment": bool(capture_alignment),
        "age_encoding": bool(age_encoding),
        "mean_source_aggregation": bool(mean_source_aggregation),
    }
    checkpoint = checkpoint_forecast_contract(provenance)
    if checkpoint != requested:
        raise RuntimeError(
            "%s forecast ablation must match checkpoint provenance" % context)
    return checkpoint


def validate_training_delay_contract(
        provenance, train_delays_ms, delay_seed,
        capture_time_evidence=None, context="checkpoint",
        forecast_representation=None, capture_alignment=None,
        age_encoding=None, mean_source_aggregation=False):
    checkpoint = legacy.validate_training_delay_contract(
        provenance, train_delays_ms, delay_seed, context=context)
    if capture_time_evidence is not None:
        if capture_alignment is None:
            capture_alignment = capture_time_evidence
        if age_encoding is None:
            age_encoding = capture_time_evidence
    if forecast_representation is not None:
        validate_forecast_contract(
            provenance, forecast_representation, capture_alignment,
            age_encoding, context=context,
            mean_source_aggregation=mean_source_aggregation)
    return checkpoint


def validate_evaluation_delay_contract(
        provenance, partner_delay_ms, capture_time_evidence=None,
        context="checkpoint", forecast_representation=None,
        capture_alignment=None, age_encoding=None,
        mean_source_aggregation=False):
    checkpoint = legacy.validate_evaluation_delay_contract(
        provenance, partner_delay_ms, context=context)
    if capture_time_evidence is not None:
        if capture_alignment is None:
            capture_alignment = capture_time_evidence
        if age_encoding is None:
            age_encoding = capture_time_evidence
    if forecast_representation is not None:
        validate_forecast_contract(
            provenance, forecast_representation, capture_alignment,
            age_encoding, context=context,
            mean_source_aggregation=mean_source_aggregation)
    return checkpoint
