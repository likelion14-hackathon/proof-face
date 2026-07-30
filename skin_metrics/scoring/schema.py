"""Pydantic output schema for skin_metrics reports."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .. import DISCLAIMER


class MetricScore(BaseModel):
    """A single 0-100 metric score with confidence and raw features.

    Attributes
    ----------
    score : float
        Normalised 0-100 "condition index" (higher = more pronounced).
    confidence : float
        Confidence in ``[0, 1]``, lowered by poor calibration / few valid ROIs.
    raw_features : dict
        The underlying physics feature values.
    is_estimate : bool
        ``True`` for proxy metrics that are not a direct measurement
        (always ``True`` for hydration).
    """

    score: float = Field(ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=1.0)
    raw_features: dict[str, Any] = Field(default_factory=dict)
    is_estimate: bool = False


class SkinReport(BaseModel):
    """Full skin analysis report for one image."""

    pigmentation: MetricScore
    erythema: MetricScore
    hydration: MetricScore
    roi_breakdown: dict[str, dict[str, Any]] = Field(default_factory=dict)
    calibration_status: Literal["reference", "grayworld", "none"]
    fitzpatrick_estimate: int = Field(ge=1, le=6)
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER
