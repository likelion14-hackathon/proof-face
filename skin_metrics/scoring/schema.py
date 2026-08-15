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
        ``True`` for proxy metrics that are not a direct measurement. All three
        current metrics measure something the camera actually resolves, so all
        three are ``False``.
    calibrated : bool
        ``True`` when the score came from a model fitted against real
        instrument readings, ``False`` when it came from the unsupervised
        composite of hand-weighted sub-features.
    predicted_value : float, optional
        The predicted instrument reading behind the score, when one exists
        (e.g. instrument pore count, expert pigmentation grade 0-5).
    predicted_units : str, optional
        Units of ``predicted_value``.
    """

    score: float = Field(ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=1.0)
    raw_features: dict[str, Any] = Field(default_factory=dict)
    is_estimate: bool = False
    calibrated: bool = False
    predicted_value: float | None = None
    predicted_units: str | None = None


class FaceScale(BaseModel):
    """How large the face was in frame, and what normalisation was applied.

    Attributes
    ----------
    eye_span_px : float
        Native outer-eye-corner separation, before resizing.
    scale_factor : float
        Resize factor applied to reach the canonical texture scale
        (``1.0`` = none).
    under_resolved : bool
        ``True`` when the face was smaller than the normalisation target, so
        texture-derived features (pores) are less reliable.
    """

    eye_span_px: float = 0.0
    scale_factor: float = 1.0
    under_resolved: bool = False


class SkinReport(BaseModel):
    """Full skin analysis report for one image."""

    pigmentation: MetricScore
    erythema: MetricScore
    pores: MetricScore
    roi_breakdown: dict[str, dict[str, Any]] = Field(default_factory=dict)
    calibration_status: Literal["reference", "grayworld", "none"]
    fitzpatrick_estimate: int = Field(ge=1, le=6)
    face_scale: FaceScale = Field(default_factory=FaceScale)
    calibration_profile: str | None = None
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER
