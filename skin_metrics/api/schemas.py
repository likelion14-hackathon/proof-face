"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from .. import DISCLAIMER, __version__
from ..scoring.schema import SkinReport

# 'x,y,w,h' of an in-frame neutral gray/white patch.
Bbox = Annotated[list[int], Field(min_length=4, max_length=4)]

# ITA -> skin_tone linear map. Del Bino boundaries put "dark" at ITA <= -30
# and "very light" at ITA > 55; the Korean cohort sits around 32 +/- 6.
_ITA_DARK = -30.0
_ITA_LIGHT = 55.0


class AnalyzeRequest(BaseModel):
    """Body of ``POST /analyze``.

    The schema example is pinned to the common case (URL only). Without it the
    docs auto-generate ``"reference_bbox": [0, 0, 0, 0]`` from the type alone,
    which the validator below rejects - a "Try it out" body that always 422s.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"image_url": "https://example.com/face.jpg"}],
        }
    )

    image_url: HttpUrl = Field(
        description="Publicly reachable http(s) URL of the face image to analyze."
    )
    reference_bbox: Optional[Bbox] = Field(
        default=None,
        description=(
            "Optional [x, y, w, h] of a neutral gray/white reference patch that is "
            "visible in the image (a gray card or sheet of white paper). Omit it "
            "unless such a patch is really there: it is taken as neutral and drives "
            "white balance, so a wrong box skews every score. When present it "
            "improves calibration and raises confidence. Width and height must be "
            "> 0."
        ),
        examples=[[10, 10, 40, 40]],
    )

    @field_validator("reference_bbox")
    @classmethod
    def _check_bbox(cls, value: Optional[list[int]]) -> Optional[list[int]]:
        """Reject negative origins and non-positive extents."""
        if value is None:
            return value
        x, y, w, h = value
        if x < 0 or y < 0:
            raise ValueError("reference_bbox x and y must be >= 0")
        if w <= 0 or h <= 0:
            raise ValueError("reference_bbox width and height must be > 0")
        return value


class AnalyzeResponse(BaseModel):
    """Body of a successful ``POST /analyze``: the three 0-100 scores.

    Same envelope shape as ``POST /analyze/simple`` (flat scores +
    ``confidence`` + ``warnings`` + ``disclaimer``), just on the 0-100 metric
    scale. Clients needing the full :class:`SkinReport` (ROI breakdown, raw
    features, Fitzpatrick estimate) should use the CLI ``analyze`` command.
    """

    pigmentation: float = Field(ge=0.0, le=100.0, description="Higher = more pigmentation.")
    erythema: float = Field(ge=0.0, le=100.0, description="Higher = more redness.")
    hydration: float = Field(
        ge=0.0, le=100.0, description="Higher = moister (PROXY estimate, not a measurement)."
    )
    confidence: dict[str, float] = Field(
        description="Per-score confidence in [0, 1], from the underlying metrics."
    )
    warnings: list[str] = Field(default_factory=list)
    elapsed_ms: float = Field(description="Server-side wall time for fetch + analysis.")
    version: str = __version__
    disclaimer: str = DISCLAIMER

    @classmethod
    def from_report(cls, report: SkinReport, elapsed_ms: float) -> "AnalyzeResponse":
        """Flatten a full :class:`SkinReport` into the three 0-100 scores.

        Parameters
        ----------
        report : SkinReport
            Output of :func:`skin_metrics.pipeline.analyze`.
        elapsed_ms : float
            Wall time to report.

        Returns
        -------
        AnalyzeResponse
            The flattened response.
        """
        return cls(
            pigmentation=round(report.pigmentation.score, 2),
            erythema=round(report.erythema.score, 2),
            hydration=round(report.hydration.score, 2),
            confidence={
                "pigmentation": round(report.pigmentation.confidence, 3),
                "erythema": round(report.erythema.confidence, 3),
                "hydration": round(report.hydration.confidence, 3),
            },
            warnings=list(report.warnings),
            elapsed_ms=elapsed_ms,
        )


class SimpleAnalyzeResponse(BaseModel):
    """Body of a successful ``POST /analyze/simple``.

    Three 0-10 consumer-facing scores derived from the full report:

    - ``skin_tone``: brightness of the skin tone, mapped linearly from the
      Individual Typology Angle (ITA in ``[-30, 55]`` degrees -> ``[0, 10]``).
      Higher = brighter. ITA is an *absolute colour* quantity, so it is the one
      field here that shifts with camera/lighting unless a ``reference_bbox``
      gray card is supplied.
    - ``dryness``: 당김·건조함 정도. Higher = drier/tighter. This is the
      hydration proxy re-oriented (``(100 - hydration score) / 10``); tightness
      is a sensation, not separately measurable from a photo, so it shares this
      value.
    - ``redness``: 붉은기. Higher = redder (``erythema score / 10``).
    """

    skin_tone: float = Field(ge=0.0, le=10.0, description="0 = dark, 10 = very bright.")
    dryness: float = Field(
        ge=0.0, le=10.0, description="당김·건조함: 0 = moist, 10 = very dry (proxy)."
    )
    redness: float = Field(ge=0.0, le=10.0, description="붉은기: 0 = none, 10 = strong.")
    confidence: dict[str, float] = Field(
        description="Per-score confidence in [0, 1], from the underlying metrics."
    )
    warnings: list[str] = Field(default_factory=list)
    elapsed_ms: float = Field(description="Server-side wall time for fetch + analysis.")
    version: str = __version__
    disclaimer: str = DISCLAIMER

    @classmethod
    def from_report(cls, report: SkinReport, elapsed_ms: float) -> "SimpleAnalyzeResponse":
        """Collapse a full :class:`SkinReport` into the three 0-10 scores.

        Parameters
        ----------
        report : SkinReport
            Output of :func:`skin_metrics.pipeline.analyze`.
        elapsed_ms : float
            Wall time to report.

        Returns
        -------
        SimpleAnalyzeResponse
            The mapped response.
        """
        ita = report.pigmentation.raw_features.get("ita")
        if ita is None:  # defensive: fall back to the Fitzpatrick bucket centre
            centres = {1: 60.0, 2: 48.0, 3: 34.5, 4: 19.0, 5: -10.0, 6: -40.0}
            ita = centres.get(report.fitzpatrick_estimate, 30.0)
        tone = 10.0 * (float(ita) - _ITA_DARK) / (_ITA_LIGHT - _ITA_DARK)

        def _clip10(v: float) -> float:
            return round(min(max(v, 0.0), 10.0), 1)

        return cls(
            skin_tone=_clip10(tone),
            dryness=_clip10((100.0 - report.hydration.score) / 10.0),
            redness=_clip10(report.erythema.score / 10.0),
            confidence={
                "skin_tone": round(report.pigmentation.confidence, 3),
                "dryness": round(report.hydration.confidence, 3),
                "redness": round(report.erythema.confidence, 3),
            },
            warnings=list(report.warnings),
            elapsed_ms=elapsed_ms,
        )


class ErrorBody(BaseModel):
    """Details of a failed request."""

    code: str = Field(description="Stable machine-readable error code.")
    message: str


class ErrorResponse(BaseModel):
    """Error envelope returned for every 4xx/5xx from this API."""

    error: ErrorBody


class HealthResponse(BaseModel):
    """Body of ``GET /healthz``."""

    status: str = "ok"
    version: str = __version__
    face_model_available: bool = Field(
        description="Whether a MediaPipe FaceLandmarker model file was found."
    )
    detection_available: bool = Field(description="Whether the 'detection' extra is importable.")
    disclaimer: str = DISCLAIMER
