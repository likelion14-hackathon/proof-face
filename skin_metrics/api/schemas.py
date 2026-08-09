"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from .. import DISCLAIMER, __version__
from ..scoring.schema import SkinReport

# 'x,y,w,h' of an in-frame neutral gray/white patch.
Bbox = Annotated[list[int], Field(min_length=4, max_length=4)]


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


class SourceInfo(BaseModel):
    """Where the analyzed image came from and what it looked like."""

    url: str
    final_url: str
    content_type: Optional[str] = None
    bytes: int
    width: int
    height: int


class AnalyzeResponse(BaseModel):
    """Body of a successful ``POST /analyze``."""

    report: SkinReport
    source: SourceInfo
    elapsed_ms: float = Field(description="Server-side wall time for fetch + analysis.")
    version: str = __version__


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
