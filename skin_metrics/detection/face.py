"""Face landmark detection and ROI extraction.

Uses MediaPipe FaceMesh (468 landmarks) to locate five skin ROIs -- forehead,
left cheek, right cheek, nose, chin -- and to mask out artifacts (specular
glare, shadows, and hair/brow/lip regions) so downstream features only see
clean skin pixels.

MediaPipe is imported lazily inside :func:`detect_landmarks` so the rest of the
package (and the Phase 1 unit tests, which build synthetic landmark arrays) can
run without the heavy optional dependency. Install it with the ``detection``
extra: ``uv sync --extra detection``.

All geometry helpers operate on a landmark array of shape ``(468, 2)`` in pixel
coordinates ``(x, y)``, so they are fully testable with synthetic inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from skimage.color import rgb2hsv

from ..calibration.color import encode_srgb

# --- ROI landmark index sets (MediaPipe FaceMesh, 468-point topology) -------
# Each ROI is the convex hull of a curated set of interior landmarks; this is
# robust without needing exact boundary loops.
ROI_LANDMARKS: dict[str, list[int]] = {
    "forehead": [10, 151, 9, 8, 107, 336, 66, 296, 69, 299, 337, 108],
    "left_cheek": [205, 50, 101, 118, 117, 123, 147, 187, 207, 216, 137],
    "right_cheek": [425, 280, 330, 347, 346, 352, 376, 411, 427, 436, 366],
    "nose": [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 45, 275],
    "chin": [152, 175, 148, 377, 400, 379, 365, 136, 149, 378, 199, 200],
}

# --- exclusion regions (hair proxy handled via shadow + these landmark sets) -
EXCLUSION_LANDMARKS: dict[str, list[int]] = {
    "left_eye": [33, 133, 159, 145, 153, 144, 163, 7, 246, 161, 160, 158],
    "right_eye": [362, 263, 386, 374, 380, 373, 390, 249, 466, 388, 387, 385],
    "left_brow": [70, 63, 105, 66, 107, 46, 53, 52, 65, 55],
    "right_brow": [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
    "lips": [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78,
    ],
}

# Outer eye corners; their separation is the scale yardstick for the face.
EYE_CORNER_LANDMARKS: tuple[int, int] = (33, 263)

_EPS = 1e-8


def eye_span(landmarks: np.ndarray) -> float:
    """Distance in pixels between the two outer eye corners.

    A stable, expression-invariant proxy for how large the face is in frame.
    Texture features (GLCM, LBP, micro-wrinkle density) operate at fixed pixel
    offsets, so they are only comparable between images taken at the same face
    scale -- this is what :func:`scale_factor_for` normalises.

    Parameters
    ----------
    landmarks : numpy.ndarray
        ``(468, 2)`` FaceMesh landmarks in pixel coordinates.

    Returns
    -------
    float
        Euclidean distance between landmarks 33 and 263.
    """
    left, right = EYE_CORNER_LANDMARKS
    return float(np.linalg.norm(np.asarray(landmarks[left], dtype=np.float64)
                                - np.asarray(landmarks[right], dtype=np.float64)))


def scale_factor_for(landmarks: np.ndarray, target_eye_span: float) -> float:
    """Resize factor that brings the face to a canonical scale.

    Downscale-only: a face already smaller than ``target_eye_span`` is left
    alone rather than upsampled, since interpolation invents no detail and
    would systematically smooth the texture features (reading as "less dry").
    Callers should instead lower confidence for such images.

    Parameters
    ----------
    landmarks : numpy.ndarray
        ``(468, 2)`` FaceMesh landmarks in pixel coordinates.
    target_eye_span : float
        Desired outer-eye-corner separation in pixels.

    Returns
    -------
    float
        Factor in ``(0, 1]`` to multiply image dimensions by. ``1.0`` means
        "leave as-is".
    """
    span = eye_span(landmarks)
    if span <= _EPS or target_eye_span <= _EPS:
        return 1.0
    return float(min(1.0, target_eye_span / span))


@dataclass
class ROIResult:
    """A single extracted ROI.

    Attributes
    ----------
    name : str
        ROI name (one of the keys of :data:`ROI_LANDMARKS`).
    region_mask : numpy.ndarray
        Boolean ``(H, W)`` mask of the ROI polygon (before artifact removal).
    valid_mask : numpy.ndarray
        Boolean ``(H, W)`` mask of clean skin pixels (artifacts removed).
    valid_ratio : float
        ``valid_mask.sum() / region_mask.sum()`` in ``[0, 1]``.
    """

    name: str
    region_mask: np.ndarray
    valid_mask: np.ndarray
    valid_ratio: float


def detect_landmarks(
    img: np.ndarray, model_path: str | None = None
) -> np.ndarray | None:
    """Detect 468 FaceMesh landmarks in an image.

    Parameters
    ----------
    img : numpy.ndarray
        RGB image ``(H, W, 3)``. ``uint8`` or float in ``[0, 1]``.
    model_path : str, optional
        Path to a ``face_landmarker.task`` model, used only by the MediaPipe
        Tasks API (>= 1.0). Ignored by the legacy solutions API.

    Returns
    -------
    numpy.ndarray or None
        ``(468, 2)`` array of ``(x, y)`` pixel coordinates, or ``None`` if no
        face was found.

    Raises
    ------
    ImportError
        If MediaPipe is not installed (install the ``detection`` extra).
    FileNotFoundError
        If only the MediaPipe *Tasks* API is available and the
        ``face_landmarker.task`` model file cannot be located (see
        :func:`resolve_face_model`).

    Notes
    -----
    Supports both MediaPipe APIs:

    - the legacy ``mp.solutions.face_mesh`` (MediaPipe 0.10.x), and
    - the newer ``mediapipe.tasks`` FaceLandmarker (MediaPipe >= 1.0), which
      needs a downloaded ``.task`` model. Set the model path via ``model_path``
      or the ``SKIN_METRICS_FACE_MODEL`` env var, or run
      :func:`ensure_face_model` once to download it.
    """
    try:
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "mediapipe is required for face detection; install with "
            "`uv sync --extra detection`."
        ) from exc

    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    h, w = arr.shape[:2]

    # --- legacy solutions API (MediaPipe 0.10.x) ---
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        mp_face = mp.solutions.face_mesh
        with mp_face.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as mesh:
            result = mesh.process(arr)
        if not result.multi_face_landmarks:
            return None
        lms = result.multi_face_landmarks[0].landmark
        return np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)

    # --- Tasks API (MediaPipe >= 1.0) ---
    return _detect_landmarks_tasks(mp, arr, model_path)


def resolve_face_model(model_path: str | None = None) -> "Path | None":
    """Locate the FaceLandmarker ``.task`` model for the Tasks API.

    Resolution order: explicit ``model_path`` -> ``SKIN_METRICS_FACE_MODEL``
    env var -> default cache ``~/.cache/skin_metrics/face_landmarker.task``.

    Parameters
    ----------
    model_path : str, optional
        Explicit path to a ``.task`` model.

    Returns
    -------
    pathlib.Path or None
        The first existing candidate path, or ``None`` if none exist.
    """
    import os
    from pathlib import Path

    candidates = []
    if model_path:
        candidates.append(Path(model_path))
    env = os.environ.get("SKIN_METRICS_FACE_MODEL")
    if env:
        candidates.append(Path(env))
    candidates.append(Path.home() / ".cache" / "skin_metrics" / "face_landmarker.task")
    for c in candidates:
        if c.exists():
            return c
    return None


FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def ensure_face_model(dest: str | None = None) -> "Path":
    """Download the FaceLandmarker ``.task`` model if not already present.

    Downloads ~3.8 MB from Google's MediaPipe model storage
    (:data:`FACE_MODEL_URL`) to the cache path (or ``dest``).

    Parameters
    ----------
    dest : str, optional
        Destination path. Defaults to
        ``~/.cache/skin_metrics/face_landmarker.task``.

    Returns
    -------
    pathlib.Path
        Path to the model file.
    """
    import urllib.request
    from pathlib import Path

    target = (
        Path(dest)
        if dest
        else Path.home() / ".cache" / "skin_metrics" / "face_landmarker.task"
    )
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(FACE_MODEL_URL, target)  # noqa: S310
    return target


def _detect_landmarks_tasks(mp, arr: np.ndarray, model_path: str | None) -> np.ndarray | None:
    """FaceLandmarker (Tasks API) landmark detection."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model = resolve_face_model(model_path)
    if model is None:
        raise FileNotFoundError(
            "MediaPipe Tasks API needs a FaceLandmarker model file. Download it "
            "once with `skin_metrics.detection.face.ensure_face_model()` or set "
            "SKIN_METRICS_FACE_MODEL to a 'face_landmarker.task' path.\n"
            f"Model URL: {FACE_MODEL_URL}"
        )

    h, w = arr.shape[:2]
    base_options = mp_python.BaseOptions(model_asset_path=str(model))
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(arr))
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None
    # Tasks API returns 478 points (468 mesh + iris); keep the base 468 mesh.
    lms = result.face_landmarks[0][:468]
    return np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)


def _polygon_mask(
    landmarks: np.ndarray,
    indices: list[int],
    shape: tuple[int, int],
) -> np.ndarray:
    """Rasterise the convex hull of selected landmarks into a boolean mask."""
    h, w = shape
    pts = landmarks[indices].astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    hull = cv2.convexHull(pts)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)


def face_mask(
    landmarks: np.ndarray,
    shape: tuple[int, int],
    dilate_frac: float = 0.35,
) -> np.ndarray:
    """Boolean mask covering the face (and a margin for hair/neck).

    Used to *exclude* skin from illuminant estimation: gray-world gains
    estimated over a face-filling portrait neutralise skin colour itself (see
    :func:`skin_metrics.calibration.color.white_balance_grayworld`).

    Parameters
    ----------
    landmarks : numpy.ndarray
        ``(468, 2)`` FaceMesh landmarks in pixel coordinates.
    shape : tuple of int
        ``(H, W)`` of the target mask.
    dilate_frac : float, optional
        Grow the region by this fraction of its size, so hair, ears, and neck
        -- which are also not neutral -- stay out of the background estimate.

    Returns
    -------
    numpy.ndarray
        Boolean ``(H, W)`` mask, ``True`` inside the face region.

    Notes
    -----
    The margin is added by pushing the hull vertices out from their centroid,
    not by a morphological dilation: at these margins the structuring element
    would be hundreds of pixels across, which costs seconds per image.
    """
    pts = np.asarray(landmarks, dtype=np.float64)
    centroid = pts.mean(axis=0)
    grown = centroid + (pts - centroid) * (1.0 + float(dilate_frac))
    return _polygon_mask(grown, list(range(len(grown))), shape)


def exclusion_mask(
    landmarks: np.ndarray,
    shape: tuple[int, int],
    dilate_px: int = 6,
) -> np.ndarray:
    """Boolean mask of eye/brow/lip regions to exclude from skin ROIs.

    Parameters
    ----------
    landmarks : numpy.ndarray
        ``(468, 2)`` landmark coordinates.
    shape : tuple of int
        ``(H, W)`` of the target image.
    dilate_px : int, optional
        Dilation radius (pixels) applied to each region for a safety margin.

    Returns
    -------
    numpy.ndarray
        Boolean ``(H, W)`` mask, ``True`` where pixels should be excluded.
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for indices in EXCLUSION_LANDMARKS.values():
        mask |= _polygon_mask(landmarks, indices, shape)
    if dilate_px > 0 and mask.any():
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    return mask


def mask_artifacts(
    rgb_linear: np.ndarray,
    lab: np.ndarray,
    region_mask: np.ndarray,
    *,
    exclude_mask: np.ndarray | None = None,
    glare_v_min: float = 0.92,
    glare_s_max: float = 0.15,
    shadow_percentile: float = 5.0,
) -> np.ndarray:
    """Return a boolean mask of clean skin pixels within a region.

    Removes, from ``region_mask``:

    - **specular glare**: HSV ``V > glare_v_min`` AND ``S < glare_s_max``,
    - **shadows**: ``L*`` below the ``shadow_percentile`` of the region,
    - **hair/brow/lip**: any pixels in ``exclude_mask`` (also serves as a
      coarse hairline guard for the forehead ROI).

    Parameters
    ----------
    rgb_linear : numpy.ndarray
        Linear RGB image ``(H, W, 3)`` in ``[0, 1]``.
    lab : numpy.ndarray
        CIELab image ``(H, W, 3)`` (``L*`` in ``[0, 100]``).
    region_mask : numpy.ndarray
        Boolean ``(H, W)`` mask of the candidate ROI.
    exclude_mask : numpy.ndarray, optional
        Boolean ``(H, W)`` mask of landmark-based exclusions (eyes/brows/lips).
    glare_v_min, glare_s_max : float, optional
        HSV thresholds for specular-highlight detection.
    shadow_percentile : float, optional
        Percentile of ``L*`` below which pixels are treated as shadow.

    Returns
    -------
    numpy.ndarray
        Boolean ``(H, W)`` mask of valid skin pixels.
    """
    valid = region_mask.copy()
    if not valid.any():
        return valid

    # HSV glare from display-encoded sRGB (perceptual V/S).
    hsv = rgb2hsv(np.clip(encode_srgb(rgb_linear), 0.0, 1.0))
    v = hsv[..., 2]
    s = hsv[..., 1]
    glare = (v > glare_v_min) & (s < glare_s_max)
    valid &= ~glare

    # Shadow: low-L* tail computed within the *region* only.
    lstar = lab[..., 0]
    region_vals = lstar[region_mask]
    if region_vals.size > 0:
        thresh = float(np.percentile(region_vals, shadow_percentile))
        valid &= lstar > thresh

    if exclude_mask is not None:
        valid &= ~exclude_mask

    return valid


def extract_rois(
    rgb_linear: np.ndarray,
    lab: np.ndarray,
    landmarks: np.ndarray,
    *,
    min_valid_ratio: float = 0.60,
    glare_v_min: float = 0.92,
    glare_s_max: float = 0.15,
    shadow_percentile: float = 5.0,
) -> dict[str, ROIResult | None]:
    """Extract all five skin ROIs with artifacts masked.

    Parameters
    ----------
    rgb_linear : numpy.ndarray
        Linear RGB image ``(H, W, 3)`` in ``[0, 1]``.
    lab : numpy.ndarray
        Matching CIELab image ``(H, W, 3)``.
    landmarks : numpy.ndarray
        ``(468, 2)`` FaceMesh landmarks.
    min_valid_ratio : float, optional
        ROIs whose valid-pixel ratio is below this are returned as ``None``.
    glare_v_min, glare_s_max, shadow_percentile : float, optional
        Passed to :func:`mask_artifacts`.

    Returns
    -------
    dict
        Maps each ROI name to a :class:`ROIResult`, or ``None`` if it failed the
        valid-ratio gate.
    """
    shape = rgb_linear.shape[:2]
    excl = exclusion_mask(landmarks, shape)
    out: dict[str, ROIResult | None] = {}
    for name, indices in ROI_LANDMARKS.items():
        region = _polygon_mask(landmarks, indices, shape)
        # The forehead ROI keeps the brow exclusion (hairline/brow guard); other
        # ROIs also drop any accidental eye/lip overlap.
        valid = mask_artifacts(
            rgb_linear,
            lab,
            region,
            exclude_mask=excl,
            glare_v_min=glare_v_min,
            glare_s_max=glare_s_max,
            shadow_percentile=shadow_percentile,
        )
        region_count = int(region.sum())
        ratio = float(valid.sum()) / max(region_count, 1)
        if region_count == 0 or ratio < min_valid_ratio:
            out[name] = None
        else:
            out[name] = ROIResult(
                name=name,
                region_mask=region,
                valid_mask=valid,
                valid_ratio=ratio,
            )
    return out


def roi_pixels(image: np.ndarray, roi: ROIResult) -> np.ndarray:
    """Return the valid pixels of an ROI as an ``(N, C)`` array.

    Parameters
    ----------
    image : numpy.ndarray
        ``(H, W)`` or ``(H, W, C)`` image to sample.
    roi : ROIResult
        ROI whose ``valid_mask`` selects the pixels.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` for single-channel input or ``(N, C)`` for multi-channel.
    """
    return image[roi.valid_mask]
