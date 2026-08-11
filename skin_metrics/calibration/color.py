"""Color calibration for skin-metric imaging.

This module standardises incoming camera imagery so downstream physics-based
features operate in a well-defined color space:

1. sRGB gamma removal (:func:`linearize_srgb`),
2. white balance (:func:`white_balance_grayworld` /
   :func:`white_balance_from_reference`),
3. optional 3x3 color-correction matrix from a color-checker
   (:func:`estimate_ccm` / :func:`apply_ccm`),
4. CIELab conversion under a D65 illuminant (:func:`rgb_to_lab`).

Every calibration entry point reports whether it succeeded so the pipeline can
lower the confidence of un-calibrated images. Use :func:`calibrate_image` for
the orchestrated flow.

Conventions
-----------
All images are float ``numpy.ndarray`` of shape ``(H, W, 3)`` in **RGB** order.
"sRGB" images are display-encoded in ``[0, 1]``; "linear" images are
scene-linear in ``[0, 1]``. Use :func:`to_float01` to convert ``uint8`` input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

try:  # colour-science is a core dependency but keep a clear error if missing.
    import colour
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "colour-science is required for skin_metrics.calibration.color; "
        "install with `uv sync` (it is a core dependency)."
    ) from exc

CalibrationStatus = Literal["reference", "grayworld", "none"]

_EPS = 1e-8


@dataclass
class CalibrationResult:
    """Outcome of a calibration pass.

    Attributes
    ----------
    image : numpy.ndarray
        Calibrated **linear** RGB image, float in ``[0, 1]``.
    status : {"reference", "grayworld", "none"}
        Which white-balance path produced ``image``.
    success : bool
        ``True`` when a *reliable* calibration was applied (reference white or a
        color-checker CCM). Grayworld and no-op both report ``False`` so the
        pipeline can down-weight confidence.
    ccm_applied : bool
        Whether a 3x3 color-correction matrix was applied.
    notes : list of str
        Human-readable diagnostics.
    """

    image: np.ndarray
    status: CalibrationStatus
    success: bool
    ccm_applied: bool = False
    notes: list[str] = field(default_factory=list)


def to_float01(img: np.ndarray) -> np.ndarray:
    """Convert an image to float32 in ``[0, 1]``.

    Parameters
    ----------
    img : numpy.ndarray
        ``uint8`` (``0-255``) or float image.

    Returns
    -------
    numpy.ndarray
        float32 image clipped to ``[0, 1]``.
    """
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
    return np.clip(arr, 0.0, 1.0)


def linearize_srgb(img: np.ndarray) -> np.ndarray:
    """Convert display sRGB values to scene-linear RGB.

    Applies the inverse sRGB electro-optical transfer function (gamma).

    Parameters
    ----------
    img : numpy.ndarray
        sRGB image, float or ``uint8``. Values are clipped to ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        Linear RGB image, float32 in ``[0, 1]``.
    """
    s = to_float01(img)
    linear = np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
    return np.clip(linear.astype(np.float32), 0.0, 1.0)


def encode_srgb(img_linear: np.ndarray) -> np.ndarray:
    """Encode scene-linear RGB back to display sRGB (inverse of :func:`linearize_srgb`).

    Parameters
    ----------
    img_linear : numpy.ndarray
        Linear RGB image, float in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        sRGB-encoded image, float32 in ``[0, 1]``.
    """
    lin = np.clip(np.asarray(img_linear, dtype=np.float32), 0.0, 1.0)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * (lin ** (1 / 2.4)) - 0.055)
    return np.clip(srgb.astype(np.float32), 0.0, 1.0)


def _apply_channel_gains(img: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Multiply per-channel gains and clip to ``[0, 1]``."""
    out = img.astype(np.float32) * gains.reshape(1, 1, 3)
    return np.clip(out, 0.0, 1.0)


def white_balance_grayworld(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    max_sample: int = 200_000,
) -> tuple[np.ndarray, bool]:
    """Gray-world white balance.

    Scales each channel so their means are equal (the gray-world assumption).
    This is a weak fallback: it reports ``success=False`` because it can bias
    scenes with a dominant true color.

    .. warning::
       **Never estimate the gains over a face-filling portrait.** The gray-world
       assumption is that the scene averages to neutral; when skin is most of
       the frame, the estimate is the skin colour itself, so applying it divides
       skin chromaticity out of the image -- ``a*`` and ``b*`` collapse to ~0
       and ITA saturates at +/-90 degrees. Pass ``mask`` to restrict the
       estimate to non-skin (background) pixels.

    Parameters
    ----------
    img : numpy.ndarray
        Linear RGB image, float in ``[0, 1]``.
    mask : numpy.ndarray, optional
        Boolean ``(H, W)`` mask selecting the pixels the gains are estimated
        from. Gains are still applied to the whole image. ``None`` uses every
        pixel (see the warning above).
    max_sample : int, optional
        Cap on the number of pixels used for the estimate. A channel mean is
        already precise to <0.1% at this many samples, and materialising a
        multi-megapixel boolean selection costs more than the estimate is
        worth.

    Returns
    -------
    balanced : numpy.ndarray
        White-balanced linear RGB, float32 in ``[0, 1]``.
    success : bool
        Always ``False`` (grayworld is a fallback, not a reliable calibration).
    """
    arr = np.asarray(img, dtype=np.float32)
    flat = arr.reshape(-1, 3)
    if mask is None:
        sample = flat[:: max(1, flat.shape[0] // max_sample + 1)]
    else:
        idx = np.flatnonzero(np.asarray(mask, dtype=bool).reshape(-1))
        if idx.size == 0:
            sample = flat[:: max(1, flat.shape[0] // max_sample + 1)]
        else:
            sample = flat[idx[:: max(1, idx.size // max_sample + 1)]]
    means = sample.mean(axis=0)
    gray = float(means.mean())
    # Guard divide-by-zero on dead channels: gain of 1.0 where mean ~ 0.
    gains = np.where(means > _EPS, gray / np.maximum(means, _EPS), 1.0)
    return _apply_channel_gains(arr, gains), False


def white_balance_from_reference(
    img: np.ndarray,
    ref_bbox: tuple[int, int, int, int],
    min_mean: float = 0.05,
    max_mean: float = 0.98,
) -> tuple[np.ndarray, bool]:
    """White balance from a neutral (gray/white) reference patch in-frame.

    The mean color of ``ref_bbox`` is forced to neutral by per-channel gains.

    Parameters
    ----------
    img : numpy.ndarray
        Linear RGB image, float in ``[0, 1]``.
    ref_bbox : tuple of int
        ``(x, y, w, h)`` bounding box of a gray card / white paper region.
    min_mean, max_mean : float, optional
        The reference region is rejected (``success=False``) if its overall mean
        brightness falls outside ``[min_mean, max_mean]`` (too dark / clipped).

    Returns
    -------
    balanced : numpy.ndarray
        White-balanced linear RGB, float32 in ``[0, 1]``.
    success : bool
        ``True`` when a usable neutral reference was found and applied.
    """
    arr = np.asarray(img, dtype=np.float32)
    h_img, w_img = arr.shape[:2]
    x, y, w, h = (int(v) for v in ref_bbox)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x + w), min(h_img, y + h)
    if x1 <= x0 or y1 <= y0:
        return arr, False

    patch = arr[y0:y1, x0:x1].reshape(-1, 3)
    if patch.size == 0:
        return arr, False

    means = patch.mean(axis=0)
    overall = float(means.mean())
    if overall < min_mean or overall > max_mean:
        # Too dark or (near-)clipped to trust as a neutral reference.
        return arr, False

    gains = np.where(means > _EPS, overall / np.maximum(means, _EPS), 1.0)
    return _apply_channel_gains(arr, gains), True


def estimate_ccm(
    detected_patches: np.ndarray,
    reference_patches: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Least-squares 3x3 color-correction matrix from color-checker patches.

    Solves ``M`` minimising ``|| detected @ M - reference ||`` so that
    ``corrected = detected @ M`` approximates the reference patch colors.

    Parameters
    ----------
    detected_patches : numpy.ndarray
        Shape ``(N, 3)`` measured RGB of the detected patches (linear RGB).
    reference_patches : numpy.ndarray
        Shape ``(N, 3)`` known reference RGB of the same patches.

    Returns
    -------
    ccm : numpy.ndarray
        ``(3, 3)`` color-correction matrix applied as ``rgb @ ccm``.
    residual_rms : float
        RMS residual per channel after correction (lower is better).

    Raises
    ------
    ValueError
        If fewer than 3 patches are given or shapes are incompatible.
    """
    det = np.asarray(detected_patches, dtype=np.float64)
    ref = np.asarray(reference_patches, dtype=np.float64)
    if det.ndim != 2 or det.shape[1] != 3 or ref.shape != det.shape:
        raise ValueError(
            f"patches must be (N, 3) with matching shapes; "
            f"got {det.shape} and {ref.shape}"
        )
    if det.shape[0] < 3:
        raise ValueError("need at least 3 patches to estimate a 3x3 CCM")

    # Solve det @ M = ref  ->  M = lstsq(det, ref)
    ccm, _residuals, _rank, _sv = np.linalg.lstsq(det, ref, rcond=None)
    corrected = det @ ccm
    residual_rms = float(np.sqrt(np.mean((corrected - ref) ** 2)))
    return ccm.astype(np.float32), residual_rms


def apply_ccm(img: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    """Apply a 3x3 color-correction matrix to an image.

    Parameters
    ----------
    img : numpy.ndarray
        Linear RGB image ``(H, W, 3)``, float in ``[0, 1]``.
    ccm : numpy.ndarray
        ``(3, 3)`` matrix from :func:`estimate_ccm`.

    Returns
    -------
    numpy.ndarray
        Corrected linear RGB, float32 clipped to ``[0, 1]``.
    """
    arr = np.asarray(img, dtype=np.float32)
    ccm = np.asarray(ccm, dtype=np.float32)
    if ccm.shape != (3, 3):
        raise ValueError(f"ccm must be (3, 3), got {ccm.shape}")
    flat = arr.reshape(-1, 3) @ ccm
    return np.clip(flat.reshape(arr.shape), 0.0, 1.0)


def rgb_to_lab(img: np.ndarray, assume_linear: bool = True) -> np.ndarray:
    """Convert RGB to CIELab (L*a*b*) under a D65 illuminant via colour-science.

    Parameters
    ----------
    img : numpy.ndarray
        RGB image ``(H, W, 3)`` in ``[0, 1]``.
    assume_linear : bool, optional
        If ``True`` (default) the input is treated as scene-linear RGB and the
        sRGB decoding step is skipped. If ``False`` the input is treated as
        display sRGB and decoded first.

    Returns
    -------
    numpy.ndarray
        CIELab image ``(H, W, 3)`` where ``L*`` is in ``[0, 100]``.

    Notes
    -----
    Uses the sRGB colourspace primaries and its D65 whitepoint, so a*/b* are
    referenced to D65. L* follows the CIE 1976 definition.
    """
    arr = np.asarray(img, dtype=np.float64)
    colourspace = colour.RGB_COLOURSPACES["sRGB"]
    xyz = colour.RGB_to_XYZ(
        arr,
        colourspace,
        apply_cctf_decoding=not assume_linear,
    )
    lab = colour.XYZ_to_Lab(xyz, colourspace.whitepoint)
    return lab.astype(np.float32)


def calibrate_image(
    img: np.ndarray,
    ref_bbox: tuple[int, int, int, int] | None = None,
    ccm: np.ndarray | None = None,
    *,
    fallback: Literal["background", "grayworld", "none"] = "background",
    background_mask: np.ndarray | None = None,
    min_background_ratio: float = 0.15,
) -> CalibrationResult:
    """Run the full calibration flow and report reliability.

    Order: linearize -> (optional CCM) -> white balance. When ``ref_bbox`` is
    given and yields a usable neutral patch, status is ``"reference"`` and
    ``success=True``. Otherwise the ``fallback`` decides what happens:

    ``"background"`` (default)
        Gray-world estimated **only over non-skin pixels** (``background_mask``),
        which keeps the gray-world assumption defensible. If too little
        background is available, degrade to ``"none"`` rather than to a
        face-driven estimate.
    ``"grayworld"``
        Legacy whole-image gray-world. Valid for scenes where the face is a
        small part of the frame; destroys skin chromaticity otherwise (see
        :func:`white_balance_grayworld`).
    ``"none"``
        No white balance -- trust the camera's own auto white balance, which is
        what the sRGB JPEG already encodes.

    Parameters
    ----------
    img : numpy.ndarray
        Input sRGB image (``uint8`` or float in ``[0, 1]``).
    ref_bbox : tuple of int, optional
        ``(x, y, w, h)`` of an in-frame neutral reference. If provided and valid,
        enables reliable reference white balance.
    ccm : numpy.ndarray, optional
        Pre-estimated ``(3, 3)`` color-correction matrix (see
        :func:`estimate_ccm`) to apply before white balance.
    fallback : {"background", "grayworld", "none"}, optional
        Strategy when no usable reference patch is present.
    background_mask : numpy.ndarray, optional
        Boolean ``(H, W)`` mask that is ``True`` on non-skin pixels. Required
        for the ``"background"`` fallback to do anything.
    min_background_ratio : float, optional
        Minimum fraction of the frame that must be background for the
        ``"background"`` fallback to trust its estimate.

    Returns
    -------
    CalibrationResult
        Calibrated linear image plus status/reliability flags.
    """
    notes: list[str] = []
    linear = linearize_srgb(img)

    ccm_applied = False
    if ccm is not None:
        linear = apply_ccm(linear, ccm)
        ccm_applied = True
        notes.append("Applied color-checker CCM.")

    if ref_bbox is not None:
        balanced, ok = white_balance_from_reference(linear, ref_bbox)
        if ok:
            notes.append("Reference white balance applied.")
            return CalibrationResult(
                image=balanced,
                status="reference",
                success=True,
                ccm_applied=ccm_applied,
                notes=notes,
            )
        notes.append(f"Reference patch unusable; fell back to '{fallback}'.")

    if fallback == "grayworld":
        balanced, _ = white_balance_grayworld(linear)
        notes.append("Whole-image gray-world white balance (no reference).")
        return CalibrationResult(
            image=balanced,
            status="grayworld",
            success=ccm_applied,
            ccm_applied=ccm_applied,
            notes=notes,
        )

    if fallback == "background" and background_mask is not None:
        bg = np.asarray(background_mask, dtype=bool)
        ratio = float(bg.mean()) if bg.size else 0.0
        if ratio >= min_background_ratio:
            balanced, _ = white_balance_grayworld(linear, mask=bg)
            notes.append(
                f"Background gray-world white balance ({ratio:.0%} of frame is "
                "non-skin)."
            )
            return CalibrationResult(
                image=balanced,
                status="grayworld",
                success=ccm_applied,
                ccm_applied=ccm_applied,
                notes=notes,
            )
        notes.append(
            f"Only {ratio:.0%} of the frame is non-skin (need "
            f"{min_background_ratio:.0%}); skipped white balance to avoid "
            "neutralising skin colour."
        )

    # No white balance: the sRGB input already carries the camera's own AWB.
    notes.append("No white balance applied; relying on camera AWB.")
    return CalibrationResult(
        image=np.asarray(linear, dtype=np.float32),
        status="none",
        success=ccm_applied,
        ccm_applied=ccm_applied,
        notes=notes,
    )
