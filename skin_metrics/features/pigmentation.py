"""Pigmentation features (Phase 1, physics-based).

Provides the Individual Typology Angle (ITA), a melanin index from red-channel
reflectance, dark-spot detection, and tone evenness. All functions defend
against divide-by-zero and non-positive log inputs.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops

_EPS = 1e-6


def ita(lstar: np.ndarray | float, bstar: np.ndarray | float) -> np.ndarray | float:
    """Individual Typology Angle (degrees).

    ``ITA = arctan((L* - 50) / b*) * 180 / pi``.

    Parameters
    ----------
    lstar : float or numpy.ndarray
        CIE ``L*`` (``0-100``).
    bstar : float or numpy.ndarray
        CIE ``b*``.

    Returns
    -------
    float or numpy.ndarray
        ITA in degrees. Higher = lighter skin. ``b*`` values near zero are
        floored to a small epsilon (sign-preserving) to avoid division blow-ups.

    Notes
    -----
    Fitzpatrick-style typing thresholds on ITA: >55 very light, 41-55 light,
    28-41 intermediate, 10-28 tan, -30-10 brown, <-30 dark (Del Bino et al.).
    """
    L = np.asarray(lstar, dtype=np.float64)
    b = np.asarray(bstar, dtype=np.float64)
    sign = np.where(b < 0, -1.0, 1.0)
    b_safe = np.where(np.abs(b) < _EPS, sign * _EPS, b)
    result = np.arctan((L - 50.0) / b_safe) * 180.0 / np.pi
    if np.isscalar(lstar) and np.isscalar(bstar):
        return float(result)
    return result


def melanin_index(r_red: np.ndarray | float) -> np.ndarray | float:
    """Reflectance-based melanin index.

    ``MI = 100 * log10(1 / R_red)`` where ``R_red`` is red-channel reflectance
    in ``(0, 1]``.

    Parameters
    ----------
    r_red : float or numpy.ndarray
        Red-channel reflectance. Values are clipped to ``[eps, 1]`` so the log
        is always finite and non-negative.

    Returns
    -------
    float or numpy.ndarray
        Melanin index (higher = more melanin / darker).
    """
    r = np.clip(np.asarray(r_red, dtype=np.float64), _EPS, 1.0)
    mi = 100.0 * np.log10(1.0 / r)
    if np.isscalar(r_red):
        return float(mi)
    return mi


def spot_detection(
    l_channel: np.ndarray,
    mask: np.ndarray | None = None,
    sigma: float = 15.0,
    contrast_thresh: float = 6.0,
    min_area: int = 8,
) -> dict[str, float]:
    """Detect dark spots as local ``L*`` depressions.

    A local mean is computed with a Gaussian (``sigma`` px); pixels darker than
    the local mean by more than ``contrast_thresh`` are labelled as spots.

    Parameters
    ----------
    l_channel : numpy.ndarray
        ``(H, W)`` CIE ``L*`` image.
    mask : numpy.ndarray, optional
        Boolean ``(H, W)`` mask of valid pixels. If ``None``, all pixels are used.
    sigma : float, optional
        Gaussian sigma (pixels) for the local-mean background.
    contrast_thresh : float, optional
        Minimum ``L*`` deficit (local_mean - L) for a pixel to count as a spot.
    min_area : int, optional
        Minimum connected-component area (pixels) to count as a spot.

    Returns
    -------
    dict
        ``{"spot_area_ratio", "spot_count", "mean_contrast"}``. Ratios/contrast
        are ``0.0`` when there are no valid pixels or no spots.
    """
    L = np.asarray(l_channel, dtype=np.float64)
    if mask is None:
        mask = np.ones(L.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    n_valid = int(mask.sum())
    if n_valid == 0:
        return {"spot_area_ratio": 0.0, "spot_count": 0.0, "mean_contrast": 0.0}

    # Normalised Gaussian smoothing that ignores invalid pixels, so masked-out
    # regions don't drag the local mean.
    m = mask.astype(np.float64)
    smoothed_num = gaussian_filter(L * m, sigma=sigma)
    smoothed_den = gaussian_filter(m, sigma=sigma)
    local_mean = smoothed_num / np.maximum(smoothed_den, _EPS)

    deficit = local_mean - L  # positive where darker than surroundings
    spot_pixels = (deficit > contrast_thresh) & mask

    labeled = label(spot_pixels)
    count = 0
    contrasts: list[float] = []
    total_area = 0
    for region in regionprops(labeled):
        if region.area < min_area:
            continue
        count += 1
        total_area += region.area
        coords = region.coords
        contrasts.append(float(deficit[coords[:, 0], coords[:, 1]].mean()))

    return {
        "spot_area_ratio": float(total_area) / float(n_valid),
        "spot_count": float(count),
        "mean_contrast": float(np.mean(contrasts)) if contrasts else 0.0,
    }


def evenness(l_channel: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Tone evenness = standard deviation of ``L*`` over valid pixels.

    Parameters
    ----------
    l_channel : numpy.ndarray
        ``(H, W)`` CIE ``L*`` image.
    mask : numpy.ndarray, optional
        Boolean ``(H, W)`` mask of valid pixels.

    Returns
    -------
    float
        ``L*`` standard deviation (higher = less even tone). ``0.0`` if fewer
        than two valid pixels.
    """
    L = np.asarray(l_channel, dtype=np.float64)
    if mask is None:
        vals = L.ravel()
    else:
        vals = L[np.asarray(mask, dtype=bool)]
    if vals.size < 2:
        return 0.0
    return float(np.std(vals))


def estimate_fitzpatrick(ita_value: float, ita_boundaries: list[float]) -> int:
    """Estimate a Fitzpatrick type (1-6) from an ITA value.

    Parameters
    ----------
    ita_value : float
        Aggregate ITA (degrees) for the face.
    ita_boundaries : list of float
        Five descending ITA cut-offs separating types 1|2|3|4|5|6.

    Returns
    -------
    int
        Fitzpatrick type in ``1..6`` (higher = darker).
    """
    fitz = 1
    for boundary in ita_boundaries:
        if ita_value <= boundary:
            fitz += 1
    return int(min(max(fitz, 1), 6))
