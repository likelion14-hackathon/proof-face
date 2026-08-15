"""Surface-texture features, which drive the pore metric (Phase 1).

Pore openings are one of the few skin properties a camera really does resolve:
they are sub-millimetre pits that scatter light differently from the surrounding
surface, so they show up directly as high-frequency texture. That is why these
functions carry no "proxy" hedge -- unlike the moisture metric they replaced,
which tried to infer stratum-corneum water content (a Corneometer capacitance
reading) from surface optics that barely encode it.

Three of these -- :func:`texture_stats` (``lbp_uniformity``, ``glcm_contrast``)
and :func:`scaling_index` -- make up the fitted pore composite; see
``composite.pores`` in ``config.yaml`` for the weights and the measured
agreement. The rest are extracted and carried in the report, but the subset
search left them out.

All functions guard divide-by-zero and empty inputs.
"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2hsv
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import frangi

from ..calibration.color import encode_srgb

_EPS = 1e-8


def _as_gray_u8(gray: np.ndarray, levels: int = 32) -> np.ndarray:
    """Quantise a float/uint8 grayscale image to ``levels`` for GLCM/LBP."""
    g = np.asarray(gray)
    if g.dtype == np.uint8:
        g = g.astype(np.float64) / 255.0
    g = np.clip(g.astype(np.float64), 0.0, 1.0)
    q = np.round(g * (levels - 1)).astype(np.uint8)
    return q


def specular_ratio(
    rgb_linear: np.ndarray,
    mask: np.ndarray | None = None,
    v_min: float = 0.90,
    s_max: float = 0.20,
) -> float:
    """Fraction of specular (shiny) pixels within the region.

    Specular highlights (high HSV ``V``, low ``S``) track surface oil and
    shine. Extracted and reported, but not part of the pore composite: on the
    cheeks it adds nothing over the texture features (a fourth-feature search
    moved held-out agreement by +0.001).

    Parameters
    ----------
    rgb_linear : numpy.ndarray
        Linear RGB image ``(H, W, 3)`` in ``[0, 1]``.
    mask : numpy.ndarray, optional
        Boolean ``(H, W)`` mask restricting the estimate to valid pixels.
    v_min, s_max : float, optional
        HSV thresholds for the specular condition.

    Returns
    -------
    float
        Specular pixel ratio in ``[0, 1]`` (``0.0`` if no valid pixels).
    """
    hsv = rgb2hsv(np.clip(encode_srgb(rgb_linear), 0.0, 1.0))
    v, s = hsv[..., 2], hsv[..., 1]
    specular = (v > v_min) & (s < s_max)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        n = int(mask.sum())
        if n == 0:
            return 0.0
        return float((specular & mask).sum()) / float(n)
    return float(specular.mean())


def texture_stats(
    gray: np.ndarray,
    mask: np.ndarray | None = None,
    levels: int = 32,
    lbp_radius: int = 2,
    lbp_points: int = 16,
) -> dict[str, float]:
    """GLCM (contrast/correlation/energy) and LBP texture summary.

    ``lbp_uniformity`` is the single strongest feature in the whole system: on
    held-out cheeks it reaches -0.581 Spearman against the instrument pore
    count, and -0.557 once age is partialled out, so it is reading pores rather
    than reading age. It falls as texture gets busier, hence the negative
    composite weight.

    Parameters
    ----------
    gray : numpy.ndarray
        ``(H, W)`` grayscale image (float ``[0, 1]`` or ``uint8``).
    mask : numpy.ndarray, optional
        Boolean ``(H, W)`` mask. GLCM is computed on the bounding crop; LBP
        statistics are restricted to masked pixels.
    levels : int, optional
        Gray levels for the GLCM.
    lbp_radius, lbp_points : int, optional
        LBP neighbourhood radius and sample count.

    Returns
    -------
    dict
        ``{"glcm_contrast", "glcm_correlation", "glcm_energy",
           "lbp_uniformity"}``. Zeros for degenerate inputs.
    """
    q = _as_gray_u8(gray, levels=levels)
    if q.size == 0 or q.shape[0] < 3 or q.shape[1] < 3:
        return {
            "glcm_contrast": 0.0,
            "glcm_correlation": 0.0,
            "glcm_energy": 0.0,
            "lbp_uniformity": 0.0,
        }

    glcm = graycomatrix(
        q,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True,
    )
    contrast = float(np.mean(graycoprops(glcm, "contrast")))
    correlation = float(np.mean(graycoprops(glcm, "correlation")))
    energy = float(np.mean(graycoprops(glcm, "energy")))

    # LBP uniformity: normalised entropy-like spread of the LBP histogram.
    # Computed on the integer-quantised image (LBP is defined for integer dtype).
    lbp = local_binary_pattern(q, lbp_points, lbp_radius, method="uniform")
    sel = lbp.ravel() if mask is None else lbp[np.asarray(mask, dtype=bool)]
    if sel.size == 0:
        lbp_uniformity = 0.0
    else:
        n_bins = lbp_points + 2
        hist, _ = np.histogram(sel, bins=n_bins, range=(0, n_bins), density=True)
        lbp_uniformity = float(np.sum(hist**2))  # higher = more uniform texture

    return {
        "glcm_contrast": contrast,
        "glcm_correlation": correlation,
        "glcm_energy": energy,
        "lbp_uniformity": lbp_uniformity,
    }


def scaling_index(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """High-frequency band energy (surface relief / roughness).

    A Laplacian high-pass response; its variance over valid pixels rises with
    visible surface relief -- pore openings, flaking, coarse texture.

    Parameters
    ----------
    gray : numpy.ndarray
        ``(H, W)`` grayscale image (float ``[0, 1]`` or ``uint8``).
    mask : numpy.ndarray, optional
        Boolean ``(H, W)`` mask of valid pixels.

    Returns
    -------
    float
        Normalised high-frequency energy (``>= 0``); ``0.0`` for empty input.
    """
    from scipy.ndimage import laplace

    g = np.asarray(gray, dtype=np.float64)
    if g.dtype == np.uint8:
        g = g / 255.0
    if g.size == 0:
        return 0.0
    hp = laplace(g)
    sel = hp.ravel() if mask is None else hp[np.asarray(mask, dtype=bool)]
    if sel.size == 0:
        return 0.0
    return float(np.var(sel))


def micro_wrinkle_density(
    gray: np.ndarray,
    mask: np.ndarray | None = None,
    threshold: float | None = None,
) -> float:
    """Micro-wrinkle density via a Frangi vesselness filter.

    The Frangi/Hessian filter highlights elongated line-like structures (fine
    wrinkles/creases). Extracted and reported, but outside the pore composite:
    pores are round pits, and this filter is built to suppress exactly that in
    favour of ridges.

    Parameters
    ----------
    gray : numpy.ndarray
        ``(H, W)`` grayscale image (float ``[0, 1]`` or ``uint8``).
    mask : numpy.ndarray, optional
        Boolean ``(H, W)`` mask of valid pixels.
    threshold : float, optional
        Ridge-response threshold. If ``None``, uses ``mean + std`` of the
        response over valid pixels.

    Returns
    -------
    float
        Fraction of valid pixels flagged as line structures, in ``[0, 1]``.
    """
    g = np.asarray(gray, dtype=np.float64)
    if g.dtype == np.uint8:
        g = g / 255.0
    if g.size == 0 or g.shape[0] < 5 or g.shape[1] < 5:
        return 0.0

    ridges = frangi(g, black_ridges=True)
    if mask is None:
        sel = ridges.ravel()
        m = np.ones(ridges.shape, dtype=bool)
    else:
        m = np.asarray(mask, dtype=bool)
        sel = ridges[m]
    if sel.size == 0:
        return 0.0

    thr = float(sel.mean() + sel.std()) if threshold is None else float(threshold)
    line_pixels = (ridges > thr) & m
    return float(line_pixels.sum()) / float(int(m.sum()) or 1)
