"""Erythema (redness) features (Phase 1, physics-based).

Includes a reflectance-based erythema index, CIE ``a*`` statistics, and a
Tsumura-style melanin/hemoglobin color separation via ICA. All functions guard
divide-by-zero and non-positive log inputs.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-6

# Reference absorbance directions in RGB optical-density space, used only to
# resolve the sign/order ambiguity of the ICA components. These are approximate
# directions (not calibrated pigment spectra): hemoglobin absorbs strongly in
# green, melanin broadly with a red-ward bias.
# TODO(pigment-basis): replace with measured pigment absorbance for the target camera.
_HEMOGLOBIN_DIR = np.array([0.30, 0.70, 0.64], dtype=np.float64)
_MELANIN_DIR = np.array([0.50, 0.60, 0.62], dtype=np.float64)


def erythema_index(
    r_red: np.ndarray | float, r_green: np.ndarray | float
) -> np.ndarray | float:
    """Reflectance-based erythema index.

    ``EI = 100 * (log10(1 / R_green) - log10(1 / R_red))``.

    Parameters
    ----------
    r_red, r_green : float or numpy.ndarray
        Red / green channel reflectance in ``(0, 1]`` (clipped to ``[eps, 1]``).

    Returns
    -------
    float or numpy.ndarray
        Erythema index (higher = redder).
    """
    r = np.clip(np.asarray(r_red, dtype=np.float64), _EPS, 1.0)
    g = np.clip(np.asarray(r_green, dtype=np.float64), _EPS, 1.0)
    ei = 100.0 * (np.log10(1.0 / g) - np.log10(1.0 / r))
    if np.isscalar(r_red) and np.isscalar(r_green):
        return float(ei)
    return ei


def mean_a_star(
    lab: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, float]:
    """Mean and upper-percentile of CIE ``a*`` over valid pixels.

    Parameters
    ----------
    lab : numpy.ndarray
        CIELab image ``(H, W, 3)``.
    mask : numpy.ndarray, optional
        Boolean ``(H, W)`` mask of valid pixels.

    Returns
    -------
    dict
        ``{"mean_a_star", "p90_a_star"}``. Zeros if no valid pixels.
    """
    a = np.asarray(lab, dtype=np.float64)[..., 1]
    vals = a.ravel() if mask is None else a[np.asarray(mask, dtype=bool)]
    if vals.size == 0:
        return {"mean_a_star": 0.0, "p90_a_star": 0.0}
    return {
        "mean_a_star": float(np.mean(vals)),
        "p90_a_star": float(np.percentile(vals, 90)),
    }


def hemoglobin_map(
    rgb: np.ndarray,
    mask: np.ndarray | None = None,
    random_state: int = 0,
) -> dict[str, object]:
    """Tsumura-style melanin/hemoglobin separation via ICA.

    Skin optical density ``-log(reflectance)`` is modelled as a linear mixture
    of melanin and hemoglobin contributions plus shading. Independent Component
    Analysis (FastICA) separates the two pigment sources; the hemoglobin source
    is identified by cosine similarity of the ICA mixing columns to a reference
    hemoglobin absorbance direction, and its sign is fixed so that *more
    hemoglobin -> larger value*.

    Parameters
    ----------
    rgb : numpy.ndarray
        Linear RGB image ``(H, W, 3)`` in ``[0, 1]``.
    mask : numpy.ndarray, optional
        Boolean ``(H, W)`` mask of valid skin pixels. Pixels outside the mask
        are excluded from the ICA fit and set to ``0`` in the output map.
    random_state : int, optional
        Seed for FastICA reproducibility.

    Returns
    -------
    dict
        ``{"hemoglobin_map": (H, W) float array,
           "hemoglobin_score": float,   # mean over valid pixels
           "separation_ok": bool}``.
        On degenerate input (too few / constant pixels) ``separation_ok`` is
        ``False`` and the map is all zeros.

    .. warning::
       ``hemoglobin_score`` is **not usable as an erythema level**. FastICA
       centres its input, so the extracted source is zero-mean by construction
       over the very pixels it was fitted on; across the calibration cohort the
       score never exceeded 1.5e-12 in magnitude. It is therefore excluded from
       the erythema composite in ``config.yaml``. The map itself is still
       meaningful *relatively* (which pixels are more hemoglobin-rich than the
       ROI average). Getting an absolute level needs the ICA fitted once per
       face with per-ROI means taken afterwards, or the ICA dropped in favour
       of projecting optical density onto a measured absorbance direction.
    """
    from sklearn.decomposition import FastICA

    arr = np.asarray(rgb, dtype=np.float64)
    h, w = arr.shape[:2]
    hb_map = np.zeros((h, w), dtype=np.float64)

    if mask is None:
        mask = np.ones((h, w), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    coords = np.argwhere(mask)
    if coords.shape[0] < 50:
        return {"hemoglobin_map": hb_map, "hemoglobin_score": 0.0, "separation_ok": False}

    # Optical density; clip reflectance to (0, 1] to keep the log finite.
    density = -np.log(np.clip(arr[mask], _EPS, 1.0))  # (N, 3)
    if np.allclose(density.std(axis=0), 0.0):
        return {"hemoglobin_map": hb_map, "hemoglobin_score": 0.0, "separation_ok": False}

    try:
        ica = FastICA(n_components=2, random_state=random_state, max_iter=500, whiten="unit-variance")
        sources = ica.fit_transform(density)  # (N, 2)
        mixing = ica.mixing_  # (3, 2): color vector of each source
    except Exception:  # pragma: no cover - numerical failure fallback
        return {"hemoglobin_map": hb_map, "hemoglobin_score": 0.0, "separation_ok": False}

    # Identify the hemoglobin source by cosine similarity of mixing columns.
    def _cos(u: np.ndarray, v: np.ndarray) -> float:
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < _EPS or nv < _EPS:
            return 0.0
        return float(np.dot(u, v) / (nu * nv))

    sims_hb = [abs(_cos(mixing[:, k], _HEMOGLOBIN_DIR)) for k in range(2)]
    sims_mel = [abs(_cos(mixing[:, k], _MELANIN_DIR)) for k in range(2)]
    # Prefer the column that is relatively more hemoglobin-like than melanin-like.
    hb_idx = int(np.argmax(np.array(sims_hb) - np.array(sims_mel)))

    # Sign disambiguation: make the hemoglobin color vector point the same way
    # as the reference direction (more hemoglobin -> larger source value).
    sign = 1.0 if _cos(mixing[:, hb_idx], _HEMOGLOBIN_DIR) >= 0 else -1.0
    hb_source = sign * sources[:, hb_idx]

    hb_map[mask] = hb_source
    return {
        "hemoglobin_map": hb_map,
        "hemoglobin_score": float(np.mean(hb_source)),
        "separation_ok": True,
    }
