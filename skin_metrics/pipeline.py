"""Phase 1 orchestration: image -> SkinReport.

Ties together calibration, ROI detection, physics feature extraction, and
percentile scoring. Landmark detection (MediaPipe) is optional at call time:
pass a precomputed ``landmarks`` array to run the whole pipeline without the
``detection`` extra (used by the unit tests).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .calibration import color as cal
from .config import load_config
from .detection.face import ROIResult, detect_landmarks, extract_rois
from .features import erythema as ery
from .features import hydration_proxy as hyd
from .features import pigmentation as pig
from .scoring.normalize import score_metric
from .scoring.schema import MetricScore, SkinReport

_EPS = 1e-8

# Confidence multiplier by calibration outcome.
_CAL_CONF = {"reference": 1.0, "grayworld": 0.6, "none": 0.3}


def _bbox_crop(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop ``image`` and ``mask`` to the mask's bounding box."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return image, mask
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def _roi_features(
    rgb_linear: np.ndarray,
    lab: np.ndarray,
    roi: ROIResult,
    config: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Compute pigmentation / erythema / hydration sub-features for one ROI."""
    vmask = roi.valid_mask
    lstar = lab[..., 0]
    bstar = lab[..., 2]

    valid_rgb = rgb_linear[vmask]           # (N, 3)
    mean_r = float(np.mean(valid_rgb[:, 0])) if valid_rgb.size else 0.0
    mean_g = float(np.mean(valid_rgb[:, 1])) if valid_rgb.size else 0.0

    med_l = float(np.median(lstar[vmask])) if vmask.any() else 0.0
    med_b = float(np.median(bstar[vmask])) if vmask.any() else 1.0

    # --- pigmentation ---
    ita_val = float(pig.ita(med_l, med_b))
    spots = pig.spot_detection(lstar, mask=vmask)
    pigmentation = {
        "melanin_index": float(pig.melanin_index(mean_r)),
        "ita": ita_val,
        "ita_inv": -ita_val,
        "evenness": pig.evenness(lstar, mask=vmask),
        "spot_area_ratio": spots["spot_area_ratio"],
        "spot_count": spots["spot_count"],
        "spot_mean_contrast": spots["mean_contrast"],
    }

    # --- erythema ---
    a_stats = ery.mean_a_star(lab, mask=vmask)
    hb = ery.hemoglobin_map(rgb_linear, mask=vmask)
    erythema = {
        "erythema_index": float(ery.erythema_index(mean_r, mean_g)),
        "mean_a_star": a_stats["mean_a_star"],
        "p90_a_star": a_stats["p90_a_star"],
        "hemoglobin": float(hb["hemoglobin_score"]),
        "hemoglobin_ok": float(bool(hb["separation_ok"])),
    }

    # --- hydration (proxy) ---
    gray = np.clip(lstar / 100.0, 0.0, 1.0)
    gray_crop, mask_crop = _bbox_crop(gray, vmask)
    tex = hyd.texture_features_proxy(gray_crop, mask=mask_crop)
    spec = hyd.specular_ratio_proxy(
        rgb_linear,
        mask=vmask,
        v_min=config["hydration_proxy"]["specular_v_min"],
        s_max=config["hydration_proxy"]["specular_s_max"],
    )
    hydration = {
        "specular_ratio": spec,
        "specular_inv": -spec,
        "glcm_contrast": tex["glcm_contrast"],
        "glcm_correlation": tex["glcm_correlation"],
        "glcm_energy": tex["glcm_energy"],
        "lbp_uniformity": tex["lbp_uniformity"],
        "scaling_index": hyd.scaling_index_proxy(gray_crop, mask=mask_crop),
        "wrinkle_density": hyd.micro_wrinkle_density_proxy(gray_crop, mask=mask_crop),
    }

    return {"pigmentation": pigmentation, "erythema": erythema, "hydration": hydration}


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    """Weighted mean with a zero-weight guard."""
    if not values:
        return 0.0
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    tot = float(w.sum())
    if tot <= _EPS:
        return float(v.mean())
    return float((v * w).sum() / tot)


def analyze(
    img: np.ndarray,
    *,
    ref_bbox: tuple[int, int, int, int] | None = None,
    ccm: np.ndarray | None = None,
    landmarks: np.ndarray | None = None,
    model_path: str | None = None,
    config: dict[str, Any] | None = None,
) -> SkinReport:
    """Run the full Phase 1 pipeline on one image.

    Parameters
    ----------
    img : numpy.ndarray
        Input sRGB image (``uint8`` or float in ``[0, 1]``), shape ``(H, W, 3)``.
    ref_bbox : tuple of int, optional
        ``(x, y, w, h)`` of an in-frame neutral gray/white reference for reliable
        white balance.
    ccm : numpy.ndarray, optional
        Pre-estimated ``(3, 3)`` color-correction matrix (see
        :func:`skin_metrics.calibration.color.estimate_ccm`).
    landmarks : numpy.ndarray, optional
        Precomputed ``(468, 2)`` FaceMesh landmarks. If ``None``, MediaPipe is
        invoked (requires the ``detection`` extra).
    config : dict, optional
        Loaded configuration; defaults to the packaged ``config.yaml``.

    Returns
    -------
    SkinReport
        Pigmentation / erythema / hydration scores, ROI breakdown, calibration
        status, Fitzpatrick estimate, and warnings.

    Raises
    ------
    ValueError
        If no face landmarks are available (no face detected and none supplied).
    """
    if config is None:
        config = load_config()

    warnings: list[str] = []

    # 1. calibration --------------------------------------------------------
    cal_result = cal.calibrate_image(img, ref_bbox=ref_bbox, ccm=ccm)
    rgb_linear = cal_result.image
    lab = cal.rgb_to_lab(rgb_linear, assume_linear=True)
    if cal_result.status != "reference":
        warnings.append(
            f"Calibration is '{cal_result.status}'; scores less reliable. "
            "Provide a gray/white reference (--reference-bbox) for best results."
        )

    # 2. landmarks ----------------------------------------------------------
    if landmarks is None:
        landmarks = detect_landmarks(img, model_path=model_path)
    if landmarks is None:
        raise ValueError("No face detected and no landmarks supplied.")

    # 3. ROIs ---------------------------------------------------------------
    masking = config["masking"]
    rois = extract_rois(
        rgb_linear,
        lab,
        landmarks,
        min_valid_ratio=masking["min_valid_ratio"],
        glare_v_min=masking["glare_v_min"],
        glare_s_max=masking["glare_s_max"],
        shadow_percentile=masking["shadow_percentile"],
    )
    valid_rois = {k: v for k, v in rois.items() if v is not None}
    dropped = [k for k, v in rois.items() if v is None]
    if dropped:
        warnings.append(f"ROIs dropped (low valid-pixel ratio): {', '.join(dropped)}.")
    if not valid_rois:
        raise ValueError("All ROIs failed the valid-pixel gate; cannot score.")

    # 4. per-ROI features ---------------------------------------------------
    roi_feats: dict[str, dict[str, dict[str, float]]] = {}
    roi_weights: dict[str, float] = {}
    for name, roi in valid_rois.items():
        roi_feats[name] = _roi_features(rgb_linear, lab, roi, config)
        roi_weights[name] = float(roi.valid_mask.sum())

    # 5. face-level aggregation (weighted by valid-pixel count) -------------
    def _agg(metric: str, key: str) -> float:
        vals = [roi_feats[n][metric][key] for n in valid_rois]
        wts = [roi_weights[n] for n in valid_rois]
        return _weighted_mean(vals, wts)

    agg: dict[str, dict[str, float]] = {"pigmentation": {}, "erythema": {}, "hydration": {}}
    for metric in agg:
        sample_keys = next(iter(roi_feats.values()))[metric].keys()
        for key in sample_keys:
            agg[metric][key] = _agg(metric, key)

    # 6. Fitzpatrick estimate from aggregate ITA ----------------------------
    fitz = pig.estimate_fitzpatrick(
        agg["pigmentation"]["ita"], config["fitzpatrick"]["ita_boundaries"]
    )

    # 7. scores -------------------------------------------------------------
    scores = {
        m: score_metric(m, agg[m], fitz, config)
        for m in ("pigmentation", "erythema", "hydration")
    }

    # 8. confidence ---------------------------------------------------------
    roi_fraction = len(valid_rois) / len(rois)
    base_conf = _CAL_CONF[cal_result.status]
    if cal_result.success and cal_result.status == "grayworld":
        base_conf = max(base_conf, 0.8)  # CCM applied without reference patch
    common_conf = base_conf * roi_fraction

    hemo_ok = _agg("erythema", "hemoglobin_ok") >= 0.5
    ery_conf = common_conf * (1.0 if hemo_ok else 0.7)
    if not hemo_ok:
        warnings.append("Hemoglobin ICA separation was unreliable for some ROIs.")

    warnings.append(
        "Hydration is a PROXY estimate from surface optics/texture, not a "
        "moisture measurement."
    )

    def _clean(d: dict[str, float]) -> dict[str, float]:
        return {k: round(float(v), 4) for k, v in d.items()}

    report = SkinReport(
        pigmentation=MetricScore(
            score=round(scores["pigmentation"], 2),
            confidence=round(common_conf, 3),
            raw_features=_clean(agg["pigmentation"]),
            is_estimate=False,
        ),
        erythema=MetricScore(
            score=round(scores["erythema"], 2),
            confidence=round(ery_conf, 3),
            raw_features=_clean(agg["erythema"]),
            is_estimate=False,
        ),
        hydration=MetricScore(
            score=round(scores["hydration"], 2),
            confidence=round(common_conf, 3),
            raw_features=_clean(agg["hydration"]),
            is_estimate=True,  # fixed: hydration is always a proxy
        ),
        roi_breakdown={
            name: {
                "valid_ratio": round(valid_rois[name].valid_ratio, 3),
                "features": {m: _clean(roi_feats[name][m]) for m in agg},
            }
            for name in valid_rois
        },
        calibration_status=cal_result.status,
        fitzpatrick_estimate=fitz,
        warnings=warnings,
    )
    return report
