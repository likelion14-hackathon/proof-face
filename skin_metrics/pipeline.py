"""Phase 1 orchestration: image -> SkinReport.

Ties together calibration, ROI detection, physics feature extraction, and
percentile scoring. Landmark detection (MediaPipe) is optional at call time:
pass a precomputed ``landmarks`` array to run the whole pipeline without the
``detection`` extra (used by the unit tests).

The image -> raw-feature half is exposed separately as :func:`extract_raw` so
offline calibration (``skin_metrics.calibrate``) can fit reference
distributions on exactly the feature values the scorer sees at inference time,
at full precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .calibration import color as cal
from .config import load_config
from .detection.face import (
    ROIResult,
    detect_landmarks,
    extract_rois,
    eye_span,
    face_mask,
    scale_factor_for,
)
from .features import erythema as ery
from .features import hydration_proxy as hyd
from .features import pigmentation as pig
from .scoring.normalize import (
    composite_raw,
    predict_instrument,
    score_from_raw,
)
from .scoring.schema import FaceScale, MetricScore, SkinReport

_EPS = 1e-8

# Confidence multiplier by calibration outcome. "none" (trust the camera's AWB)
# is no longer a failure mode -- it is the default path, and the one the shipped
# reference cohort was captured and fitted under -- so it is not penalised
# relative to gray-world. Only an in-frame neutral reference earns full marks.
_CAL_CONF = {"reference": 1.0, "grayworld": 0.6, "none": 0.6}


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


def _normalize_face_scale(
    rgb_linear: np.ndarray,
    landmarks: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Resize a linear-RGB image so the face sits at a canonical pixel scale.

    Resampling happens in **linear** light so ROI means are preserved; doing it
    on sRGB-encoded data would bias every colour feature through the gamma
    curve.

    Parameters
    ----------
    rgb_linear : numpy.ndarray
        Calibrated linear RGB image ``(H, W, 3)``.
    landmarks : numpy.ndarray
        ``(468, 2)`` landmarks in the coordinates of ``rgb_linear``.
    config : dict
        Loaded configuration; reads the ``normalization`` block.

    Returns
    -------
    tuple
        ``(rgb_linear, landmarks, native_eye_span, factor)`` -- resized image
        and rescaled landmarks, plus the pre-resize eye span and the factor
        applied (``1.0`` when normalisation is disabled or the face is already
        at or below target).
    """
    from skimage.transform import resize as _resize

    span = eye_span(landmarks)
    norm = config.get("normalization") or {}
    if not norm.get("enabled", False):
        return rgb_linear, landmarks, span, 1.0

    factor = scale_factor_for(landmarks, float(norm.get("target_eye_span_px", 512.0)))
    if factor >= 1.0 - 1e-3:
        return rgb_linear, landmarks, span, 1.0

    h, w = rgb_linear.shape[:2]
    new_shape = (max(1, int(round(h * factor))), max(1, int(round(w * factor))))
    resized = _resize(
        rgb_linear,
        new_shape,
        order=1,
        anti_aliasing=True,       # required: we are downsampling
        preserve_range=True,
        mode="reflect",
    ).astype(np.float64, copy=False)
    # Use the realised factors, which differ from `factor` by the pixel rounding.
    scaled = np.asarray(landmarks, dtype=np.float64).copy()
    scaled[:, 0] *= new_shape[1] / w
    scaled[:, 1] *= new_shape[0] / h
    return resized, scaled, span, factor


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


def _aggregation_rois(
    metric: str,
    valid: list[str],
    config: dict[str, Any],
    warnings: list[str],
) -> list[str]:
    """ROIs that feed a metric's face-level aggregate.

    A metric may declare ``composite.<metric>.rois`` to restrict aggregation to
    the sites where its signal was actually measured -- hydration is validated
    against a Corneometer reading taken on the cheeks, and averaging in
    forehead/nose/chin texture only dilutes it. Metrics without the key keep
    every valid ROI.

    Parameters
    ----------
    metric : str
        Metric name, e.g. ``"hydration"``.
    valid : list of str
        ROIs that passed the valid-pixel gate.
    config : dict
        Parsed configuration.
    warnings : list of str
        Appended to when the restriction cannot be honoured.

    Returns
    -------
    list of str
        ROI names to aggregate over; never empty.
    """
    wanted = ((config.get("composite") or {}).get(metric) or {}).get("rois")
    if not wanted:
        return valid
    names = [n for n in valid if n in wanted]
    if names:
        return names
    warnings.append(
        f"None of the ROIs {metric} is calibrated on ({', '.join(wanted)}) survived "
        "the valid-pixel gate; aggregated over the remaining ROIs instead, which "
        "makes this score less reliable."
    )
    return valid


@dataclass
class RawExtraction:
    """Un-rounded physics features for one image, before any scoring.

    Attributes
    ----------
    roi_features : dict
        ``{roi_name: {metric: {feature: value}}}`` for ROIs that passed the
        valid-pixel gate.
    roi_valid_ratio : dict
        ``{roi_name: valid-pixel ratio}`` for the same ROIs.
    roi_weights : dict
        ``{roi_name: valid-pixel count}``; the aggregation weights.
    aggregate : dict
        ``{metric: {feature: value}}`` face-level weighted means.
    fitzpatrick : int
        Fitzpatrick type (1-6) estimated from the aggregate ITA.
    calibration_status : str
        ``"reference"`` | ``"grayworld"`` | ``"none"``.
    calibration_success : bool
        Whether a CCM was successfully estimated and applied.
    n_rois_total : int
        Number of ROIs attempted (valid + dropped).
    eye_span_px : float
        Native outer-eye-corner separation before scale normalisation. Texture
        features are only trustworthy at or above the configured target.
    scale_factor : float
        Resize factor actually applied by scale normalisation (``1.0`` = none).
    under_resolved : bool
        ``True`` when the native face was smaller than the normalisation
        target, so texture features were computed below canonical scale.
    dropped_rois : list of str
        ROIs that failed the valid-pixel gate.
    warnings : list of str
        Human-readable quality warnings accumulated during extraction.
    """

    roi_features: dict[str, dict[str, dict[str, float]]]
    roi_valid_ratio: dict[str, float]
    roi_weights: dict[str, float]
    aggregate: dict[str, dict[str, float]]
    fitzpatrick: int
    calibration_status: str
    calibration_success: bool
    n_rois_total: int
    eye_span_px: float = 0.0
    scale_factor: float = 1.0
    under_resolved: bool = False
    dropped_rois: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def extract_raw(
    img: np.ndarray,
    *,
    ref_bbox: tuple[int, int, int, int] | None = None,
    ccm: np.ndarray | None = None,
    landmarks: np.ndarray | None = None,
    model_path: str | None = None,
    config: dict[str, Any] | None = None,
) -> RawExtraction:
    """Run calibration -> ROI detection -> feature extraction (no scoring).

    This is the first half of :func:`analyze`, split out so calibration
    tooling can fit on full-precision feature values.

    Parameters
    ----------
    img : numpy.ndarray
        Input sRGB image (``uint8`` or float in ``[0, 1]``), shape ``(H, W, 3)``.
    ref_bbox : tuple of int, optional
        ``(x, y, w, h)`` of an in-frame neutral gray/white reference.
    ccm : numpy.ndarray, optional
        Pre-estimated ``(3, 3)`` color-correction matrix.
    landmarks : numpy.ndarray, optional
        Precomputed ``(468, 2)`` FaceMesh landmarks. If ``None``, MediaPipe is
        invoked (requires the ``detection`` extra).
    model_path : str, optional
        FaceLandmarker model path (Tasks API).
    config : dict, optional
        Loaded configuration; defaults to the packaged ``config.yaml``.

    Returns
    -------
    RawExtraction
        Per-ROI and face-level features at full precision.

    Raises
    ------
    ValueError
        If no face landmarks are available, or every ROI fails the valid-pixel
        gate.
    """
    if config is None:
        config = load_config()

    warnings: list[str] = []

    # 1. landmarks ----------------------------------------------------------
    # Landmarks come first: the illuminant estimate in step 2 must know where
    # the skin is so it can exclude it.
    if landmarks is None:
        landmarks = detect_landmarks(img, model_path=model_path)
    if landmarks is None:
        raise ValueError("No face detected and no landmarks supplied.")

    # 2. calibration --------------------------------------------------------
    cal_cfg = config.get("calibration") or {}
    background = ~face_mask(
        landmarks,
        img.shape[:2],
        dilate_frac=float(cal_cfg.get("face_dilate_frac", 0.35)),
    )
    cal_result = cal.calibrate_image(
        img,
        ref_bbox=ref_bbox,
        ccm=ccm,
        fallback=cal_cfg.get("fallback", "background"),
        background_mask=background,
        min_background_ratio=float(cal_cfg.get("min_background_ratio", 0.15)),
    )
    rgb_linear = cal_result.image
    if cal_result.status != "reference":
        warnings.append(
            f"Colour calibration is '{cal_result.status}' (no in-frame neutral "
            "reference), so absolute colour depends on the camera. Include a "
            "gray card or white paper and pass --reference-bbox to make scores "
            "comparable across devices."
        )

    # 3. face-scale normalisation ------------------------------------------
    # Bring every face to the same pixel scale before any texture feature is
    # computed; otherwise a 1140px-wide face (DSLR) and a 560px one (phone)
    # produce incomparable GLCM/LBP/wrinkle values.
    rgb_linear, landmarks, native_span, scale_factor = _normalize_face_scale(
        rgb_linear, landmarks, config
    )
    norm_cfg = config.get("normalization") or {}
    target_span = float(norm_cfg.get("target_eye_span_px", 512.0))
    under_resolved = bool(norm_cfg.get("enabled", False)) and native_span < target_span
    if under_resolved:
        warnings.append(
            f"Face is under-resolved (eye span {native_span:.0f}px < "
            f"{target_span:.0f}px target); texture-based features "
            "(hydration) are less reliable. Move closer or use a higher-"
            "resolution photo."
        )

    lab = cal.rgb_to_lab(rgb_linear, assume_linear=True)

    # 4. ROIs ---------------------------------------------------------------
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

    # 5. per-ROI features ---------------------------------------------------
    roi_feats: dict[str, dict[str, dict[str, float]]] = {}
    roi_weights: dict[str, float] = {}
    for name, roi in valid_rois.items():
        roi_feats[name] = _roi_features(rgb_linear, lab, roi, config)
        roi_weights[name] = float(roi.valid_mask.sum())

    # 6. face-level aggregation (weighted by valid-pixel count) -------------
    def _agg(metric: str, key: str, names: list[str]) -> float:
        vals = [roi_feats[n][metric][key] for n in names]
        wts = [roi_weights[n] for n in names]
        return _weighted_mean(vals, wts)

    agg: dict[str, dict[str, float]] = {"pigmentation": {}, "erythema": {}, "hydration": {}}
    for metric in agg:
        names = _aggregation_rois(metric, list(valid_rois), config, warnings)
        sample_keys = next(iter(roi_feats.values()))[metric].keys()
        for key in sample_keys:
            agg[metric][key] = _agg(metric, key, names)

    # 7. Fitzpatrick estimate from aggregate ITA ----------------------------
    fitz = pig.estimate_fitzpatrick(
        agg["pigmentation"]["ita"], config["fitzpatrick"]["ita_boundaries"]
    )

    return RawExtraction(
        roi_features=roi_feats,
        roi_valid_ratio={n: valid_rois[n].valid_ratio for n in valid_rois},
        roi_weights=roi_weights,
        aggregate=agg,
        fitzpatrick=fitz,
        calibration_status=cal_result.status,
        calibration_success=bool(cal_result.success),
        n_rois_total=len(rois),
        eye_span_px=native_span,
        scale_factor=scale_factor,
        under_resolved=under_resolved,
        dropped_rois=dropped,
        warnings=warnings,
    )


@dataclass
class _Prediction:
    """How one metric's score was produced.

    Attributes
    ----------
    calibrated : bool
        ``True`` when a supervised instrument model supplied the driver.
    value : float or None
        Predicted instrument reading, in its own units.
    units : str or None
        Units of ``value``.
    """

    calibrated: bool = False
    value: float | None = None
    units: str | None = None


def _score_metric(
    metric: str,
    raw: RawExtraction,
    config: dict[str, Any],
    warnings: list[str],
) -> tuple[float, _Prediction]:
    """Score one metric, preferring a fitted instrument model over the composite.

    Parameters
    ----------
    metric : str
        ``"pigmentation"`` | ``"erythema"`` | ``"hydration"``.
    raw : RawExtraction
        Extracted features for the image.
    config : dict
        Loaded configuration.
    warnings : list of str
        Appended to when a calibrated model was configured but could not be
        applied, so the caller can surface the degraded path.

    Returns
    -------
    tuple
        ``(score, prediction)``.
    """
    def _finish(driver: float) -> float:
        """Percentile-map the driver, then orient it for the reader.

        Everything upstream -- features, composite weights, the fitted quantile
        grids -- works in "more pronounced condition" units, which for hydration
        means DRYNESS. `scoring.report_inverted` flips the finished percentile
        for metrics whose public name reads the other way round, so a field
        called `hydration` means what a reader expects it to mean. Flipping here
        and not earlier keeps the calibration and its validation numbers in the
        single orientation they were fitted in.
        """
        score = score_from_raw(driver, metric, raw.fitzpatrick, config)
        inverted = (config.get("scoring") or {}).get("report_inverted") or ()
        return 100.0 - score if metric in inverted else score

    model = (config.get("supervised") or {}).get(metric)
    if model:
        # Unweighted across ROIs on purpose: the instrument took one reading per
        # site regardless of how large that site is in frame, so the cohort
        # face-level label -- and hence the reference grid this score is looked
        # up in -- is a plain mean over sites. Area-weighting here would score
        # against a distribution built a different way.
        predicted = predict_instrument(model, raw.roi_features)
        if predicted is not None:
            # `score_sign` orients the driver so higher always means "more
            # pronounced condition" (moisture is inverted: drier scores higher).
            driver = float(model.get("score_sign", 1.0)) * predicted
            return (
                _finish(driver),
                _Prediction(True, predicted, model.get("target_units")),
            )
        warnings.append(
            f"Calibrated {metric} model could not be applied (no usable ROI); "
            "fell back to the uncalibrated composite score."
        )

    driver = composite_raw(metric, raw.aggregate[metric], config)
    return _finish(driver), _Prediction()


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

    raw = extract_raw(
        img,
        ref_bbox=ref_bbox,
        ccm=ccm,
        landmarks=landmarks,
        model_path=model_path,
        config=config,
    )
    agg = raw.aggregate
    fitz = raw.fitzpatrick
    warnings = list(raw.warnings)
    roi_feats = raw.roi_features
    valid_rois = raw.roi_valid_ratio

    # 8. scores -------------------------------------------------------------
    scores: dict[str, float] = {}
    predictions: dict[str, _Prediction] = {}
    for metric in ("pigmentation", "erythema", "hydration"):
        scores[metric], predictions[metric] = _score_metric(
            metric, raw, config, warnings
        )

    # 9. confidence ---------------------------------------------------------
    roi_fraction = len(valid_rois) / raw.n_rois_total
    base_conf = _CAL_CONF[raw.calibration_status]
    if raw.calibration_success:
        base_conf = max(base_conf, 0.8)  # CCM applied without reference patch
    common_conf = base_conf * roi_fraction

    hemo_ok = agg["erythema"]["hemoglobin_ok"] >= 0.5
    ery_conf = common_conf * (1.0 if hemo_ok else 0.7)
    if not hemo_ok:
        warnings.append("Hemoglobin ICA separation was unreliable for some ROIs.")

    # Texture features drive hydration, so an under-resolved face hurts it most.
    hyd_conf = common_conf * (0.6 if raw.under_resolved else 1.0)

    if predictions["hydration"].calibrated:
        warnings.append(
            "Hydration is an ESTIMATE of a Corneometer reading regressed from "
            "surface optics/texture, not a moisture measurement."
        )
    else:
        warnings.append(
            "Hydration is a PROXY estimate from cheek surface texture, not a "
            "moisture measurement: a higher score means this face ranks as "
            "moister than the reference cohort, not that a given amount of "
            "water was measured."
        )
    if not predictions["erythema"].calibrated:
        warnings.append(
            "Erythema is normalised against the reference cohort but has no "
            "instrument ground truth; treat it as relative, not absolute."
        )

    def _clean(d: dict[str, float]) -> dict[str, float]:
        # 6 decimals: several texture features (scaling_index, specular_ratio)
        # sit at 1e-3..1e-4, where 4 decimals would quantise them away.
        return {k: round(float(v), 6) for k, v in d.items()}

    def _metric_score(
        metric: str, confidence: float, is_estimate: bool
    ) -> MetricScore:
        """Assemble one metric's report entry from its score and prediction."""
        pred = predictions[metric]
        return MetricScore(
            score=round(scores[metric], 2),
            confidence=round(confidence, 3),
            raw_features=_clean(agg[metric]),
            is_estimate=is_estimate,
            calibrated=pred.calibrated,
            predicted_value=(
                None if pred.value is None else round(float(pred.value), 3)
            ),
            predicted_units=pred.units,
        )

    report = SkinReport(
        pigmentation=_metric_score("pigmentation", common_conf, False),
        erythema=_metric_score("erythema", ery_conf, False),
        # Hydration is inferred from surface optics, never measured, so it
        # stays flagged as an estimate even once calibrated.
        hydration=_metric_score("hydration", hyd_conf, True),
        roi_breakdown={
            name: {
                "valid_ratio": round(valid_rois[name], 3),
                "features": {m: _clean(roi_feats[name][m]) for m in agg},
            }
            for name in valid_rois
        },
        calibration_status=raw.calibration_status,
        fitzpatrick_estimate=fitz,
        face_scale=FaceScale(
            eye_span_px=round(raw.eye_span_px, 1),
            scale_factor=round(raw.scale_factor, 4),
            under_resolved=raw.under_resolved,
        ),
        calibration_profile=(config.get("provenance") or {}).get("profile"),
        warnings=warnings,
    )
    return report


__all__ = ["RawExtraction", "analyze", "extract_raw"]
