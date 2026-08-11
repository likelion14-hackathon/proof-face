"""Normalisation and scoring: raw features -> 0-100 percentile scores.

Every metric ends up as a percentile of a **score driver** against a
**Fitzpatrick-type-specific** reference distribution. Splitting the reference by
skin type reduces bias for darker skin, where absolute pigmentation/erythema
features differ systematically. There are two ways to obtain that driver:

*Calibrated* (``config["supervised"][metric]`` present)
    A ridge regression fitted on a labelled cohort predicts the actual
    instrument reading (Corneometer moisture, expert pigmentation grade) from
    the per-ROI physics features. The driver is that prediction, signed so
    higher always means "more pronounced condition".

*Uncalibrated* (fallback)
    The z-scored sub-features are combined into a composite raw value using the
    weights and anchors in ``config.yaml``.

The reference distribution is looked up as an **empirical percentile grid**
when the config carries one, falling back to a normal CDF. Real cohort
distributions of spot counts and predicted grades are skewed, so the Gaussian
assumption misplaces the tails.
"""

from __future__ import annotations

import bisect
import math
from typing import Any, Mapping, Sequence

_EPS = 1e-8


def _zscore(value: float, mean: float, std: float) -> float:
    """Standard score with a divide-by-zero guard."""
    return (value - mean) / (std if abs(std) > _EPS else _EPS)


def _normal_cdf(z: float) -> float:
    """Standard-normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def composite_raw(
    metric: str,
    subfeatures: dict[str, float],
    config: dict[str, Any],
) -> float:
    """Combine z-scored sub-features into one composite raw value.

    Parameters
    ----------
    metric : str
        One of ``"pigmentation"``, ``"erythema"``, ``"hydration"``.
    subfeatures : dict
        Sub-feature name -> value. Only keys listed in the config weights are
        used; missing keys contribute ``0`` (their weight is skipped and the
        result renormalised over the available weights).
    config : dict
        Loaded configuration (see ``config.yaml``).

    Returns
    -------
    float
        Weighted sum of standardised sub-features (approximately zero-mean,
        unit-scale by construction of the anchors).

    Notes
    -----
    Renormalisation uses the sum of |weight|, not the signed sum. Fitted
    weights can legitimately be negative (a suppressor variable), and a signed
    denominator would then shrink toward zero and blow the result up.
    """
    spec = config["composite"][metric]
    weights: dict[str, float] = spec["weights"]
    anchors: dict[str, dict[str, float]] = spec["anchors"]

    total_w = 0.0
    acc = 0.0
    for name, w in weights.items():
        if name not in subfeatures or name not in anchors:
            continue
        anchor = anchors[name]
        z = _zscore(float(subfeatures[name]), anchor["mean"], anchor["std"])
        acc += w * z
        total_w += abs(w)
    if total_w <= _EPS:
        return 0.0
    return acc / total_w


def _percentile_from_quantiles(value: float, quantiles: Sequence[float]) -> float:
    """Percentile of ``value`` within an empirical quantile grid.

    Parameters
    ----------
    value : float
        Score driver to locate.
    quantiles : sequence of float
        Non-decreasing values at percentiles ``0, 1, ..., 100``.

    Returns
    -------
    float
        Percentile in ``[0, 100]``, linearly interpolated between grid points
        and saturating at the ends.
    """
    n = len(quantiles)
    if n < 2:
        return 50.0
    step = 100.0 / (n - 1)
    if value <= quantiles[0]:
        return 0.0
    if value >= quantiles[-1]:
        return 100.0
    # Index of the first grid point strictly greater than `value`.
    hi = bisect.bisect_right(quantiles, value)
    lo = hi - 1
    span = quantiles[hi] - quantiles[lo]
    frac = 0.0 if span <= _EPS else (value - quantiles[lo]) / span
    return float((lo + frac) * step)


def score_from_raw(
    raw: float,
    metric: str,
    fitzpatrick: int,
    config: dict[str, Any],
) -> float:
    """Map a score driver to a 0-100 percentile score.

    Uses the empirical quantile grid fitted on the reference cohort when the
    config carries one, otherwise a normal CDF around ``mean``/``std``.

    Parameters
    ----------
    raw : float
        Score driver -- a composite raw value from :func:`composite_raw`, or a
        signed instrument prediction from :func:`predict_instrument`.
    metric : str
        Metric name (selects the reference distribution).
    fitzpatrick : int
        Fitzpatrick type (1-6); selects the type-specific reference, falling
        back to ``default`` if that type is absent.
    config : dict
        Loaded configuration.

    Returns
    -------
    float
        Score in ``[0, 100]`` (higher = more pronounced condition).
    """
    ref_all = config["reference"][metric]
    ref = ref_all.get(str(fitzpatrick), ref_all["default"])

    quantiles = ref.get("quantiles")
    if quantiles:
        return _percentile_from_quantiles(raw, quantiles)

    clip_z = float(config.get("scoring", {}).get("clip_z", 4.0))
    z = _zscore(raw, ref["mean"], ref["std"])
    z = max(-clip_z, min(clip_z, z))
    return float(_normal_cdf(z) * 100.0)


def flatten_features(roi_features: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Flatten one ROI's ``{metric: {feature: value}}`` to ``f_<metric>_<name>``.

    This naming is the contract between the offline calibration tables
    (:mod:`skin_metrics.calibrate.extract`) and the fitted models applied here,
    so both sides go through this function.

    Parameters
    ----------
    roi_features : mapping
        Per-metric feature dictionaries for a single ROI.

    Returns
    -------
    dict
        Flat ``{column_name: value}``.
    """
    return {
        f"f_{metric}_{name}": float(value)
        for metric, features in roi_features.items()
        for name, value in features.items()
    }


def predict_instrument(
    model: Mapping[str, Any],
    roi_features: Mapping[str, Mapping[str, Mapping[str, float]]],
    roi_weights: Mapping[str, float] | None = None,
) -> float | None:
    """Predict a face-level instrument reading from per-ROI features.

    The model is applied only to the ROIs it was fitted on
    (``model["applies_to_rois"]``); predicting a cheek Corneometer value from
    nose features would be extrapolation, since the instrument never measured
    there.

    Parameters
    ----------
    model : mapping
        Fitted model from :func:`skin_metrics.calibrate.fit.fit_ridge`.
    roi_features : mapping
        ``{roi: {metric: {feature: value}}}`` for the ROIs that passed the
        valid-pixel gate.
    roi_weights : mapping, optional
        ``{roi: weight}``. Defaults to equal weights, which is what the
        pipeline uses: the reference instrument took one reading per site
        irrespective of the site's area in frame, so the cohort distribution
        this prediction is scored against is a plain mean over sites.

    Returns
    -------
    float or None
        Mean prediction across applicable ROIs, or ``None`` when no ROI
        supplies the full feature set.
    """
    names: Sequence[str] = model["features"]
    mean: Sequence[float] = model["feature_mean"]
    std: Sequence[float] = model["feature_std"]
    coef: Sequence[float] = model["coef"]
    intercept = float(model["intercept"])
    applicable = set(model.get("applies_to_rois") or roi_features.keys())

    total_w = 0.0
    acc = 0.0
    for roi, features in roi_features.items():
        if roi not in applicable:
            continue
        flat = flatten_features(features)
        if any(name not in flat for name in names):
            continue
        value = intercept + sum(
            c * (flat[n] - m) / (s if abs(s) > _EPS else _EPS)
            for n, m, s, c in zip(names, mean, std, coef)
        )
        weight = float(roi_weights.get(roi, 1.0)) if roi_weights else 1.0
        acc += value * weight
        total_w += weight

    if total_w <= _EPS:
        return None
    return acc / total_w


def score_metric(
    metric: str,
    subfeatures: dict[str, float],
    fitzpatrick: int,
    config: dict[str, Any],
) -> float:
    """Convenience: composite raw -> 0-100 score in one call."""
    raw = composite_raw(metric, subfeatures, config)
    return score_from_raw(raw, metric, fitzpatrick, config)


def compare(
    current: dict[str, float],
    baseline: dict[str, float],
    min_delta: float = 5.0,
) -> dict[str, dict[str, Any]]:
    """Compare two sets of metric scores for time-series tracking.

    Parameters
    ----------
    current, baseline : dict
        Metric name -> 0-100 score. Only metrics present in both are compared.
    min_delta : float, optional
        Absolute score change considered meaningful. Because a single image has
        no per-metric variance, this fixed threshold stands in for significance;
        replace with a measured within-subject SD when available.

    Returns
    -------
    dict
        Per metric: ``{"delta", "direction", "significant"}`` where ``delta`` is
        ``current - baseline`` and ``direction`` is ``"up"|"down"|"flat"``.
    """
    out: dict[str, dict[str, Any]] = {}
    for metric in current:
        if metric not in baseline:
            continue
        delta = float(current[metric]) - float(baseline[metric])
        if delta > _EPS:
            direction = "up"
        elif delta < -_EPS:
            direction = "down"
        else:
            direction = "flat"
        out[metric] = {
            "delta": delta,
            "direction": direction,
            "significant": bool(abs(delta) >= min_delta),
        }
    return out
