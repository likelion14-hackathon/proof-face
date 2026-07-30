"""Normalisation and scoring: raw features -> 0-100 percentile scores.

Each metric combines its z-scored sub-features into a single composite raw
value (weights + anchors from ``config.yaml``), then maps that value to a 0-100
score via a normal-CDF percentile against a **Fitzpatrick-type-specific**
reference distribution. Splitting the reference by skin type reduces bias for
darker skin, where absolute pigmentation/erythema features differ systematically.
"""

from __future__ import annotations

import math
from typing import Any

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
    """
    spec = config["composite"][metric]
    weights: dict[str, float] = spec["weights"]
    anchors: dict[str, dict[str, float]] = spec["anchors"]

    total_w = 0.0
    acc = 0.0
    for name, w in weights.items():
        if name not in subfeatures:
            continue
        anchor = anchors[name]
        z = _zscore(float(subfeatures[name]), anchor["mean"], anchor["std"])
        acc += w * z
        total_w += w
    if total_w <= _EPS:
        return 0.0
    return acc / total_w


def score_from_raw(
    raw: float,
    metric: str,
    fitzpatrick: int,
    config: dict[str, Any],
) -> float:
    """Map a composite raw value to a 0-100 percentile score.

    Parameters
    ----------
    raw : float
        Composite raw value from :func:`composite_raw`.
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
    clip_z = float(config.get("scoring", {}).get("clip_z", 4.0))
    z = _zscore(raw, ref["mean"], ref["std"])
    z = max(-clip_z, min(clip_z, z))
    return float(_normal_cdf(z) * 100.0)


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
