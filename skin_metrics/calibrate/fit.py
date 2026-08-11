"""Fit ``calibration_profile.yaml`` from an extracted cohort.

Three things are fitted, all from the CSVs written by
:mod:`skin_metrics.calibrate.extract`, and written out by
:func:`write_profile` for :func:`skin_metrics.config.load_config` to merge over
``config.yaml``:

1. **Composite anchors** -- the mean/std used to standardise each physics
   sub-feature. The shipped placeholders were off by up to two orders of
   magnitude (``glcm_contrast`` anchored at 50 against a real value of ~0.8),
   which alone made the composite scores meaningless.
2. **Supervised instrument models** -- ridge regressions from the per-ROI
   physics features onto the instrument readings measured on that same ROI
   (Corneometer moisture; expert pigmentation grade). A model only ships if it
   clears :func:`accept_model` on held-out subjects; otherwise the metric keeps
   the unsupervised composite and the profile records the rejection.
3. **Reference distributions** -- empirical percentile grids of the score
   driver over the cohort, split by Fitzpatrick type, so a 0-100 score means
   "this percentile of the reference population". The grid is always built on
   whichever driver the scorer will actually use for that metric.

Erythema has no instrument ground truth in this corpus, so it can only ever
take the composite path.

Fitting always happens on the ``train`` split; every number reported back is
computed on the held-out ``val`` split (disjoint subjects).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# Sub-feature blocks handed to each supervised model. Deliberately restricted to
# the metric's own physics block: letting the hydration model see melanin/ITA
# would have it predict moisture from skin tone, which does not transfer.
HYDRATION_FEATURES = (
    "f_hydration_scaling_index",
    "f_hydration_glcm_contrast",
    "f_hydration_glcm_correlation",
    "f_hydration_glcm_energy",
    "f_hydration_lbp_uniformity",
    "f_hydration_wrinkle_density",
    "f_hydration_specular_ratio",
)
PIGMENTATION_FEATURES = (
    "f_pigmentation_melanin_index",
    "f_pigmentation_ita",
    "f_pigmentation_evenness",
    "f_pigmentation_spot_area_ratio",
    "f_pigmentation_spot_count",
    "f_pigmentation_spot_mean_contrast",
)

# ROIs that carry each instrument label, hence the ROIs a model may be applied
# to at inference time. (The nose is never measured by the instruments.)
MOISTURE_ROIS = ("forehead", "left_cheek", "right_cheek", "chin")
PIGMENTATION_ROIS = ("forehead", "left_cheek", "right_cheek")

# A Fitzpatrick bucket needs at least this many cohort images to get its own
# reference grid; otherwise it falls back to `default`.
MIN_BUCKET_N = 80

# Acceptance gate for a supervised model. A model that barely beats "always
# predict the cohort mean" must not be shipped: the report would print a
# concrete instrument value (e.g. "moisture 62 a.u.") that carries almost no
# information about the individual. Failing the gate is not an error -- the
# metric falls back to the cohort-percentile composite and the profile records
# why, which is the honest outcome.
MIN_HELDOUT_N = 30
MIN_PEARSON = 0.25
MIN_MAE_IMPROVEMENT = 0.05  # fraction of the predict-the-mean baseline

# Percentile grid stored per reference distribution (0, 1, ..., 100).
_QUANTILE_GRID = np.arange(0, 101, dtype=np.float64)

_EPS = 1e-8

# Anchors and reference moments are rounded to 6 decimals on the way out, so an
# epsilon-sized floor would round to exactly 0.0 and become a division hazard
# for whoever reads the profile back. Floor above that rounding instead.
_MIN_STD = 1e-6


def _column(rows: Sequence[dict[str, Any]], name: str) -> np.ndarray:
    """Extract one column as a float array with ``None`` mapped to ``nan``."""
    return np.array(
        [np.nan if r.get(name) is None else float(r[name]) for r in rows],
        dtype=np.float64,
    )


def _design_matrix(
    rows: Sequence[dict[str, Any]],
    features: Sequence[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(X, y, keep)`` dropping rows with any missing value.

    Parameters
    ----------
    rows : sequence of dict
        Feature rows from :func:`skin_metrics.calibrate.extract.load_rows`.
    features : sequence of str
        Column names forming the design matrix.
    target : str
        Column name of the regression target.

    Returns
    -------
    tuple
        ``X`` of shape ``(n, len(features))``, ``y`` of shape ``(n,)``, and the
        boolean mask of retained rows.
    """
    cols = [_column(rows, f) for f in features]
    y = _column(rows, target)
    X = np.column_stack(cols) if cols else np.empty((len(rows), 0))
    keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[keep], y[keep], keep


def trimmed_moments(values: np.ndarray, trim_pct: float = 1.0) -> tuple[float, float]:
    """Mean and std after dropping the extreme tails.

    Anchors are used to standardise features, so a single blown-out image (a
    failed detection, a blurred capture) should not set the scale.

    Parameters
    ----------
    values : numpy.ndarray
        Sample values; non-finite entries are ignored.
    trim_pct : float, optional
        Percentage trimmed from each tail.

    Returns
    -------
    tuple of float
        ``(mean, std)``. ``std`` is floored at a small epsilon so downstream
        z-scores can never divide by zero.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [trim_pct, 100.0 - trim_pct])
    core = v[(v >= lo) & (v <= hi)]
    if core.size < 2:
        core = v
    # Floor above the rounding used when these are serialised, so a constant
    # feature can never be written out as `std: 0.0` and then divided by.
    return float(core.mean()), float(max(core.std(), _MIN_STD))


def fit_anchors(
    face_rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    """Re-estimate ``composite.<metric>.anchors`` from the cohort.

    Parameters
    ----------
    face_rows : sequence of dict
        Face-level feature rows (training split).
    config : dict
        Current configuration; its ``composite.<metric>.weights`` decides which
        sub-features need anchors.

    Returns
    -------
    dict
        ``{metric: {feature: {"mean": m, "std": s}}}``.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for metric, spec in config["composite"].items():
        anchors: dict[str, dict[str, float]] = {}
        for feature in spec["weights"]:
            mean, std = trimmed_moments(_column(face_rows, f"f_{metric}_{feature}"))
            anchors[feature] = {"mean": round(mean, 6), "std": round(std, 6)}
        out[metric] = anchors
    return out


def fit_ridge(
    train_rows: Sequence[dict[str, Any]],
    features: Sequence[str],
    target: str,
    *,
    alphas: Iterable[float] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
) -> dict[str, Any] | None:
    """Fit a standardised ridge regression from physics features to a label.

    Parameters
    ----------
    train_rows : sequence of dict
        Per-ROI rows from the training split, already filtered to the ROIs the
        label exists for.
    features : sequence of str
        Design-matrix columns.
    target : str
        Label column.
    alphas : iterable of float, optional
        Ridge penalties searched by leave-one-out cross-validation.

    Returns
    -------
    dict or None
        Serialisable model (feature names, standardisation moments,
        coefficients, intercept, training size), or ``None`` if there was not
        enough data to fit.
    """
    from sklearn.linear_model import RidgeCV

    X, y, _ = _design_matrix(train_rows, features, target)
    if X.shape[0] < 50:
        return None

    mean = X.mean(axis=0)
    std = np.maximum(X.std(axis=0), _EPS)
    Xz = (X - mean) / std

    model = RidgeCV(alphas=list(alphas))
    model.fit(Xz, y)

    return {
        "features": list(features),
        "feature_mean": [round(float(v), 8) for v in mean],
        "feature_std": [round(float(v), 8) for v in std],
        "coef": [round(float(v), 6) for v in np.ravel(model.coef_)],
        "intercept": round(float(model.intercept_), 6),
        "alpha": float(model.alpha_),
        "n_train": int(X.shape[0]),
    }


def predict_ridge(model: dict[str, Any], row: dict[str, Any]) -> float | None:
    """Apply a fitted model to one feature row.

    Parameters
    ----------
    model : dict
        Output of :func:`fit_ridge`.
    row : dict
        Feature row containing every column in ``model["features"]``.

    Returns
    -------
    float or None
        Prediction, or ``None`` if any required feature is missing.
    """
    values = []
    for name in model["features"]:
        v = row.get(name)
        if v is None or not np.isfinite(float(v)):
            return None
        values.append(float(v))
    x = np.asarray(values, dtype=np.float64)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    std = np.asarray(model["feature_std"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    return float(np.dot(coef, (x - mean) / std) + model["intercept"])


def evaluate_ridge(
    model: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    target: str,
) -> dict[str, float]:
    """Score a fitted model on held-out rows.

    Parameters
    ----------
    model : dict
        Output of :func:`fit_ridge`.
    rows : sequence of dict
        Held-out rows (validation split).
    target : str
        Label column.

    Returns
    -------
    dict
        ``{"n", "pearson", "spearman", "mae", "rmse", "baseline_mae"}`` where
        ``baseline_mae`` is the error of always predicting the training mean --
        the number the model has to beat to be worth anything.
    """
    from scipy.stats import pearsonr, spearmanr

    X, y, keep = _design_matrix(rows, model["features"], target)
    if X.shape[0] < 5:
        return {"n": int(X.shape[0])}
    kept_rows = [r for r, k in zip(rows, keep) if k]
    pred = np.array([predict_ridge(model, r) for r in kept_rows], dtype=np.float64)

    resid = pred - y
    return {
        "n": int(y.size),
        "pearson": round(float(pearsonr(pred, y).statistic), 4),
        "spearman": round(float(spearmanr(pred, y).statistic), 4),
        "mae": round(float(np.mean(np.abs(resid))), 4),
        "rmse": round(float(np.sqrt(np.mean(resid**2))), 4),
        "baseline_mae": round(float(np.mean(np.abs(y - y.mean()))), 4),
    }


# Instrument targets the composite weights can be fitted against. The instrument
# count is preferred over the expert grade: it is objective and present on every
# image.
COMPOSITE_TARGETS: dict[str, str] = {
    "pigmentation": "label_pigmentation_count",
    "hydration": "label_moisture",
}

# Which table each metric is fitted on. Pigmentation's instrument reading is one
# whole-face spot count, so it can only be fitted face-level. The Corneometer
# reads *per site*, and averaging those readings into one face number throws
# away most of the signal -- fitting it ROI-level (restricted to the sites the
# metric declares in `composite.<metric>.rois`) measured +0.176 -> +0.280 on
# held-out cheeks where the face-level fit had collapsed to -0.208.
COMPOSITE_LEVEL: dict[str, str] = {"pigmentation": "face", "hydration": "roi"}

# Sign that maps the target onto the score's orientation (higher = more
# pronounced). Moisture runs the other way: more moisture, less dryness.
COMPOSITE_TARGET_SIGN: dict[str, float] = {"pigmentation": 1.0, "hydration": -1.0}

# Fitted weights must beat the hand-set ones by at least this much Spearman on
# held-out subjects before they are shipped.
MIN_WEIGHT_GAIN = 0.05


def _rank_agreement(
    rows: Sequence[dict[str, Any]],
    metric: str,
    weights: dict[str, float],
    anchors: dict[str, dict[str, float]],
    target: str,
    sign: float,
) -> tuple[float, int]:
    """Spearman between a weighted composite and the instrument target.

    Parameters
    ----------
    rows : sequence of dict
        Face-level rows (held-out split).
    metric : str
        Metric whose composite is being scored.
    weights, anchors : dict
        The weight set and matching anchors to evaluate.
    target : str
        Face-level label column.
    sign : float
        ``+1``/``-1`` orienting the target like the score.

    Returns
    -------
    tuple
        ``(spearman, n)``; ``(0.0, 0)`` when there is nothing to compare.
    """
    from scipy.stats import spearmanr

    usable = [r for r in rows if r.get(target) is not None]
    if len(usable) < 10:
        return 0.0, len(usable)
    composite = np.array(
        [_composite_raw(r, metric, weights, anchors) for r in usable], dtype=np.float64
    )
    y = sign * np.array([float(r[target]) for r in usable], dtype=np.float64)
    if not np.isfinite(composite).all() or composite.std() <= _EPS:
        return 0.0, len(usable)
    return float(spearmanr(composite, y).statistic), len(usable)


def fit_composite_weights(
    train_rows: Sequence[dict[str, Any]],
    metric: str,
    features: Sequence[str],
    target: str,
    anchors: dict[str, dict[str, float]],
    *,
    sign: float = 1.0,
    alphas: Iterable[float] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
) -> dict[str, float] | None:
    """Fit composite weights by regressing the target on standardised features.

    The composite is a weighted sum of z-scored sub-features, so a ridge fitted
    in the same standardised space yields exactly those weights. They are
    rescaled to sum to 1 in absolute value, which keeps the composite on the
    familiar unit scale regardless of the target's units.

    Parameters
    ----------
    train_rows : sequence of dict
        Face-level rows from the training split.
    metric : str
        Metric being fitted.
    features : sequence of str
        Sub-feature names (without the ``f_<metric>_`` prefix).
    target : str
        Instrument label column.
    anchors : dict
        Fitted anchors, used to standardise exactly as the scorer will.
    sign : float, optional
        :data:`COMPOSITE_TARGET_SIGN` for this metric. The composite driver is
        oriented "higher = more pronounced", which for hydration is the
        *opposite* of the instrument (more moisture, less dryness). Regressing
        on the raw target here would return weights of the wrong sign, and the
        gate would then reject them for being anti-correlated -- which is what
        happened to hydration on every device before this argument existed.
    alphas : iterable of float, optional
        Ridge penalties searched by cross-validation.

    Returns
    -------
    dict or None
        ``{feature: weight}`` summing to 1 in absolute value, or ``None`` when
        there is too little data or the target is degenerate.
    """
    from sklearn.linear_model import RidgeCV

    usable = [r for r in train_rows if r.get(target) is not None]
    if len(usable) < 100:
        return None

    columns = [f"f_{metric}_{name}" for name in features]
    X = np.array(
        [[_zscore_value(r.get(c), anchors[n]) for c, n in zip(columns, features)]
         for r in usable],
        dtype=np.float64,
    )
    y = np.array([sign * float(r[target]) for r in usable], dtype=np.float64)
    keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[keep], y[keep]
    if X.shape[0] < 100 or y.std() <= _EPS:
        return None

    model = RidgeCV(alphas=list(alphas)).fit(X, y)
    coef = np.ravel(model.coef_)
    scale = float(np.abs(coef).sum())
    if scale <= _EPS:
        return None
    return {name: round(float(c / scale), 4) for name, c in zip(features, coef)}


def _zscore_value(value: Any, anchor: dict[str, float]) -> float:
    """Standardise one raw feature value against its anchor."""
    if value is None:
        return np.nan
    return (float(value) - anchor["mean"]) / max(abs(anchor["std"]), _EPS)


def accept_model(stats: dict[str, float]) -> tuple[bool, str]:
    """Decide whether held-out performance justifies shipping a model.

    Parameters
    ----------
    stats : dict
        Output of :func:`evaluate_ridge`.

    Returns
    -------
    tuple
        ``(accepted, reason)``. ``reason`` is always populated so the profile
        can explain a rejection.
    """
    n = int(stats.get("n", 0))
    if n < MIN_HELDOUT_N:
        return False, f"only {n} held-out rows (need {MIN_HELDOUT_N})"

    pearson = float(stats.get("pearson", 0.0))
    baseline = float(stats.get("baseline_mae", 0.0))
    mae = float(stats.get("mae", 0.0))
    improvement = (baseline - mae) / baseline if baseline > _EPS else 0.0

    if abs(pearson) < MIN_PEARSON:
        return False, (
            f"held-out correlation {pearson:+.3f} is below {MIN_PEARSON:+.2f}"
        )
    if improvement < MIN_MAE_IMPROVEMENT:
        return False, (
            f"held-out MAE {mae:.3f} improves on the predict-the-mean baseline "
            f"{baseline:.3f} by only {improvement:.1%} "
            f"(need {MIN_MAE_IMPROVEMENT:.0%})"
        )
    return True, (
        f"held-out r={pearson:+.3f}, MAE {mae:.3f} vs {baseline:.3f} baseline "
        f"({improvement:.1%} better)"
    )


def fit_reference(
    values: np.ndarray,
    fitzpatrick: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Build per-Fitzpatrick empirical percentile grids for a score driver.

    An empirical grid beats the normal-CDF assumption the scorer shipped with:
    spot counts and predicted grades are visibly skewed, so a Gaussian
    percentile misplaces the tails.

    Parameters
    ----------
    values : numpy.ndarray
        Score-driver value per cohort image (higher = more pronounced).
    fitzpatrick : numpy.ndarray
        Matching Fitzpatrick type per image.

    Returns
    -------
    dict
        ``{"default": entry, "3": entry, ...}`` where each entry has
        ``{"mean", "std", "n", "quantiles"}`` and ``quantiles`` is the value at
        each whole percentile from 0 to 100.
    """
    out: dict[str, dict[str, Any]] = {}

    def _entry(sample: np.ndarray) -> dict[str, Any]:
        """Moments plus the whole-percentile grid for one bucket."""
        mean, std = trimmed_moments(sample)
        return {
            "mean": round(mean, 6),
            "std": round(std, 6),
            "n": int(sample.size),
            "quantiles": [
                round(float(v), 6) for v in np.percentile(sample, _QUANTILE_GRID)
            ],
        }

    finite = np.isfinite(values)
    out["default"] = _entry(values[finite])
    for t in range(1, 7):
        sel = finite & (fitzpatrick == t)
        if int(sel.sum()) >= MIN_BUCKET_N:
            out[str(t)] = _entry(values[sel])
    return out


def _composite_raw(
    row: dict[str, Any],
    metric: str,
    weights: dict[str, float],
    anchors: dict[str, dict[str, float]],
) -> float:
    """Composite raw value for one flat feature row.

    Delegates to the runtime scorer rather than reimplementing it. An earlier
    local copy drifted out of sync -- it renormalised by the *signed* weight
    sum, so any fitted weight set summing to <= 0 silently produced an
    all-zeros composite -- and the reference grids fitted here must be built by
    exactly the code that will later look values up in them.

    Parameters
    ----------
    row : dict
        Face-level row with ``f_<metric>_<name>`` columns.
    metric : str
        Metric whose composite is wanted.
    weights, anchors : dict
        Weight set and matching anchors.

    Returns
    -------
    float
        The composite raw value.
    """
    from ..scoring.normalize import composite_raw

    prefix = f"f_{metric}_"
    subfeatures = {
        key[len(prefix):]: value
        for key, value in row.items()
        if key.startswith(prefix) and value is not None
    }
    stub = {"composite": {metric: {"weights": weights, "anchors": anchors}}}
    return composite_raw(metric, subfeatures, stub)


def fit_calibration(
    roi_rows: Sequence[dict[str, Any]],
    face_rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
    *,
    device: str | None = None,
) -> dict[str, Any]:
    """Fit anchors, supervised models, and reference grids in one pass.

    Parameters
    ----------
    roi_rows, face_rows : sequence of dict
        Rows loaded from ``features_roi.csv`` / ``features_face.csv``.
    config : dict
        Current configuration (supplies the composite weights).
    device : str, optional
        Restrict the fit to one capture device; ``None`` pools all devices,
        which is what a general-purpose deployment should use.

    Returns
    -------
    dict
        ``{"composite", "reference", "supervised", "validation", "provenance"}``
        ready to merge into ``config.yaml``.
    """
    if device is not None:
        roi_rows = [r for r in roi_rows if r["device"] == device]
        face_rows = [r for r in face_rows if r["device"] == device]

    roi_train = [r for r in roi_rows if r["split"] == "train"]
    roi_val = [r for r in roi_rows if r["split"] == "val"]
    face_train = [r for r in face_rows if r["split"] == "train"]

    # --- 1. anchors ------------------------------------------------------
    anchors = fit_anchors(face_train, config)

    # --- 1b. composite weights -------------------------------------------
    # The hand-set weights in config.yaml were never measured against anything.
    # Refit them, but only ship the result if it actually ranks held-out
    # subjects better than what is already there.
    composite_weights: dict[str, dict[str, float]] = {}
    validation_weights: dict[str, Any] = {}
    face_val = [r for r in face_rows if r["split"] == "val"]
    for metric, target in COMPOSITE_TARGETS.items():
        if metric not in config["composite"]:
            continue
        spec = config["composite"][metric]
        declared = spec["weights"]
        sign = COMPOSITE_TARGET_SIGN[metric]
        if COMPOSITE_LEVEL[metric] == "roi":
            keep = set(spec.get("rois") or ())
            w_train = [r for r in roi_train if not keep or r["roi"] in keep]
            w_val = [r for r in roi_val if not keep or r["roi"] in keep]
        else:
            w_train, w_val = face_train, face_val
        fitted = fit_composite_weights(
            w_train, metric, list(declared), target, anchors[metric], sign=sign
        )
        base_rho, n = _rank_agreement(
            w_val, metric, declared, anchors[metric], target, sign
        )
        entry: dict[str, Any] = {
            "target": target,
            "n_heldout": n,
            "spearman_declared": round(base_rho, 4),
        }
        if fitted is None:
            entry.update({"accepted": False, "reason": "too few labelled rows to fit"})
        else:
            new_rho, _ = _rank_agreement(
                w_val, metric, fitted, anchors[metric], target, sign
            )
            entry["spearman_fitted"] = round(new_rho, 4)
            gain = new_rho - base_rho
            if gain >= MIN_WEIGHT_GAIN:
                composite_weights[metric] = fitted
                entry.update(
                    {"accepted": True,
                     "reason": f"held-out Spearman {base_rho:+.3f} -> {new_rho:+.3f} "
                               f"({gain:+.3f})"}
                )
            else:
                entry.update(
                    {"accepted": False,
                     "reason": f"held-out Spearman {base_rho:+.3f} -> {new_rho:+.3f} "
                               f"({gain:+.3f}); needs {MIN_WEIGHT_GAIN:+.2f}"}
                )
        validation_weights[metric] = entry

    # Score everything below with whatever weights actually won.
    effective_weights = {
        metric: composite_weights.get(metric, spec["weights"])
        for metric, spec in config["composite"].items()
    }

    # --- 2. supervised instrument models ---------------------------------
    supervised: dict[str, Any] = {}
    validation: dict[str, Any] = {}

    specs = (
        (
            "hydration",
            HYDRATION_FEATURES,
            "label_moisture",
            MOISTURE_ROIS,
            {
                "target": "corneometer_moisture",
                "target_units": "a.u.",
                # The score reads "how dry", so more moisture must score lower.
                "score_sign": -1.0,
            },
        ),
        (
            "pigmentation",
            PIGMENTATION_FEATURES,
            "label_pigmentation_grade",
            PIGMENTATION_ROIS,
            {
                "target": "expert_pigmentation_grade",
                "target_units": "grade 0-5",
                "score_sign": 1.0,
            },
        ),
    )

    for metric, features, target, rois, meta in specs:
        train_subset = [r for r in roi_train if r["roi"] in rois]
        val_subset = [r for r in roi_val if r["roi"] in rois]
        model = fit_ridge(train_subset, features, target)
        if model is None:
            validation[metric] = {
                "accepted": False,
                "reason": "too few training rows to fit",
            }
            continue

        model.update(meta)
        model["applies_to_rois"] = list(rois)

        stats = evaluate_ridge(model, val_subset, target)
        accepted, reason = accept_model(stats)
        validation[metric] = {**stats, "accepted": accepted, "reason": reason}
        if accepted:
            supervised[metric] = model

    # --- 3. reference distributions --------------------------------------
    # Each grid must be built on the very quantity the scorer will later look up
    # in it. So a metric whose model was accepted gets a grid over model
    # predictions; a metric that fell back (rejected model, or erythema which
    # has no label at all) gets one over its composite raw value.
    reference: dict[str, Any] = {}
    fitz_train = _column(face_train, "fitzpatrick")
    roi_by_key = _index_by_key(roi_train)

    # Driven by what the config actually declares, not a hard-coded triple, so
    # a config carrying a subset of the metrics still fits cleanly.
    for metric in config["composite"]:
        model = supervised.get(metric)
        if model is not None:
            sign = float(model.get("score_sign", 1.0))
            driver = sign * np.array(
                [_face_prediction(r, roi_by_key, model) for r in face_train],
                dtype=np.float64,
            )
        else:
            # The composite weights are already written so that higher means
            # "more pronounced" for every metric, so no sign flip here.
            driver = np.array(
                [
                    _composite_raw(r, metric, effective_weights[metric], anchors[metric])
                    for r in face_train
                ],
                dtype=np.float64,
            )
        reference[metric] = fit_reference(driver, fitz_train)

    return {
        "composite_anchors": anchors,
        "composite_weights": composite_weights,
        "reference": reference,
        "supervised": supervised,
        "validation": validation,
        "validation_weights": validation_weights,
        "provenance": {
            "corpus": "AI-Hub 028. 한국인 피부상태 측정 데이터",
            "device": device or "all",
            "n_train_images": len(face_train),
            "n_train_rois": len(roi_train),
            "n_val_rois": len(roi_val),
        },
    }


def write_profile(
    fitted: dict[str, Any],
    path: str | Path,
    *,
    profile_name: str = "korean_cohort_2023",
) -> Path:
    """Write a fitted calibration as ``calibration_profile.yaml``.

    Kept separate from ``config.yaml`` so regenerating the fit never clobbers
    the hand-written rationale there; :func:`skin_metrics.config.load_config`
    merges the two.

    Parameters
    ----------
    fitted : dict
        Output of :func:`fit_calibration`.
    path : str or pathlib.Path
        Destination file.
    profile_name : str, optional
        Identifier surfaced in reports as ``SkinReport.calibration_profile``.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    import yaml

    provenance = dict(fitted["provenance"])
    provenance["profile"] = profile_name
    provenance["fitted_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    document = {
        "provenance": provenance,
        "composite_anchors": fitted["composite_anchors"],
        "composite_weights": fitted["composite_weights"],
        "supervised": fitted["supervised"],
        "reference": fitted["reference"],
        "validation": fitted["validation"],
        "validation_weights": fitted["validation_weights"],
    }

    out = Path(path)
    header = (
        "# GENERATED FILE -- do not edit by hand.\n"
        "#\n"
        "# Fitted by `skin-metrics calibrate fit` from a labelled cohort; see\n"
        "# skin_metrics/calibrate/fit.py. Hand-maintained settings and their\n"
        "# rationale live in config.yaml, which this file is merged over at\n"
        "# load time (skin_metrics.config.load_config).\n"
        "#\n"
        "# `validation` numbers are measured on held-out subjects, disjoint\n"
        "# from the ones the models were fitted on.\n"
    )
    with out.open("w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump(document, fh, sort_keys=False, allow_unicode=True,
                       default_flow_style=False, width=100)
    return out


def format_validation(fitted: dict[str, Any]) -> str:
    """Render the held-out validation numbers as a readable block.

    Parameters
    ----------
    fitted : dict
        Output of :func:`fit_calibration`.

    Returns
    -------
    str
        Multi-line summary suitable for a CLI or a commit message.
    """
    prov = fitted["provenance"]
    lines = [
        f"corpus:  {prov['corpus']}  (device: {prov['device']})",
        f"fitted:  {prov['n_train_images']} images / {prov['n_train_rois']} ROI rows",
        f"held out: {prov['n_val_rois']} ROI rows from disjoint subjects",
        "",
    ]
    lines.append("composite weights (rank agreement with the instrument):")
    for metric, w in sorted(fitted["validation_weights"].items()):
        mark = "FITTED  " if w.get("accepted") else "DECLARED"
        detail = ""
        if "spearman_fitted" in w:
            detail = (f"  declared={w['spearman_declared']:+.3f} "
                      f"fitted={w['spearman_fitted']:+.3f}")
        lines.append(f"  {metric:14s} {mark}  n={w.get('n_heldout', 0)}{detail}")
        lines.append(f"                 {w.get('reason', '')}")
    lines.append("")

    lines.append("supervised instrument models:")
    for metric, stats in sorted(fitted["validation"].items()):
        mark = "ACCEPTED" if stats.get("accepted") else "REJECTED"
        lines.append(f"{metric}: {mark}")
        if stats.get("n", 0) >= 5:
            lines.append(
                f"    n={stats['n']}  pearson={stats['pearson']:+.3f}  "
                f"spearman={stats['spearman']:+.3f}  "
                f"MAE={stats['mae']:.3f} vs {stats['baseline_mae']:.3f} baseline "
                f"({_improvement(stats):+.1f}%)"
            )
        lines.append(f"    {stats.get('reason', '')}")
        if not stats.get("accepted"):
            lines.append(
                f"    -> {metric} keeps the unsupervised composite, scored "
                "against the cohort percentile grid."
            )
        lines.append("")

    lines.append(
        "erythema: no instrument ground truth in this corpus; cohort percentile "
        "only."
    )
    return "\n".join(lines)


def _improvement(stats: dict[str, float]) -> float:
    """Percent reduction in MAE relative to predicting the mean."""
    base = stats.get("baseline_mae", 0.0)
    if base <= _EPS:
        return 0.0
    return 100.0 * (base - stats["mae"]) / base


def _index_by_key(
    roi_rows: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group per-ROI rows by their image key (avoids an O(n^2) rescan)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for row in roi_rows:
        out.setdefault(row["key"], []).append(row)
    return out


def _face_prediction(
    face_row: dict[str, Any],
    roi_by_key: dict[str, list[dict[str, Any]]],
    model: dict[str, Any],
) -> float:
    """Face-level model prediction: mean over the ROIs the model applies to.

    Mirrors what the runtime scorer does, so the reference grid is built on the
    same quantity that will later be looked up in it.
    """
    wanted = set(model.get("applies_to_rois", ()))
    preds = [
        p
        for r in roi_by_key.get(face_row["key"], ())
        if r["roi"] in wanted
        for p in (predict_ridge(model, r),)
        if p is not None
    ]
    return float(np.mean(preds)) if preds else np.nan
