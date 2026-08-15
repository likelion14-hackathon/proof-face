"""Tests for the offline calibration tooling.

Everything here runs on synthetic tables -- no corpus, no images, no network --
so the suite stays runnable without the 43GB AI-Hub download.
"""

from __future__ import annotations

import numpy as np
import pytest

from skin_metrics.calibrate import fit as cfit
from skin_metrics.config import merge_profile
from skin_metrics.scoring.normalize import (
    _percentile_from_quantiles,
    flatten_features,
    predict_instrument,
    score_from_raw,
)


# --- helpers ---------------------------------------------------------------


def _roi_row(key, roi, split, pore_count, contrast, grade=None, **extra):
    """One synthetic per-ROI feature row."""
    row = {
        "key": key,
        "roi": roi,
        "split": split,
        "device": "phone",
        "subject": key.split("/")[-1],
        "label_pore_count": pore_count,
        "label_pigmentation_grade": grade,
        "f_pores_scaling_index": 0.001 + 0.0001 * contrast,
        "f_pores_glcm_contrast": contrast,
        "f_pores_glcm_correlation": 0.95,
        "f_pores_glcm_energy": 0.26,
        "f_pores_lbp_uniformity": 0.2,
        "f_pores_wrinkle_density": 0.07,
        "f_pores_specular_ratio": 0.002,
    }
    row.update(extra)
    return row


# --- trimmed_moments -------------------------------------------------------


def test_trimmed_moments_ignores_outliers_and_nans():
    values = np.concatenate([np.full(200, 5.0), [1e9], [np.nan]])
    mean, std = cfit.trimmed_moments(values)
    assert mean == pytest.approx(5.0)
    assert std > 0.0  # floored, never zero


def test_trimmed_moments_empty_is_safe():
    mean, std = cfit.trimmed_moments(np.array([np.nan, np.nan]))
    assert (mean, std) == (0.0, 1.0)


# --- ridge round-trip ------------------------------------------------------


def test_ridge_recovers_a_linear_relationship():
    """A pore-count label that is an exact linear function of one feature."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(300):
        contrast = float(rng.uniform(0.2, 2.0))
        # more pores as texture contrast rises
        rows.append(_roi_row(f"train/{i:04d}", "left_cheek", "train",
                             pore_count=200.0 + 400.0 * contrast, contrast=contrast))

    model = cfit.fit_ridge(rows, cfit.PORE_FEATURES, "label_pore_count")
    assert model is not None

    stats = cfit.evaluate_ridge(model, rows, "label_pore_count")
    assert stats["pearson"] > 0.98
    assert stats["mae"] < stats["baseline_mae"]


def test_fit_ridge_declines_tiny_samples():
    rows = [_roi_row(f"k{i}", "left_cheek", "train", 600.0, 1.0) for i in range(10)]
    assert cfit.fit_ridge(rows, cfit.PORE_FEATURES, "label_pore_count") is None


def test_predict_ridge_returns_none_on_missing_feature():
    rows = [
        _roi_row(f"k{i}", "left_cheek", "train", 600.0 + i % 7, 0.5 + 0.01 * i)
        for i in range(120)
    ]
    model = cfit.fit_ridge(rows, cfit.PORE_FEATURES, "label_pore_count")
    incomplete = dict(rows[0])
    del incomplete["f_pores_glcm_contrast"]
    assert cfit.predict_ridge(model, incomplete) is None


# --- acceptance gate -------------------------------------------------------


def test_accept_model_requires_a_real_improvement():
    good = {"n": 500, "pearson": 0.54, "mae": 0.78, "baseline_mae": 0.97}
    accepted, reason = cfit.accept_model(good)
    assert accepted and "better" in reason


def test_accept_model_rejects_a_barely_better_than_mean_model():
    """Statistically real, practically useless -- the moisture metric's fate."""
    weak = {"n": 800, "pearson": 0.185, "mae": 7.566, "baseline_mae": 7.741}
    accepted, reason = cfit.accept_model(weak)
    assert not accepted
    assert "correlation" in reason


def test_accept_model_rejects_when_mae_gain_is_tiny_despite_correlation():
    stats = {"n": 500, "pearson": 0.40, "mae": 9.8, "baseline_mae": 10.0}
    accepted, reason = cfit.accept_model(stats)
    assert not accepted and "MAE" in reason


def test_accept_model_rejects_without_enough_held_out_rows():
    accepted, reason = cfit.accept_model(
        {"n": 5, "pearson": 0.9, "mae": 0.1, "baseline_mae": 1.0}
    )
    assert not accepted and "held-out" in reason


# --- composite weights -----------------------------------------------------


def _face_row(key, split, spot_count, count_label, device="phone"):
    """One synthetic face-level row with a linear instrument label."""
    return {
        "key": key,
        "subject": key.split("/")[-1],
        "split": split,
        "device": device,
        "fitzpatrick": 3.0,
        "f_pigmentation_spot_count": spot_count,
        "f_pigmentation_spot_area_ratio": 0.01 + 0.0001 * spot_count,
        "f_pigmentation_spot_mean_contrast": 7.0,
        "f_pigmentation_evenness": 6.0,
        "label_pigmentation_count": count_label,
    }


def test_composite_raw_handles_negative_weights():
    """Renormalising by the SIGNED weight sum silently zeroes the composite.

    A fitted weight set summing to <= 0 is legitimate (suppressor variables);
    the denominator must be the sum of absolute weights.
    """
    config = {
        "composite": {
            "pigmentation": {
                "weights": {"a": -0.7, "b": 0.3},
                "anchors": {
                    "a": {"mean": 0.0, "std": 1.0},
                    "b": {"mean": 0.0, "std": 1.0},
                },
            }
        }
    }
    from skin_metrics.scoring.normalize import composite_raw

    # (-0.7*2 + 0.3*4) / (0.7 + 0.3) = -0.2
    assert composite_raw("pigmentation", {"a": 2.0, "b": 4.0}, config) == pytest.approx(-0.2)


def test_fit_composite_weights_recovers_the_driving_feature():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(400):
        spots = float(rng.uniform(1, 40))
        rows.append(_face_row(f"train/{i:04d}", "train", spots, 10.0 * spots))
    anchors = {
        "spot_count": {"mean": 20.0, "std": 10.0},
        "spot_area_ratio": {"mean": 0.012, "std": 0.001},
        "spot_mean_contrast": {"mean": 7.0, "std": 1.0},
        "evenness": {"mean": 6.0, "std": 1.0},
    }
    weights = cfit.fit_composite_weights(
        rows, "pigmentation", list(anchors), "label_pigmentation_count", anchors
    )
    assert weights is not None
    assert sum(abs(w) for w in weights.values()) == pytest.approx(1.0, abs=1e-3)
    # The two informative features must carry essentially all the weight.
    informative = abs(weights["spot_count"]) + abs(weights["spot_area_ratio"])
    assert informative > 0.9


def test_fit_composite_weights_honours_an_inverted_target():
    """A sign=-1 target must not come back as sign-flipped weights.

    No shipped metric currently needs this -- both targets in
    :data:`COMPOSITE_TARGET_SIGN` are +1 -- but any instrument that reads the
    *good* end of a scale does, and the failure is silent: the fit returns
    exactly inverted weights and the gate then rejects them as
    "anti-correlated" rather than as a bug. That is how the moisture metric
    failed on every device before the ``sign`` argument existed, so the
    mechanism is pinned here even while it is unused.
    """
    rng = np.random.default_rng(11)
    rows = []
    for i in range(400):
        severity = float(rng.uniform(0.0, 1.0))
        rows.append(_face_row(f"train/{i:04d}", "train", severity, 0.0))
        # A hypothetical instrument reading the healthy end of the scale: it
        # falls as the condition the score measures gets worse.
        rows[-1]["label_inverted_instrument"] = 100.0 - 50.0 * severity

    anchors = {"spot_count": {"mean": 0.5, "std": 0.25}}
    plain = cfit.fit_composite_weights(
        rows, "pigmentation", ["spot_count"], "label_inverted_instrument", anchors
    )
    signed = cfit.fit_composite_weights(
        rows, "pigmentation", ["spot_count"], "label_inverted_instrument", anchors,
        sign=-1.0,
    )
    assert plain["spot_count"] < 0        # tracks the instrument
    assert signed["spot_count"] > 0       # tracks severity, which is what we score


def test_fit_composite_weights_declines_tiny_samples():
    rows = [_face_row(f"k{i}", "train", 10.0, 100.0) for i in range(20)]
    anchors = {"spot_count": {"mean": 10.0, "std": 5.0}}
    assert cfit.fit_composite_weights(
        rows, "pigmentation", ["spot_count"], "label_pigmentation_count", anchors
    ) is None


def test_fit_calibration_keeps_declared_weights_when_fitting_does_not_help():
    """Weights only ship when they beat the declared set on held-out rows."""
    rng = np.random.default_rng(4)
    face, roi = [], []
    for i in range(400):
        spots = float(rng.uniform(1, 40))
        # Label is pure noise -> no weight set can rank held-out subjects.
        face.append(_face_row(f"train/{i:04d}", "train", spots, float(rng.uniform(0, 100))))
    for i in range(120):
        spots = float(rng.uniform(1, 40))
        face.append(_face_row(f"val/{i:04d}", "val", spots, float(rng.uniform(0, 100))))

    config = {
        "composite": {
            "pigmentation": {
                "weights": {"spot_count": 1.0},
                "anchors": {"spot_count": {"mean": 20.0, "std": 10.0}},
            }
        }
    }
    fitted = cfit.fit_calibration(roi, face, config)
    assert "pigmentation" not in fitted["composite_weights"]
    assert fitted["validation_weights"]["pigmentation"]["accepted"] is False


# --- reference grids -------------------------------------------------------


def test_fit_reference_buckets_only_well_populated_types():
    rng = np.random.default_rng(1)
    values = rng.normal(size=500)
    fitz = np.array([3] * 400 + [5] * 100)  # both clear MIN_BUCKET_N
    ref = cfit.fit_reference(values, fitz)
    assert "default" in ref and "3" in ref and "5" in ref

    sparse = np.array([3] * 490 + [6] * 10)
    ref2 = cfit.fit_reference(values, sparse)
    assert "6" not in ref2, "a 10-image bucket must fall back to default"


def test_reference_quantiles_are_monotonic():
    values = np.random.default_rng(2).gamma(2.0, 1.0, size=400)
    ref = cfit.fit_reference(values, np.full(400, 3))
    q = ref["default"]["quantiles"]
    assert len(q) == 101
    assert all(a <= b for a, b in zip(q, q[1:]))


# --- percentile mapping ----------------------------------------------------


def test_percentile_from_quantiles_matches_the_grid():
    quantiles = list(np.linspace(0.0, 100.0, 101))  # value == percentile
    assert _percentile_from_quantiles(0.0, quantiles) == 0.0
    assert _percentile_from_quantiles(100.0, quantiles) == 100.0
    assert _percentile_from_quantiles(50.0, quantiles) == pytest.approx(50.0)
    assert _percentile_from_quantiles(25.5, quantiles) == pytest.approx(25.5)


def test_percentile_saturates_outside_the_grid():
    quantiles = list(np.linspace(10.0, 20.0, 101))
    assert _percentile_from_quantiles(-5.0, quantiles) == 0.0
    assert _percentile_from_quantiles(999.0, quantiles) == 100.0


def test_score_from_raw_prefers_quantiles_over_normal_cdf():
    """An empirical grid must win when present; the Gaussian is the fallback."""
    config = {
        "reference": {
            "pores": {
                "default": {
                    "mean": 0.0,
                    "std": 1.0,
                    # Skewed grid: the median sits at 9, not at 0.
                    "quantiles": list(np.linspace(0.0, 100.0, 101) ** 0.5 * 10.0 / 10.0),
                }
            }
        },
        "scoring": {"clip_z": 4.0},
    }
    empirical = score_from_raw(5.0, "pores", 3, config)

    del config["reference"]["pores"]["default"]["quantiles"]
    gaussian = score_from_raw(5.0, "pores", 3, config)

    assert empirical != pytest.approx(gaussian)
    assert 0.0 <= empirical <= 100.0


def test_score_falls_back_to_default_bucket_for_unseen_fitzpatrick():
    config = {
        "reference": {"erythema": {"default": {"mean": 0.0, "std": 1.0}}},
        "scoring": {"clip_z": 4.0},
    }
    assert score_from_raw(0.0, "erythema", 6, config) == pytest.approx(50.0)


# --- runtime application of a fitted model ---------------------------------


def test_flatten_features_uses_the_csv_column_names():
    flat = flatten_features({"pores": {"glcm_contrast": 0.8}})
    assert flat == {"f_pores_glcm_contrast": 0.8}


def test_predict_instrument_skips_rois_the_model_was_not_fitted_on():
    model = {
        "features": ["f_pores_glcm_contrast"],
        "feature_mean": [1.0],
        "feature_std": [1.0],
        "coef": [10.0],
        "intercept": 50.0,
        "applies_to_rois": ["forehead"],
    }
    roi_features = {
        "forehead": {"pores": {"glcm_contrast": 2.0}},   # -> 50 + 10*1 = 60
        "nose": {"pores": {"glcm_contrast": 100.0}},     # must be ignored
    }
    assert predict_instrument(model, roi_features) == pytest.approx(60.0)


def test_predict_instrument_weights_by_roi_area():
    model = {
        "features": ["f_pores_glcm_contrast"],
        "feature_mean": [0.0],
        "feature_std": [1.0],
        "coef": [1.0],
        "intercept": 0.0,
        "applies_to_rois": ["forehead", "chin"],
    }
    roi_features = {
        "forehead": {"pores": {"glcm_contrast": 10.0}},
        "chin": {"pores": {"glcm_contrast": 20.0}},
    }
    weights = {"forehead": 3.0, "chin": 1.0}
    assert predict_instrument(model, roi_features, weights) == pytest.approx(12.5)


def test_predict_instrument_returns_none_without_usable_rois():
    model = {
        "features": ["f_pores_glcm_contrast"],
        "feature_mean": [0.0],
        "feature_std": [1.0],
        "coef": [1.0],
        "intercept": 0.0,
        "applies_to_rois": ["forehead"],
    }
    assert predict_instrument(model, {"nose": {"pores": {"glcm_contrast": 1.0}}}) is None


# --- profile merging -------------------------------------------------------


def test_merge_profile_overlays_anchors_and_models():
    config = {
        "composite": {
            "pores": {
                "weights": {"glcm_contrast": 1.0},
                "anchors": {"glcm_contrast": {"mean": 50.0, "std": 30.0}},
            }
        }
    }
    profile = {
        "composite_anchors": {"pores": {"glcm_contrast": {"mean": 0.8, "std": 0.2}}},
        "reference": {"pores": {"default": {"mean": 0.0, "std": 1.0}}},
        "supervised": {"pores": {"target": "instrument_pore_count"}},
        "provenance": {"profile": "test_cohort"},
    }
    merged = merge_profile(config, profile)

    assert merged["composite"]["pores"]["anchors"]["glcm_contrast"]["mean"] == 0.8
    assert merged["supervised"]["pores"]["target"] == "instrument_pore_count"
    assert merged["provenance"]["profile"] == "test_cohort"


def test_merge_profile_ignores_metrics_absent_from_the_base_config():
    config = {"composite": {}}
    merged = merge_profile(config, {"composite_anchors": {"nonexistent": {}}})
    assert merged["composite"] == {}


# --- corpus index ----------------------------------------------------------


def _write_corpus(root, subject="0007", device="phone", split="train", angle="F"):
    """Lay out a minimal AI-Hub-shaped corpus for one subject and angle."""
    import json

    from skin_metrics.calibrate.aihub import DEVICE_CODES, DEVICE_DIRS, SPLIT_DIRS

    top, src_sub, label_sub = SPLIT_DIRS[split]
    dev_dir = DEVICE_DIRS[device]
    stem = f"{subject}_{DEVICE_CODES[device]}_{angle}"

    src = root / top / src_sub / dev_dir / subject
    src.mkdir(parents=True, exist_ok=True)
    (src / f"{stem}.jpg").write_bytes(b"not-a-real-jpeg")

    labels = root / top / label_sub / dev_dir / subject
    labels.mkdir(parents=True, exist_ok=True)
    info = {"filename": f"{stem}.jpg", "id": subject, "gender": "F", "age": 41,
            "date": "2023-08-02", "skin_type": 3, "sensitive": 0}
    docs = {
        0: {"annotations": {"acne": [{"name": "papule", "points": [1, 2]}]},
            "equipment": {"pigmentation_count": 127}},
        1: {"annotations": {"forehead_pigmentation": 2},
            "equipment": {"forehead_moisture": 70.5, "forehead_elasticity_R2": 0.61}},
        5: {"annotations": {"l_cheek_pigmentation": 3, "l_cheek_pore": 2},
            "equipment": {"l_cheek_moisture": 63.0, "l_cheek_pore": 631.0}},
        6: {"annotations": {"r_cheek_pigmentation": 3},
            "equipment": {"r_cheek_moisture": 64.0}},
        8: {"annotations": {"chin_sagging": 1}, "equipment": {"chin_moisture": 66.0}},
    }
    for part, body in docs.items():
        doc = {"info": info,
               "images": {"device": 2, "width": 100, "height": 200, "angle": 0,
                          "facepart": part, "bbox": [0, 0, 100, 200]},
               **body}
        (labels / f"{stem}_{part:02d}.json").write_text(
            json.dumps(doc, ensure_ascii=False), encoding="utf-8"
        )


def test_index_dataset_joins_labels_to_images(tmp_path):
    from skin_metrics.calibrate.aihub import index_dataset

    _write_corpus(tmp_path)
    samples = index_dataset(tmp_path, splits=("train",), devices=("phone",))
    assert len(samples) == 1

    s = samples[0]
    assert s.subject == "0007" and s.age == 41 and s.skin_type == 3
    assert s.pigmentation_count == 127.0
    assert s.acne_count == 1
    assert set(s.roi_labels) == {"forehead", "left_cheek", "right_cheek", "chin"}
    # Dataset `l_*` is image-left, which is where our `left_cheek` ROI sits.
    assert s.roi_labels["left_cheek"].moisture == 63.0
    assert s.roi_labels["left_cheek"].pigmentation_grade == 3
    assert s.roi_labels["left_cheek"].pore_count == 631.0
    assert s.roi_labels["chin"].pigmentation_grade is None  # never graded there


def test_index_dataset_walks_every_requested_angle(tmp_path):
    from skin_metrics.calibrate.aihub import index_dataset

    for angle in ("F", "L", "R"):
        _write_corpus(tmp_path, angle=angle)
    samples = index_dataset(tmp_path, splits=("train",), devices=("phone",),
                            angles=("F", "L", "R"))
    assert [s.angle for s in samples] == ["F", "L", "R"]
    # Same subject, same labels -- only the viewpoint differs.
    assert {s.roi_labels["left_cheek"].moisture for s in samples} == {63.0}


def test_sample_keys_stay_unique_across_angles(tmp_path):
    from skin_metrics.calibrate.aihub import index_dataset

    for angle in ("F", "L", "R"):
        _write_corpus(tmp_path, angle=angle)
    samples = index_dataset(tmp_path, splits=("train",), devices=("phone",),
                            angles=("F", "L", "R"))
    keys = [s.key for s in samples]
    assert len(set(keys)) == 3
    # Frontal keys are unsuffixed so pre-multi-angle CSVs still resume.
    assert keys[0] == "train/phone/0007"


def test_index_dataset_skips_angles_a_device_did_not_shoot(tmp_path):
    from skin_metrics.calibrate.aihub import index_dataset

    _write_corpus(tmp_path, angle="F")
    samples = index_dataset(tmp_path, splits=("train",), devices=("phone",),
                            angles=("F", "L30"))
    assert [s.angle for s in samples] == ["F"]


def test_index_dataset_skips_subjects_without_images(tmp_path):
    from skin_metrics.calibrate.aihub import index_dataset

    _write_corpus(tmp_path)
    for jpg in tmp_path.rglob("*.jpg"):
        jpg.unlink()
    assert index_dataset(tmp_path, splits=("train",), devices=("phone",)) == []


def test_resolve_data_root_rejects_a_wrong_directory(tmp_path):
    from skin_metrics.calibrate.aihub import resolve_data_root

    with pytest.raises(FileNotFoundError, match="Training"):
        resolve_data_root(tmp_path)


# --- column-name contract --------------------------------------------------


def test_feature_columns_match_the_pipeline(synthetic_image, synthetic_landmarks):
    """The hard-coded CSV layout must equal what the pipeline really emits."""
    from skin_metrics.calibrate.extract import FEATURE_COLUMNS
    from skin_metrics.pipeline import extract_raw

    raw = extract_raw(synthetic_image, landmarks=synthetic_landmarks)
    produced = set(flatten_features(next(iter(raw.roi_features.values()))))
    assert produced == set(FEATURE_COLUMNS)


# --- end-to-end: a profile actually drives the report ----------------------


def test_a_supervised_profile_reaches_the_report(synthetic_image, synthetic_landmarks):
    """With a model in the config, the report must carry its prediction."""
    from skin_metrics.config import load_config
    from skin_metrics.pipeline import analyze

    config = load_config(use_profile=False)
    # A trivial model: predicted grade = 2.5 regardless of the features.
    config["supervised"] = {
        "pigmentation": {
            "features": ["f_pigmentation_ita"],
            "feature_mean": [0.0],
            "feature_std": [1.0],
            "coef": [0.0],
            "intercept": 2.5,
            "applies_to_rois": ["forehead", "left_cheek", "right_cheek"],
            "target": "expert_pigmentation_grade",
            "target_units": "grade 0-5",
            "score_sign": 1.0,
        }
    }
    # Replace the whole metric entry, not just "default": per-Fitzpatrick
    # buckets take precedence, and a real fitted profile likewise replaces
    # config["reference"] wholesale rather than merging into it.
    config["reference"]["pigmentation"] = {
        "default": {
            "mean": 2.5,
            "std": 1.0,
            "quantiles": list(np.linspace(0.0, 5.0, 101)),
        }
    }
    config["provenance"] = {"profile": "unit_test_profile"}

    report = analyze(synthetic_image, landmarks=synthetic_landmarks, config=config)

    assert report.pigmentation.calibrated is True
    assert report.pigmentation.predicted_value == pytest.approx(2.5)
    assert report.pigmentation.predicted_units == "grade 0-5"
    assert report.pigmentation.score == pytest.approx(50.0, abs=1.0)
    assert report.calibration_profile == "unit_test_profile"

    # Metrics without a model stay on the composite path.
    assert report.pores.calibrated is False
    assert report.pores.predicted_value is None


def test_an_inapplicable_model_degrades_to_the_composite(
    synthetic_image, synthetic_landmarks
):
    """A model whose ROIs are all missing must warn, not crash or silently lie."""
    from skin_metrics.config import load_config
    from skin_metrics.pipeline import analyze

    config = load_config(use_profile=False)
    config["supervised"] = {
        "pigmentation": {
            "features": ["f_pigmentation_ita"],
            "feature_mean": [0.0],
            "feature_std": [1.0],
            "coef": [1.0],
            "intercept": 0.0,
            "applies_to_rois": ["a_roi_that_does_not_exist"],
            "target": "expert_pigmentation_grade",
            "target_units": "grade 0-5",
            "score_sign": 1.0,
        }
    }

    report = analyze(synthetic_image, landmarks=synthetic_landmarks, config=config)

    assert report.pigmentation.calibrated is False
    assert report.pigmentation.predicted_value is None
    assert any("could not be applied" in w for w in report.warnings)
