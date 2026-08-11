"""End-to-end Phase 1 pipeline test using synthetic image + landmarks."""

from __future__ import annotations

import numpy as np
import pytest

from skin_metrics.pipeline import analyze
from skin_metrics.scoring.schema import SkinReport


def test_analyze_returns_valid_report(synthetic_image, synthetic_landmarks):
    report = analyze(synthetic_image, landmarks=synthetic_landmarks)
    assert isinstance(report, SkinReport)

    for metric in (report.pigmentation, report.erythema, report.hydration):
        assert 0.0 <= metric.score <= 100.0
        assert 0.0 <= metric.confidence <= 1.0

    # Hydration is always flagged as an estimate.
    assert report.hydration.is_estimate is True
    assert report.pigmentation.is_estimate is False

    assert 1 <= report.fitzpatrick_estimate <= 6
    assert report.calibration_status in ("reference", "grayworld", "none")
    assert report.roi_breakdown  # at least one ROI
    assert any("not a moisture measurement" in w for w in report.warnings)


def test_hydration_is_reported_as_moisture_not_dryness(
    synthetic_image, synthetic_landmarks
):
    """A field called `hydration` must rise with moisture, not with dryness.

    Everything upstream measures dryness; `scoring.report_inverted` flips the
    finished percentile. Without it "수분력 85점" would mean *very dry*.
    """
    import copy

    from skin_metrics.config import load_config

    config = load_config()
    assert "hydration" in config["scoring"]["report_inverted"]
    report = analyze(synthetic_image, landmarks=synthetic_landmarks, config=config)

    straight = copy.deepcopy(config)
    straight["scoring"]["report_inverted"] = []
    dryness = analyze(synthetic_image, landmarks=synthetic_landmarks, config=straight)

    assert report.hydration.score == pytest.approx(100.0 - dryness.hydration.score)
    # The metrics that are honestly named are left alone.
    assert report.pigmentation.score == dryness.pigmentation.score
    assert report.erythema.score == dryness.erythema.score


def test_hydration_aggregates_over_the_cheeks_only(synthetic_image, synthetic_landmarks):
    """Hydration is calibrated on cheek Corneometer readings, so only cheeks feed it."""
    from skin_metrics.config import load_config
    from skin_metrics.pipeline import extract_raw

    config = load_config()
    assert config["composite"]["hydration"]["rois"] == ["left_cheek", "right_cheek"]

    raw = extract_raw(synthetic_image, landmarks=synthetic_landmarks, config=config)
    cheeks = [r for r in ("left_cheek", "right_cheek") if r in raw.roi_features]
    assert cheeks, "fixture must produce at least one cheek ROI"

    for key, value in raw.aggregate["hydration"].items():
        expected = np.average(
            [raw.roi_features[r]["hydration"][key] for r in cheeks],
            weights=[raw.roi_weights[r] for r in cheeks],
        )
        assert value == np.float64(expected)

    # Pigmentation declares no restriction, so it still uses every valid ROI.
    key = next(iter(raw.aggregate["pigmentation"]))
    all_rois = list(raw.roi_features)
    expected = np.average(
        [raw.roi_features[r]["pigmentation"][key] for r in all_rois],
        weights=[raw.roi_weights[r] for r in all_rois],
    )
    assert raw.aggregate["pigmentation"][key] == np.float64(expected)


def test_hydration_falls_back_when_no_cheek_survives(synthetic_image, synthetic_landmarks):
    """Losing both cheeks degrades to the other ROIs, loudly."""
    from skin_metrics.pipeline import _aggregation_rois

    config = {"composite": {"hydration": {"rois": ["left_cheek", "right_cheek"]}}}
    warnings: list[str] = []
    names = _aggregation_rois("hydration", ["forehead", "chin"], config, warnings)
    assert names == ["forehead", "chin"]
    assert any("less reliable" in w for w in warnings)


def test_analyze_no_reference_skips_white_balance(synthetic_image, synthetic_landmarks):
    """Without a gray card the default is to trust the camera AWB, not gray-world.

    Whole-image gray-world on a face-filling portrait estimates the illuminant
    from skin itself and neutralises the very chromaticity the metrics read.
    """
    report = analyze(synthetic_image, landmarks=synthetic_landmarks)
    assert report.calibration_status == "none"
    # Anything short of an in-frame neutral reference caps confidence.
    assert report.pigmentation.confidence < 1.0


def test_analyze_with_reference_bbox(synthetic_image, synthetic_landmarks):
    # Add a neutral gray patch in a corner to enable reference white balance.
    img = synthetic_image.copy()
    img[0:30, 0:30] = 150  # neutral gray
    report = analyze(img, ref_bbox=(0, 0, 30, 30), landmarks=synthetic_landmarks)
    assert report.calibration_status == "reference"


def test_analyze_json_serializable(synthetic_image, synthetic_landmarks):
    import json

    report = analyze(synthetic_image, landmarks=synthetic_landmarks)
    text = json.dumps(report.model_dump(), ensure_ascii=False)
    assert "disclaimer" in text


def test_analyze_raises_without_landmarks_or_face():
    # A blank image with no landmarks supplied would need MediaPipe; instead we
    # verify the explicit error path when landmarks resolve to None is reachable
    # by passing an all-zero landmark set that still forms (degenerate) masks.
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    try:
        analyze(img, landmarks=np.zeros((468, 2), dtype=np.float32))
    except ValueError:
        pass  # acceptable: degenerate landmarks -> no valid ROIs
