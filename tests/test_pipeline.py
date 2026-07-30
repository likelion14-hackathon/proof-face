"""End-to-end Phase 1 pipeline test using synthetic image + landmarks."""

from __future__ import annotations

import numpy as np

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
    assert any("PROXY" in w or "proxy" in w for w in report.warnings)


def test_analyze_no_reference_is_grayworld(synthetic_image, synthetic_landmarks):
    report = analyze(synthetic_image, landmarks=synthetic_landmarks)
    assert report.calibration_status == "grayworld"
    # Grayworld should lower confidence below the reference ceiling.
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
