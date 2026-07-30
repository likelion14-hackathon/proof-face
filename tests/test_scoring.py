"""Unit tests for normalization/scoring and the output schema."""

from __future__ import annotations

import pytest

from skin_metrics.config import load_config
from skin_metrics.scoring.normalize import (
    compare,
    composite_raw,
    score_from_raw,
    score_metric,
)
from skin_metrics.scoring.schema import MetricScore, SkinReport


@pytest.fixture
def config():
    return load_config()


def test_composite_raw_uses_available_weights(config):
    raw = composite_raw("pigmentation", {"melanin_index": 40.0}, config)
    assert isinstance(raw, float)
    # At the anchor mean, z=0 -> composite 0.
    assert raw == pytest.approx(0.0, abs=1e-6)


def test_composite_raw_missing_all_features(config):
    assert composite_raw("erythema", {}, config) == 0.0


def test_score_from_raw_bounds_and_monotonic(config):
    lo = score_from_raw(-3.0, "pigmentation", 3, config)
    mid = score_from_raw(0.0, "pigmentation", 3, config)
    hi = score_from_raw(3.0, "pigmentation", 3, config)
    assert 0.0 <= lo < mid < hi <= 100.0
    assert mid == pytest.approx(50.0, abs=1.0)


def test_score_metric_fitzpatrick_reference_differs(config):
    feats = {"melanin_index": 60.0, "spot_area_ratio": 0.05, "evenness": 6.0, "ita_inv": 10.0}
    s1 = score_metric("pigmentation", feats, 1, config)
    s6 = score_metric("pigmentation", feats, 6, config)
    # Same raw features score differently under type-1 vs type-6 references.
    assert s1 != s6


def test_score_metric_unknown_fitz_falls_back(config):
    # Type present in reference; ensure no KeyError for a normal call.
    s = score_metric("hydration", {"scaling_index": 0.1}, 4, config)
    assert 0.0 <= s <= 100.0


def test_compare_deltas_and_significance():
    baseline = {"pigmentation": 50.0, "erythema": 40.0, "hydration": 30.0}
    current = {"pigmentation": 58.0, "erythema": 41.0, "hydration": 30.0}
    res = compare(current, baseline, min_delta=5.0)
    assert res["pigmentation"]["delta"] == pytest.approx(8.0)
    assert res["pigmentation"]["significant"] is True
    assert res["pigmentation"]["direction"] == "up"
    assert res["erythema"]["significant"] is False
    assert res["hydration"]["direction"] == "flat"


def test_schema_validates_and_dumps():
    ms = MetricScore(score=50.0, confidence=0.8, raw_features={"x": 1.0})
    report = SkinReport(
        pigmentation=ms,
        erythema=ms,
        hydration=MetricScore(score=20.0, confidence=0.5, is_estimate=True),
        calibration_status="grayworld",
        fitzpatrick_estimate=3,
        warnings=["test"],
    )
    payload = report.model_dump()
    assert payload["hydration"]["is_estimate"] is True
    assert "disclaimer" in payload
    assert payload["calibration_status"] == "grayworld"


def test_schema_rejects_out_of_range():
    with pytest.raises(Exception):
        MetricScore(score=150.0, confidence=0.5)
