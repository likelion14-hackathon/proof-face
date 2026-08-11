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
    """The config as shipped, calibration profile included."""
    return load_config()


@pytest.fixture
def fixed_config():
    """A config with known anchors and references.

    Tests that assert exact arithmetic use this rather than the shipped
    config: the shipped anchors and reference grids are refitted from a cohort
    whenever `skin-metrics calibrate fit` runs, so pinning numbers to them
    would make an ordinary recalibration look like a regression.
    """
    return {
        "composite": {
            "pigmentation": {
                "weights": {"melanin_index": 0.6, "evenness": 0.4},
                "anchors": {
                    "melanin_index": {"mean": 40.0, "std": 10.0},
                    "evenness": {"mean": 4.0, "std": 2.0},
                },
            }
        },
        "reference": {
            "pigmentation": {
                "default": {"mean": 0.0, "std": 1.0},
                "1": {"mean": -1.0, "std": 1.0},
                "6": {"mean": 1.0, "std": 1.0},
            }
        },
        "scoring": {"clip_z": 4.0},
    }


def test_composite_raw_uses_available_weights(fixed_config):
    raw = composite_raw("pigmentation", {"melanin_index": 40.0}, fixed_config)
    assert isinstance(raw, float)
    # At the anchor mean, z=0 -> composite 0, with `evenness` absent its weight
    # is dropped and the result renormalised over what is available.
    assert raw == pytest.approx(0.0, abs=1e-6)


def test_composite_raw_missing_all_features(config):
    assert composite_raw("erythema", {}, config) == 0.0


def test_score_from_raw_bounds_and_monotonic(fixed_config):
    lo = score_from_raw(-3.0, "pigmentation", 3, fixed_config)
    mid = score_from_raw(0.0, "pigmentation", 3, fixed_config)
    hi = score_from_raw(3.0, "pigmentation", 3, fixed_config)
    assert 0.0 <= lo < mid < hi <= 100.0
    assert mid == pytest.approx(50.0, abs=1.0)


def test_shipped_config_scores_stay_in_range(config):
    """Whatever the current calibration is, scores must stay bounded."""
    for metric in ("pigmentation", "erythema", "hydration"):
        for raw in (-1e6, -1.0, 0.0, 1.0, 1e6):
            assert 0.0 <= score_from_raw(raw, metric, 3, config) <= 100.0


def test_score_metric_fitzpatrick_reference_differs(fixed_config):
    feats = {"melanin_index": 60.0, "evenness": 6.0}
    s1 = score_metric("pigmentation", feats, 1, fixed_config)
    s6 = score_metric("pigmentation", feats, 6, fixed_config)
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
