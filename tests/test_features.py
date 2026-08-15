"""Unit tests for the physics feature modules."""

from __future__ import annotations

import numpy as np
import pytest

from skin_metrics.features import erythema as ery
from skin_metrics.features import pores as por
from skin_metrics.features import pigmentation as pig


# --- pigmentation ----------------------------------------------------------
def test_ita_known_value():
    # L=70, b=20 -> arctan(20/20)=45deg
    assert pig.ita(70.0, 20.0) == pytest.approx(45.0, abs=1e-6)


def test_ita_zero_b_guarded():
    val = pig.ita(70.0, 0.0)
    assert np.isfinite(val)
    assert abs(val) == pytest.approx(90.0, abs=1e-3)


def test_melanin_index_guards_nonpositive():
    assert np.isfinite(pig.melanin_index(0.0))
    assert pig.melanin_index(0.0) > 0
    # Lower reflectance -> higher melanin index.
    assert pig.melanin_index(0.2) > pig.melanin_index(0.8)


def test_spot_detection_finds_dark_blob():
    L = np.full((80, 80), 70.0)
    L[38:44, 38:44] = 40.0  # a dark spot
    res = pig.spot_detection(L, sigma=8.0, contrast_thresh=5.0, min_area=4)
    assert res["spot_count"] >= 1
    assert res["spot_area_ratio"] > 0
    assert res["mean_contrast"] > 0


def test_spot_detection_empty_mask():
    L = np.full((20, 20), 70.0)
    res = pig.spot_detection(L, mask=np.zeros((20, 20), dtype=bool))
    assert res == {"spot_area_ratio": 0.0, "spot_count": 0.0, "mean_contrast": 0.0}


def test_evenness_constant_is_zero():
    assert pig.evenness(np.full((10, 10), 50.0)) == 0.0


def test_estimate_fitzpatrick_monotonic():
    boundaries = [55.0, 41.0, 28.0, 10.0, -30.0]
    assert pig.estimate_fitzpatrick(70.0, boundaries) == 1
    assert pig.estimate_fitzpatrick(-40.0, boundaries) == 6
    assert pig.estimate_fitzpatrick(30.0, boundaries) == 3


# --- erythema --------------------------------------------------------------
def test_erythema_index_sign():
    # Redder skin: red reflectance high, green low -> positive EI.
    assert ery.erythema_index(0.8, 0.4) > 0
    assert ery.erythema_index(0.4, 0.8) < 0


def test_erythema_index_guards_zero():
    assert np.isfinite(ery.erythema_index(0.0, 0.0))


def test_mean_a_star_stats():
    lab = np.zeros((10, 10, 3))
    lab[..., 1] = np.linspace(0, 20, 100).reshape(10, 10)
    stats = ery.mean_a_star(lab)
    assert stats["mean_a_star"] == pytest.approx(10.0, abs=0.5)
    assert stats["p90_a_star"] > stats["mean_a_star"]


def test_mean_a_star_empty():
    lab = np.zeros((5, 5, 3))
    stats = ery.mean_a_star(lab, mask=np.zeros((5, 5), dtype=bool))
    assert stats == {"mean_a_star": 0.0, "p90_a_star": 0.0}


def test_hemoglobin_map_runs_on_varied_skin():
    rng = np.random.default_rng(3)
    base = np.array([0.55, 0.38, 0.34])
    img = base + rng.normal(0, 0.05, size=(40, 40, 3))
    img = np.clip(img, 0.01, 0.99)
    res = ery.hemoglobin_map(img)
    assert res["separation_ok"] is True
    assert res["hemoglobin_map"].shape == (40, 40)
    assert np.isfinite(res["hemoglobin_score"])


def test_hemoglobin_map_degenerate_constant():
    img = np.full((40, 40, 3), 0.5)
    res = ery.hemoglobin_map(img)
    assert res["separation_ok"] is False
    assert res["hemoglobin_score"] == 0.0


def test_hemoglobin_map_too_few_pixels():
    img = np.full((5, 5, 3), 0.5)
    res = ery.hemoglobin_map(img)
    assert res["separation_ok"] is False


# --- pores / surface texture -----------------------------------------------
def test_specular_ratio_detects_bright_patch():
    img = np.full((30, 30, 3), 0.3, dtype=np.float64)  # matte
    img[0:10, 0:10] = 0.99  # bright, low-saturation -> specular
    ratio = por.specular_ratio(img)
    assert 0.0 < ratio < 1.0


def test_specular_ratio_empty_mask():
    img = np.full((10, 10, 3), 0.99)
    assert por.specular_ratio(img, mask=np.zeros((10, 10), dtype=bool)) == 0.0


def test_texture_features_ranges():
    rng = np.random.default_rng(5)
    gray = rng.uniform(0, 1, size=(40, 40))
    feats = por.texture_stats(gray)
    for key in ("glcm_contrast", "glcm_correlation", "glcm_energy", "lbp_uniformity"):
        assert key in feats and np.isfinite(feats[key])
    assert feats["glcm_energy"] >= 0.0


def test_texture_features_tiny_input():
    feats = por.texture_stats(np.zeros((2, 2)))
    assert feats["glcm_contrast"] == 0.0


def test_scaling_index_higher_for_rough():
    smooth = np.full((40, 40), 0.5)
    rng = np.random.default_rng(9)
    rough = rng.uniform(0, 1, size=(40, 40))
    assert por.scaling_index(rough) > por.scaling_index(smooth)


def test_micro_wrinkle_density_range():
    rng = np.random.default_rng(11)
    gray = rng.uniform(0, 1, size=(40, 40))
    d = por.micro_wrinkle_density(gray)
    assert 0.0 <= d <= 1.0


def test_micro_wrinkle_tiny_input():
    assert por.micro_wrinkle_density(np.zeros((3, 3))) == 0.0
