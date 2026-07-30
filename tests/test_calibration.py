"""Unit tests for skin_metrics.calibration.color."""

from __future__ import annotations

import numpy as np
import pytest

from skin_metrics.calibration import color as cal


def test_linearize_srgb_endpoints_and_midpoint():
    img = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    lin = cal.linearize_srgb(img)
    assert lin[0, 0, 0] == pytest.approx(0.0, abs=1e-6)
    assert lin[0, 0, 2] == pytest.approx(1.0, abs=1e-6)
    # sRGB 0.5 -> ~0.214 linear
    assert lin[0, 0, 1] == pytest.approx(0.2140, abs=1e-3)


def test_linearize_roundtrip():
    rng = np.random.default_rng(0)
    srgb = rng.uniform(0, 1, size=(4, 4, 3)).astype(np.float32)
    back = cal.encode_srgb(cal.linearize_srgb(srgb))
    assert np.allclose(srgb, back, atol=1e-4)


def test_uint8_input_scaled():
    img = np.full((2, 2, 3), 128, dtype=np.uint8)
    lin = cal.linearize_srgb(img)
    assert lin.max() <= 1.0 and lin.min() >= 0.0


def test_grayworld_neutralizes_channels():
    img = np.zeros((10, 10, 3), dtype=np.float32)
    img[..., 0] = 0.6
    img[..., 1] = 0.3
    img[..., 2] = 0.3
    balanced, success = cal.white_balance_grayworld(img)
    assert success is False  # grayworld is a weak fallback
    means = balanced.reshape(-1, 3).mean(axis=0)
    assert np.allclose(means, means[0], atol=1e-3)


def test_grayworld_zero_channel_guarded():
    img = np.zeros((5, 5, 3), dtype=np.float32)  # all-black: no divide-by-zero
    balanced, _ = cal.white_balance_grayworld(img)
    assert np.all(np.isfinite(balanced))


def test_reference_white_balance_success_and_reject():
    img = np.zeros((20, 20, 3), dtype=np.float32)
    img[..., 0] = 0.7
    img[..., 1] = 0.5
    img[..., 2] = 0.5
    # A bright neutral-ish patch in the corner.
    balanced, ok = cal.white_balance_from_reference(img, (0, 0, 6, 6))
    assert ok is True
    patch_means = balanced[0:6, 0:6].reshape(-1, 3).mean(axis=0)
    assert np.allclose(patch_means, patch_means[0], atol=1e-3)

    # Too-dark reference is rejected.
    dark = np.full((20, 20, 3), 0.01, dtype=np.float32)
    _, ok2 = cal.white_balance_from_reference(dark, (0, 0, 6, 6))
    assert ok2 is False

    # Out-of-bounds bbox -> failure, not a crash.
    _, ok3 = cal.white_balance_from_reference(img, (100, 100, 6, 6))
    assert ok3 is False


def test_estimate_ccm_recovers_matrix():
    rng = np.random.default_rng(1)
    ref = rng.uniform(0.1, 0.9, size=(24, 3))
    true_m = np.array([[1.05, -0.02, 0.0], [0.01, 0.98, 0.02], [0.0, 0.03, 1.02]])
    detected = ref @ np.linalg.inv(true_m)  # so detected @ true_m == ref
    ccm, residual = cal.estimate_ccm(detected, ref)
    assert residual < 1e-6
    assert np.allclose(ccm, true_m, atol=1e-4)


def test_estimate_ccm_requires_three_patches():
    with pytest.raises(ValueError):
        cal.estimate_ccm(np.zeros((2, 3)), np.zeros((2, 3)))


def test_apply_ccm_shape_and_clip():
    img = np.full((3, 3, 3), 0.5, dtype=np.float32)
    out = cal.apply_ccm(img, np.eye(3))
    assert out.shape == img.shape
    assert np.allclose(out, 0.5, atol=1e-6)


def test_rgb_to_lab_white_is_l100():
    white = np.ones((2, 2, 3), dtype=np.float32)
    lab = cal.rgb_to_lab(white, assume_linear=True)
    assert lab[..., 0].mean() == pytest.approx(100.0, abs=0.5)
    assert abs(lab[..., 1].mean()) < 1.0
    assert abs(lab[..., 2].mean()) < 1.0


def test_calibrate_image_status():
    img = np.full((20, 20, 3), 0.5, dtype=np.float32)
    # No reference -> grayworld, not reliable.
    res = cal.calibrate_image(img)
    assert res.status == "grayworld"
    assert res.success is False

    # Valid reference -> reference, reliable.
    res2 = cal.calibrate_image(img, ref_bbox=(0, 0, 6, 6))
    assert res2.status == "reference"
    assert res2.success is True

    # CCM without reference -> grayworld status but reliable (success True).
    res3 = cal.calibrate_image(img, ccm=np.eye(3))
    assert res3.ccm_applied is True
    assert res3.success is True
