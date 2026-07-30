"""Unit tests for ROI geometry and artifact masking (no MediaPipe needed)."""

from __future__ import annotations

import numpy as np

from skin_metrics.calibration import color as cal
from skin_metrics.detection import face


def test_polygon_mask_nonempty(synthetic_landmarks, image_size):
    shape = image_size
    for name, idxs in face.ROI_LANDMARKS.items():
        mask = face._polygon_mask(synthetic_landmarks, idxs, shape)
        assert mask.dtype == bool
        assert mask.sum() > 0, f"empty ROI mask for {name}"


def test_exclusion_mask_covers_regions(synthetic_landmarks, image_size):
    excl = face.exclusion_mask(synthetic_landmarks, image_size)
    assert excl.sum() > 0
    assert excl.shape == image_size


def test_mask_artifacts_removes_glare(synthetic_image, synthetic_landmarks, image_size):
    linear = cal.linearize_srgb(synthetic_image)
    lab = cal.rgb_to_lab(linear)
    region = face._polygon_mask(
        synthetic_landmarks, face.ROI_LANDMARKS["left_cheek"], image_size
    )
    # Inject a glare patch inside the region.
    linear_glare = linear.copy()
    ys, xs = np.where(region)
    cy, cx = int(ys.mean()), int(xs.mean())
    linear_glare[cy - 3 : cy + 3, cx - 3 : cx + 3] = 0.99
    lab_glare = cal.rgb_to_lab(linear_glare)
    valid = face.mask_artifacts(linear_glare, lab_glare, region)
    assert valid.sum() < region.sum()  # some pixels removed
    assert valid[valid].size > 0


def test_extract_rois_returns_valid(synthetic_image, synthetic_landmarks, image_size):
    linear = cal.linearize_srgb(synthetic_image)
    lab = cal.rgb_to_lab(linear)
    rois = face.extract_rois(linear, lab, synthetic_landmarks, min_valid_ratio=0.5)
    valid = {k: v for k, v in rois.items() if v is not None}
    # Cheeks and nose should survive; require at least 3 valid ROIs.
    assert len(valid) >= 3, f"only got {list(valid)}"
    for roi in valid.values():
        assert 0.0 <= roi.valid_ratio <= 1.0
        assert roi.valid_mask.sum() > 0


def test_extract_rois_drops_low_ratio(synthetic_image, synthetic_landmarks):
    linear = cal.linearize_srgb(synthetic_image)
    lab = cal.rgb_to_lab(linear)
    # Impossibly strict ratio -> everything dropped.
    rois = face.extract_rois(linear, lab, synthetic_landmarks, min_valid_ratio=1.01)
    assert all(v is None for v in rois.values())
