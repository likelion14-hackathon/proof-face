"""Shared pytest fixtures: synthetic images and synthetic FaceMesh landmarks.

These let the whole Phase 1 pipeline be exercised without MediaPipe (the
``detection`` extra). Landmarks are placed at controlled cluster positions so
the ROI polygons and artifact-exclusion regions land where we expect.
"""

from __future__ import annotations

import numpy as np
import pytest

from skin_metrics.detection.face import EXCLUSION_LANDMARKS, ROI_LANDMARKS


def _place(pts: np.ndarray, indices, center, spread, rng) -> None:
    cx, cy = center
    sx, sy = spread
    for i in indices:
        pts[i, 0] = cx + rng.uniform(-sx, sx)
        pts[i, 1] = cy + rng.uniform(-sy, sy)


@pytest.fixture
def image_size() -> tuple[int, int]:
    return (500, 500)  # (H, W)


@pytest.fixture
def synthetic_landmarks(image_size) -> np.ndarray:
    """A (468, 2) landmark array with realistic cluster geometry."""
    h, w = image_size
    rng = np.random.default_rng(42)
    pts = np.full((468, 2), [w / 2.0, h / 2.0], dtype=np.float32)

    # Exclusions first; ROIs afterwards so shared indices resolve to the ROI.
    excl_centers = {
        "left_eye": (0.36 * w, 0.36 * h),
        "right_eye": (0.64 * w, 0.36 * h),
        "left_brow": (0.36 * w, 0.30 * h),
        "right_brow": (0.64 * w, 0.30 * h),
        "lips": (0.50 * w, 0.72 * h),
    }
    for name, idxs in EXCLUSION_LANDMARKS.items():
        _place(pts, idxs, excl_centers[name], (0.05 * w, 0.02 * h), rng)

    roi_centers = {
        "forehead": (0.50 * w, 0.17 * h),
        "left_cheek": (0.30 * w, 0.55 * h),
        "right_cheek": (0.70 * w, 0.55 * h),
        "nose": (0.50 * w, 0.47 * h),
        "chin": (0.50 * w, 0.83 * h),
    }
    roi_spreads = {
        "forehead": (0.17 * w, 0.06 * h),
        "left_cheek": (0.09 * w, 0.11 * h),
        "right_cheek": (0.09 * w, 0.11 * h),
        "nose": (0.05 * w, 0.11 * h),
        "chin": (0.11 * w, 0.05 * h),
    }
    for name, idxs in ROI_LANDMARKS.items():
        _place(pts, idxs, roi_centers[name], roi_spreads[name], rng)

    return pts


@pytest.fixture
def synthetic_image(image_size) -> np.ndarray:
    """A skin-toned sRGB uint8 face image with spots, redness, and texture."""
    h, w = image_size
    rng = np.random.default_rng(7)
    img = np.zeros((h, w, 3), dtype=np.float64)
    # Base skin tone (sRGB): warm, R > G > B.
    img[..., 0] = 0.80
    img[..., 1] = 0.62
    img[..., 2] = 0.55

    # Fine texture noise.
    img += rng.normal(0, 0.015, size=img.shape)

    # A few dark spots on the left cheek.
    for _ in range(6):
        cy = int(rng.uniform(0.45, 0.65) * h)
        cx = int(rng.uniform(0.24, 0.36) * w)
        yy, xx = np.ogrid[:h, :w]
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 < (6 ** 2)
        img[blob] *= 0.65

    # Mild redness gradient on the right cheek.
    yy, xx = np.ogrid[:h, :w]
    right = (xx > 0.6 * w) & (yy > 0.45 * h) & (yy < 0.65 * h)
    img[right, 0] = np.clip(img[right, 0] + 0.06, 0, 1)
    img[right, 1] = np.clip(img[right, 1] - 0.03, 0, 1)

    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)
