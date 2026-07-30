"""Phase 2 dataset: ROI crops + Phase-1 physics feature vectors.

Each sample bundles:

- an ROI crop image tensor (``3 x H x W``),
- a Phase-1 **physics feature vector** (the aggregated raw features),
- multitask regression labels (pigmentation / erythema / hydration),
- a Fitzpatrick type (for per-type reporting / reference splitting),
- an illumination-condition bucket (for the domain-adversarial head).

Labels come from a CSV of instrument ground truth (Corneometer for hydration,
Mexameter for melanin/erythema). Because such labels are often unavailable, a
:class:`DummyLabelGenerator` produces a fully self-consistent synthetic dataset
so the entire training loop can be validated immediately, with **no labels and
no image files required**.

Requires the ``dl`` extra (``uv sync --extra dl``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Canonical ordering of the Phase-1 physics features fed to the model.
# Keep in sync with skin_metrics.pipeline aggregation keys.
PHYSICS_FEATURE_NAMES: tuple[str, ...] = (
    "melanin_index",
    "ita",
    "evenness",
    "spot_area_ratio",
    "erythema_index",
    "mean_a_star",
    "hemoglobin",
    "specular_ratio",
    "glcm_contrast",
    "scaling_index",
    "wrinkle_density",
    "lbp_uniformity",
)
PHYSICS_DIM = len(PHYSICS_FEATURE_NAMES)

TARGET_NAMES: tuple[str, ...] = ("pigmentation", "erythema", "hydration")
N_ILLUMINATION_BUCKETS = 4


@dataclass
class Sample:
    """One training sample (framework-agnostic container)."""

    image: np.ndarray            # (H, W, 3) float32 in [0, 1]
    physics: np.ndarray          # (PHYSICS_DIM,) float32
    targets: np.ndarray          # (3,) float32 -> TARGET_NAMES
    fitzpatrick: int             # 1..6
    illumination: int            # 0..N_ILLUMINATION_BUCKETS-1
    meta: dict[str, Any] = field(default_factory=dict)


class DummyLabelGenerator:
    """Generate a synthetic, internally-consistent labelled dataset.

    Targets are deterministic (noisy) functions of the physics vector, so a
    model *can* actually fit them -- this validates the full training loop end
    to end without any real labels.
    """

    def __init__(self, image_size: int = 64, seed: int = 0):
        self.image_size = image_size
        self.rng = np.random.default_rng(seed)
        # Random-but-fixed linear maps from physics -> each target.
        self._w = self.rng.normal(0, 1, size=(PHYSICS_DIM, len(TARGET_NAMES)))

    def _physics_vector(self, fitz: int) -> np.ndarray:
        base = self.rng.normal(0, 1, size=PHYSICS_DIM).astype(np.float32)
        # Nudge a couple of features by skin type so per-Fitzpatrick reports vary.
        base[0] += 0.4 * (fitz - 3)   # melanin_index
        base[1] -= 0.5 * (fitz - 3)   # ita
        return base

    def sample(self) -> Sample:
        fitz = int(self.rng.integers(1, 7))
        illum = int(self.rng.integers(0, N_ILLUMINATION_BUCKETS))
        phys = self._physics_vector(fitz)

        raw = phys @ self._w + self.rng.normal(0, 0.3, size=len(TARGET_NAMES))
        # Squash to a 0-100 target range.
        targets = (50.0 + 12.0 * raw).clip(0, 100).astype(np.float32)

        # A cheap image whose brightness tracks the illumination bucket, so the
        # domain-adversarial head has real signal to (be forced to) ignore.
        s = self.image_size
        img = self.rng.uniform(0, 1, size=(s, s, 3)).astype(np.float32)
        img = np.clip(img * (0.6 + 0.15 * illum), 0, 1).astype(np.float32)

        return Sample(
            image=img,
            physics=phys,
            targets=targets,
            fitzpatrick=fitz,
            illumination=illum,
            meta={"synthetic": True},
        )

    def build(self, n: int) -> list[Sample]:
        """Return a list of ``n`` synthetic samples."""
        return [self.sample() for _ in range(n)]


def load_label_csv(path: str | Path) -> list[dict[str, Any]]:
    """Load an instrument-label CSV into row dicts.

    Expected columns (missing target columns are allowed for ranking mode):
    ``image_path``, physics feature columns (any of :data:`PHYSICS_FEATURE_NAMES`),
    ``pigmentation``, ``erythema``, ``hydration``, ``fitzpatrick``,
    ``illumination`` (optional).

    Parameters
    ----------
    path : str or pathlib.Path
        CSV file path.

    Returns
    -------
    list of dict
        One dict per row (values as strings; the dataset casts them).
    """
    import csv

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(dict(row))
    return rows


def _build_augment(train: bool):
    """Build an albumentations pipeline: color transforms > geometric.

    Color/illumination changes are applied aggressively (color-temperature
    shift, exposure, noise, JPEG) since illumination robustness is the goal;
    geometric changes are kept mild.
    """
    import albumentations as A

    if not train:
        return A.Compose([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    return A.Compose(
        [
            # --- aggressive color / illumination ---
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.3, p=0.7),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.6),
            A.GaussNoise(p=0.4),
            A.ImageCompression(quality_range=(40, 90), p=0.4),
            # --- mild geometric ---
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=0.05, scale=(0.95, 1.05), rotate=(-8, 8), p=0.3),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


class SkinDataset:
    """A ``torch.utils.data.Dataset`` of :class:`Sample` records.

    Parameters
    ----------
    samples : list of Sample
        Pre-built samples (from :class:`DummyLabelGenerator` or a CSV loader).
    train : bool, optional
        If ``True``, applies the training augmentation pipeline.
    """

    def __init__(self, samples: list[Sample], train: bool = True):
        # Import torch lazily so the module is importable without the dl extra
        # for documentation / introspection.
        import torch  # noqa: F401

        self.samples = samples
        self.train = train
        self._augment = _build_augment(train)

    @classmethod
    def dummy(cls, n: int = 128, image_size: int = 64, train: bool = True, seed: int = 0):
        """Construct a dummy dataset (no labels/images needed)."""
        gen = DummyLabelGenerator(image_size=image_size, seed=seed)
        return cls(gen.build(n), train=train)

    @classmethod
    def from_csv(cls, path: str | Path, image_root: str | Path | None = None, train: bool = True):
        """Construct a dataset from an instrument-label CSV.

        Missing physics columns default to ``0``; missing targets default to
        ``NaN`` (valid for ranking mode, which ignores absolute values).
        """
        from skimage.io import imread

        rows = load_label_csv(path)
        root = Path(image_root) if image_root else None
        samples: list[Sample] = []
        for r in rows:
            phys = np.array(
                [float(r.get(name, 0.0) or 0.0) for name in PHYSICS_FEATURE_NAMES],
                dtype=np.float32,
            )
            targets = np.array(
                [float(r[t]) if r.get(t) not in (None, "") else np.nan for t in TARGET_NAMES],
                dtype=np.float32,
            )
            img_path = r.get("image_path")
            if img_path:
                p = root / img_path if root else Path(img_path)
                img = imread(str(p)).astype(np.float32) / 255.0
            else:
                img = np.zeros((64, 64, 3), dtype=np.float32)
            samples.append(
                Sample(
                    image=img,
                    physics=phys,
                    targets=targets,
                    fitzpatrick=int(float(r.get("fitzpatrick", 3) or 3)),
                    illumination=int(float(r.get("illumination", 0) or 0)),
                )
            )
        return cls(samples, train=train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch

        s = self.samples[idx]
        img_u8 = np.clip(s.image * 255.0, 0, 255).astype(np.uint8)
        augmented = self._augment(image=img_u8)["image"]  # HWC normalized float
        img_t = torch.from_numpy(augmented.transpose(2, 0, 1)).float()
        return {
            "image": img_t,
            "physics": torch.from_numpy(s.physics).float(),
            "targets": torch.from_numpy(s.targets).float(),
            "fitzpatrick": torch.tensor(s.fitzpatrick, dtype=torch.long),
            "illumination": torch.tensor(s.illumination, dtype=torch.long),
        }
