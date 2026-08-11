"""Batch feature extraction over a labelled cohort.

Runs :func:`skin_metrics.pipeline.extract_raw` across the corpus indexed by
:mod:`skin_metrics.calibrate.aihub` and writes two CSVs:

``features_roi.csv``
    One row per (image, ROI). Carries the per-ROI physics features next to the
    instrument labels measured on that same ROI -- the table the supervised
    regressions in :mod:`skin_metrics.calibrate.fit` are fitted on.
``features_face.csv``
    One row per image: the face-level weighted-mean features, Fitzpatrick
    estimate, calibration status, and whole-face labels. This is the table the
    reference distributions are fitted on, because it matches what the scorer
    sees at inference.

Extraction is multiprocess (each worker builds its own MediaPipe landmarker)
and resumable -- rerunning skips images already present in the output.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable, Sequence

from .aihub import Sample, index_dataset

METRICS = ("pigmentation", "erythema", "hydration")

# Identity/label columns written ahead of the feature columns.
_ROI_ID_COLUMNS = (
    "key", "subject", "device", "split", "angle", "roi",
    "age", "gender", "skin_type", "sensitive",
    "valid_ratio", "eye_span_px", "scale_factor", "under_resolved",
    "calibration_status", "fitzpatrick",
    "label_moisture", "label_pigmentation_grade", "label_pore_grade",
    "label_pore_count", "label_elasticity_r2",
)
_FACE_ID_COLUMNS = (
    "key", "subject", "device", "split", "angle",
    "age", "gender", "skin_type", "sensitive",
    "n_valid_rois", "eye_span_px", "scale_factor", "under_resolved",
    "calibration_status", "fitzpatrick",
    "label_pigmentation_count", "label_acne_count",
    "label_moisture_mean", "label_pigmentation_grade_mean",
)

# Set once per worker process so the config is parsed only on first use.
_WORKER_CONFIG: dict[str, Any] | None = None
_WORKER_CONFIG_PATH: str | None = None


def _feature_columns() -> list[str]:
    """Column names for the flattened feature block, in a stable order.

    Spelled out rather than derived from a sample extraction so CSV column
    order is stable across runs and machines.
    ``tests/test_calibrate.py::test_feature_columns_match_the_pipeline`` asserts
    this stays in sync with what :func:`skin_metrics.pipeline._roi_features`
    actually produces.
    """
    layout = {
        "pigmentation": (
            "melanin_index", "ita", "ita_inv", "evenness",
            "spot_area_ratio", "spot_count", "spot_mean_contrast",
        ),
        "erythema": (
            "erythema_index", "mean_a_star", "p90_a_star",
            "hemoglobin", "hemoglobin_ok",
        ),
        "hydration": (
            "specular_ratio", "specular_inv", "glcm_contrast",
            "glcm_correlation", "glcm_energy", "lbp_uniformity",
            "scaling_index", "wrinkle_density",
        ),
    }
    return [f"f_{m}_{name}" for m, names in layout.items() for name in names]


FEATURE_COLUMNS = _feature_columns()
ROI_COLUMNS = list(_ROI_ID_COLUMNS) + FEATURE_COLUMNS
FACE_COLUMNS = list(_FACE_ID_COLUMNS) + FEATURE_COLUMNS


# BLAS/OpenMP backends each default to one thread per core. Multiplied by the
# worker count that oversubscribes the machine by an order of magnitude (a
# 12-core box hit load average 140 and ground to a halt). Children inherit
# these, so they are set in the parent before the pool is forked/spawned.
_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _limit_threads() -> None:
    """Pin every numeric backend to a single thread in this process."""
    for var in _THREAD_ENV:
        os.environ[var] = "1"


def _init_worker(config_path: str | None) -> None:
    """Pool initialiser: remember which config the workers should load."""
    global _WORKER_CONFIG_PATH
    _WORKER_CONFIG_PATH = config_path
    _limit_threads()
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:  # pragma: no cover - cv2 always present in practice
        pass


def _worker_config() -> dict[str, Any]:
    """Lazily load (and cache) this worker's configuration."""
    global _WORKER_CONFIG
    if _WORKER_CONFIG is None:
        from ..config import load_config

        _WORKER_CONFIG = load_config(_WORKER_CONFIG_PATH)
    return _WORKER_CONFIG


@dataclass
class ExtractResult:
    """One worker's output, self-identifying so it survives an unordered pool.

    Attributes
    ----------
    key : str
        ``Sample.key`` of the source sample.
    image_path : str
        Source image, recorded in the error log.
    roi_rows : list of dict
        Per-ROI feature rows (empty on failure).
    face_row : dict or None
        Face-level feature row, ``None`` on failure.
    error : str
        Empty on success, otherwise the reason extraction failed.
    """

    key: str
    image_path: str
    roi_rows: list[dict[str, Any]]
    face_row: dict[str, Any] | None
    error: str


def _mean(values: Iterable[float | None]) -> float | None:
    """Mean of the non-``None`` values, or ``None`` if there are none."""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def extract_one(sample: Sample) -> "ExtractResult":
    """Extract features for one sample.

    Parameters
    ----------
    sample : Sample
        Indexed corpus entry.

    Returns
    -------
    ExtractResult
        Carries the sample's own identity so callers can attribute results
        arriving out of order from an unordered pool.
    """
    from skimage.io import imread

    from ..pipeline import extract_raw

    ident = (sample.key, str(sample.image_path))
    try:
        img = imread(sample.image_path)
    except Exception as exc:  # unreadable/corrupt file
        return ExtractResult(*ident, [], None, f"imread: {exc}")

    try:
        raw = extract_raw(img, config=_worker_config())
    except Exception as exc:  # no face, all ROIs gated out, etc.
        return ExtractResult(*ident, [], None, f"extract: {exc}")

    common = {
        "key": sample.key,
        "subject": sample.subject,
        "device": sample.device,
        "split": sample.split,
        "angle": sample.angle,
        "age": sample.age,
        "gender": sample.gender,
        "skin_type": sample.skin_type,
        "sensitive": sample.sensitive,
        "eye_span_px": round(raw.eye_span_px, 2),
        "scale_factor": round(raw.scale_factor, 4),
        "under_resolved": int(raw.under_resolved),
        "calibration_status": raw.calibration_status,
        "fitzpatrick": raw.fitzpatrick,
    }

    roi_rows: list[dict[str, Any]] = []
    for roi, feats in raw.roi_features.items():
        labels = sample.roi_labels.get(roi)
        row = dict(common)
        row.update(
            {
                "roi": roi,
                "valid_ratio": round(raw.roi_valid_ratio[roi], 4),
                "label_moisture": labels.moisture if labels else None,
                "label_pigmentation_grade": labels.pigmentation_grade if labels else None,
                "label_pore_grade": labels.pore_grade if labels else None,
                "label_pore_count": labels.pore_count if labels else None,
                "label_elasticity_r2": labels.elasticity_r2 if labels else None,
            }
        )
        for metric in METRICS:
            for name, value in feats[metric].items():
                row[f"f_{metric}_{name}"] = value
        roi_rows.append(row)

    face_row = dict(common)
    face_row.update(
        {
            "n_valid_rois": len(raw.roi_features),
            "label_pigmentation_count": sample.pigmentation_count,
            "label_acne_count": sample.acne_count,
            "label_moisture_mean": _mean(
                lb.moisture for lb in sample.roi_labels.values()
            ),
            "label_pigmentation_grade_mean": _mean(
                lb.pigmentation_grade for lb in sample.roi_labels.values()
            ),
        }
    )
    for metric in METRICS:
        for name, value in raw.aggregate[metric].items():
            face_row[f"f_{metric}_{name}"] = value

    return ExtractResult(*ident, roi_rows, face_row, "")


def _existing_keys(path: Path) -> set[str]:
    """Keys already written to ``path`` (for resuming an interrupted run)."""
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        return {row["key"] for row in csv.DictReader(fh) if row.get("key")}


def _open_writer(path: Path, columns: Sequence[str]) -> tuple[Any, csv.DictWriter]:
    """Open ``path`` for append, writing the header only when it is new."""
    is_new = not path.is_file() or path.stat().st_size == 0
    fh = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
    if is_new:
        writer.writeheader()
    return fh, writer


def run_extraction(
    samples: list[Sample],
    out_dir: str | Path,
    *,
    workers: int = 8,
    config_path: str | Path | None = None,
    resume: bool = True,
    progress_every: int = 25,
) -> dict[str, int]:
    """Extract features for many samples in parallel, appending to CSVs.

    Parameters
    ----------
    samples : list of Sample
        Corpus entries from :func:`skin_metrics.calibrate.aihub.index_dataset`.
    out_dir : str or pathlib.Path
        Directory for ``features_roi.csv``, ``features_face.csv``, and
        ``extract_errors.csv``. Created if absent.
    workers : int, optional
        Process-pool size.
    config_path : str or pathlib.Path, optional
        Config the workers load; ``None`` uses the packaged default.
    resume : bool, optional
        Skip samples whose key already appears in ``features_face.csv``.
    progress_every : int, optional
        Emit a progress line to stderr every N samples.

    Returns
    -------
    dict
        ``{"done", "skipped", "failed"}`` counts.
    """
    _limit_threads()  # set before the pool starts so children inherit it
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    roi_path = out / "features_roi.csv"
    face_path = out / "features_face.csv"
    err_path = out / "extract_errors.csv"

    pending = samples
    skipped = 0
    if resume:
        done_keys = _existing_keys(face_path) | _existing_keys(err_path)
        pending = [s for s in samples if s.key not in done_keys]
        skipped = len(samples) - len(pending)

    if not pending:
        return {"done": 0, "skipped": skipped, "failed": 0}

    roi_fh, roi_writer = _open_writer(roi_path, ROI_COLUMNS)
    face_fh, face_writer = _open_writer(face_path, FACE_COLUMNS)
    err_fh, err_writer = _open_writer(err_path, ("key", "image_path", "error"))

    done = failed = 0
    started = time.time()
    try:
        with Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(str(config_path) if config_path else None,),
        ) as pool:
            # imap_unordered decouples result order from `pending`, which is why
            # each result carries its own key rather than being zipped back.
            for res in pool.imap_unordered(extract_one, pending, chunksize=1):
                if res.error or res.face_row is None:
                    failed += 1
                    err_writer.writerow(
                        {"key": res.key, "image_path": res.image_path,
                         "error": res.error or "unknown"}
                    )
                else:
                    face_writer.writerow(res.face_row)
                    roi_writer.writerows(res.roi_rows)
                    done += 1
                if (done + failed) % progress_every == 0:
                    elapsed = time.time() - started
                    rate = (done + failed) / max(elapsed, 1e-6)
                    remaining = (len(pending) - done - failed) / max(rate, 1e-6)
                    print(
                        f"[extract] {done + failed}/{len(pending)} "
                        f"({rate:.2f}/s, ~{remaining / 60:.1f} min left, "
                        f"{failed} failed)",
                        file=sys.stderr,
                        flush=True,
                    )
                    roi_fh.flush()
                    face_fh.flush()
                    err_fh.flush()
    finally:
        roi_fh.close()
        face_fh.close()
        err_fh.close()

    return {"done": done, "skipped": skipped, "failed": failed}


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a feature CSV back, coercing numeric columns to float.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to ``features_roi.csv`` or ``features_face.csv``.

    Returns
    -------
    list of dict
        Rows with numeric strings converted to ``float`` and empty strings to
        ``None``; identity columns stay strings.
    """
    text_columns = {"key", "subject", "device", "split", "angle", "roi",
                    "gender", "calibration_status"}
    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            clean: dict[str, Any] = {}
            for k, v in row.items():
                if k in text_columns:
                    clean[k] = v
                elif v is None or v == "":
                    clean[k] = None
                else:
                    try:
                        clean[k] = float(v)
                    except ValueError:
                        clean[k] = v
            out.append(clean)
    return out


def index_and_extract(
    data_root: str | Path,
    out_dir: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "val"),
    devices: tuple[str, ...] | None = None,
    angles: tuple[str, ...] = ("F",),
    workers: int = 8,
    config_path: str | Path | None = None,
    limit: int | None = None,
    resume: bool = True,
    progress_every: int = 25,
) -> dict[str, int]:
    """Index the corpus and extract features in one call.

    Parameters
    ----------
    data_root : str or pathlib.Path
        Corpus path (see :func:`skin_metrics.calibrate.aihub.resolve_data_root`).
    out_dir : str or pathlib.Path
        Output directory for the CSVs.
    splits, devices, angles, limit : optional
        Passed through to :func:`~skin_metrics.calibrate.aihub.index_dataset`.
    workers, config_path, resume, progress_every : optional
        Passed through to :func:`run_extraction`.

    Returns
    -------
    dict
        Counts from :func:`run_extraction`.
    """
    samples = index_dataset(data_root, splits=splits, devices=devices,
                            angles=angles, limit=limit)
    print(f"[extract] indexed {len(samples)} samples", file=sys.stderr, flush=True)
    return run_extraction(
        samples,
        out_dir,
        workers=workers,
        config_path=config_path,
        resume=resume,
        progress_every=progress_every,
    )
