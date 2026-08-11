"""Index the AI-Hub *028. 한국인 피부상태 측정 데이터* corpus.

Corpus layout (``<root>`` = the ``028.…`` directory or its ``1.데이터`` child)::

    3.개방데이터/1.데이터/
      Training/01.원천데이터/TS/<device>/<subject>/<subject>_<dev>_<angle>.jpg
      Training/02.라벨링데이터/TL/<device>/<subject>/<subject>_<dev>_<angle>_<part>.json
      Validation/01.원천데이터/VS/…   Validation/02.라벨링데이터/VL/…
      Other/Other/메타데이터/{meta_data.csv,measurement_data.csv}

Each image has nine sibling label JSONs, one per *facepart*. A facepart JSON
carries the ROI ``bbox``, expert ``annotations`` (ordinal grades), and
``equipment`` readings — the instrument ground truth we calibrate against:

===========  ==============  =========================================
facepart     ROI             instrument / expert labels
===========  ==============  =========================================
0            whole face      ``pigmentation_count`` (spot counter), acne points
1            forehead        Corneometer moisture, Cutometer R0-R9/Q0-Q3, pigmentation + wrinkle grade
2            glabella        wrinkle grade
3 / 4        peri-ocular     Visioline Ra/Rq/Rmax/Rz…, wrinkle grade
5 / 6        left/right cheek  moisture, elasticity, pore count, pigmentation + pore grade
7            lips            dryness grade
8            chin            moisture, elasticity, sagging grade
===========  ==============  =========================================

Laterality: dataset ``l_*`` faceparts sit on the **image left** (verified
against the bboxes), which is also where :data:`skin_metrics.detection.face.
ROI_LANDMARKS` puts ``left_cheek`` — so the names map straight across.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# Device sub-directory names, keyed by the ASCII slug used in output files.
DEVICE_DIRS: dict[str, str] = {
    "digital_camera": "1. 디지털카메라",
    "tablet": "2. 스마트패드",
    "phone": "3. 스마트폰",
}

# Filename infix identifying the device (``0112_01_F.jpg`` -> digital camera).
DEVICE_CODES: dict[str, str] = {
    "digital_camera": "01",
    "tablet": "02",
    "phone": "03",
}

# split -> (top directory, source sub-path, label sub-path)
SPLIT_DIRS: dict[str, tuple[str, str, str]] = {
    "train": ("Training", "01.원천데이터/TS", "02.라벨링데이터/TL"),
    "val": ("Validation", "01.원천데이터/VS", "02.라벨링데이터/VL"),
}

# facepart index -> skin_metrics ROI name (only the ROIs our pipeline extracts)
FACEPART_TO_ROI: dict[int, str] = {
    1: "forehead",
    5: "left_cheek",
    6: "right_cheek",
    8: "chin",
}

# ROI name -> label-key prefix used inside the JSON
ROI_LABEL_PREFIX: dict[str, str] = {
    "forehead": "forehead",
    "left_cheek": "l_cheek",
    "right_cheek": "r_cheek",
    "chin": "chin",
}

_FACE_LEVEL_PART = 0


@dataclass
class RoiLabels:
    """Instrument and expert labels for one ROI of one subject.

    Attributes
    ----------
    moisture : float or None
        Corneometer reading in arbitrary units (~24-84 in this cohort). The
        regression target for the hydration metric.
    pigmentation_grade : int or None
        Expert ordinal pigmentation grade, 0 (none) to 5 (severe). Absent for
        ``chin``.
    pore_grade : int or None
        Expert ordinal pore grade, 0-5. Cheeks only.
    pore_count : float or None
        Instrument pore count. Cheeks only.
    elasticity_r2 : float or None
        Cutometer R2 (gross elasticity), 0-1.
    bbox : tuple of int or None
        ``(x1, y1, x2, y2)`` ROI box in image pixels, as annotated.
    """

    moisture: float | None = None
    pigmentation_grade: int | None = None
    pore_grade: int | None = None
    pore_count: float | None = None
    elasticity_r2: float | None = None
    bbox: tuple[int, int, int, int] | None = None


@dataclass
class Sample:
    """One (subject, device, angle) image plus every label attached to it.

    Attributes
    ----------
    subject : str
        Zero-padded subject id, e.g. ``"0112"``.
    device : str
        Device slug (key of :data:`DEVICE_DIRS`).
    split : str
        ``"train"`` or ``"val"``.
    image_path : pathlib.Path
        Source image.
    angle : str
        Filename angle code the image was taken from (``"F"`` frontal,
        ``"L"``/``"R"`` three-quarter, ``"L15"``..``"R30"`` on the DSLR).
        The instrument labels are identical across angles -- the same
        measurement is simply photographed from several viewpoints.
    age, gender, skin_type, sensitive : int / str / int / int
        Subject metadata copied from the label JSON ``info`` block.
        ``skin_type`` is the survey's *facial skin type* code (dry/neutral/
        oily/combination), **not** a Fitzpatrick type.
    pigmentation_count : float or None
        Whole-face instrument spot count (facepart 0).
    acne_count : int or None
        Number of annotated acne lesions (facepart 0).
    roi_labels : dict
        ``{roi_name: RoiLabels}`` for the ROIs this corpus labels.
    """

    subject: str
    device: str
    split: str
    image_path: Path
    angle: str = "F"
    age: int | None = None
    gender: str | None = None
    skin_type: int | None = None
    sensitive: int | None = None
    pigmentation_count: float | None = None
    acne_count: int | None = None
    roi_labels: dict[str, RoiLabels] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Stable identifier used to resume interrupted extraction runs.

        The frontal angle is left unsuffixed so keys written before multi-angle
        support stay valid and a frontal run can still be resumed.
        """
        stem = f"{self.split}/{self.device}/{self.subject}"
        return stem if self.angle == "F" else f"{stem}/{self.angle}"


def resolve_data_root(root: str | Path) -> Path:
    """Locate the ``1.데이터`` directory from a user-supplied corpus path.

    Parameters
    ----------
    root : str or pathlib.Path
        Either the ``028.…`` download directory, its ``3.개방데이터`` child, or
        the ``1.데이터`` directory itself.

    Returns
    -------
    pathlib.Path
        The directory directly containing ``Training`` / ``Validation``.

    Raises
    ------
    FileNotFoundError
        If no ``Training`` directory can be found from ``root``.
    """
    base = Path(root)
    candidates = [base, base / "1.데이터", base / "3.개방데이터" / "1.데이터"]
    for cand in candidates:
        if (cand / "Training").is_dir():
            return cand
    raise FileNotFoundError(
        f"Could not find a 'Training' directory under {base}. "
        "Point --data-root at the '028. 한국인 피부상태 측정 데이터' folder."
    )


def _as_float(value: Any) -> float | None:
    """Coerce a JSON value to float, mapping non-numerics to ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    """Coerce a JSON value to int, mapping non-numerics to ``None``."""
    f = _as_float(value)
    return None if f is None else int(round(f))


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a label JSON (BOM-tolerant), returning ``None`` if unreadable."""
    try:
        with path.open("r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _parse_roi_labels(part: int, doc: dict[str, Any]) -> RoiLabels:
    """Pull the labels for one facepart JSON into a :class:`RoiLabels`."""
    roi = FACEPART_TO_ROI[part]
    prefix = ROI_LABEL_PREFIX[roi]
    ann = doc.get("annotations") or {}
    eq = doc.get("equipment") or {}
    bbox_raw = (doc.get("images") or {}).get("bbox")
    bbox = None
    if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
        bbox = tuple(int(v) for v in bbox_raw)  # type: ignore[assignment]
    return RoiLabels(
        moisture=_as_float(eq.get(f"{prefix}_moisture")),
        pigmentation_grade=_as_int(ann.get(f"{prefix}_pigmentation")),
        pore_grade=_as_int(ann.get(f"{prefix}_pore")),
        pore_count=_as_float(eq.get(f"{prefix}_pore")),
        elasticity_r2=_as_float(eq.get(f"{prefix}_elasticity_R2")),
        bbox=bbox,
    )


def _label_dir(data_root: Path, split: str, device: str) -> Path:
    """Directory holding label JSONs for one split/device."""
    top, _, label_sub = SPLIT_DIRS[split]
    return data_root / top / label_sub / DEVICE_DIRS[device]


def _source_dir(data_root: Path, split: str, device: str) -> Path:
    """Directory holding source images for one split/device."""
    top, src_sub, _ = SPLIT_DIRS[split]
    return data_root / top / src_sub / DEVICE_DIRS[device]


def build_sample(
    data_root: Path,
    split: str,
    device: str,
    subject: str,
    angle: str = "F",
) -> Sample | None:
    """Assemble one :class:`Sample` from its image and nine label JSONs.

    Parameters
    ----------
    data_root : pathlib.Path
        Output of :func:`resolve_data_root`.
    split : {"train", "val"}
    device : str
        Key of :data:`DEVICE_DIRS`.
    subject : str
        Zero-padded subject id.
    angle : str, optional
        Filename angle code; ``"F"`` (frontal) is what the service sees.

    Returns
    -------
    Sample or None
        ``None`` when the image or every label JSON is missing.
    """
    stem = f"{subject}_{DEVICE_CODES[device]}_{angle}"
    image_path = _source_dir(data_root, split, device) / subject / f"{stem}.jpg"
    if not image_path.is_file():
        return None

    label_dir = _label_dir(data_root, split, device) / subject
    sample = Sample(subject=subject, device=device, split=split,
                    image_path=image_path, angle=angle)

    seen_any = False
    for part in (_FACE_LEVEL_PART, *FACEPART_TO_ROI):
        doc = _read_json(label_dir / f"{stem}_{part:02d}.json")
        if doc is None:
            continue
        seen_any = True
        info = doc.get("info") or {}
        if sample.age is None:
            sample.age = _as_int(info.get("age"))
            sample.gender = info.get("gender")
            sample.skin_type = _as_int(info.get("skin_type"))
            sample.sensitive = _as_int(info.get("sensitive"))
        if part == _FACE_LEVEL_PART:
            eq = doc.get("equipment") or {}
            sample.pigmentation_count = _as_float(eq.get("pigmentation_count"))
            acne = (doc.get("annotations") or {}).get("acne")
            sample.acne_count = len(acne) if isinstance(acne, list) else None
        else:
            sample.roi_labels[FACEPART_TO_ROI[part]] = _parse_roi_labels(part, doc)

    return sample if seen_any else None


def index_dataset(
    root: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "val"),
    devices: tuple[str, ...] | None = None,
    angles: tuple[str, ...] = ("F",),
    limit: int | None = None,
) -> list[Sample]:
    """Walk the corpus and return every labelled sample.

    Parameters
    ----------
    root : str or pathlib.Path
        Corpus path (see :func:`resolve_data_root`).
    splits : tuple of str, optional
        Subset of ``("train", "val")``.
    devices : tuple of str, optional
        Subset of :data:`DEVICE_DIRS` keys; ``None`` means all three.
    angles : tuple of str, optional
        Filename angle codes to include, default frontal only. Angles a device
        did not shoot are skipped silently (the DSLR has ``L15``/``L30``, the
        phone and tablet have ``L``/``R``).
    limit : int, optional
        Stop after this many subjects (per split/device), for smoke runs.

    Returns
    -------
    list of Sample
        Sorted by ``(split, device, subject, angle)``.
    """
    data_root = resolve_data_root(root)
    devices = devices or tuple(DEVICE_DIRS)
    out: list[Sample] = []
    for split in splits:
        for device in devices:
            src = _source_dir(data_root, split, device)
            if not src.is_dir():
                continue
            subjects = sorted(p.name for p in src.iterdir() if p.is_dir())
            if limit is not None:
                subjects = subjects[:limit]
            for subject in subjects:
                for angle in angles:
                    sample = build_sample(data_root, split, device, subject, angle=angle)
                    if sample is not None:
                        out.append(sample)
    return out


def read_subject_metadata(root: str | Path) -> dict[str, dict[str, str]]:
    """Read ``meta_data.csv`` (survey metadata), keyed by zero-padded subject id.

    Parameters
    ----------
    root : str or pathlib.Path
        Corpus path (see :func:`resolve_data_root`).

    Returns
    -------
    dict
        ``{subject_id: {column: value}}``. Empty if the file is absent — the
        per-image label JSONs already carry the fields we need, so this is a
        cross-check rather than a dependency.
    """
    data_root = resolve_data_root(root)
    path = data_root / "Other" / "Other" / "메타데이터" / "meta_data.csv"
    if not path.is_file():
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            subject_no = (row.get("subject_no") or "").strip()
            if subject_no:
                out[subject_no.zfill(4)] = row
    return out


def iter_roi_rows(samples: list[Sample]) -> Iterator[dict[str, Any]]:
    """Flatten samples into one dict per (sample, labelled ROI).

    Yields
    ------
    dict
        Label columns for a single ROI, prefixed ``label_``, plus the sample
        identity columns. Feature columns are joined on later by
        :mod:`skin_metrics.calibrate.extract`.
    """
    for s in samples:
        for roi, labels in s.roi_labels.items():
            yield {
                "key": s.key,
                "subject": s.subject,
                "device": s.device,
                "split": s.split,
                "roi": roi,
                "age": s.age,
                "gender": s.gender,
                "skin_type": s.skin_type,
                "sensitive": s.sensitive,
                "label_moisture": labels.moisture,
                "label_pigmentation_grade": labels.pigmentation_grade,
                "label_pore_grade": labels.pore_grade,
                "label_pore_count": labels.pore_count,
                "label_elasticity_r2": labels.elasticity_r2,
                "label_face_pigmentation_count": s.pigmentation_count,
                "label_face_acne_count": s.acne_count,
            }
