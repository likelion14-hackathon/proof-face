"""Command-line interface for skin_metrics.

Examples
--------
    skin-metrics analyze face.jpg --reference-bbox 10,10,40,40 --output report.json
    skin-metrics compare before.jpg after.jpg
    skin-metrics train --data data/labels.csv --config skin_metrics/config.yaml
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from . import DISCLAIMER
from .config import load_config

app = typer.Typer(
    add_completion=False,
    help="Camera-based skin metric quantification (NOT a medical device).",
)


def _read_image(path: Path) -> np.ndarray:
    """Read an image file as an RGB array."""
    from skimage.io import imread

    img = imread(str(path))
    if img.ndim == 2:  # grayscale -> RGB
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:  # drop alpha
        img = img[..., :3]
    return img


def _parse_bbox(bbox: Optional[str]) -> Optional[tuple[int, int, int, int]]:
    if bbox is None:
        return None
    parts = [int(p) for p in bbox.split(",")]
    if len(parts) != 4:
        raise typer.BadParameter("reference-bbox must be 'x,y,w,h'")
    return (parts[0], parts[1], parts[2], parts[3])


@app.command()
def analyze(
    image: Path = typer.Argument(..., exists=True, help="Face image file."),
    reference_bbox: Optional[str] = typer.Option(
        None, "--reference-bbox", help="Neutral gray/white patch 'x,y,w,h'."
    ),
    output: Optional[Path] = typer.Option(None, "--output", help="Write JSON report here."),
    model: Optional[Path] = typer.Option(
        None, "--model", help="face_landmarker.task path (MediaPipe Tasks API)."
    ),
    download_model: bool = typer.Option(
        False, "--download-model", help="Download the FaceLandmarker model (~3.8MB) if missing."
    ),
    config: Optional[Path] = typer.Option(None, "--config", help="Config YAML path."),
) -> None:
    """Analyze one face image and print/write a JSON SkinReport."""
    from .pipeline import analyze as run_analyze

    cfg = load_config(config)
    model_path = str(model) if model else None
    if download_model:
        from .detection.face import ensure_face_model

        model_path = str(ensure_face_model(model_path))
        typer.echo(f"Face model ready at {model_path}")

    img = _read_image(image)
    report = run_analyze(
        img, ref_bbox=_parse_bbox(reference_bbox), model_path=model_path, config=cfg
    )

    payload = report.model_dump()
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if output is not None:
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Wrote report to {output}")
    else:
        typer.echo(text)


@app.command()
def compare(
    image1: Path = typer.Argument(..., exists=True, help="Baseline image."),
    image2: Path = typer.Argument(..., exists=True, help="Current image."),
    reference_bbox: Optional[str] = typer.Option(None, "--reference-bbox"),
    output: Optional[Path] = typer.Option(None, "--output"),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Compare two images (baseline vs current) and report score changes."""
    from .pipeline import analyze as run_analyze
    from .scoring.normalize import compare as compare_scores

    cfg = load_config(config)
    bbox = _parse_bbox(reference_bbox)
    r1 = run_analyze(_read_image(image1), ref_bbox=bbox, config=cfg)
    r2 = run_analyze(_read_image(image2), ref_bbox=bbox, config=cfg)

    baseline = {
        "pigmentation": r1.pigmentation.score,
        "erythema": r1.erythema.score,
        "hydration": r1.hydration.score,
    }
    current = {
        "pigmentation": r2.pigmentation.score,
        "erythema": r2.erythema.score,
        "hydration": r2.hydration.score,
    }
    result = {
        "baseline": baseline,
        "current": current,
        "changes": compare_scores(current, baseline),
        "disclaimer": DISCLAIMER,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if output is not None:
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Wrote comparison to {output}")
    else:
        typer.echo(text)


@app.command()
def train(
    data: Optional[Path] = typer.Option(None, "--data", help="Label CSV (optional)."),
    config: Optional[Path] = typer.Option(None, "--config"),
    mode: str = typer.Option("regression", "--mode", help="'regression' or 'ranking'."),
    epochs: int = typer.Option(3, "--epochs"),
    dummy: bool = typer.Option(
        False, "--dummy", help="Use the synthetic dummy dataset (no labels needed)."
    ),
) -> None:
    """Train the Phase 2 model (requires the 'dl' extra: `uv sync --extra dl`)."""
    try:
        from .models.train import run_training
    except ImportError as exc:
        typer.echo(
            "Phase 2 requires the 'dl' extra. Install with: uv sync --extra dl\n"
            f"({exc})"
        )
        raise typer.Exit(code=1)

    cfg = load_config(config)
    run_training(
        data_csv=data, config=cfg, mode=mode, epochs=epochs, use_dummy=dummy or data is None
    )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port", help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev)."),
    download_model: bool = typer.Option(
        False, "--download-model", help="Download the FaceLandmarker model at startup if missing."
    ),
    config: Optional[Path] = typer.Option(None, "--config", help="Config YAML path."),
    allow_private_hosts: bool = typer.Option(
        False,
        "--allow-private-hosts",
        help="DEV ONLY: allow image URLs pointing at localhost/private networks (disables the "
        "SSRF guard).",
    ),
) -> None:
    """Serve the HTTP API (requires the 'api' extra: `uv sync --extra api`)."""
    import os

    try:
        import uvicorn
    except ImportError as exc:
        typer.echo(f"The API requires the 'api' extra. Install with: uv sync --extra api\n({exc})")
        raise typer.Exit(code=1)

    # create_app() reads settings from the environment; mirror the flags there so
    # that --reload (which re-imports the app in a subprocess) sees them too.
    if config is not None:
        os.environ["SKIN_METRICS_API_CONFIG"] = str(config)
    if download_model:
        os.environ["SKIN_METRICS_API_DOWNLOAD_MODEL"] = "1"
    if allow_private_hosts:
        os.environ["SKIN_METRICS_API_ALLOW_PRIVATE_HOSTS"] = "1"
        typer.echo("WARNING: private/loopback image URLs are allowed (SSRF guard off).")

    typer.echo(f"Serving skin-metrics API on http://{host}:{port} (docs at /docs)")
    uvicorn.run("skin_metrics.api.app:app", host=host, port=port, reload=reload)


calibrate_app = typer.Typer(
    add_completion=False,
    help="Fit the scoring calibration from a labelled reference cohort.",
)
app.add_typer(calibrate_app, name="calibrate")


@calibrate_app.command("extract")
def calibrate_extract(
    data_root: Path = typer.Option(
        ...,
        "--data-root",
        exists=True,
        help="AI-Hub '028. 한국인 피부상태 측정 데이터' corpus directory.",
    ),
    out_dir: Path = typer.Option(
        Path("calibration_work"), "--out-dir", help="Where the feature CSVs go."
    ),
    workers: int = typer.Option(6, "--workers", help="Process-pool size."),
    devices: Optional[str] = typer.Option(
        None,
        "--devices",
        help="Comma-separated subset: digital_camera,tablet,phone.",
    ),
    angles: str = typer.Option(
        "F",
        "--angles",
        help="Comma-separated capture angles: F (frontal), L/R (phone+tablet "
        "three-quarter), L15,L30,R15,R30,Ft,Fb (digital camera).",
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Subjects per split/device (smoke runs)."
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        help="Alternative config.yaml for the workers (e.g. to sweep "
        "normalization.target_eye_span_px). Default: the packaged one.",
    ),
    no_resume: bool = typer.Option(
        False, "--no-resume", help="Re-extract images already in the CSVs."
    ),
) -> None:
    """Extract physics features for every labelled image in the corpus.

    Resumable: rerunning picks up where an interrupted run stopped.
    """
    from .calibrate.extract import index_and_extract

    device_tuple = tuple(d.strip() for d in devices.split(",")) if devices else None
    angle_tuple = tuple(a.strip() for a in angles.split(",") if a.strip())
    stats = index_and_extract(
        data_root,
        out_dir,
        devices=device_tuple,
        angles=angle_tuple,
        workers=workers,
        config_path=config,
        limit=limit,
        resume=not no_resume,
    )
    typer.echo(
        f"extracted={stats['done']} skipped={stats['skipped']} "
        f"failed={stats['failed']} -> {out_dir}"
    )


@calibrate_app.command("fit")
def calibrate_fit(
    features_dir: Path = typer.Option(
        Path("calibration_work"),
        "--features-dir",
        exists=True,
        help="Directory holding features_roi.csv / features_face.csv.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help="Profile destination (default: packaged calibration_profile.yaml).",
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="Fit on one capture device only (default: pooled)."
    ),
    profile_name: str = typer.Option(
        "korean_cohort_2023", "--profile-name", help="Identifier stored in reports."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print validation numbers without writing."
    ),
) -> None:
    """Fit anchors, instrument models, and reference grids; write the profile."""
    from .calibrate.extract import load_rows
    from .calibrate.fit import fit_calibration, format_validation, write_profile
    from .config import DEFAULT_PROFILE_PATH

    roi_rows = load_rows(features_dir / "features_roi.csv")
    face_rows = load_rows(features_dir / "features_face.csv")
    # Fit against the base config only: a previously written profile must not
    # feed back into the anchors it is about to replace.
    config = load_config(use_profile=False)

    fitted = fit_calibration(roi_rows, face_rows, config, device=device)
    typer.echo(format_validation(fitted))

    if dry_run:
        typer.echo("(--dry-run: profile not written)")
        return

    dest = write_profile(fitted, output or DEFAULT_PROFILE_PATH,
                         profile_name=profile_name)
    typer.echo(f"wrote {dest}")


@app.callback()
def _main() -> None:
    """skin-metrics CLI."""


if __name__ == "__main__":  # pragma: no cover
    app()
