# Repository Guidelines

## Project Structure & Module Organization

`skin_metrics/` contains the package. The main flow starts in `pipeline.py`, with CLI commands in `cli.py` and default thresholds in `config.yaml`. Subpackages separate concerns: `calibration/` for color handling, `detection/` for face landmarks and ROIs, `features/` for pigmentation/erythema/hydration proxy metrics, `scoring/` for normalization and schemas, `models/` for Phase 2 training scaffolding, and `api/` for FastAPI endpoints. Tests live in `tests/` and use synthetic data; `data/` is for local sample images and should not contain committed personal photos.

## Build, Test, and Development Commands

Use Python 3.11 with `uv`.

```bash
uv sync --extra dev
uv sync --extra detection --extra dl --extra api --extra dev
uv run pytest -q
uv run pytest tests/test_api.py -q
uv run skin-metrics analyze data/face.jpg --download-model --output report.json
uv run skin-metrics train --dummy --mode ranking --epochs 1
uv run skin-metrics serve --download-model
docker compose up --build
```

`uv sync` installs only the selected extras, so use the full extras command when working across API, detection, and model code.

## Coding Style & Naming Conventions

Write typed Python with clear numpy-style docstrings for public functions. Keep images as `(H, W, 3)` float arrays and document whether values are linear RGB or sRGB. Defend numeric code with epsilon clipping before division or logarithms. Heavy dependencies such as `mediapipe`, `torch`, and `timm` should stay lazily imported unless the module is specific to that extra, as in `skin_metrics/api/`. Hydration functionality must remain labeled as a proxy or estimate.

## Testing Guidelines

The test suite uses `pytest` and synthetic fixtures in `tests/conftest.py`; tests should not require real face photos or external network access. Add focused tests beside the affected behavior, using names like `test_pipeline.py` and functions named `test_<behavior>`. Run `uv run pytest -q` before submitting, plus targeted files for changed subsystems.

## Commit & Pull Request Guidelines

Recent commits use short Conventional Commit-style prefixes, for example `feat:` and `settings:`. Keep messages imperative and scoped, such as `feat: add api timeout setting`. Pull requests should include a summary, test results, linked issues when relevant, and screenshots or sample JSON for user-visible CLI/API changes. Note any optional extras, model downloads, or Docker impacts.

## Security & Configuration Tips

Do not commit local images, generated reports, credentials, or downloaded model files. Treat `--allow-private-hosts` and `SKIN_METRICS_API_ALLOW_PRIVATE_HOSTS` as development-only SSRF bypasses. Preserve the medical-device disclaimer in outputs and user-facing docs.
