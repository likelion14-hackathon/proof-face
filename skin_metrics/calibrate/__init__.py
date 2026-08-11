"""Offline calibration tooling (not needed at inference time).

Fits the reference distributions, composite anchors, and supervised
instrument-regression coefficients in ``config.yaml`` from a labelled cohort.
The shipped calibration was fitted on the AI-Hub open dataset
*028. 한국인 피부상태 측정 데이터* (Korean skin-condition measurement data);
see :mod:`skin_metrics.calibrate.aihub` for the corpus layout.

Nothing here is imported by the runtime pipeline or the HTTP API.
"""

__all__ = ["aihub", "extract", "fit"]
