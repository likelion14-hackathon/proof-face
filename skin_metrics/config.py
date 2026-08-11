"""Configuration loading for skin_metrics.

Configuration comes from two files that are merged at load time:

``config.yaml``
    Hand-maintained knobs and their rationale (thresholds, composite weights,
    white-balance policy). Heavily commented; never machine-written.
``calibration_profile.yaml``
    Machine-fitted numbers produced by :mod:`skin_metrics.calibrate.fit` from a
    labelled cohort: composite anchors, reference percentile grids, and the
    supervised instrument models. Regenerating it must not clobber the prose in
    ``config.yaml``, which is why the two are kept apart.

The profile is optional. Without it the pipeline still runs, using the
uncalibrated composite scores and whatever anchors ``config.yaml`` carries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_PROFILE_PATH = Path(__file__).with_name("calibration_profile.yaml")


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping, raising a clear error if it is not one."""
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data)!r}: {path}")
    return data


def merge_profile(config: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Overlay a fitted calibration profile onto a base configuration.

    Parameters
    ----------
    config : dict
        Parsed ``config.yaml``. Mutated in place and returned.
    profile : dict
        Parsed ``calibration_profile.yaml``. Recognised keys:
        ``composite_anchors`` (merged per metric into
        ``composite.<metric>.anchors``), ``reference``, ``supervised``,
        ``validation``, and ``provenance``.

    Returns
    -------
    dict
        The merged configuration.
    """
    for metric, anchors in (profile.get("composite_anchors") or {}).items():
        if metric in config.get("composite", {}):
            config["composite"][metric]["anchors"].update(anchors)

    # Weights are replaced, not merged: a fitted set is a complete alternative
    # to the declared one, and mixing the two would produce a combination that
    # was never validated.
    for metric, weights in (profile.get("composite_weights") or {}).items():
        if metric in config.get("composite", {}):
            config["composite"][metric]["weights"] = dict(weights)

    for key in (
        "reference",
        "supervised",
        "validation",
        "validation_weights",
        "provenance",
    ):
        if key in profile:
            config[key] = profile[key]

    return config


def load_config(
    path: str | Path | None = None,
    *,
    profile: str | Path | None = None,
    use_profile: bool = True,
) -> dict[str, Any]:
    """Load a skin_metrics configuration dictionary.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Path to a YAML config file. If ``None``, the packaged default
        ``config.yaml`` is used.
    profile : str or pathlib.Path, optional
        Path to a calibration profile. If ``None``, the packaged
        ``calibration_profile.yaml`` is used when it exists.
    use_profile : bool, optional
        Set ``False`` to load the base config alone -- useful for refitting a
        profile without the previous one influencing the result.

    Returns
    -------
    dict
        Parsed configuration, with the calibration profile merged in.

    Raises
    ------
    FileNotFoundError
        If ``path`` (or an explicitly given ``profile``) does not exist.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    data = _read_yaml(cfg_path)

    if not use_profile:
        return data

    if profile is not None:
        profile_path = Path(profile)
        if not profile_path.exists():
            raise FileNotFoundError(f"Calibration profile not found: {profile_path}")
    else:
        profile_path = DEFAULT_PROFILE_PATH
        if not profile_path.exists():
            return data

    return merge_profile(data, _read_yaml(profile_path))
