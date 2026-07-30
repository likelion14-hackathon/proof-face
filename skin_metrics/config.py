"""Configuration loading for skin_metrics.

The default configuration ships as ``config.yaml`` next to this module. All
reference-distribution numbers there are placeholders (see the file header);
swap in a ``config.yaml`` fitted to your own camera/cohort via ``load_config``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a skin_metrics configuration dictionary.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Path to a YAML config file. If ``None``, the packaged default
        ``config.yaml`` is used.

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If ``path`` is given but does not exist.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data)!r}")
    return data
