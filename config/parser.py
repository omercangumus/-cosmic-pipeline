"""Configuration loader — YAML file with dict fallback."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    import yaml

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config or {}


def get_default_config() -> dict[str, Any]:
    """Return sensible defaults as a plain dict (no file dependency)."""
    return {
        "dsp_detector": {
            "zscore_threshold": 2.0,
            "sliding_window": 50,
            "sliding_window_threshold": 3.0,
            "max_gap_seconds": 60,
            "range_std_multiplier": 10.0,
        },
        "lstm_detector": {
            "model_path": "models/lstm_ae.pt",
            "contamination": 0.05,
            "threshold_percentile": 95,
        },
        "ensemble": {
            "strategy": "hybrid",
            "min_agreement": 2,
        },
        "classic_filter": {
            "median_window": 5,
        },
    }
