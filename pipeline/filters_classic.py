"""Classic signal filters: interpolation, detrend, median — layered pipeline."""

import logging

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter as _medfilt
from scipy.signal import detrend as _scipy_detrend

logger = logging.getLogger(__name__)


def interpolate_gaps(
    df: pd.DataFrame, mask: pd.Series, method: str = "linear"
) -> pd.DataFrame:
    """
    Fill gaps (NaN / masked points) via interpolation.

    Args:
        df: DataFrame with a 'value' column.
        mask: Boolean mask (True = gap to fill).
        method: Interpolation method ('linear', 'cubic', 'nearest').

    Returns:
        Corrected DataFrame.
    """
    out = df.copy()
    values = out["value"].values.astype(np.float64)

    # Set masked positions to NaN so pandas can interpolate them
    indices = np.where(mask.values)[0]
    values[indices] = np.nan

    out["value"] = values
    out["value"] = out["value"].interpolate(method=method, limit_direction="both")

    # If edges remain NaN after interpolation, forward/back fill
    out["value"] = out["value"].ffill().bfill()

    n_filled = len(indices)
    remaining_nan = out["value"].isna().sum()
    logger.info("Interpolation (%s): filled %d gaps, %d NaN remaining",
                method, n_filled, remaining_nan)
    return out


def detrend_signal(
    df: pd.DataFrame,
    type: str = "linear",
) -> pd.DataFrame:
    """
    Remove linear trend from the signal.

    NaN values are preserved — only finite spans are detrended.

    Args:
        df: DataFrame with a 'value' column.
        type: 'linear' or 'constant' (passed to scipy.signal.detrend).

    Returns:
        New DataFrame with trend removed.
    """
    out = df.copy()
    values = out["value"].values.astype(np.float64)

    finite = np.isfinite(values)
    if finite.sum() < 2:
        return out

    detrended = values.copy()
    detrended[finite] = _scipy_detrend(values[finite], type=type)

    out["value"] = detrended
    logger.info("Detrend (%s): removed trend from %d finite points", type, finite.sum())
    return out


def median_filter(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Apply median filter to the entire signal for spike removal.

    Args:
        df: DataFrame with a 'value' column.
        window: Median filter kernel size (must be odd).

    Returns:
        Filtered DataFrame.
    """
    out = df.copy()
    values = out["value"].values.astype(np.float64)

    if window % 2 == 0:
        window += 1

    out["value"] = _medfilt(values, size=window)
    logger.info("Median filter (w=%d): filtered %d points", window, len(values))
    return out


def apply_classic_filters(
    df: pd.DataFrame,
    mask: pd.Series,
    median_window: int = 5,
    return_intermediates: bool = False,
    **kwargs,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Apply the layered classic filtering pipeline.

    Three sequential layers — each step's output feeds the next:
      1. Interpolate NaN gaps.
      2. Detrend (remove linear drift from entire signal).
      3. Median filter (remove spikes from entire signal).

    Args:
        df: DataFrame with 'value' column.
        mask: Boolean mask (True = anomaly). Used to identify gaps.
        median_window: Kernel size for median filter.
        return_intermediates: If True, return (result, intermediates_dict).
        **kwargs: Ignored (backward compatibility for removed SG/wavelet params).

    Returns:
        Corrected DataFrame, or (DataFrame, intermediates) if requested.
    """
    intermediates: dict[str, np.ndarray] = {}
    intermediates["step_0_raw"] = df["value"].values.copy()

    # Step 1: Interpolate NaN gaps
    gap_mask = df["value"].isna()
    result = interpolate_gaps(df, gap_mask)
    intermediates["step_1_interpolated"] = result["value"].values.copy()

    # Step 2: Detrend
    result = detrend_signal(result)
    intermediates["step_2_detrended"] = result["value"].values.copy()

    # Step 3: Median filter (entire signal)
    result = median_filter(result, window=median_window)
    intermediates["step_3_median"] = result["value"].values.copy()

    logger.info(
        "Classic filter pipeline complete: %d gaps filled, %d total points processed",
        int(gap_mask.sum()), len(df),
    )

    if return_intermediates:
        return result, intermediates
    return result
