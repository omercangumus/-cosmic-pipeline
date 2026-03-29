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
    Layered repair — only anomalous points are modified.

    Steps:
      1. Set fault points to NaN.
      2. Interpolate from clean neighbors.
      3. Median filter on fault points only (spike smoothing).

    Clean points (mask == False) are NEVER modified.

    Args:
        df: DataFrame with 'value' column.
        mask: Boolean mask (True = anomaly to repair).
        median_window: Kernel size for median filter.
        return_intermediates: If True, return (result, intermediates_dict).
        **kwargs: Ignored (backward compatibility).

    Returns:
        Corrected DataFrame, or (DataFrame, intermediates) if requested.
    """
    result = df.copy()
    original_clean = df["value"].values.copy()
    intermediates: dict[str, np.ndarray] = {}
    intermediates["step_0_raw"] = original_clean.copy()

    # Step 1: Set ALL fault points to NaN (including existing NaN gaps)
    nan_mask = result["value"].isna()
    all_bad = mask | nan_mask

    values = result["value"].values.astype(np.float64)
    values[all_bad.values] = np.nan
    result["value"] = values
    intermediates["step_1_nan"] = result["value"].values.copy()

    # Step 2: Interpolate from clean neighbors
    result["value"] = result["value"].interpolate(method="linear", limit_direction="both")
    result["value"] = result["value"].ffill().bfill()
    intermediates["step_2_interpolated"] = result["value"].values.copy()

    # Step 3: Median filter — compute on full signal, apply only to fault points
    if mask.any() and median_window > 1:
        filtered_values = result["value"].values.astype(np.float64).copy()
        if median_window % 2 == 0:
            median_window += 1
        med_result = _medfilt(filtered_values, size=median_window)
        result_values = result["value"].values.copy()
        result_values[mask.values] = med_result[mask.values]
        result["value"] = result_values
    intermediates["step_3_median"] = result["value"].values.copy()

    # Guarantee: clean points with original finite values stay untouched
    clean_finite = ~mask & np.isfinite(original_clean)
    result.loc[clean_finite, "value"] = original_clean[clean_finite]

    n_nan = int(nan_mask.sum())
    n_fault = int(mask.sum())
    n_clean = int(clean_finite.sum())
    logger.info(
        "Classic filter pipeline complete: %d gaps filled, %d fault points repaired, %d clean preserved",
        n_nan, n_fault, n_clean,
    )

    if return_intermediates:
        return result, intermediates
    return result
