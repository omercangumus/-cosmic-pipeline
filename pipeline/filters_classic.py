"""Classic signal filters: median, Savitzky-Golay, wavelet denoising, interpolation."""

import logging

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter as _savgol

logger = logging.getLogger(__name__)


def median_filter(
    df: pd.DataFrame, mask: pd.Series, window: int = 5
) -> pd.DataFrame:
    """
    Replace anomalous points with local median.

    Only points where mask=True are replaced; the rest stay untouched.

    Args:
        df: DataFrame with a 'value' column.
        mask: Boolean mask (True = anomaly to fix).
        window: Median window size (must be odd).

    Returns:
        Corrected DataFrame.
    """
    out = df.copy()
    values = out["value"].values.astype(np.float64)

    if window % 2 == 0:
        window += 1

    half = window // 2
    n = len(values)

    indices = np.where(mask.values)[0]
    for idx in indices:
        lo = max(0, idx - half)
        hi = min(n, idx + half + 1)
        neighbours = values[lo:hi]
        finite = neighbours[np.isfinite(neighbours)]
        if len(finite) > 0:
            values[idx] = np.median(finite)

    out["value"] = values
    logger.info("Median filter (w=%d): corrected %d points", window, len(indices))
    return out


def savgol_filter(
    df: pd.DataFrame,
    mask: pd.Series,
    window: int = 11,
    polyorder: int = 3,
) -> pd.DataFrame:
    """
    Apply Savitzky-Golay smoothing to anomalous regions.

    Computes the SG-smoothed version of the full signal, then replaces
    only the masked points with the smoothed values.

    Args:
        df: DataFrame with a 'value' column.
        mask: Boolean mask (True = anomaly to fix).
        window: SG window length (must be odd and > polyorder).
        polyorder: Polynomial order.

    Returns:
        Corrected DataFrame.
    """
    out = df.copy()
    values = out["value"].values.astype(np.float64)

    if window % 2 == 0:
        window += 1
    if window <= polyorder:
        window = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3

    indices = np.where(mask.values)[0]

    # Remove spikes before SG fitting: set masked points to NaN, then interpolate
    clean = values.copy()
    clean[indices] = np.nan
    filled = pd.Series(clean).interpolate(method="linear", limit_direction="both").ffill().bfill().values
    smoothed = _savgol(filled, window_length=window, polyorder=polyorder)
    values[indices] = smoothed[indices]

    out["value"] = values
    logger.info("Savitzky-Golay filter (w=%d, p=%d): corrected %d points",
                window, polyorder, len(indices))
    return out


def wavelet_filter(
    df: pd.DataFrame,
    mask: pd.Series,
    wavelet: str = "db4",
    level: int = 3,
) -> pd.DataFrame:
    """
    Apply wavelet denoising to anomalous regions.

    Uses soft thresholding on detail coefficients, then replaces
    only the masked points with the denoised values.

    Args:
        df: DataFrame with a 'value' column.
        mask: Boolean mask (True = anomaly to fix).
        wavelet: Wavelet family name.
        level: Decomposition level.

    Returns:
        Corrected DataFrame.
    """
    try:
        import pywt
    except ImportError:
        logger.warning("PyWavelets not installed, falling back to Savitzky-Golay")
        return savgol_filter(df, mask)

    out = df.copy()
    values = out["value"].values.astype(np.float64)

    # Remove spikes before wavelet fitting: set masked points to NaN, then interpolate
    clean = values.copy()
    indices = np.where(mask.values)[0]
    clean[indices] = np.nan
    filled = pd.Series(clean).interpolate(method="linear", limit_direction="both").ffill().bfill().values

    max_level = pywt.dwt_max_level(len(filled), pywt.Wavelet(wavelet).dec_len)
    level = min(level, max_level)

    coeffs = pywt.wavedec(filled, wavelet, level=level)

    # Universal threshold (VisuShrink)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(filled)))

    denoised_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, value=threshold, mode="soft"))

    denoised = pywt.waverec(denoised_coeffs, wavelet)[:len(values)]

    indices = np.where(mask.values)[0]
    values[indices] = denoised[indices]

    out["value"] = values
    logger.info("Wavelet filter (%s, L%d): corrected %d points", wavelet, level, len(indices))
    return out


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


def apply_classic_filters(
    df: pd.DataFrame,
    mask: pd.Series,
    median_window: int = 5,
    sg_window: int = 11,
    sg_polyorder: int = 3,
    wavelet_family: str = "db4",
    wavelet_level: int = 3,
) -> pd.DataFrame:
    """
    Apply the full classic filtering pipeline to anomalous points.

    Strategy:
      1. Interpolate NaN gaps first.
      2. Median filter for spike removal.
      3. Savitzky-Golay for smoothing residual noise.

    Args:
        df: DataFrame with 'value' column.
        mask: Boolean mask (True = anomaly).
        median_window: Window for median filter.
        sg_window: Window for Savitzky-Golay filter.
        sg_polyorder: Polynomial order for SG filter.
        wavelet_family: Wavelet name for wavelet filter.
        wavelet_level: Decomposition level.

    Returns:
        Corrected DataFrame.
    """
    gap_mask = df["value"].isna()
    spike_mask = mask & ~gap_mask

    # Step 1: fill gaps
    result = interpolate_gaps(df, gap_mask)

    # Step 2: median filter on spikes
    if spike_mask.any():
        result = median_filter(result, spike_mask, window=median_window)

    # Step 3: smooth remaining flagged areas
    if spike_mask.any():
        result = savgol_filter(result, spike_mask, window=sg_window, polyorder=sg_polyorder)

    logger.info("Classic filter pipeline complete: %d gap fills, %d spike fixes",
                gap_mask.sum(), spike_mask.sum())
    return result
