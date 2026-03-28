"""Classic DSP anomaly detection: Z-score, sliding window, gap detection."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_outliers_zscore(
    df: pd.DataFrame, threshold: float = 3.0
) -> pd.Series:
    """
    Detect anomalies using Z-score on the value column.

    Points where |z-score| > threshold are flagged as anomalies.

    Args:
        df: DataFrame with a 'value' column.
        threshold: Z-score threshold for anomaly detection.

    Returns:
        Boolean Series (True = anomaly detected).
    """
    values = df["value"].values.astype(np.float64)
    mask = pd.Series(np.zeros(len(values), dtype=bool), index=df.index)

    finite = np.isfinite(values)
    if finite.sum() < 2:
        logger.warning("Not enough finite values for Z-score detection")
        return mask

    mean = np.nanmean(values)
    std = np.nanstd(values, ddof=1)

    if std < 1e-12:
        logger.warning("Near-zero std (%.2e), skipping Z-score detection", std)
        return mask

    z_scores = np.abs((values - mean) / std)
    mask = pd.Series(z_scores > threshold, index=df.index)
    # NaN values are not flagged by z-score (they're handled by gap detection)
    mask[~finite] = False

    n_detected = mask.sum()
    logger.info("Z-score (threshold=%.1f): %d anomalies detected", threshold, n_detected)
    return mask


def detect_range_violation(
    df: pd.DataFrame, max_std_multiplier: float = 10.0
) -> pd.Series:
    """
    Detect physically impossible values (extreme outliers beyond N*std).

    Args:
        df: DataFrame with a 'value' column.
        max_std_multiplier: Flag values this many stds from the mean.

    Returns:
        Boolean Series (True = range violation detected).
    """
    values = df["value"].values.astype(np.float64)
    finite = values[np.isfinite(values)]

    mask = pd.Series(np.zeros(len(values), dtype=bool), index=df.index)
    if len(finite) < 2:
        return mask

    mean = np.nanmean(finite)
    std = np.nanstd(finite)
    if std < 1e-12:
        return mask

    threshold = max_std_multiplier * std
    mask = pd.Series(np.abs(values - mean) > threshold, index=df.index)
    mask[~np.isfinite(values)] = False

    n_detected = mask.sum()
    logger.info(
        "Range violation (%d*std): %d anomalies detected",
        max_std_multiplier, n_detected,
    )
    return mask


def detect_sliding_window(
    df: pd.DataFrame,
    window: int = 50,
    threshold: float = 3.0,
) -> pd.Series:
    """
    Detect anomalies using a sliding window deviation method.

    Compares each point to the local rolling median and flags points
    that deviate more than threshold * rolling_std.

    Args:
        df: DataFrame with a 'value' column.
        window: Rolling window size.
        threshold: Number of local standard deviations for flagging.

    Returns:
        Boolean Series (True = anomaly detected).
    """
    values = df["value"]

    rolling_median = values.rolling(window=window, center=True, min_periods=1).median()
    rolling_std = values.rolling(window=window, center=True, min_periods=1).std()

    # Prevent division by zero
    rolling_std = rolling_std.clip(lower=1e-12)

    deviation = np.abs(values - rolling_median) / rolling_std
    mask = deviation > threshold

    # NaN values should not be flagged here
    mask = mask.fillna(False).astype(bool)

    n_detected = mask.sum()
    logger.info(
        "Sliding window (w=%d, t=%.1f): %d anomalies detected",
        window, threshold, n_detected,
    )
    return pd.Series(mask, index=df.index)


def detect_delta_spike(
    df: pd.DataFrame, max_delta_multiplier: float = 5.0
) -> pd.Series:
    """
    Detect sudden jumps relative to the previous value.

    Flags points where the absolute difference from the prior sample
    exceeds mean(delta) + max_delta_multiplier * std(delta).

    Args:
        df: DataFrame with a 'value' column.
        max_delta_multiplier: Threshold multiplier on delta statistics.

    Returns:
        Boolean Series (True = sudden jump detected).
    """
    values = df["value"].values.astype(np.float64)
    delta = np.abs(np.diff(values, prepend=values[0]))
    finite_delta = delta[np.isfinite(delta)]

    if len(finite_delta) < 2:
        return pd.Series(np.zeros(len(values), dtype=bool), index=df.index)

    threshold = np.nanmean(finite_delta) + max_delta_multiplier * np.nanstd(finite_delta)
    mask = delta > threshold
    mask[~np.isfinite(values)] = False

    n_detected = mask.sum()
    logger.info(
        "Delta spike (%d*std): %d anomalies detected",
        max_delta_multiplier, n_detected,
    )
    return pd.Series(mask, index=df.index)


def detect_gaps(
    df: pd.DataFrame, max_gap_seconds: int = 60
) -> pd.Series:
    """
    Detect data gaps: NaN values and large timestamp discontinuities.

    Args:
        df: DataFrame with 'timestamp' and 'value' columns.
        max_gap_seconds: Maximum allowed gap between consecutive timestamps.

    Returns:
        Boolean Series (True = gap/missing data detected).
    """
    mask = df["value"].isna().copy()

    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        time_diff = df["timestamp"].diff().dt.total_seconds()
        large_gaps = time_diff > max_gap_seconds
        # Flag the point AFTER each large gap
        mask = mask | large_gaps.fillna(False)

    n_detected = mask.sum()
    logger.info(
        "Gap detection (max=%ds): %d gaps/missing detected",
        max_gap_seconds, n_detected,
    )
    return pd.Series(mask, index=df.index)


def detect_all(
    df: pd.DataFrame,
    zscore_threshold: float = 2.0,
    window: int = 50,
    window_threshold: float = 3.0,
    max_gap_seconds: int = 60,
    range_std_multiplier: float = 10.0,
    delta_multiplier: float = 5.0,
    df_original: pd.DataFrame | None = None,
    **kwargs,
) -> dict[str, pd.Series]:
    """
    Run all classic detectors and return individual masks.

    Expects pre-detrended data in *df* for statistical detectors.
    Gap, range, and delta detection runs on *df_original* (or *df* if
    not provided) so that NaN positions and original scale are preserved.

    Args:
        df: DataFrame with 'timestamp' and 'value' columns (detrended).
        zscore_threshold: Z-score threshold.
        window: Sliding window size.
        window_threshold: Sliding window deviation threshold.
        max_gap_seconds: Maximum allowed timestamp gap.
        range_std_multiplier: Multiplier for range violation detection.
        delta_multiplier: Multiplier for delta spike detection.
        df_original: Original (non-detrended) DataFrame for gap/range/delta.

    Returns:
        Dict of detector_name → boolean mask.
    """
    gap_source = df_original if df_original is not None else df

    results = {
        "zscore": detect_outliers_zscore(df, threshold=zscore_threshold),
        "sliding_window": detect_sliding_window(df, window=window, threshold=window_threshold),
        "gaps": detect_gaps(gap_source, max_gap_seconds=max_gap_seconds),
        "range": detect_range_violation(gap_source, max_std_multiplier=range_std_multiplier),
        "delta": detect_delta_spike(gap_source, max_delta_multiplier=delta_multiplier),
    }

    total = sum(m.sum() for m in results.values())
    logger.info("Classic detection complete: %d total flags across %d detectors", total, len(results))
    return results
