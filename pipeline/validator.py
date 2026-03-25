"""Output validation and quality metrics for the cosmic pipeline."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_output(df: pd.DataFrame) -> dict:
    """
    Check quality of cleaned telemetry data.

    Args:
        df: Cleaned DataFrame with at least a 'value' column.

    Returns:
        {
            'is_valid': bool,
            'issues': list[str],
            'quality_score': float  # 0.0–1.0
        }
    """
    issues: list[str] = []
    score = 1.0

    # Check NaN ratio
    nan_ratio = df["value"].isna().sum() / len(df) if len(df) > 0 else 1.0
    if nan_ratio > 0.5:
        issues.append(f"High NaN ratio: {nan_ratio:.1%}")
        score -= 0.4
    elif nan_ratio > 0.1:
        issues.append(f"Moderate NaN ratio: {nan_ratio:.1%}")
        score -= 0.2
    elif nan_ratio > 0:
        score -= nan_ratio

    # Check for infinite values
    inf_count = np.isinf(df["value"].dropna()).sum()
    if inf_count > 0:
        issues.append(f"{inf_count} infinite values found")
        score -= 0.3

    # Check signal variance (constant signal = suspicious)
    finite_vals = df["value"].dropna()
    if len(finite_vals) > 1:
        std = finite_vals.std()
        if std < 1e-10:
            issues.append("Near-zero variance — signal may be constant")
            score -= 0.2

    # Check minimum length
    if len(df) < 100:
        issues.append(f"Signal too short ({len(df)} points)")
        score -= 0.1

    # Check timestamp monotonicity
    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        if not df["timestamp"].is_monotonic_increasing:
            issues.append("Timestamps are not monotonically increasing")
            score -= 0.1

    score = max(0.0, min(1.0, score))
    is_valid = len(issues) == 0

    logger.info(
        "Validation: valid=%s, score=%.2f, issues=%d",
        is_valid, score, len(issues),
    )
    return {"is_valid": is_valid, "issues": issues, "quality_score": score}


def calculate_metrics(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    ground_truth: pd.DataFrame | None = None,
) -> dict:
    """
    Calculate quality metrics comparing original, cleaned, and optionally ground truth.

    Args:
        original: Corrupted signal DataFrame.
        cleaned: Cleaned signal DataFrame.
        ground_truth: Optional clean reference DataFrame.

    Returns:
        {
            'rmse': float,
            'mae': float,
            'r2_score': float,
            'snr': float,  # Signal-to-Noise Ratio in dB
        }
    """
    if ground_truth is not None:
        ref = ground_truth["value"].values.astype(np.float64)
        pred = cleaned["value"].values.astype(np.float64)

        # Align lengths
        n = min(len(ref), len(pred))
        ref = ref[:n]
        pred = pred[:n]

        # Use only finite points for metrics
        valid = np.isfinite(ref) & np.isfinite(pred)
        ref_v = ref[valid]
        pred_v = pred[valid]

        if len(ref_v) == 0:
            logger.warning("No valid overlapping points for metric calculation")
            return {"rmse": np.nan, "mae": np.nan, "r2_score": np.nan, "snr": np.nan}

        rmse = float(np.sqrt(np.mean((ref_v - pred_v) ** 2)))
        mae = float(np.mean(np.abs(ref_v - pred_v)))
        r2 = _r2_score(ref_v, pred_v)
        snr = _snr_db(ref_v, ref_v - pred_v)
    else:
        # Without ground truth: compare cleaned vs original
        orig = original["value"].values.astype(np.float64)
        cln = cleaned["value"].values.astype(np.float64)

        n = min(len(orig), len(cln))
        orig = orig[:n]
        cln = cln[:n]

        valid = np.isfinite(orig) & np.isfinite(cln)
        orig_v = orig[valid]
        cln_v = cln[valid]

        if len(orig_v) == 0:
            return {"rmse": np.nan, "mae": np.nan, "r2_score": np.nan, "snr": np.nan}

        diff = orig_v - cln_v
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))
        r2 = _r2_score(orig_v, cln_v)
        snr = _snr_db(cln_v, diff)

    logger.info("Metrics: RMSE=%.4f, MAE=%.4f, R²=%.4f, SNR=%.2f dB",
                rmse, mae, r2, snr)
    return {"rmse": rmse, "mae": mae, "r2_score": r2, "snr": snr}


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio in decibels."""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-20:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))
