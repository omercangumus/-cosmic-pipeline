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


def assess_repair_eligibility(
    df: pd.DataFrame,
    fault_mask: pd.Series,
    fault_timeline: pd.DataFrame,
    max_gap_ratio: float = 0.3,
) -> pd.DataFrame:
    """
    Decide repair action for each anomaly: repair / flag_only / preserve.

    Decision logic:
      - repair: Hard-rule anomaly or sufficient detector agreement.
      - flag_only: Overall fault ratio too high — bulk repair is risky.
      - preserve: Low severity (single detector), may be a real event.

    Args:
        df: Original DataFrame.
        fault_mask: Boolean anomaly mask.
        fault_timeline: Fault timeline with reason and severity columns.
        max_gap_ratio: If fault ratio exceeds this, use flag_only.

    Returns:
        fault_timeline with an added 'repair_decision' column.
    """
    result = fault_timeline.copy()

    if result.empty:
        result["repair_decision"] = pd.Series(dtype=str)
        return result

    total_points = len(df)
    total_faults = int(fault_mask.sum())
    fault_ratio = total_faults / max(total_points, 1)

    decisions: list[str] = []
    for _, row in result.iterrows():
        reason = row.get("reason", "unknown")
        severity = row.get("severity", 0)

        if reason == "hard_rule":
            decisions.append("repair")
        elif fault_ratio > max_gap_ratio:
            decisions.append("flag_only")
        elif severity <= 0.2:
            decisions.append("preserve")
        else:
            decisions.append("repair")

    result["repair_decision"] = decisions

    repair_count = decisions.count("repair")
    flag_count = decisions.count("flag_only")
    preserve_count = decisions.count("preserve")

    logger.info(
        "Repair eligibility: %d repair, %d flag_only, %d preserve (fault_ratio=%.2f)",
        repair_count, flag_count, preserve_count, fault_ratio,
    )
    return result


def calculate_repair_confidence(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    fault_mask: pd.Series,
    detector_masks: dict[str, pd.Series] | None = None,
) -> pd.Series:
    """
    Calculate a confidence score (0.0-1.0) for each repaired point.

    Factors:
      - detector_agreement: How many detectors flagged this point.
      - neighbor_quality: Are surrounding points clean?
      - change_magnitude: How large was the correction?

    Non-fault points receive 1.0 (no repair needed).

    Args:
        original: Original (corrupted) DataFrame.
        cleaned: Cleaned DataFrame.
        fault_mask: Boolean anomaly mask.
        detector_masks: Per-detector masks (optional, improves scoring).

    Returns:
        Float Series with values in [0.0, 1.0].
    """
    n = len(original)
    confidence = pd.Series(np.ones(n, dtype=np.float64), index=original.index)

    orig_values = original["value"].values.astype(np.float64)
    clean_values = cleaned["value"].values.astype(np.float64)
    fault_indices = np.where(fault_mask.values)[0]

    if len(fault_indices) == 0:
        return confidence

    finite_orig = orig_values[np.isfinite(orig_values)]
    global_std = float(np.nanstd(finite_orig)) if len(finite_orig) > 1 else 1.0
    if global_std < 1e-12:
        global_std = 1.0

    for idx in fault_indices:
        # Factor 1: Detector agreement (0.0 - 0.4)
        if detector_masks:
            n_detectors = len(detector_masks)
            n_agreed = sum(1 for m in detector_masks.values() if m.iloc[idx])
            score_agreement = 0.4 * (n_agreed / max(n_detectors, 1))
        else:
            score_agreement = 0.2

        # Factor 2: Neighbor quality (0.0 - 0.3)
        window = 10
        lo = max(0, idx - window)
        hi = min(n, idx + window + 1)
        neighbor_mask = fault_mask.values[lo:hi]
        clean_ratio = 1.0 - (neighbor_mask.sum() / len(neighbor_mask))
        score_neighbors = 0.3 * clean_ratio

        # Factor 3: Change magnitude (0.0 - 0.3)
        if np.isfinite(orig_values[idx]) and np.isfinite(clean_values[idx]):
            change = abs(orig_values[idx] - clean_values[idx])
            relative_change = change / global_std
            score_change = 0.3 * max(0.0, 1.0 - relative_change / 10.0)
        else:
            score_change = 0.1

        confidence.iloc[idx] = round(
            min(1.0, score_agreement + score_neighbors + score_change), 3,
        )

    mean_conf = float(confidence.iloc[fault_indices].mean())
    logger.info(
        "Repair confidence: %d points scored, mean=%.3f, min=%.3f",
        len(fault_indices), mean_conf, float(confidence.iloc[fault_indices].min()),
    )
    return confidence


def verify_repair(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    fault_mask: pd.Series,
) -> dict:
    """
    Post-repair quality check.

    Checks:
      - Were new NaN values introduced?
      - Were new Inf values introduced?
      - Did variance explode?
      - Are repaired points within a reasonable range?

    Args:
        original: Original (corrupted) DataFrame.
        cleaned: Cleaned DataFrame.
        fault_mask: Boolean anomaly mask.

    Returns:
        Dict with keys: passed, issues, new_nan_count, new_inf_count,
        variance_ratio, out_of_range_repairs.
    """
    issues: list[str] = []

    orig_values = original["value"].values.astype(np.float64)
    clean_values = cleaned["value"].values.astype(np.float64)
    fault_indices = np.where(fault_mask.values)[0]

    # Check 1: New NaN
    orig_nan = int(np.isnan(orig_values).sum())
    clean_nan = int(np.isnan(clean_values).sum())
    new_nan = max(0, clean_nan - orig_nan)
    if new_nan > 0:
        issues.append(f"Repair introduced {new_nan} new NaN values")

    # Check 2: New Inf
    orig_inf = int(np.isinf(orig_values).sum())
    clean_inf = int(np.isinf(clean_values).sum())
    new_inf = max(0, clean_inf - orig_inf)
    if new_inf > 0:
        issues.append(f"Repair introduced {new_inf} new Inf values")

    # Check 3: Variance explosion
    finite_orig = orig_values[np.isfinite(orig_values)]
    finite_clean = clean_values[np.isfinite(clean_values)]

    if len(finite_orig) > 1 and len(finite_clean) > 1:
        orig_var = float(np.var(finite_orig))
        clean_var = float(np.var(finite_clean))
        variance_ratio = clean_var / max(orig_var, 1e-12)
        if variance_ratio > 10.0:
            issues.append(f"Variance increased {variance_ratio:.1f}x after repair")
    else:
        variance_ratio = 1.0

    # Check 4: Out-of-range repairs
    out_of_range = 0
    if len(finite_clean) > 1 and len(fault_indices) > 0:
        clean_mean = float(np.nanmean(finite_clean))
        clean_std = float(np.nanstd(finite_clean))
        threshold = clean_mean + 10 * clean_std

        for idx in fault_indices:
            if idx < len(clean_values) and np.isfinite(clean_values[idx]):
                if abs(clean_values[idx] - clean_mean) > threshold:
                    out_of_range += 1

        if out_of_range > 0:
            issues.append(f"{out_of_range} repaired points still out of range")

    passed = len(issues) == 0

    logger.info(
        "Repair verification: %s (%d issues, %d new NaN, %d new Inf)",
        "PASSED" if passed else "FAILED", len(issues), new_nan, new_inf,
    )

    return {
        "passed": passed,
        "issues": issues,
        "new_nan_count": new_nan,
        "new_inf_count": new_inf,
        "variance_ratio": round(variance_ratio, 4),
        "out_of_range_repairs": out_of_range,
    }


def validate_sampling_rate(
    df: pd.DataFrame,
    expected_interval_seconds: float | None = None,
    max_jitter_ratio: float = 0.5,
) -> dict:
    """
    Check timestamp sampling-rate consistency.

    Detects jitter, large gaps, and unexpected compression in the
    time axis of the signal.

    Args:
        df: DataFrame with a 'timestamp' column.
        expected_interval_seconds: Expected interval. Estimated from
            the median if *None*.
        max_jitter_ratio: Acceptable std/median ratio.

    Returns:
        Dict with keys: valid, issues, detected_interval,
        expected_interval, jitter_ratio, n_irregular, irregular_ratio,
        n_large_gaps, n_compressed.
    """
    issues: list[str] = []
    result: dict = {
        "valid": True,
        "issues": issues,
        "detected_interval": 0.0,
        "expected_interval": 0.0,
        "jitter_ratio": 0.0,
        "n_irregular": 0,
        "irregular_ratio": 0.0,
        "n_large_gaps": 0,
        "n_compressed": 0,
    }

    if "timestamp" not in df.columns:
        issues.append("No timestamp column found")
        result["valid"] = False
        return result

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        issues.append("Timestamp column is not datetime type")
        result["valid"] = False
        return result

    if len(df) < 3:
        issues.append("Too few data points for sampling rate analysis")
        result["valid"] = False
        return result

    intervals = df["timestamp"].diff().dt.total_seconds().dropna().values

    if len(intervals) == 0:
        issues.append("Could not compute intervals")
        result["valid"] = False
        return result

    median_interval = float(np.median(intervals))
    if expected_interval_seconds is None:
        expected_interval_seconds = median_interval

    result["detected_interval"] = round(median_interval, 4)
    result["expected_interval"] = round(expected_interval_seconds, 4)

    # Jitter
    jitter = float(np.std(intervals)) / median_interval if median_interval > 0 else 0.0
    result["jitter_ratio"] = round(jitter, 4)
    if jitter > max_jitter_ratio:
        issues.append(f"High jitter: {jitter:.2%} (threshold: {max_jitter_ratio:.0%})")

    # Irregular intervals (>150% of median deviation)
    tolerance = median_interval * 1.5
    irregular = np.abs(intervals - median_interval) > tolerance
    n_irregular = int(irregular.sum())
    result["n_irregular"] = n_irregular
    result["irregular_ratio"] = round(n_irregular / len(intervals), 4) if len(intervals) > 0 else 0.0
    if result["irregular_ratio"] > 0.1:
        issues.append(f"{n_irregular} irregular intervals ({result['irregular_ratio']:.1%})")

    # Large gaps (>10x median)
    large_gap_threshold = median_interval * 10
    n_large_gaps = int((intervals > large_gap_threshold).sum())
    result["n_large_gaps"] = n_large_gaps
    if n_large_gaps > 0:
        issues.append(f"{n_large_gaps} large gaps (>{large_gap_threshold:.1f}s)")

    # Compressed intervals (<20% of median)
    if median_interval > 0:
        n_compressed = int((intervals < median_interval * 0.2).sum())
    else:
        n_compressed = 0
    result["n_compressed"] = n_compressed
    if n_compressed > 0:
        issues.append(f"{n_compressed} compressed intervals (unexpectedly frequent)")

    result["valid"] = len(issues) == 0

    logger.info(
        "Sampling rate: interval=%.2fs, jitter=%.2f%%, irregular=%d, gaps=%d, compressed=%d — %s",
        median_interval, jitter * 100, n_irregular, n_large_gaps, n_compressed,
        "VALID" if result["valid"] else "ISSUES FOUND",
    )
    return result
