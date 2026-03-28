"""Pipeline orchestrator: ingestion -> detection -> hybrid ensemble -> filtering -> validation."""

import logging
import time

import numpy as np
import pandas as pd

from pipeline.detector_classic import (
    detect_all as detect_classic,
    detect_gaps,
    detect_range_violation,
    detect_outliers_zscore,
    detect_sliding_window,
)
from pipeline.ensemble import hybrid_majority_vote
from pipeline.filters_classic import apply_classic_filters
from pipeline.ingestion import load_data, preprocess, validate_schema
from pipeline.validator import calculate_metrics, validate_output

logger = logging.getLogger(__name__)

VALID_METHODS = ("classic", "ml", "both")

# Detectors whose output is treated as "hard" (any-True → anomaly)
HARD_DETECTORS = {"gaps", "range", "delta"}


def run_pipeline(
    df: pd.DataFrame,
    config: dict | None = None,
    method: str = "classic",
) -> dict:
    """
    Main pipeline entry point.

    Args:
        df: Corrupted telemetry data (columns: timestamp, value).
        config: Configuration dict. If None, uses sensible defaults.
        method: Detection method — 'classic', 'ml', or 'both'.

    Returns:
        {
            'cleaned_data': pd.DataFrame,
            'fault_mask': pd.Series,
            'metrics': {
                'faults_detected': int,
                'faults_corrected': int,
                'processing_time': float,
            },
            'fault_timeline': pd.DataFrame,
        }
    """
    if method not in VALID_METHODS:
        raise ValueError(f"Invalid method '{method}'. Must be one of {VALID_METHODS}")

    if config is None:
        config = {}

    t_start = time.perf_counter()

    # --- 1. Ingestion ---
    logger.info("Step 1: Ingestion")
    data = load_data(df)
    validate_schema(data)
    data = preprocess(data)

    # --- 2. Detrend for detection (does NOT modify `data`) ---
    from pipeline.filters_classic import detrend_signal

    data_detrended = detrend_signal(data)

    # --- 3. Detection ---
    logger.info("Step 2: Anomaly detection (method=%s)", method)
    detector_masks: dict[str, pd.Series] = {}

    if method in ("classic", "both"):
        dsp_cfg = config.get("dsp_detector", {})
        classic_masks = detect_classic(
            data_detrended,
            zscore_threshold=dsp_cfg.get("zscore_threshold", 2.0),
            window=dsp_cfg.get("window", 50),
            window_threshold=dsp_cfg.get("window_threshold", 3.0),
            max_gap_seconds=dsp_cfg.get("max_gap_seconds", 60),
            range_std_multiplier=dsp_cfg.get("range_std_multiplier", 10.0),
            delta_multiplier=dsp_cfg.get("delta_multiplier", 5.0),
            df_original=data,
        )
        detector_masks.update(classic_masks)

    if method in ("ml", "both"):
        from pipeline.detector_ml import detect_all_ml

        ml_cfg = config.get("lstm_detector", {})
        ml_masks = detect_all_ml(
            data_detrended,
            model_path=ml_cfg.get("model_path", "models/lstm_ae.pt"),
            contamination=config.get("dsp_detector", {}).get("iforest_contamination", 0.05),
            threshold_percentile=ml_cfg.get("threshold_percentile", 95.0),
        )
        detector_masks.update(ml_masks)

    # --- 4. Hybrid Majority Ensemble ---
    logger.info("Step 3: Hybrid ensemble voting (%d detectors)", len(detector_masks))
    ens_cfg = config.get("ensemble", {})

    hard_masks = [m for k, m in detector_masks.items() if k in HARD_DETECTORS]
    soft_masks = [m for k, m in detector_masks.items() if k not in HARD_DETECTORS]

    # Fallback: if only soft or only hard, adapt
    if not hard_masks and not soft_masks:
        # Should not happen, but guard against it
        fault_mask = pd.Series(np.zeros(len(data), dtype=bool), index=data.index)
    elif not soft_masks:
        # Only hard detectors — union them
        fault_mask = hybrid_majority_vote(hard_masks, [], min_agreement=1)
    else:
        fault_mask = hybrid_majority_vote(
            hard_masks,
            soft_masks,
            min_agreement=ens_cfg.get("min_agreement", 2),
        )

    # Log pyramid layers
    _empty = pd.Series(dtype=bool)
    layer1_count = sum(
        int(detector_masks.get(k, _empty).sum())
        for k in ["gaps", "range", "delta"] if k in detector_masks
    )
    layer2_count = (
        int(detector_masks.get("zscore", _empty).sum())
        + int(detector_masks.get("sliding_window", _empty).sum())
    )
    layer3_count = int(detector_masks.get("isolation_forest", _empty).sum()) if "isolation_forest" in detector_masks else 0
    layer4_count = int(detector_masks.get("lstm_ae", _empty).sum()) if "lstm_ae" in detector_masks else 0

    logger.info(
        "Pyramid detection — L1(hard): %d, L2(statistical): %d, L3(ML): %d, L4(temporal): %d",
        layer1_count, layer2_count, layer3_count, layer4_count,
    )

    # --- 5. Filtering ---
    logger.info("Step 4: Filtering (method=%s)", method)
    flt_cfg = config.get("classic_filter", {})
    cleaned_data = apply_classic_filters(
        data,
        fault_mask,
        median_window=flt_cfg.get("median_window", 5),
    )

    # --- 6. Validation ---
    logger.info("Step 5: Validation")
    validate_output(cleaned_data)

    processing_time = time.perf_counter() - t_start
    faults_detected = int(fault_mask.sum())
    faults_corrected = faults_detected

    fault_timeline = _build_fault_timeline(data, detector_masks, fault_mask)

    logger.info(
        "Pipeline complete (method=%s): %d faults detected, %.3fs elapsed",
        method, faults_detected, processing_time,
    )

    return {
        "cleaned_data": cleaned_data,
        "fault_mask": fault_mask,
        "metrics": {
            "faults_detected": faults_detected,
            "faults_corrected": faults_corrected,
            "processing_time": processing_time,
        },
        "fault_timeline": fault_timeline,
    }


def _build_fault_timeline(
    data: pd.DataFrame,
    detector_masks: dict[str, pd.Series],
    combined_mask: pd.Series,
) -> pd.DataFrame:
    """
    Build a timeline of detected faults with type and severity.

    Args:
        data: Original preprocessed DataFrame.
        detector_masks: Individual detector results.
        combined_mask: Ensemble-voted mask.

    Returns:
        DataFrame with columns [timestamp, fault_type, severity, reason].
    """
    rows: list[dict] = []
    fault_indices = combined_mask[combined_mask].index

    for idx in fault_indices:
        types = [name for name, mask in detector_masks.items() if mask.iloc[idx]]
        fault_type = "+".join(types) if types else "unknown"
        severity = len(types) / max(len(detector_masks), 1)
        ts = data["timestamp"].iloc[idx] if "timestamp" in data.columns else idx

        if any(t in ("gaps", "range", "delta") for t in types):
            reason = "hard_rule"
        elif any(t in ("zscore", "sliding_window") for t in types):
            reason = "statistical"
        elif any(t == "isolation_forest" for t in types):
            reason = "ml_outlier"
        elif any(t == "lstm_ae" for t in types):
            reason = "temporal_pattern"
        else:
            reason = "unknown"

        rows.append({
            "timestamp": ts,
            "fault_type": fault_type,
            "severity": round(severity, 2),
            "reason": reason,
        })

    return pd.DataFrame(rows, columns=["timestamp", "fault_type", "severity", "reason"])
