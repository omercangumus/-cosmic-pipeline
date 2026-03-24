"""Pipeline orchestrator: ingestion -> detection -> ensemble -> filtering -> validation."""

import logging
import time

import pandas as pd

from pipeline.detector_classic import detect_all as detect_classic
from pipeline.ensemble import ensemble_vote
from pipeline.filters_classic import apply_classic_filters
from pipeline.ingestion import load_data, preprocess, validate_schema
from pipeline.validator import calculate_metrics, validate_output

logger = logging.getLogger(__name__)

VALID_METHODS = ("classic", "ml", "both")


def run_pipeline(
    df: pd.DataFrame,
    config: dict | None = None,
    method: str = "classic",
) -> dict:
    """
    Main pipeline entry point.

    Args:
        df: Corrupted telemetry data (columns: timestamp, value).
        config: Configuration dict (from config/default.yaml).
                If None, uses sensible defaults.
        method: Detection/filtering method — 'classic', 'ml', or 'both'.
                'both' runs classic and ML detectors, ensembles them,
                and uses ML filtering with classic fallback.

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

    # --- 2. Detection ---
    logger.info("Step 2: Anomaly detection (method=%s)", method)
    detector_masks: dict[str, pd.Series] = {}

    if method in ("classic", "both"):
        dsp_cfg = config.get("dsp_detector", {})
        classic_masks = detect_classic(
            data,
            zscore_threshold=dsp_cfg.get("zscore_threshold", 3.5),
            iqr_multiplier=dsp_cfg.get("iqr_multiplier", 1.5),
            window=dsp_cfg.get("window", 50),
            window_threshold=dsp_cfg.get("window_threshold", 3.0),
            max_gap_seconds=dsp_cfg.get("max_gap_seconds", 60),
        )
        detector_masks.update(classic_masks)

    if method in ("ml", "both"):
        from pipeline.detector_ml import detect_all_ml

        ml_cfg = config.get("lstm_detector", {})
        ml_masks = detect_all_ml(
            data,
            model_path=ml_cfg.get("model_path", "models/lstm_ae.pt"),
            contamination=config.get("dsp_detector", {}).get("iforest_contamination", 0.05),
            threshold_percentile=ml_cfg.get("threshold_percentile", 95.0),
        )
        detector_masks.update(ml_masks)

    # --- 3. Ensemble ---
    logger.info("Step 3: Ensemble voting (%d detectors)", len(detector_masks))
    ens_cfg = config.get("ensemble", {})
    mask_list = list(detector_masks.values())
    fault_mask = ensemble_vote(
        mask_list,
        strategy="majority",
        min_agreement=ens_cfg.get("min_agreement"),
    )

    # --- 4. Filtering ---
    logger.info("Step 4: Filtering (method=%s)", method)
    if method in ("ml", "both"):
        cleaned_data = _apply_ml_filter(data, fault_mask, config)
        # Fill any remaining NaN gaps with classic interpolation
        remaining_nan = cleaned_data["value"].isna()
        if remaining_nan.any():
            from pipeline.filters_classic import interpolate_gaps
            cleaned_data = interpolate_gaps(cleaned_data, remaining_nan)
    else:
        flt_cfg = config.get("classic_filter", {})
        cleaned_data = apply_classic_filters(
            data,
            fault_mask,
            median_window=flt_cfg.get("median_window", 5),
            sg_window=flt_cfg.get("sg_window", 11),
            sg_polyorder=flt_cfg.get("sg_polyorder", 3),
            wavelet_family=flt_cfg.get("wavelet_family", "db4"),
            wavelet_level=flt_cfg.get("wavelet_level", 3),
        )

    # --- 5. Validation ---
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


def _apply_ml_filter(
    data: pd.DataFrame,
    fault_mask: pd.Series,
    config: dict,
) -> pd.DataFrame:
    """Apply ML-based filtering with interpolation fallback."""
    from pipeline.filters_ml import reconstruct_with_lstm

    ml_cfg = config.get("ml_reconstructor", {})
    return reconstruct_with_lstm(
        data,
        fault_mask,
        model_path=config.get("lstm_detector", {}).get("model_path", "models/lstm_ae.pt"),
        blend_window=ml_cfg.get("blend_window", 10),
    )


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
        DataFrame with columns [timestamp, fault_type, severity].
    """
    rows: list[dict] = []
    fault_indices = combined_mask[combined_mask].index

    for idx in fault_indices:
        types = [name for name, mask in detector_masks.items() if mask.iloc[idx]]
        fault_type = "+".join(types) if types else "unknown"
        severity = len(types) / len(detector_masks)
        ts = data["timestamp"].iloc[idx] if "timestamp" in data.columns else idx

        rows.append({
            "timestamp": ts,
            "fault_type": fault_type,
            "severity": round(severity, 2),
        })

    return pd.DataFrame(rows, columns=["timestamp", "fault_type", "severity"])
