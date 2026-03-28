"""Pipeline orchestrator: ingestion -> detection -> hybrid ensemble -> filtering -> validation."""

import copy
import logging
import time

import numpy as np
import pandas as pd

from pipeline.detector_classic import detect_all as detect_classic
from pipeline.ensemble import hybrid_majority_vote
from pipeline.filters_classic import apply_classic_filters
from pipeline.ingestion import load_data, preprocess, validate_schema
from pipeline.tracer import PipelineTracer
from pipeline.validator import calculate_metrics, validate_output

logger = logging.getLogger(__name__)

VALID_METHODS = ("classic", "ml", "both")

# Detectors whose output is treated as "hard" (any-True → anomaly)
HARD_DETECTORS = {"gaps", "range", "delta", "flatline", "duplicates"}

DEFAULT_CONFIG = {
    "dsp_detector": {
        "zscore_threshold": 2.0,
        "window": 50,
        "window_threshold": 3.0,
        "max_gap_seconds": 60,
        "range_std_multiplier": 10.0,
        "delta_multiplier": 5.0,
        "iforest_contamination": 0.05,
    },
    "lstm_detector": {
        "model_path": "models/lstm_ae.pt",
        "threshold_percentile": 95.0,
    },
    "ensemble": {
        "min_agreement": 2,
    },
    "classic_filter": {
        "median_window": 5,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


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

    config = _deep_merge(DEFAULT_CONFIG, config or {})

    t_start = time.perf_counter()
    tracer = PipelineTracer()

    # --- 1. Ingestion ---
    logger.info("Step 1: Ingestion")
    data = load_data(df)
    validate_schema(data)
    data = preprocess(data)
    tracer.snapshot("Veri Alimi", "Yukleme, sema kontrolu, on isleme", df, data)

    # Sampling rate validation
    from pipeline.validator import validate_sampling_rate
    sampling_info = validate_sampling_rate(data)
    if sampling_info["issues"]:
        logger.warning("Sampling rate issues: %s", ", ".join(sampling_info["issues"]))

    # --- 2. Detrend for detection (does NOT modify `data`) ---
    from pipeline.filters_classic import detrend_signal

    data_detrended = detrend_signal(data)
    tracer.snapshot("Trend Kaldirma", "Lineer trend (TID drift) cikarildi", data, data_detrended)

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
        for det_name, det_mask in classic_masks.items():
            tracer.snapshot_detection(
                f"Detektor: {det_name}", f"{det_name} anomali taramasi",
                det_mask, data, detector_name=det_name,
            )

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
        for det_name, det_mask in ml_masks.items():
            tracer.snapshot_detection(
                f"Detektor: {det_name}", f"{det_name} ML anomali taramasi",
                det_mask, data, detector_name=det_name,
            )

    # --- detector counts for dashboard ---
    detector_counts = {name: int(mask.sum()) for name, mask in detector_masks.items()}
    for name, count in detector_counts.items():
        logger.info("Detector '%s': %d anomalies", name, count)

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
        for k in ["gaps", "range", "delta", "flatline", "duplicates"] if k in detector_masks
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

    _soft_only = max(0, int(fault_mask.sum()) - layer1_count)
    tracer.snapshot_ensemble(
        hard_count=layer1_count, soft_count=_soft_only,
        total_count=int(fault_mask.sum()),
    )

    # --- 5. Repair eligibility (BEFORE filtering) ---
    faults_detected = int(fault_mask.sum())
    fault_timeline = _build_fault_timeline(data, detector_masks, fault_mask)

    from pipeline.validator import assess_repair_eligibility
    fault_timeline = assess_repair_eligibility(data, fault_mask, fault_timeline)

    # Remove preserve/flag_only points from fault_mask before filtering
    repair_mask = fault_mask.copy()
    if not fault_timeline.empty and "repair_decision" in fault_timeline.columns:
        skip_indices: set[int] = set()
        for _, row in fault_timeline.iterrows():
            if row.get("repair_decision") in ("preserve", "flag_only"):
                ts = row["timestamp"]
                matching = data.index[data["timestamp"] == ts]
                skip_indices.update(matching.tolist())
        if skip_indices:
            for idx in skip_indices:
                if idx in repair_mask.index:
                    repair_mask.iloc[idx] = False
            logger.info(
                "Repair decision: %d points skipped (preserve/flag_only)",
                len(skip_indices),
            )

    # --- 6. Filtering (only on repairable points) ---
    logger.info("Step 4: Filtering (method=%s)", method)
    flt_cfg = config.get("classic_filter", {})
    cleaned_data = apply_classic_filters(
        data,
        repair_mask,
        median_window=flt_cfg.get("median_window", 5),
    )

    tracer.snapshot("Filtreleme", "Interpolation -> Detrend -> Median (sadece repair noktalar)", data, cleaned_data)

    # --- 7. Validation ---
    logger.info("Step 5: Validation")
    validate_output(cleaned_data)
    tracer.snapshot("Dogrulama", "Temizlenmis sinyal kalite kontrolu", cleaned_data, cleaned_data)

    processing_time = time.perf_counter() - t_start
    faults_corrected = int(repair_mask.sum())

    # Repair confidence scoring
    from pipeline.validator import calculate_repair_confidence, verify_repair
    repair_confidence = calculate_repair_confidence(
        data, cleaned_data, repair_mask, detector_masks,
    )

    # Repair verification
    repair_verification = verify_repair(data, cleaned_data, repair_mask)

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
        "detector_counts": detector_counts,
        "repair_confidence": repair_confidence,
        "repair_verification": repair_verification,
        "sampling_info": sampling_info,
        "tracer_table": tracer.to_dataframe(),
        "tracer_summary": tracer.to_summary(),
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

        if any(t in ("gaps", "range", "delta", "flatline", "duplicates") for t in types):
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


# ── Multi-channel wrapper ─────────────────────────────────────────────────────

_TIME_ALIASES = ("timestamp", "time_tag", "date", "datetime", "time", "ds")


def run_pipeline_multi(
    df: pd.DataFrame,
    config: dict | None = None,
    method: str = "classic",
    columns: list[str] | None = None,
) -> dict:
    """
    Multi-channel pipeline: run pipeline separately for each numeric column.

    Args:
        df: DataFrame with a timestamp column + N numeric columns.
        config: Pipeline configuration dict.
        method: Detection method — 'classic', 'ml', or 'both'.
        columns: Specific columns to process. If None, all numeric columns.

    Returns:
        {
            'channels': {col_name: run_pipeline() result, ...},
            'summary': {
                'total_channels': int,
                'total_faults': int,
                'processing_time': float,
                'per_channel': {col: faults_detected, ...},
            },
        }
    """
    t_start = time.perf_counter()

    # Resolve timestamp column
    time_col = None
    for alias in _TIME_ALIASES:
        if alias in df.columns:
            time_col = alias
            break

    if time_col is None:
        df = df.copy()
        df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="1s")
        time_col = "timestamp"

    # Resolve numeric columns
    exclude = {time_col, "label", "timestamp"}
    available = [c for c in df.select_dtypes(include=["number"]).columns if c not in exclude]

    if columns:
        numeric_cols = [c for c in columns if c in available]
        if not numeric_cols:
            raise ValueError(f"None of {columns} found in numeric columns: {available}")
    else:
        numeric_cols = available

    if not numeric_cols:
        raise ValueError("No numeric columns found for pipeline processing")

    # Single-column shortcut
    if len(numeric_cols) == 1:
        col = numeric_cols[0]
        single_df = df[[time_col, col]].rename(
            columns={time_col: "timestamp", col: "value"},
        )
        result = run_pipeline(single_df, config=config, method=method)
        return {
            "channels": {col: result},
            "summary": {
                "total_channels": 1,
                "total_faults": result["metrics"]["faults_detected"],
                "processing_time": time.perf_counter() - t_start,
                "per_channel": {col: result["metrics"]["faults_detected"]},
            },
        }

    # Multi-channel: run pipeline per column
    channels: dict[str, dict] = {}
    total_faults = 0
    per_channel: dict[str, int] = {}

    for col in numeric_cols:
        logger.info("Processing channel: %s", col)
        channel_df = df[[time_col, col]].copy()
        channel_df.columns = ["timestamp", "value"]

        try:
            result = run_pipeline(channel_df, config=config, method=method)
            channels[col] = result
            faults = result["metrics"]["faults_detected"]
            total_faults += faults
            per_channel[col] = faults
        except Exception as e:
            logger.error("Channel %s failed: %s", col, e)
            channels[col] = {"error": str(e)}
            per_channel[col] = -1

    return {
        "channels": channels,
        "summary": {
            "total_channels": len(numeric_cols),
            "total_faults": total_faults,
            "processing_time": time.perf_counter() - t_start,
            "per_channel": per_channel,
        },
    }
