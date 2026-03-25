"""Benchmark: generate synthetic data, run pipeline in 3 modes, compare metrics."""

import logging
import sys
import time
import traceback

import numpy as np
import pandas as pd

from data.synthetic_generator import generate_corrupted_dataset
from pipeline.orchestrator import run_pipeline
from pipeline.validator import calculate_metrics

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)

N_SAMPLES = 10_000
SEED = 42
METHODS = ["classic", "ml", "both"]


def ground_truth_to_bool_mask(gt_mask: dict, n: int) -> np.ndarray:
    """Convert ground truth dict to boolean array for precision/recall."""
    mask = np.zeros(n, dtype=bool)

    for idx in gt_mask.get("seu", []):
        if 0 <= idx < n:
            mask[idx] = True

    for start, end in gt_mask.get("gap", []):
        mask[start:min(end, n)] = True

    # Noise indices where std > 1.0
    for idx in gt_mask.get("noise", []):
        if 0 <= idx < n:
            mask[idx] = True

    # TID drift affects entire signal — but it's a gradual effect,
    # so we mark indices where drift magnitude exceeds a threshold
    # (roughly the latter 60% of the signal where drift is significant)
    tid_indices = gt_mask.get("tid", [])
    if tid_indices:
        start_idx = int(n * 0.4)
        for idx in range(start_idx, n):
            mask[idx] = True

    return mask


def precision_recall(detected: np.ndarray, truth: np.ndarray):
    """Compute precision, recall, F1 from boolean masks."""
    tp = np.sum(detected & truth)
    fp = np.sum(detected & ~truth)
    fn = np.sum(~detected & truth)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def main():
    print("=" * 72)
    print("  COSMIC PIPELINE BENCHMARK — Synthetic Data, 3 Modes")
    print("=" * 72)

    # --- 1. Generate synthetic data ---
    print(f"\n[1] Generating synthetic data ({N_SAMPLES} samples, seed={SEED}) ...")
    clean_df, corrupted_df, gt_mask = generate_corrupted_dataset(n=N_SAMPLES, seed=SEED)
    gt_bool = ground_truth_to_bool_mask(gt_mask, N_SAMPLES)

    print(f"    Clean signal : {len(clean_df)} points")
    print(f"    Corrupted    : {len(corrupted_df)} points")
    print(f"    NaN count    : {corrupted_df['value'].isna().sum()}")
    print(f"    Ground truth anomalies: {gt_bool.sum()} / {N_SAMPLES}")
    print(f"    Fault types  : SEU={len(gt_mask['seu'])}, "
          f"TID=whole-signal, "
          f"Gaps={len(gt_mask['gap'])}, "
          f"Noise={len(gt_mask['noise'])}")

    # --- 2. Run pipeline in each mode ---
    results = {}
    errors = {}

    for method in METHODS:
        print(f"\n[2] Running pipeline: method='{method}' ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            result = run_pipeline(corrupted_df.copy(), method=method)
            elapsed = time.perf_counter() - t0
            results[method] = result
            results[method]["elapsed"] = elapsed
            print(f"OK ({elapsed:.3f}s)")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            errors[method] = traceback.format_exc()
            print(f"FAIL ({elapsed:.3f}s)")
            print(f"    ERROR: {e}")

    # --- 3. Compute metrics for each mode ---
    print("\n" + "=" * 72)
    print("  METRICS COMPARISON")
    print("=" * 72)

    header = f"{'Metric':<28} | {'classic':>12} | {'ml':>12} | {'both':>12}"
    sep = "-" * len(header)
    print(f"\n{header}")
    print(sep)

    all_metrics = {}
    for method in METHODS:
        if method not in results:
            all_metrics[method] = None
            continue

        r = results[method]
        cleaned = r["cleaned_data"]
        fault_mask = r["fault_mask"]

        # Quality metrics vs ground truth (clean signal)
        qm = calculate_metrics(corrupted_df, cleaned, ground_truth=clean_df)

        # Detection precision/recall
        detected_bool = fault_mask.values if hasattr(fault_mask, "values") else np.array(fault_mask)
        prec, rec, f1 = precision_recall(detected_bool, gt_bool)

        all_metrics[method] = {
            "faults_detected": r["metrics"]["faults_detected"],
            "processing_time": r["elapsed"],
            "rmse": qm["rmse"],
            "mae": qm["mae"],
            "r2_score": qm["r2_score"],
            "snr_db": qm["snr"],
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "quality_score": None,
        }

    def fmt(val, fmt_str=".4f"):
        if val is None:
            return "N/A"
        if isinstance(val, float):
            return f"{val:{fmt_str}}"
        return str(val)

    def row(label, key, fmt_str=".4f"):
        vals = []
        for m in METHODS:
            if all_metrics[m] is None:
                vals.append("ERROR")
            else:
                vals.append(fmt(all_metrics[m][key], fmt_str))
        print(f"{label:<28} | {vals[0]:>12} | {vals[1]:>12} | {vals[2]:>12}")

    row("Faults Detected", "faults_detected", "d")
    row("Processing Time (s)", "processing_time", ".3f")
    row("RMSE (vs clean)", "rmse")
    row("MAE  (vs clean)", "mae")
    row("R² Score", "r2_score")
    row("SNR (dB)", "snr_db", ".2f")
    row("Precision", "precision")
    row("Recall", "recall")
    row("F1 Score", "f1_score")
    print(sep)

    # --- 4. Best method summary ---
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    valid = {m: v for m, v in all_metrics.items() if v is not None}
    if valid:
        best_rmse = min(valid, key=lambda m: valid[m]["rmse"])
        best_snr = max(valid, key=lambda m: valid[m]["snr_db"])
        best_f1 = max(valid, key=lambda m: valid[m]["f1_score"])
        fastest = min(valid, key=lambda m: valid[m]["processing_time"])

        print(f"  Best RMSE     : {best_rmse:<10} ({valid[best_rmse]['rmse']:.4f})")
        print(f"  Best SNR      : {best_snr:<10} ({valid[best_snr]['snr_db']:.2f} dB)")
        print(f"  Best F1       : {best_f1:<10} ({valid[best_f1]['f1_score']:.4f})")
        print(f"  Fastest       : {fastest:<10} ({valid[fastest]['processing_time']:.3f}s)")

    # --- 5. Report errors ---
    if errors:
        print("\n" + "=" * 72)
        print("  ERRORS")
        print("=" * 72)
        for method, tb in errors.items():
            print(f"\n--- {method} ---")
            print(tb)

    print()
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
