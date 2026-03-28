"""Export pipeline metrics in various formats."""

import json

import pandas as pd


def export_metrics(result: dict, fmt: str = "json") -> str:
    """
    Export pipeline result metrics as JSON or CSV string.

    Args:
        result: Pipeline result dict from run_pipeline().
        fmt: Output format — 'json' or 'csv'.

    Returns:
        Formatted string.
    """
    metrics = result.get("metrics", {})
    detector_counts = result.get("detector_counts", {})

    if fmt == "json":
        export_data = {**metrics, "detector_counts": detector_counts}
        return json.dumps(export_data, indent=2, default=str)

    if fmt == "csv":
        flat = {**metrics}
        for k, v in detector_counts.items():
            flat[f"detector_{k}"] = v
        return pd.DataFrame([flat]).to_csv(index=False)

    raise ValueError(f"Desteklenmeyen format: {fmt}")
