"""Convert SMD (Server Machine Dataset) CSV to pipeline-compatible format.

SMD has integer timestamps (0, 1, 2, ...), 38 numeric columns, and a label column.
This script converts a selected column to the pipeline's [timestamp, value] format
and optionally extracts ground truth labels for evaluation.

Usage:
    python scripts/convert_smd.py                          # default: cpu_r
    python scripts/convert_smd.py --column mem_u
    python scripts/convert_smd.py --column load_1 --output data/test_samples/smd_load1.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INPUT = "data/test_samples/smd_cleaned_named.csv"
DEFAULT_COLUMN = "cpu_r"
DEFAULT_START = "2024-01-01"


def convert_smd(
    input_path: str,
    column: str = DEFAULT_COLUMN,
    output_path: str | None = None,
    start_time: str = DEFAULT_START,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convert an SMD CSV to pipeline format.

    Args:
        input_path: Path to smd_cleaned_named.csv.
        column: Numeric column to use as 'value'.
        output_path: If given, write the result CSV here.
        start_time: Datetime origin for the generated timestamps.

    Returns:
        (pipeline_df, labels) where pipeline_df has [timestamp, value]
        and labels is the original label column (1 = anomaly).
    """
    df = pd.read_csv(input_path)
    logger.info("Loaded %s: %d rows, %d columns", input_path, len(df), len(df.columns))

    if column not in df.columns:
        available = [c for c in df.columns if c not in ("timestamp", "label")]
        raise ValueError(
            f"Column '{column}' not found. Available: {', '.join(available)}"
        )

    # Integer timestamp → datetime (1-second intervals)
    pipeline_df = pd.DataFrame({
        "timestamp": pd.date_range(start_time, periods=len(df), freq="1s"),
        "value": df[column].astype(float),
    })

    labels = df["label"].copy() if "label" in df.columns else pd.Series(dtype=int)

    logger.info(
        "Converted column '%s': range=[%.4f, %.4f], NaN=%d",
        column, pipeline_df["value"].min(), pipeline_df["value"].max(),
        pipeline_df["value"].isna().sum(),
    )

    if labels.any():
        n_anomaly = int((labels == 1).sum())
        logger.info("Ground truth: %d anomalies (%.1f%%)", n_anomaly, 100 * n_anomaly / len(labels))

    if output_path:
        pipeline_df.to_csv(output_path, index=False)
        labels.to_csv(output_path.replace(".csv", "_labels.csv"), index=False, header=True)
        logger.info("Saved to %s", output_path)

    return pipeline_df, labels


def main():
    parser = argparse.ArgumentParser(description="Convert SMD CSV to pipeline format")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input SMD CSV path")
    parser.add_argument("--column", default=DEFAULT_COLUMN, help="Column to extract as value")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--start", default=DEFAULT_START, help="Start datetime for timestamps")
    parser.add_argument("--list-columns", action="store_true", help="List available columns and exit")
    args = parser.parse_args()

    if args.list_columns:
        df = pd.read_csv(args.input, nrows=0)
        cols = [c for c in df.columns if c not in ("timestamp", "label")]
        print(f"Available columns ({len(cols)}):")
        for c in cols:
            print(f"  {c}")
        return

    convert_smd(args.input, args.column, args.output, args.start)


if __name__ == "__main__":
    main()
