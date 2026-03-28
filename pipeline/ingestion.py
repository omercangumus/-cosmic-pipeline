"""Data ingestion and preprocessing for the cosmic pipeline."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"timestamp", "value"}


def load_data(source: str | pd.DataFrame) -> pd.DataFrame:
    """
    Load telemetry data from CSV path, netCDF4 path, or existing DataFrame.

    Args:
        source: File path (CSV or netCDF4) or DataFrame.

    Returns:
        DataFrame with at least [timestamp, value] columns.

    Raises:
        FileNotFoundError: If file path does not exist.
        ValueError: If file format is unsupported.
    """
    if isinstance(source, pd.DataFrame):
        logger.info("Received DataFrame with %d rows", len(source))
        return source.copy()

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
        logger.info("Loaded CSV %s with %d rows", path.name, len(df))
        return df

    if suffix in (".nc", ".nc4", ".netcdf"):
        return _load_netcdf(path)

    raise ValueError(f"Unsupported file format: {suffix}")


def _load_netcdf(path: Path) -> pd.DataFrame:
    """
    Load a netCDF4 file and convert to standard DataFrame.

    Args:
        path: Path to the netCDF4 file.

    Returns:
        DataFrame with [timestamp, value] columns.
    """
    try:
        import netCDF4 as nc
    except ImportError:
        raise ImportError("netCDF4 package required for .nc files: pip install netCDF4")

    ds = nc.Dataset(str(path), "r")
    try:
        # Find time variable
        time_var = None
        for name in ("time", "timestamp", "Time", "epoch"):
            if name in ds.variables:
                time_var = name
                break

        if time_var is None:
            raise ValueError(f"No time variable found in {path.name}")

        time_data = ds.variables[time_var][:]

        # Find first numeric data variable (skip time/coordinate vars)
        skip = {time_var, "lat", "latitude", "lon", "longitude"}
        value_var = None
        for name, var in ds.variables.items():
            if name.lower() not in skip and var.ndim >= 1:
                value_var = name
                break

        if value_var is None:
            raise ValueError(f"No data variable found in {path.name}")

        values = ds.variables[value_var][:].flatten()

        # Convert time: try num2date, fallback to raw numeric
        try:
            time_unit = ds.variables[time_var].units
            calendar = getattr(ds.variables[time_var], "calendar", "standard")
            timestamps = nc.num2date(time_data, time_unit, calendar=calendar)
            timestamps = pd.DatetimeIndex(timestamps)
        except (AttributeError, ValueError):
            timestamps = pd.to_datetime(time_data, unit="s", origin="unix")

        n = min(len(timestamps), len(values))
        df = pd.DataFrame({
            "timestamp": timestamps[:n],
            "value": values[:n].astype(np.float64),
        })

        logger.info("Loaded netCDF4 %s (%s) with %d rows", path.name, value_var, n)
        return df
    finally:
        ds.close()


def validate_schema(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has the required columns and types.

    Args:
        df: Input DataFrame.

    Returns:
        True if schema is valid.

    Raises:
        ValueError: If required columns are missing or data is invalid.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("DataFrame is empty")

    if not pd.api.types.is_numeric_dtype(df["value"]):
        raise ValueError(f"'value' column must be numeric, got {df['value'].dtype}")

    logger.debug("Schema validation passed: %d rows, columns=%s", len(df), list(df.columns))
    return True


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess telemetry data: parse timestamps, sort, handle duplicates.

    Args:
        df: Raw DataFrame with [timestamp, value] columns.

    Returns:
        Preprocessed DataFrame sorted by timestamp with parsed datetime index.
    """
    out = df.copy()

    # Parse timestamps if not already datetime
    if not pd.api.types.is_datetime64_any_dtype(out["timestamp"]):
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        coerced_nans = out["timestamp"].isna().sum()
        if coerced_nans > 0:
            logger.warning("Dropped %d rows with unparseable timestamps", coerced_nans)
            out = out.dropna(subset=["timestamp"])

    # Sort by timestamp
    out = out.sort_values("timestamp").reset_index(drop=True)

    # Remove duplicate timestamps (keep first)
    dup_count = out["timestamp"].duplicated().sum()
    if dup_count > 0:
        logger.warning("Removed %d duplicate timestamps", dup_count)
        out = out.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)

    # Ensure value is float64
    out["value"] = out["value"].astype(np.float64)

    logger.info("Preprocessed: %d rows, time range %s → %s",
                len(out), out["timestamp"].iloc[0], out["timestamp"].iloc[-1])
    return out
