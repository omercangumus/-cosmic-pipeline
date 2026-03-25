"""Data validation utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def validate_dataframe_schema(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if df is None or df.empty:
        issues.append("DataFrame is None or empty")
        return False, issues
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    return len(issues) == 0, issues


def validate_telemetry_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate telemetry data format and content.
    
    Args:
        df: Telemetry DataFrame
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check schema
    is_valid, schema_issues = validate_dataframe_schema(df, ['timestamp', 'value'])
    if not is_valid:
        return False, schema_issues
    
    # Check timestamp type
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        issues.append("timestamp column is not datetime type")
    
    # Check value type
    if not pd.api.types.is_numeric_dtype(df['value']):
        issues.append("value column is not numeric type")
    
    # Check for all NaN
    if df['value'].isna().all():
        issues.append("All values are NaN")
    
    # Check for constant signal
    if df['value'].nunique() == 1:
        issues.append("Signal is constant (all values are the same)")
    
    # Check timestamp monotonicity
    if not df['timestamp'].is_monotonic_increasing:
        issues.append("Timestamps are not monotonically increasing")
    
    # Check for duplicates
    if df['timestamp'].duplicated().any():
        issues.append(f"Found {df['timestamp'].duplicated().sum()} duplicate timestamps")
    
    return len(issues) == 0, issues


def validate_signal_quality(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate signal quality metrics.
    
    Args:
        df: Telemetry DataFrame
        
    Returns:
        Dictionary of quality metrics
    """
    quality = {}
    
    # NaN ratio
    quality['nan_ratio'] = df['value'].isna().sum() / len(df)
    
    # Infinite values
    quality['inf_count'] = np.isinf(df['value']).sum()
    
    # Value range
    valid_values = df['value'].dropna()
    if len(valid_values) > 0:
        quality['min_value'] = valid_values.min()
        quality['max_value'] = valid_values.max()
        quality['mean_value'] = valid_values.mean()
        quality['std_value'] = valid_values.std()
        quality['value_range'] = quality['max_value'] - quality['min_value']
    else:
        quality['min_value'] = np.nan
        quality['max_value'] = np.nan
        quality['mean_value'] = np.nan
        quality['std_value'] = np.nan
        quality['value_range'] = np.nan
    
    # Overall quality score (0-1)
    score = 1.0
    score -= quality['nan_ratio'] * 0.5  # Penalize NaN
    score -= min(quality['inf_count'] / len(df), 0.3)  # Penalize inf
    
    if quality['std_value'] == 0:
        score -= 0.2  # Penalize constant signal
    
    quality['quality_score'] = max(0.0, min(1.0, score))
    
    return quality


def check_data_gaps(df: pd.DataFrame, max_gap_seconds: int = 60) -> List[Tuple[int, int]]:
    """
    Find data gaps (missing timestamps).
    
    Args:
        df: Telemetry DataFrame
        max_gap_seconds: Maximum allowed gap in seconds
        
    Returns:
        List of (start_index, end_index) tuples for gaps
    """
    gaps = []
    
    if len(df) < 2:
        return gaps
    
    time_diffs = df['timestamp'].diff().dt.total_seconds()
    gap_indices = time_diffs[time_diffs > max_gap_seconds].index
    
    for idx in gap_indices:
        gaps.append((idx - 1, idx))
    
    return gaps
