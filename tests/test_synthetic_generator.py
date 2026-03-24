"""Unit tests for synthetic telemetry generator."""

import pytest
import numpy as np
import pandas as pd
from data.synthetic_generator import generate_clean_signal, inject_faults


def test_clean_signal_shape():
    """Test that generated signal has correct shape."""
    df = generate_clean_signal(5000)
    assert len(df) == 5000
    assert list(df.columns) == ["timestamp", "value"]


def test_clean_signal_no_nan():
    """Test that clean signal contains no NaN values."""
    df = generate_clean_signal(5000)
    assert df["value"].isna().sum() == 0


def test_clean_signal_timestamp_type():
    """Test that timestamp column has correct dtype."""
    df = generate_clean_signal(100)
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_inject_faults_returns_mask():
    """Test that inject_faults returns ground truth mask with correct keys."""
    df = generate_clean_signal(1000)
    _, mask = inject_faults(df)
    assert set(mask.keys()) == {"seu", "tid", "gap", "noise"}


def test_seu_indices_in_range():
    """Test that SEU indices are within valid range."""
    df = generate_clean_signal(1000)
    _, mask = inject_faults(df, seu_count=10)
    assert all(0 <= i < 1000 for i in mask["seu"])


def test_gap_creates_nan():
    """Test that data gaps create NaN values at correct positions."""
    df = generate_clean_signal(1000)
    corrupted, mask = inject_faults(df, gap_count=3)
    
    for start, end in mask["gap"]:
        assert corrupted["value"].iloc[start:end].isna().all()


def test_tid_covers_full_signal():
    """Test that TID drift affects entire signal."""
    df = generate_clean_signal(500)
    _, mask = inject_faults(df)
    assert len(mask["tid"]) == 500


def test_does_not_modify_input():
    """Test that inject_faults does not modify input DataFrame."""
    df = generate_clean_signal(500)
    original_values = df["value"].values.copy()
    inject_faults(df)
    np.testing.assert_array_equal(df["value"].values, original_values)


def test_reproducibility():
    """Test that same seed produces identical results."""
    df = generate_clean_signal(1000)
    c1, m1 = inject_faults(df, seed=42)
    c2, m2 = inject_faults(df, seed=42)
    
    pd.testing.assert_frame_equal(c1, c2)
    assert m1["seu"] == m2["seu"]


def test_noise_indices_above_threshold():
    """Test that noise indices are correctly identified."""
    df = generate_clean_signal(1000)
    _, mask = inject_faults(df, noise_std_max=3.0)
    assert len(mask["noise"]) > 0
