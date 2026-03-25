"""Unit tests for pipeline.detector_classic module."""

import numpy as np
import pandas as pd
import pytest

from pipeline.detector_classic import (
    detect_all,
    detect_gaps,
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_sliding_window,
)


@pytest.fixture
def normal_df():
    """DataFrame with clean sinusoidal signal."""
    n = 1000
    t = np.arange(n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
        "value": np.sin(2 * np.pi * 0.01 * t) * 10,
    })


@pytest.fixture
def spiked_df(normal_df):
    """DataFrame with a few extreme spikes injected."""
    df = normal_df.copy()
    spike_indices = [100, 300, 700]
    for i in spike_indices:
        df.loc[i, "value"] = 500.0
    return df, spike_indices


@pytest.fixture
def gapped_df(normal_df):
    """DataFrame with NaN gaps and a timestamp jump."""
    df = normal_df.copy()
    df.loc[200:210, "value"] = np.nan
    # Inject a large timestamp gap at index 500
    df.loc[500, "timestamp"] = df.loc[499, "timestamp"] + pd.Timedelta(seconds=120)
    return df


# --- Z-score ---

def test_zscore_detects_spikes(spiked_df):
    df, spike_indices = spiked_df
    mask = detect_outliers_zscore(df, threshold=3.0)
    for idx in spike_indices:
        assert mask.iloc[idx], f"Spike at {idx} not detected"


def test_zscore_clean_signal_few_flags(normal_df):
    mask = detect_outliers_zscore(normal_df, threshold=3.0)
    assert mask.sum() < len(normal_df) * 0.05


def test_zscore_returns_bool_series(normal_df):
    mask = detect_outliers_zscore(normal_df)
    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool


def test_zscore_nan_not_flagged():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1s"),
        "value": [1.0, 2.0, np.nan, 1.5, 1.0, 1.5, 2.0, 1.0, 1.5, 1.0],
    })
    mask = detect_outliers_zscore(df)
    assert not mask.iloc[2]


# --- IQR ---

def test_iqr_detects_spikes(spiked_df):
    df, spike_indices = spiked_df
    mask = detect_outliers_iqr(df)
    for idx in spike_indices:
        assert mask.iloc[idx], f"Spike at {idx} not detected by IQR"


def test_iqr_clean_signal_few_flags(normal_df):
    mask = detect_outliers_iqr(normal_df)
    assert mask.sum() < len(normal_df) * 0.1


def test_iqr_strict_multiplier_catches_more(spiked_df):
    df, _ = spiked_df
    loose = detect_outliers_iqr(df, multiplier=3.0)
    strict = detect_outliers_iqr(df, multiplier=1.0)
    assert strict.sum() >= loose.sum()


# --- Sliding Window ---

def test_sliding_window_detects_local_spike():
    values = np.ones(200)
    values[100] = 100.0  # obvious local spike
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=200, freq="1s"),
        "value": values,
    })
    mask = detect_sliding_window(df, window=20, threshold=3.0)
    assert mask.iloc[100]


def test_sliding_window_returns_bool_series(normal_df):
    mask = detect_sliding_window(normal_df)
    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool


# --- Gap Detection ---

def test_gaps_detects_nan(gapped_df):
    mask = detect_gaps(gapped_df)
    assert mask.iloc[200:211].all()


def test_gaps_detects_timestamp_jump(gapped_df):
    mask = detect_gaps(gapped_df, max_gap_seconds=60)
    assert mask.iloc[500]


def test_gaps_no_false_positives(normal_df):
    mask = detect_gaps(normal_df, max_gap_seconds=60)
    assert mask.sum() == 0


# --- detect_all ---

def test_detect_all_returns_all_detectors(normal_df):
    results = detect_all(normal_df)
    assert set(results.keys()) == {"zscore", "iqr", "sliding_window", "gaps"}
    for name, mask in results.items():
        assert isinstance(mask, pd.Series), f"{name} is not a Series"
        assert mask.dtype == bool, f"{name} is not bool dtype"


def test_detect_all_finds_spikes(spiked_df):
    df, spike_indices = spiked_df
    results = detect_all(df)
    combined = results["zscore"] | results["iqr"]
    for idx in spike_indices:
        assert combined.iloc[idx], f"Spike at {idx} not detected by any method"
