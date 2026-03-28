"""Unit tests for pipeline.detector_classic module."""

import numpy as np
import pandas as pd
import pytest

from pipeline.detector_classic import (
    detect_all,
    detect_delta_spike,
    detect_duplicates,
    detect_flatline,
    detect_gaps,
    detect_outliers_zscore,
    detect_range_violation,
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


# --- Range Violation ---

def test_range_detects_extreme_spike():
    n = 500
    values = np.sin(np.linspace(0, 4 * np.pi, n)) * 10
    values[100] = 5000.0  # way beyond 10*std
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
        "value": values,
    })
    mask = detect_range_violation(df, max_std_multiplier=10.0)
    assert mask.iloc[100]


def test_range_clean_signal_no_flags(normal_df):
    mask = detect_range_violation(normal_df)
    assert mask.sum() == 0


def test_range_returns_bool_series(normal_df):
    mask = detect_range_violation(normal_df)
    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool


# --- Delta Spike ---

def test_delta_detects_sudden_jump():
    n = 200
    values = np.ones(n) * 10.0
    values[100] = 500.0  # sudden jump
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
        "value": values,
    })
    mask = detect_delta_spike(df)
    assert mask.iloc[100]  # jump point
    assert mask.iloc[101]  # return point


def test_delta_clean_signal_no_flags():
    n = 200
    values = np.sin(np.linspace(0, 4 * np.pi, n)) * 10
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
        "value": values,
    })
    mask = detect_delta_spike(df, max_delta_multiplier=10.0)
    assert mask.sum() == 0


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
    assert set(results.keys()) == {"zscore", "sliding_window", "gaps", "range", "delta", "flatline", "duplicates"}
    for name, mask in results.items():
        assert isinstance(mask, pd.Series), f"{name} is not a Series"
        assert mask.dtype == bool, f"{name} is not bool dtype"


def test_detect_all_finds_spikes(spiked_df):
    df, spike_indices = spiked_df
    results = detect_all(df)
    combined = results["zscore"] | results["range"]
    for idx in spike_indices:
        assert combined.iloc[idx], f"Spike at {idx} not detected by any method"


# --- Flatline ---

def test_flatline_detects_stuck_sensor():
    n = 200
    values = np.sin(np.linspace(0, 4 * np.pi, n)) * 10
    values[50:80] = 5.0  # 30 points same value
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
        "value": values,
    })
    mask = detect_flatline(df, min_duration=20)
    assert mask.iloc[60], "Flatline region not detected"
    assert not mask.iloc[0], "Normal point wrongly flagged"


def test_flatline_short_run_not_flagged():
    n = 100
    values = np.sin(np.linspace(0, 4 * np.pi, n)) * 10
    values[50:55] = 5.0
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
        "value": values,
    })
    mask = detect_flatline(df, min_duration=20)
    assert mask.iloc[50:55].sum() == 0


def test_flatline_entire_constant_signal():
    n = 100
    values = np.ones(n) * 42.0
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
        "value": values,
    })
    mask = detect_flatline(df, min_duration=20)
    assert mask.all()


# --- Duplicates ---

def test_duplicates_detects_repeated_timestamps():
    timestamps = pd.date_range("2024-01-01", periods=10, freq="1s").tolist()
    timestamps[5] = timestamps[4]
    df = pd.DataFrame({"timestamp": timestamps, "value": range(10)})
    mask = detect_duplicates(df)
    assert mask.iloc[4] and mask.iloc[5]


def test_duplicates_clean_data():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=50, freq="1s"),
        "value": range(50),
    })
    mask = detect_duplicates(df)
    assert mask.sum() == 0
