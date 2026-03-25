"""Unit tests for pipeline.filters_classic module."""

import numpy as np
import pandas as pd
import pytest

from pipeline.filters_classic import (
    apply_classic_filters,
    interpolate_gaps,
    median_filter,
    savgol_filter,
    wavelet_filter,
)


@pytest.fixture
def smooth_df():
    """Smooth sinusoidal signal with known shape."""
    n = 500
    t = np.arange(n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
        "value": np.sin(2 * np.pi * 0.01 * t) * 10,
    })


@pytest.fixture
def spike_mask():
    """Mask with three spike positions."""
    mask = pd.Series(np.zeros(500, dtype=bool))
    mask.iloc[[50, 200, 400]] = True
    return mask


@pytest.fixture
def spiked_df(smooth_df, spike_mask):
    """Signal with extreme spikes at known positions."""
    df = smooth_df.copy()
    df.loc[spike_mask, "value"] = 999.0
    return df


# --- Median Filter ---

def test_median_removes_spikes(spiked_df, spike_mask):
    result = median_filter(spiked_df, spike_mask, window=5)
    for idx in [50, 200, 400]:
        assert abs(result["value"].iloc[idx]) < 100, f"Spike at {idx} not removed"


def test_median_preserves_clean_points(spiked_df, spike_mask):
    result = median_filter(spiked_df, spike_mask, window=5)
    clean = ~spike_mask
    pd.testing.assert_series_equal(
        result.loc[clean, "value"].reset_index(drop=True),
        spiked_df.loc[clean, "value"].reset_index(drop=True),
    )


def test_median_does_not_modify_input(spiked_df, spike_mask):
    original = spiked_df["value"].values.copy()
    median_filter(spiked_df, spike_mask)
    np.testing.assert_array_equal(spiked_df["value"].values, original)


def test_median_even_window_handled(spiked_df, spike_mask):
    result = median_filter(spiked_df, spike_mask, window=4)
    assert result["value"].iloc[50] != 999.0


# --- Savitzky-Golay ---

def test_savgol_smooths_spikes(spiked_df, spike_mask):
    result = savgol_filter(spiked_df, spike_mask, window=11, polyorder=3)
    for idx in [50, 200, 400]:
        assert abs(result["value"].iloc[idx]) < 100


def test_savgol_preserves_clean_points(spiked_df, spike_mask):
    result = savgol_filter(spiked_df, spike_mask)
    clean = ~spike_mask
    pd.testing.assert_series_equal(
        result.loc[clean, "value"].reset_index(drop=True),
        spiked_df.loc[clean, "value"].reset_index(drop=True),
    )


# --- Wavelet ---

def test_wavelet_smooths_spikes(spiked_df, spike_mask):
    result = wavelet_filter(spiked_df, spike_mask, wavelet="db4", level=3)
    for idx in [50, 200, 400]:
        assert abs(result["value"].iloc[idx]) < 100


# --- Interpolation ---

def test_interpolate_fills_nan():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1s"),
        "value": [1.0, 2.0, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    })
    mask = df["value"].isna()
    result = interpolate_gaps(df, mask)
    assert result["value"].isna().sum() == 0


def test_interpolate_linear_values():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1s"),
        "value": [0.0, np.nan, np.nan, np.nan, 4.0],
    })
    mask = df["value"].isna()
    result = interpolate_gaps(df, mask)
    np.testing.assert_allclose(result["value"].values, [0.0, 1.0, 2.0, 3.0, 4.0])


def test_interpolate_does_not_modify_input():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1s"),
        "value": [1.0, np.nan, 3.0, np.nan, 5.0],
    })
    original = df["value"].values.copy()
    mask = df["value"].isna()
    interpolate_gaps(df, mask)
    np.testing.assert_array_equal(df["value"].values, original)


# --- apply_classic_filters ---

def test_apply_classic_fixes_spikes_and_gaps():
    n = 200
    t = np.arange(n)
    values = np.sin(2 * np.pi * 0.02 * t) * 5.0

    # Inject a spike and a gap
    values[50] = 500.0
    values[120:125] = np.nan

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
        "value": values,
    })
    mask = pd.Series(np.zeros(n, dtype=bool))
    mask.iloc[50] = True
    mask.iloc[120:125] = True

    result = apply_classic_filters(df, mask)
    assert result["value"].isna().sum() == 0
    assert abs(result["value"].iloc[50]) < 50


def test_apply_classic_no_anomalies(smooth_df):
    mask = pd.Series(np.zeros(len(smooth_df), dtype=bool))
    result = apply_classic_filters(smooth_df, mask)
    pd.testing.assert_frame_equal(result, smooth_df)
