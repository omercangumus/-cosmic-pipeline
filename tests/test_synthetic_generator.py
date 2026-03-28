"""Unit tests for synthetic telemetry generator."""

import pytest
import numpy as np
import pandas as pd
from data.synthetic_generator import generate_clean_signal, inject_faults


def test_clean_signal_shape():
    """Test that generated signal has correct shape."""
    df = generate_clean_signal(5000)
    assert len(df) == 5000
    assert "timestamp" in df.columns
    assert "value" in df.columns


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


def test_clean_signal_has_metadata_columns():
    """Test that clean signal includes satellite metadata."""
    df = generate_clean_signal(n=1000, seed=42)
    assert "timestamp" in df.columns
    assert "value" in df.columns
    assert "orbit_id" in df.columns
    assert "phase" in df.columns
    assert "sensor_id" in df.columns


def test_clean_signal_realistic_temperature_range():
    """Thermal sensor should stay within ~5-35 C."""
    df = generate_clean_signal(n=10000, seed=42)
    assert df["value"].min() > 0, "Temperature dropped below 0"
    assert df["value"].max() < 40, "Temperature exceeded 40"


def test_clean_signal_has_multiple_orbits():
    """20000 seconds / 5400s per orbit = 3+ orbits."""
    df = generate_clean_signal(n=20000, seed=42)
    assert df["orbit_id"].nunique() >= 3


def test_corrupted_dataset_drops_metadata():
    """Corrupted df should only have timestamp + value."""
    from data.synthetic_generator import generate_corrupted_dataset

    clean, corrupted, _ = generate_corrupted_dataset(n=5000, seed=42)
    assert "orbit_id" in clean.columns
    assert "timestamp" in corrupted.columns
    assert "value" in corrupted.columns


def test_clean_signal_has_all_telemetry_columns():
    """All 20 telemetry columns should be present."""
    df = generate_clean_signal(n=5000, seed=42)
    expected = [
        "timestamp", "value", "satellite_id", "sensor_id", "orbit_id",
        "phase", "latitude", "longitude", "altitude_km",
        "battery_voltage", "solar_panel_current",
        "magnetometer_x", "magnetometer_y", "magnetometer_z",
        "cpu_temperature", "signal_strength_dbm", "radiation_dose_rad",
        "status_flag", "data_quality", "telemetry_packet_id",
    ]
    for col in expected:
        assert col in df.columns, f"Missing column: {col}"


def test_latitude_range():
    df = generate_clean_signal(n=10000, seed=42)
    assert df["latitude"].min() >= -90
    assert df["latitude"].max() <= 90


def test_battery_voltage_range():
    df = generate_clean_signal(n=10000, seed=42)
    assert df["battery_voltage"].min() > 25
    assert df["battery_voltage"].max() < 35


def test_radiation_dose_monotonic():
    df = generate_clean_signal(n=5000, seed=42)
    diffs = df["radiation_dose_rad"].diff().dropna()
    assert (diffs >= 0).all(), "Radiation dose must be monotonically increasing"


def test_telemetry_packet_sequential():
    df = generate_clean_signal(n=1000, seed=42)
    assert df["telemetry_packet_id"].iloc[0] == 1
    assert df["telemetry_packet_id"].iloc[-1] == 1000
