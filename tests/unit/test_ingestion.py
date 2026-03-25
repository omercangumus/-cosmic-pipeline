"""Unit tests for pipeline.ingestion module."""

import numpy as np
import pandas as pd
import pytest

from pipeline.ingestion import load_data, preprocess, validate_schema


@pytest.fixture
def clean_df():
    """Minimal valid telemetry DataFrame."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="1s"),
        "value": np.sin(np.linspace(0, 2 * np.pi, 100)),
    })


# --- load_data ---

def test_load_data_from_dataframe(clean_df):
    result = load_data(clean_df)
    pd.testing.assert_frame_equal(result, clean_df)


def test_load_data_does_not_modify_input(clean_df):
    original = clean_df.copy()
    result = load_data(clean_df)
    result["value"].iloc[0] = 999.0
    pd.testing.assert_frame_equal(clean_df, original)


def test_load_data_csv(tmp_path, clean_df):
    csv_path = tmp_path / "test.csv"
    clean_df.to_csv(csv_path, index=False)
    result = load_data(str(csv_path))
    assert len(result) == 100
    assert "timestamp" in result.columns
    assert "value" in result.columns


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("/nonexistent/path.csv")


def test_load_data_unsupported_format(tmp_path):
    path = tmp_path / "data.xyz"
    path.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported"):
        load_data(str(path))


# --- validate_schema ---

def test_validate_schema_valid(clean_df):
    assert validate_schema(clean_df) is True


def test_validate_schema_missing_column():
    df = pd.DataFrame({"timestamp": [1, 2], "other": [3, 4]})
    with pytest.raises(ValueError, match="Missing"):
        validate_schema(df)


def test_validate_schema_empty():
    df = pd.DataFrame({"timestamp": [], "value": []})
    with pytest.raises(ValueError, match="empty"):
        validate_schema(df)


def test_validate_schema_non_numeric_value():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="1s"),
        "value": ["a", "b", "c"],
    })
    with pytest.raises(ValueError, match="numeric"):
        validate_schema(df)


# --- preprocess ---

def test_preprocess_sorts_by_timestamp():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]),
        "value": [3.0, 1.0, 2.0],
    })
    result = preprocess(df)
    assert list(result["value"]) == [1.0, 2.0, 3.0]


def test_preprocess_removes_duplicates():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
        "value": [1.0, 2.0, 3.0],
    })
    result = preprocess(df)
    assert len(result) == 2


def test_preprocess_parses_string_timestamps():
    df = pd.DataFrame({
        "timestamp": ["2024-01-01 00:00:00", "2024-01-01 00:00:01"],
        "value": [1.0, 2.0],
    })
    result = preprocess(df)
    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])


def test_preprocess_drops_unparseable_timestamps():
    df = pd.DataFrame({
        "timestamp": ["2024-01-01", "not_a_date", "2024-01-03"],
        "value": [1.0, 2.0, 3.0],
    })
    result = preprocess(df)
    assert len(result) == 2


def test_preprocess_value_dtype_float64(clean_df):
    clean_df["value"] = clean_df["value"].astype(np.float32)
    result = preprocess(clean_df)
    assert result["value"].dtype == np.float64
