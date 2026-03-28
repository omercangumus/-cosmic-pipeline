"""Unit tests for utils.csv_parser — CSV upload validation and normalization."""

import base64

import numpy as np
import pandas as pd
import pytest

from utils.csv_parser import parse_uploaded_csv


class TestCSVUploadParsing:
    """CSV upload parsing and format conversion tests."""

    def _make_upload_contents(self, df: pd.DataFrame) -> str:
        """Convert a DataFrame into Dash-style base64 upload contents."""
        csv_string = df.to_csv(index=False)
        encoded = base64.b64encode(csv_string.encode()).decode()
        return f"data:text/csv;base64,{encoded}"

    def test_valid_csv_with_timestamp_and_value(self):
        """Standard format: timestamp + value columns."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="1s"),
            "value": np.sin(np.linspace(0, 4 * np.pi, 100)),
        })
        contents = self._make_upload_contents(df)
        result, error = parse_uploaded_csv(contents, "test.csv")
        assert error is None
        assert len(result) == 100
        assert "timestamp" in result.columns
        assert "value" in result.columns

    def test_valid_csv_with_time_tag(self):
        """GOES format: time_tag column."""
        df = pd.DataFrame({
            "time_tag": pd.date_range("2024-01-01", periods=50, freq="5min"),
            "flux": np.random.default_rng(42).uniform(0.1, 100, 50),
        })
        contents = self._make_upload_contents(df)
        result, error = parse_uploaded_csv(contents, "goes.csv")
        assert error is None
        assert len(result) == 50
        assert result.columns.tolist() == ["timestamp", "value"]

    def test_csv_without_timestamp_auto_generates(self):
        """Missing timestamp column should be auto-generated."""
        df = pd.DataFrame({
            "sensor_reading": np.ones(100) * 42.0,
        })
        contents = self._make_upload_contents(df)
        result, error = parse_uploaded_csv(contents, "no_time.csv")
        assert error is None
        assert len(result) == 100
        assert result["timestamp"].notna().all()

    def test_csv_without_value_uses_first_numeric(self):
        """Missing value column should pick the first numeric column."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="1s"),
            "temperature": np.linspace(20, 30, 50),
            "pressure": np.linspace(1000, 1010, 50),
        })
        contents = self._make_upload_contents(df)
        result, error = parse_uploaded_csv(contents, "sensors.csv")
        assert error is None
        assert result["value"].iloc[0] == pytest.approx(20.0, abs=0.1)

    def test_empty_file_returns_error(self):
        """Empty file should return an error."""
        encoded = base64.b64encode(b"").decode()
        contents = f"data:text/csv;base64,{encoded}"
        result, error = parse_uploaded_csv(contents, "empty.csv")
        assert result is None
        assert error is not None

    def test_no_numeric_columns_returns_error(self):
        """No numeric columns should return an error."""
        df = pd.DataFrame({
            "name": ["a", "b", "c"] * 10,
            "category": ["x", "y", "z"] * 10,
        })
        contents = self._make_upload_contents(df)
        result, error = parse_uploaded_csv(contents, "text_only.csv")
        assert result is None
        assert "sayısal" in error.lower() or "sütun" in error.lower()

    def test_too_few_rows_returns_error(self):
        """Fewer than 10 rows should return an error."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1s"),
            "value": [1, 2, 3, 4, 5],
        })
        contents = self._make_upload_contents(df)
        result, error = parse_uploaded_csv(contents, "tiny.csv")
        assert result is None
        assert "az" in error.lower() or "minimum" in error.lower()

    def test_all_nan_values_returns_error(self):
        """All-NaN values should return an error."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="1s"),
            "value": [np.nan] * 20,
        })
        contents = self._make_upload_contents(df)
        result, error = parse_uploaded_csv(contents, "all_nan.csv")
        assert result is None
        assert "NaN" in error or "nan" in error.lower()

    def test_none_contents_returns_error(self):
        """None contents should return an error."""
        result, error = parse_uploaded_csv(None, "nothing.csv")
        assert result is None
        assert error is not None

    def test_invalid_base64_returns_error(self):
        """Invalid base64 should return an error."""
        result, error = parse_uploaded_csv(
            "data:text/csv;base64,NOT_VALID!!!", "bad.csv",
        )
        assert result is None
        assert error is not None

    def test_output_sorted_by_timestamp(self):
        """Output should be sorted by timestamp."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(
                ["2024-01-03", "2024-01-01", "2024-01-02"] * 10,
            ),
            "value": range(30),
        })
        contents = self._make_upload_contents(df)
        result, error = parse_uploaded_csv(contents, "unsorted.csv")
        assert error is None
        assert result["timestamp"].is_monotonic_increasing
