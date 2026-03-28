"""Unit tests for multi-channel pipeline support."""

import numpy as np
import pandas as pd
import pytest

from pipeline.orchestrator import run_pipeline_multi


class TestMultiChannelBasic:
    def test_two_channels(self):
        """Multi-channel pipeline processes 2+ columns."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=200, freq="1s"),
            "temp": np.random.default_rng(42).normal(20, 1, 200),
            "pressure": np.random.default_rng(43).normal(1013, 5, 200),
        })
        # Inject spikes
        df.loc[50, "temp"] = 999.0
        df.loc[60, "pressure"] = -500.0

        result = run_pipeline_multi(df, method="classic")
        assert result["summary"]["total_channels"] == 2
        assert "temp" in result["channels"]
        assert "pressure" in result["channels"]
        assert result["summary"]["total_faults"] > 0

    def test_single_column_fallback(self):
        """Single numeric column still works through multi-channel wrapper."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=200, freq="1s"),
            "value": np.random.default_rng(42).normal(5, 0.1, 200),
        })
        result = run_pipeline_multi(df, method="classic")
        assert result["summary"]["total_channels"] == 1

    def test_column_selection(self):
        """Only selected columns are processed."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=200, freq="1s"),
            "a": np.random.default_rng(1).normal(0, 1, 200),
            "b": np.random.default_rng(2).normal(0, 1, 200),
            "c": np.random.default_rng(3).normal(0, 1, 200),
        })
        result = run_pipeline_multi(df, method="classic", columns=["a", "c"])
        assert result["summary"]["total_channels"] == 2
        assert "a" in result["channels"]
        assert "c" in result["channels"]
        assert "b" not in result["channels"]

    def test_per_channel_metrics(self):
        """Each channel has its own metrics."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=200, freq="1s"),
            "sensor1": np.random.default_rng(42).normal(10, 1, 200),
            "sensor2": np.random.default_rng(43).normal(50, 2, 200),
        })
        result = run_pipeline_multi(df, method="classic")
        for col in ["sensor1", "sensor2"]:
            ch = result["channels"][col]
            assert "metrics" in ch
            assert "faults_detected" in ch["metrics"]
            assert "cleaned_data" in ch
            assert "fault_mask" in ch

    def test_no_numeric_columns_raises(self):
        """Error when no numeric columns are found."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1s"),
            "label": ["a"] * 10,
        })
        with pytest.raises(ValueError, match="No numeric columns"):
            run_pipeline_multi(df, method="classic")

    def test_invalid_column_selection(self):
        """Error when selected columns don't exist."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=200, freq="1s"),
            "a": np.ones(200),
        })
        with pytest.raises(ValueError, match="None of"):
            run_pipeline_multi(df, method="classic", columns=["nonexistent"])

    def test_label_column_excluded(self):
        """Label column is not processed as a channel."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=200, freq="1s"),
            "value": np.random.default_rng(42).normal(0, 1, 200),
            "label": np.zeros(200, dtype=int),
        })
        result = run_pipeline_multi(df, method="classic")
        assert result["summary"]["total_channels"] == 1
        assert "label" not in result["channels"]

    def test_auto_timestamp_generation(self):
        """Timestamp is auto-generated when not present."""
        df = pd.DataFrame({
            "a": np.random.default_rng(42).normal(0, 1, 200),
            "b": np.random.default_rng(43).normal(0, 1, 200),
        })
        result = run_pipeline_multi(df, method="classic")
        assert result["summary"]["total_channels"] == 2

    def test_summary_processing_time(self):
        """Summary includes total processing time."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=200, freq="1s"),
            "x": np.random.default_rng(42).normal(0, 1, 200),
        })
        result = run_pipeline_multi(df, method="classic")
        assert result["summary"]["processing_time"] > 0

    def test_per_channel_fault_counts(self):
        """per_channel dict has fault count per channel."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=200, freq="1s"),
            "a": np.random.default_rng(42).normal(0, 1, 200),
            "b": np.random.default_rng(43).normal(0, 1, 200),
        })
        result = run_pipeline_multi(df, method="classic")
        assert "per_channel" in result["summary"]
        assert "a" in result["summary"]["per_channel"]
        assert "b" in result["summary"]["per_channel"]
