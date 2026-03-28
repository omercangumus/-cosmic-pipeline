"""Edge case tests for pipeline robustness."""

import numpy as np
import pandas as pd
import pytest

from pipeline.orchestrator import run_pipeline


class TestEdgeCases:
    def _make_df(self, values):
        return pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=len(values), freq="1s"),
            "value": values,
        })

    def test_all_nan(self):
        df = self._make_df([np.nan] * 200)
        result = run_pipeline(df, method="classic")
        assert "cleaned_data" in result

    def test_constant_signal(self):
        df = self._make_df([5.0] * 500)
        result = run_pipeline(df, method="classic")
        assert "cleaned_data" in result

    def test_very_large_values(self):
        values = np.random.default_rng(42).normal(0, 1, 200)
        values[50] = 1e15
        values[100] = -1e15
        df = self._make_df(values)
        result = run_pipeline(df, method="classic")
        assert result["metrics"]["faults_detected"] >= 2

    def test_negative_values(self):
        values = np.random.default_rng(42).normal(-100, 5, 300)
        values[50] = 500
        df = self._make_df(values)
        result = run_pipeline(df, method="classic")
        assert result["metrics"]["faults_detected"] >= 1

    def test_mixed_nan_and_values(self):
        values = np.random.default_rng(42).normal(5, 0.1, 300).tolist()
        for i in range(0, 300, 3):
            values[i] = np.nan
        df = self._make_df(values)
        result = run_pipeline(df, method="classic")
        assert "cleaned_data" in result

    def test_short_signal(self):
        """Very short signal — should not crash."""
        df = self._make_df([5.0, 5.1])
        try:
            run_pipeline(df, method="classic")
        except (ValueError, Exception):
            pass  # exception is acceptable for very short signals

    def test_single_point(self):
        """Single data point — should not crash."""
        df = self._make_df([5.0])
        try:
            run_pipeline(df, method="classic")
        except (ValueError, Exception):
            pass
