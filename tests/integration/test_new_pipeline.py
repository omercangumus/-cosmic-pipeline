"""Integration tests for the simplified mercek-model pipeline."""

import numpy as np
import pandas as pd
import pytest

from data.synthetic_generator import generate_corrupted_dataset
from pipeline.orchestrator import run_pipeline
from pipeline.filters_classic import (
    apply_classic_filters,
    detrend_signal,
    interpolate_gaps,
    median_filter,
)
from pipeline.ensemble import hybrid_majority_vote
from pipeline.detector_classic import detect_all, detect_range_violation


class TestMercekFilterPipeline:
    """Mercek modeli: her filtre oncekinin ciktisini aliyor mu?"""

    def test_intermediates_have_all_steps(self):
        _, corrupted, _ = generate_corrupted_dataset(n=1000, seed=42)
        mask = pd.Series(np.zeros(len(corrupted), dtype=bool))
        result, intermediates = apply_classic_filters(
            corrupted, mask, return_intermediates=True,
        )
        assert "step_0_raw" in intermediates
        assert "step_1_nan" in intermediates
        assert "step_2_interpolated" in intermediates
        assert "step_3_median" in intermediates

    def test_each_step_changes_signal(self):
        _, corrupted, _ = generate_corrupted_dataset(n=1000, seed=42)
        mask = corrupted["value"].isna()
        result, intermediates = apply_classic_filters(
            corrupted, mask, return_intermediates=True,
        )
        # Each step should differ from the previous
        assert not np.array_equal(
            intermediates["step_0_raw"], intermediates["step_1_nan"],
        )
        assert not np.array_equal(
            intermediates["step_1_nan"], intermediates["step_2_interpolated"],
        )

    def test_detrend_reduces_trend(self):
        """A linearly trending signal should have lower std after detrend."""
        n = 500
        values = np.linspace(0, 50, n) + np.sin(np.linspace(0, 4 * np.pi, n))
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": values,
        })
        result = detrend_signal(df)
        assert result["value"].std() < df["value"].std()

    def test_median_filter_removes_spike(self):
        """Median filter should eliminate a single spike."""
        n = 100
        values = np.ones(n) * 10.0
        values[50] = 9999.0
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": values,
        })
        result = median_filter(df, window=5)
        assert abs(result["value"].iloc[50] - 10.0) < 1.0


class TestHybridMajorityEnsemble:
    """Hybrid majority: hard anomaliler direkt, soft'lar oylama ile."""

    def test_gap_always_detected(self):
        """Every NaN point should be flagged (hard rule via gap detector)."""
        _, corrupted, _ = generate_corrupted_dataset(n=1000, seed=42)
        result = run_pipeline(corrupted, method="classic")
        fault_mask = result["fault_mask"]
        nan_indices = corrupted["value"].isna()
        for idx in nan_indices[nan_indices].index:
            assert fault_mask.iloc[idx], f"Gap at index {idx} not detected"

    def test_range_violation_always_detected(self):
        """Physically impossible values must always be flagged."""
        n = 500
        values = np.sin(np.linspace(0, 4 * np.pi, n)) * 10
        values[100] = 50000.0
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": values,
        })
        mask = detect_range_violation(df, max_std_multiplier=10.0)
        assert mask.iloc[100]

    def test_single_soft_detector_not_enough(self):
        """A single soft detector vote should not suffice (min_agreement=2)."""
        s1 = pd.Series([True, False, False, False])
        s2 = pd.Series([False, False, False, False])
        hard = [pd.Series([False, False, False, False])]
        result = hybrid_majority_vote(hard, [s1, s2], min_agreement=2)
        assert not result.iloc[0]


class TestEndToEndPipeline:
    """Full end-to-end pipeline tests."""

    def test_classic_pipeline_runs(self):
        _, corrupted, _ = generate_corrupted_dataset(n=2000, seed=42)
        result = run_pipeline(corrupted, method="classic")
        assert "cleaned_data" in result
        assert "fault_mask" in result
        assert "metrics" in result
        assert result["metrics"]["faults_detected"] >= 0
        assert result["metrics"]["processing_time"] > 0
        assert result["cleaned_data"]["value"].isna().sum() == 0

    def test_pipeline_improves_signal(self):
        """RMSE should drop after pipeline processing."""
        clean, corrupted, _ = generate_corrupted_dataset(n=2000, seed=42)
        result = run_pipeline(corrupted, method="classic")

        clean_vals = clean["value"].values
        corrupted_vals = corrupted["value"].values
        cleaned_vals = result["cleaned_data"]["value"].values

        # Compare only finite points (corrupted may have NaN)
        finite = np.isfinite(clean_vals) & np.isfinite(corrupted_vals)
        rmse_before = np.sqrt(np.mean((clean_vals[finite] - corrupted_vals[finite]) ** 2))

        finite_after = np.isfinite(clean_vals) & np.isfinite(cleaned_vals)
        rmse_after = np.sqrt(np.mean((clean_vals[finite_after] - cleaned_vals[finite_after]) ** 2))

        assert rmse_after < rmse_before, (
            f"RMSE did not improve: {rmse_before:.2f} -> {rmse_after:.2f}"
        )

    def test_all_three_methods_run(self):
        _, corrupted, _ = generate_corrupted_dataset(n=1000, seed=42)
        for method in ["classic", "ml", "both"]:
            result = run_pipeline(corrupted, method=method)
            assert result["cleaned_data"]["value"].isna().sum() == 0, (
                f"Method '{method}' left NaN in output"
            )
