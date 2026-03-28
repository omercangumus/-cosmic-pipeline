"""Integration tests for pyramid detection features: delta spike, reason column, hybrid ensemble."""

import numpy as np
import pandas as pd
import pytest

from data.synthetic_generator import generate_corrupted_dataset
from pipeline.orchestrator import run_pipeline
from pipeline.detector_classic import (
    detect_delta_spike,
    detect_all,
    detect_range_violation,
    detect_gaps,
)
from pipeline.ensemble import hybrid_majority_vote


class TestDeltaSpikeDetector:
    """Delta spike detector testleri."""

    def test_single_spike_detected(self):
        """Tek bir ani sıçrama tespit edilmeli."""
        n = 300
        values = np.ones(n) * 10.0
        values[150] = 1000.0
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": values,
        })
        mask = detect_delta_spike(df)
        assert mask.iloc[150], "Spike noktası tespit edilmedi"
        assert mask.iloc[151], "Spike dönüş noktası tespit edilmedi"

    def test_multiple_spikes(self):
        """Birden fazla sıçrama tespit edilmeli."""
        n = 500
        values = np.ones(n) * 5.0
        values[100] = 500.0
        values[300] = -500.0
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": values,
        })
        mask = detect_delta_spike(df)
        assert mask.iloc[100]
        assert mask.iloc[300]

    def test_gradual_change_not_flagged(self):
        """Yavaş değişen sinyal flag'lenmemeli."""
        n = 200
        values = np.linspace(0, 100, n)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": values,
        })
        mask = detect_delta_spike(df, max_delta_multiplier=10.0)
        assert mask.sum() == 0

    def test_nan_values_not_flagged(self):
        """NaN değerler delta spike olarak işaretlenmemeli."""
        n = 100
        values = np.ones(n) * 10.0
        values[50] = np.nan
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": values,
        })
        mask = detect_delta_spike(df)
        assert not mask.iloc[50]

    def test_returns_correct_type(self):
        n = 50
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": np.ones(n),
        })
        mask = detect_delta_spike(df)
        assert isinstance(mask, pd.Series)
        assert mask.dtype == bool
        assert len(mask) == n


class TestDeltaInDetectAll:
    """Delta'nın detect_all içinde çalıştığını doğrula."""

    def test_delta_in_detect_all_keys(self):
        n = 200
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": np.sin(np.linspace(0, 4 * np.pi, n)) * 10,
        })
        results = detect_all(df)
        assert "delta" in results

    def test_delta_finds_spike_in_detect_all(self):
        n = 200
        values = np.ones(n) * 10.0
        values[100] = 5000.0
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": values,
        })
        results = detect_all(df)
        assert results["delta"].iloc[100]


class TestDeltaAsHardRule:
    """Delta'nın hybrid majority'de hard rule olarak çalıştığını doğrula."""

    def test_delta_bypasses_voting(self):
        """Delta hard rule — tek başına yeterli, oylama gerektirmez."""
        hard_delta = [pd.Series([False, True, False, False])]
        soft_zscore = [pd.Series([False, False, False, False])]
        soft_if = [pd.Series([False, False, False, False])]
        result = hybrid_majority_vote(
            hard_delta, [soft_zscore[0], soft_if[0]], min_agreement=2,
        )
        assert result.iloc[1], "Delta hard rule olmasına rağmen anomali olarak işaretlenmedi"

    def test_soft_alone_needs_agreement(self):
        """Soft dedektörler tek başına yetmemeli."""
        hard = [pd.Series([False, False, False])]
        soft1 = pd.Series([True, False, False])
        soft2 = pd.Series([False, False, False])
        result = hybrid_majority_vote(hard, [soft1, soft2], min_agreement=2)
        assert not result.iloc[0], "Tek soft dedektör yetmemeli"


class TestFaultTimelineReason:
    """fault_timeline'daki reason sütunu testleri."""

    def test_reason_column_exists(self):
        _, corrupted, _ = generate_corrupted_dataset(n=2000, seed=42)
        result = run_pipeline(corrupted, method="classic")
        ft = result["fault_timeline"]
        assert "reason" in ft.columns

    def test_reason_values_valid(self):
        _, corrupted, _ = generate_corrupted_dataset(n=2000, seed=42)
        result = run_pipeline(corrupted, method="classic")
        ft = result["fault_timeline"]
        valid_reasons = {
            "hard_rule", "statistical", "ml_outlier",
            "temporal_pattern", "unknown",
        }
        for reason in ft["reason"].unique():
            assert reason in valid_reasons, f"Geçersiz reason: {reason}"

    def test_gap_anomalies_are_hard_rule(self):
        """NaN içeren noktalar hard_rule olarak etiketlenmeli."""
        n = 500
        values = np.sin(np.linspace(0, 4 * np.pi, n)) * 10
        values[100:110] = np.nan
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": values,
        })
        result = run_pipeline(df, method="classic")
        ft = result["fault_timeline"]
        gap_faults = ft[ft["fault_type"].str.contains("gaps")]
        if len(gap_faults) > 0:
            assert all(gap_faults["reason"] == "hard_rule")


class TestPyramidEndToEnd:
    """Tüm piramit katmanlarının birlikte çalıştığını doğrula."""

    def test_all_methods_produce_timeline(self):
        import os

        has_model = os.path.exists("models/lstm_ae.pt")
        _, corrupted, _ = generate_corrupted_dataset(n=2000, seed=42)
        for method in ["classic", "ml", "both"]:
            result = run_pipeline(corrupted, method=method)
            ft = result["fault_timeline"]
            assert isinstance(ft, pd.DataFrame)
            assert "timestamp" in ft.columns
            assert "fault_type" in ft.columns
            assert "severity" in ft.columns
            assert "reason" in ft.columns
            assert "repair_decision" in ft.columns
            # ml-only without model may produce empty timeline
            if method == "classic" or has_model:
                assert len(ft) > 0, f"{method} should produce timeline"

    def test_pipeline_with_clean_signal(self):
        """Temiz sinyal az anomali üretmeli."""
        from data.synthetic_generator import generate_clean_signal

        clean = generate_clean_signal(n=1000, seed=42)
        result = run_pipeline(clean, method="classic")
        assert result["metrics"]["faults_detected"] < 50

    def test_corrupted_signal_more_faults_than_clean(self):
        clean, corrupted, _ = generate_corrupted_dataset(n=2000, seed=42)
        result_clean = run_pipeline(clean, method="classic")
        result_corrupted = run_pipeline(corrupted, method="classic")
        assert (
            result_corrupted["metrics"]["faults_detected"]
            > result_clean["metrics"]["faults_detected"]
        )


class TestRepairEligibility:
    """Repair eligibility assessment tests."""

    def test_repair_eligibility_column_exists(self):
        _, corrupted, _ = generate_corrupted_dataset(n=2000, seed=42)
        result = run_pipeline(corrupted, method="classic")
        ft = result["fault_timeline"]
        assert "repair_decision" in ft.columns

    def test_repair_eligibility_valid_values(self):
        _, corrupted, _ = generate_corrupted_dataset(n=2000, seed=42)
        result = run_pipeline(corrupted, method="classic")
        ft = result["fault_timeline"]
        valid_decisions = {"repair", "flag_only", "preserve"}
        for d in ft["repair_decision"].unique():
            assert d in valid_decisions, f"Invalid decision: {d}"

    def test_repair_hard_rules_get_repair(self):
        _, corrupted, _ = generate_corrupted_dataset(n=2000, seed=42)
        result = run_pipeline(corrupted, method="classic")
        ft = result["fault_timeline"]
        hard_faults = ft[ft["reason"] == "hard_rule"]
        if len(hard_faults) > 0:
            assert (hard_faults["repair_decision"] == "repair").all()

    def test_flatline_in_pipeline(self):
        """Flatline detector should work inside the pipeline."""
        n = 500
        values = np.sin(np.linspace(0, 4 * np.pi, n)) * 10
        values[200:250] = 7.0  # 50 points stuck
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": values,
        })
        result = run_pipeline(df, method="classic")
        ft = result["fault_timeline"]
        flatline_faults = ft[ft["fault_type"].str.contains("flatline")]
        assert len(flatline_faults) > 0, "Flatline not detected in pipeline"
