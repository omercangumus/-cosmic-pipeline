"""End-to-end integration test: synthetic data -> pipeline -> metrics check."""

import numpy as np
import pandas as pd
import pytest

from data.synthetic_generator import generate_clean_signal, inject_faults
from pipeline.orchestrator import run_pipeline
from pipeline.validator import calculate_metrics


@pytest.fixture
def synthetic_data():
    """Generate clean + corrupted synthetic telemetry pair."""
    clean = generate_clean_signal(n=5000, seed=42)
    corrupted, ground_truth = inject_faults(clean, seed=42)
    return clean, corrupted, ground_truth


def test_pipeline_returns_expected_keys(synthetic_data):
    _, corrupted, _ = synthetic_data
    result = run_pipeline(corrupted)

    assert "cleaned_data" in result
    assert "fault_mask" in result
    assert "metrics" in result
    assert "fault_timeline" in result


def test_pipeline_cleaned_data_shape(synthetic_data):
    _, corrupted, _ = synthetic_data
    result = run_pipeline(corrupted)

    assert len(result["cleaned_data"]) == len(corrupted)
    assert "timestamp" in result["cleaned_data"].columns
    assert "value" in result["cleaned_data"].columns


def test_pipeline_no_nan_after_cleaning(synthetic_data):
    _, corrupted, _ = synthetic_data
    result = run_pipeline(corrupted)

    nan_count = result["cleaned_data"]["value"].isna().sum()
    assert nan_count == 0, f"Cleaned data still has {nan_count} NaN values"


def test_pipeline_detects_faults(synthetic_data):
    _, corrupted, _ = synthetic_data
    result = run_pipeline(corrupted)

    assert result["metrics"]["faults_detected"] > 0
    assert result["metrics"]["faults_corrected"] > 0
    assert result["metrics"]["processing_time"] > 0


def test_pipeline_improves_signal(synthetic_data):
    clean, corrupted, _ = synthetic_data
    result = run_pipeline(corrupted)

    before = calculate_metrics(corrupted, corrupted, ground_truth=clean)
    after = calculate_metrics(corrupted, result["cleaned_data"], ground_truth=clean)

    # RMSE should decrease after cleaning
    assert after["rmse"] < before["rmse"], (
        f"RMSE did not improve: before={before['rmse']:.4f}, after={after['rmse']:.4f}"
    )


def test_pipeline_fault_mask_is_bool(synthetic_data):
    _, corrupted, _ = synthetic_data
    result = run_pipeline(corrupted)

    assert result["fault_mask"].dtype == bool
    assert len(result["fault_mask"]) == len(corrupted)


def test_pipeline_fault_timeline_structure(synthetic_data):
    _, corrupted, _ = synthetic_data
    result = run_pipeline(corrupted)

    ft = result["fault_timeline"]
    assert isinstance(ft, pd.DataFrame)
    assert list(ft.columns) == ["timestamp", "fault_type", "severity", "reason", "repair_decision"]
    assert len(ft) == result["metrics"]["faults_detected"]
    assert (ft["severity"] >= 0).all() and (ft["severity"] <= 1).all()


def test_pipeline_with_config(synthetic_data):
    _, corrupted, _ = synthetic_data
    config = {
        "dsp_detector": {"zscore_threshold": 4.0},
        "ensemble": {"min_agreement": 2},
        "classic_filter": {"median_window": 7},
    }
    result = run_pipeline(corrupted, config=config)

    assert result["metrics"]["faults_detected"] >= 0
    assert result["cleaned_data"]["value"].isna().sum() == 0


def test_pipeline_clean_signal_minimal_changes():
    """Running pipeline on clean data should not introduce large artifacts."""
    clean = generate_clean_signal(n=2000, seed=99)
    result = run_pipeline(clean)

    cleaned_std = result["cleaned_data"]["value"].std()
    original_std = clean["value"].std()
    # Detrend removes the DC offset (~20 C baseline), but signal shape
    # (orbital cycle amplitude) should be roughly preserved.
    assert cleaned_std > original_std * 0.3, "Pipeline destroyed the signal"
    assert result["cleaned_data"]["value"].isna().sum() == 0


def test_pipeline_invalid_method():
    df = generate_clean_signal(n=500)
    with pytest.raises(ValueError, match="Invalid method"):
        run_pipeline(df, method="invalid")


# --- ML method tests (require trained model) ---

from pathlib import Path
MODEL_EXISTS = Path("models/lstm_ae.pt").exists()


@pytest.mark.skipif(not MODEL_EXISTS, reason="Trained model not found")
class TestPipelineML:
    def test_ml_method_runs(self, synthetic_data):
        _, corrupted, _ = synthetic_data
        result = run_pipeline(corrupted, method="ml")
        assert result["metrics"]["faults_detected"] > 0
        assert result["cleaned_data"]["value"].isna().sum() == 0

    def test_both_method_runs(self, synthetic_data):
        _, corrupted, _ = synthetic_data
        result = run_pipeline(corrupted, method="both")
        assert result["metrics"]["faults_detected"] > 0
        assert result["cleaned_data"]["value"].isna().sum() == 0

    def test_both_uses_more_detectors(self, synthetic_data):
        _, corrupted, _ = synthetic_data
        classic = run_pipeline(corrupted, method="classic")
        both = run_pipeline(corrupted, method="both")
        # 'both' has more detector types in fault_timeline
        classic_types = set()
        both_types = set()
        for ft in classic["fault_timeline"]["fault_type"]:
            classic_types.update(ft.split("+"))
        for ft in both["fault_timeline"]["fault_type"]:
            both_types.update(ft.split("+"))
        assert len(both_types) >= len(classic_types)

    def test_ml_improves_signal(self, synthetic_data):
        clean, corrupted, _ = synthetic_data
        result = run_pipeline(corrupted, method="ml")
        before = calculate_metrics(corrupted, corrupted, ground_truth=clean)
        after = calculate_metrics(corrupted, result["cleaned_data"], ground_truth=clean)
        assert after["rmse"] < before["rmse"]
