"""Unit tests for pipeline.validator module."""

import numpy as np
import pandas as pd
import pytest

from pipeline.validator import calculate_metrics, validate_output


@pytest.fixture
def good_df():
    """Clean, valid signal."""
    n = 500
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
        "value": np.sin(np.linspace(0, 4 * np.pi, n)) * 10,
    })


# --- validate_output ---

def test_valid_signal(good_df):
    result = validate_output(good_df)
    assert result["is_valid"] is True
    assert result["quality_score"] == 1.0
    assert result["issues"] == []


def test_high_nan_ratio():
    values = np.full(100, np.nan)
    values[:10] = 1.0
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="1s"),
        "value": values,
    })
    result = validate_output(df)
    assert result["is_valid"] is False
    assert result["quality_score"] < 0.7
    assert any("NaN" in issue for issue in result["issues"])


def test_infinite_values():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1s"),
        "value": [1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    })
    result = validate_output(df)
    assert result["is_valid"] is False
    assert any("infinite" in issue for issue in result["issues"])


def test_constant_signal():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=200, freq="1s"),
        "value": np.ones(200) * 5.0,
    })
    result = validate_output(df)
    assert result["is_valid"] is False
    assert any("variance" in issue for issue in result["issues"])


def test_short_signal():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1s"),
        "value": np.arange(10, dtype=float),
    })
    result = validate_output(df)
    assert any("short" in issue for issue in result["issues"])


def test_non_monotonic_timestamps():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]),
        "value": [1.0, 2.0, 3.0],
    })
    result = validate_output(df)
    assert any("monoton" in issue for issue in result["issues"])


def test_quality_score_clamped():
    # Many issues should not push score below 0
    values = np.full(50, np.nan)
    values[0] = np.inf
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-03", "2024-01-01"] * 25),
        "value": values,
    })
    result = validate_output(df)
    assert 0.0 <= result["quality_score"] <= 1.0


# --- calculate_metrics ---

def test_perfect_reconstruction():
    gt = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    cleaned = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    original = pd.DataFrame({"value": [10.0, 20.0, 30.0, 40.0, 50.0]})

    metrics = calculate_metrics(original, cleaned, ground_truth=gt)
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2_score"] == 1.0


def test_metrics_with_noise():
    t = np.linspace(0, 4 * np.pi, 100)
    gt_vals = np.sin(t) * 10
    gt = pd.DataFrame({"value": gt_vals})
    cleaned = pd.DataFrame({"value": gt_vals + np.random.default_rng(42).normal(0, 0.01, 100)})
    original = pd.DataFrame({"value": gt_vals + np.random.default_rng(7).normal(0, 5.0, 100)})

    metrics = calculate_metrics(original, cleaned, ground_truth=gt)
    assert metrics["rmse"] < 0.05
    assert metrics["mae"] < 0.05
    assert metrics["r2_score"] > 0.99
    assert metrics["snr"] > 20


def test_metrics_without_ground_truth():
    original = pd.DataFrame({"value": [1.0, 5.0, 3.0]})
    cleaned = pd.DataFrame({"value": [1.0, 2.0, 3.0]})

    metrics = calculate_metrics(original, cleaned)
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert "r2_score" in metrics
    assert "snr" in metrics


def test_metrics_handles_nan():
    gt = pd.DataFrame({"value": [1.0, np.nan, 3.0, 4.0]})
    cleaned = pd.DataFrame({"value": [1.0, 2.0, np.nan, 4.0]})
    original = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]})

    metrics = calculate_metrics(original, cleaned, ground_truth=gt)
    # Only points [0] and [3] are both finite in gt and cleaned
    assert np.isfinite(metrics["rmse"])


def test_metrics_keys():
    original = pd.DataFrame({"value": [1.0, 2.0]})
    cleaned = pd.DataFrame({"value": [1.0, 2.0]})
    metrics = calculate_metrics(original, cleaned)
    assert set(metrics.keys()) == {"rmse", "mae", "r2_score", "snr"}


# --- calculate_repair_confidence ---

from pipeline.validator import calculate_repair_confidence, verify_repair


class TestRepairConfidence:

    def test_clean_signal_full_confidence(self):
        n = 100
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": np.ones(n) * 10.0,
        })
        mask = pd.Series(np.zeros(n, dtype=bool))
        conf = calculate_repair_confidence(df, df, mask)
        assert (conf == 1.0).all()

    def test_fault_points_have_lower_confidence(self):
        n = 100
        orig_vals = np.ones(n) * 10.0
        orig_vals[50] = 9999.0
        orig = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": orig_vals,
        })
        cleaned = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": np.ones(n) * 10.0,
        })
        mask = pd.Series(np.zeros(n, dtype=bool))
        mask.iloc[50] = True
        conf = calculate_repair_confidence(orig, cleaned, mask)
        assert conf.iloc[50] < 1.0
        assert conf.iloc[0] == 1.0

    def test_confidence_with_detector_masks(self):
        n = 50
        orig_vals = np.ones(n) * 10.0
        orig_vals[25] = 500.0
        orig = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": orig_vals,
        })
        cleaned = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": np.ones(n) * 10.0,
        })
        mask = pd.Series(np.zeros(n, dtype=bool))
        mask.iloc[25] = True
        det_masks = {
            "zscore": mask.copy(),
            "range": mask.copy(),
            "delta": mask.copy(),
        }
        conf = calculate_repair_confidence(orig, cleaned, mask, det_masks)
        assert conf.iloc[25] > 0.3


# --- verify_repair ---

class TestRepairVerification:

    def test_good_repair_passes(self):
        n = 100
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": np.ones(n) * 10.0,
        })
        mask = pd.Series(np.zeros(n, dtype=bool))
        result = verify_repair(df, df, mask)
        assert result["passed"]
        assert result["new_nan_count"] == 0
        assert result["new_inf_count"] == 0

    def test_repair_with_new_nan_fails(self):
        n = 100
        orig = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": np.ones(n) * 10.0,
        })
        clean_vals = np.ones(n) * 10.0
        clean_vals[50] = np.nan
        cleaned = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": clean_vals,
        })
        mask = pd.Series(np.zeros(n, dtype=bool))
        mask.iloc[50] = True
        result = verify_repair(orig, cleaned, mask)
        assert not result["passed"]
        assert result["new_nan_count"] > 0

    def test_repair_with_new_inf_fails(self):
        n = 100
        orig = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": np.ones(n) * 10.0,
        })
        clean_vals = np.ones(n) * 10.0
        clean_vals[50] = np.inf
        cleaned = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": clean_vals,
        })
        mask = pd.Series(np.zeros(n, dtype=bool))
        result = verify_repair(orig, cleaned, mask)
        assert not result["passed"]


# --- end-to-end confidence ---

class TestEndToEndConfidence:

    def test_pipeline_returns_confidence(self):
        from data.synthetic_generator import generate_corrupted_dataset
        from pipeline.orchestrator import run_pipeline

        _, corrupted, _ = generate_corrupted_dataset(n=2000, seed=42)
        result = run_pipeline(corrupted, method="classic")
        assert "repair_confidence" in result
        assert "repair_verification" in result
        assert len(result["repair_confidence"]) == len(corrupted)
        assert result["repair_verification"]["passed"] is not None

    def test_pipeline_confidence_range(self):
        from data.synthetic_generator import generate_corrupted_dataset
        from pipeline.orchestrator import run_pipeline

        _, corrupted, _ = generate_corrupted_dataset(n=2000, seed=42)
        result = run_pipeline(corrupted, method="classic")
        conf = result["repair_confidence"]
        assert (conf >= 0.0).all()
        assert (conf <= 1.0).all()
