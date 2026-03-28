"""Tests for ML detector using the trained LSTM model."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

MODEL_PATH = Path("models/lstm_ae.pt")
pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="Trained model not found")


@pytest.fixture
def corrupted_signal():
    """Signal with obvious spikes and gaps."""
    n = 1000
    t = np.arange(n)
    values = np.sin(2 * np.pi * 0.01 * t) * 10
    # Inject spikes
    values[100] = 500.0
    values[300] = -500.0
    # Inject gap
    values[600:610] = np.nan
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
        "value": values,
    })
    mask = pd.Series(np.zeros(n, dtype=bool))
    mask.iloc[100] = True
    mask.iloc[300] = True
    mask.iloc[600:610] = True
    return df, mask


class TestLSTMDetectorTrained:
    def test_detects_anomalies(self, corrupted_signal):
        from pipeline.detector_ml import detect_with_lstm
        df, _ = corrupted_signal
        mask = detect_with_lstm(df, model_path=MODEL_PATH, threshold_percentile=95)
        assert mask.sum() > 0

    def test_returns_correct_length(self, corrupted_signal):
        from pipeline.detector_ml import detect_with_lstm
        df, _ = corrupted_signal
        mask = detect_with_lstm(df, model_path=MODEL_PATH)
        assert len(mask) == len(df)

    def test_higher_percentile_fewer_detections(self, corrupted_signal):
        from pipeline.detector_ml import detect_with_lstm
        df, _ = corrupted_signal
        loose = detect_with_lstm(df, model_path=MODEL_PATH, threshold_percentile=90)
        strict = detect_with_lstm(df, model_path=MODEL_PATH, threshold_percentile=99)
        assert strict.sum() <= loose.sum()


class TestDetectAllML:
    def test_returns_both_detectors(self, corrupted_signal):
        from pipeline.detector_ml import detect_all_ml
        df, _ = corrupted_signal
        results = detect_all_ml(df, model_path=MODEL_PATH)
        assert "isolation_forest" in results
        assert "lstm_ae" in results
        for name, mask in results.items():
            assert isinstance(mask, pd.Series)
            assert mask.dtype == bool
            assert len(mask) == len(df)
