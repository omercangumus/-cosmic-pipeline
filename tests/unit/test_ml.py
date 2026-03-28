"""Unit tests for ML modules: lstm_autoencoder, detector_ml, filters_ml."""

import numpy as np
import pandas as pd
import pytest
import torch

from models.lstm_autoencoder import LSTMAutoencoder
from models.train import create_sequences, normalize


# === LSTMAutoencoder ===

class TestLSTMAutoencoder:
    def test_forward_shape(self):
        model = LSTMAutoencoder(input_dim=1, hidden_dim=64, latent_dim=32, num_layers=2)
        x = torch.randn(4, 50, 1)
        out = model(x)
        assert out.shape == (4, 50, 1)

    def test_reconstruction_error_shape(self):
        model = LSTMAutoencoder(input_dim=1, hidden_dim=64, latent_dim=32, num_layers=2)
        x = torch.randn(4, 50, 1)
        err = model.reconstruction_error(x)
        assert err.shape == (4, 50)

    def test_reconstruction_error_nonnegative(self):
        model = LSTMAutoencoder()
        x = torch.randn(2, 30, 1)
        err = model.reconstruction_error(x)
        assert (err >= 0).all()

    def test_different_sequence_lengths(self):
        model = LSTMAutoencoder()
        for seq_len in [10, 50, 100]:
            x = torch.randn(2, seq_len, 1)
            out = model(x)
            assert out.shape == (2, seq_len, 1)

    def test_encoder_decoder_dimensions(self):
        model = LSTMAutoencoder(input_dim=1, hidden_dim=32, latent_dim=16, num_layers=1)
        assert model.encoder.hidden_dim == 32
        assert model.encoder.latent_dim == 16
        assert model.decoder.hidden_dim == 32


# === Training utilities ===

class TestTrainUtils:
    def test_create_sequences_shape(self):
        values = np.sin(np.linspace(0, 10, 200))
        seqs = create_sequences(values, window_size=50)
        assert seqs.shape == (151, 50, 1)

    def test_create_sequences_too_short(self):
        values = np.ones(10)
        with pytest.raises(ValueError, match="Signal length"):
            create_sequences(values, window_size=50)

    def test_normalize_roundtrip(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normed, mean, std = normalize(values)
        recovered = normed * std + mean
        np.testing.assert_allclose(recovered, values, atol=1e-10)

    def test_normalize_zero_std(self):
        values = np.ones(10) * 5.0
        normed, mean, std = normalize(values)
        assert std == 1.0
        assert mean == 5.0


# === Isolation Forest ===

class TestIsolationForest:
    def test_detects_spikes(self):
        from pipeline.detector_ml import detect_isolation_forest

        n = 1000
        values = np.sin(np.linspace(0, 4 * np.pi, n)) * 10
        values[100] = 500.0
        values[500] = -500.0
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": values,
        })
        mask = detect_isolation_forest(df, contamination=0.05)
        assert mask.iloc[100], "Spike at 100 not detected"
        assert mask.iloc[500], "Spike at 500 not detected"

    def test_returns_bool_series(self):
        from pipeline.detector_ml import detect_isolation_forest

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=200, freq="1s"),
            "value": np.sin(np.linspace(0, 2 * np.pi, 200)),
        })
        mask = detect_isolation_forest(df)
        assert isinstance(mask, pd.Series)
        assert mask.dtype == bool

    def test_contamination_controls_count(self):
        from pipeline.detector_ml import detect_isolation_forest

        n = 1000
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
            "value": np.random.default_rng(42).normal(0, 1, n),
        })
        low = detect_isolation_forest(df, contamination=0.01)
        high = detect_isolation_forest(df, contamination=0.1)
        assert high.sum() > low.sum()


# === LSTM Detector (without trained model) ===

class TestLSTMDetectorNoModel:
    def test_missing_model_returns_empty_mask(self):
        from pipeline.detector_ml import detect_with_lstm

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="1s"),
            "value": np.ones(100),
        })
        mask = detect_with_lstm(df, model_path="nonexistent/model.pt")
        assert mask.sum() == 0
        assert len(mask) == 100


# === ML Filter stub (always delegates to interpolation) ===

class TestMLFilterStub:
    def test_stub_delegates_to_interpolation(self):
        from pipeline.filters_ml import reconstruct_with_lstm

        values = np.array([1.0, 2.0, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1s"),
            "value": values,
        })
        mask = pd.Series([False, False, True, True, False, False, False, False, False, False])

        result = reconstruct_with_lstm(df, mask)
        assert result["value"].isna().sum() == 0
        assert len(result) == 10
