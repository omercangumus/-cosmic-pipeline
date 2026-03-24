"""Unit tests for dashboard charts and components."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from dashboard.charts import (
    plot_signal,
    plot_comparison,
    plot_metrics_bar,
    plot_anomaly_timeline
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    n = 200
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
        "value": np.sin(np.linspace(0, 4 * np.pi, n))
    })


def test_plot_signal_returns_figure(sample_df):
    """Test that plot_signal returns a Plotly Figure."""
    import plotly.graph_objects as go
    fig = plot_signal(sample_df)
    assert isinstance(fig, go.Figure)


def test_plot_signal_with_mask(sample_df):
    """Test that plot_signal handles anomaly mask correctly."""
    mask = np.zeros(len(sample_df), dtype=bool)
    mask[10:15] = True
    fig = plot_signal(sample_df, anomaly_mask=mask)
    assert len(fig.data) == 2  # signal + anomaly scatter


def test_plot_comparison_returns_figure(sample_df):
    """Test that plot_comparison returns a Plotly Figure."""
    import plotly.graph_objects as go
    fig = plot_comparison(sample_df, sample_df, sample_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 3


def test_plot_metrics_bar_returns_figure():
    """Test that plot_metrics_bar returns a Plotly Figure."""
    import plotly.graph_objects as go
    classic = {"precision": 0.72, "recall": 0.61, "f1": 0.66}
    ml = {"precision": 0.91, "recall": 0.89, "f1": 0.90}
    fig = plot_metrics_bar(classic, ml)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_plot_anomaly_timeline_returns_figure():
    """Test that plot_anomaly_timeline returns a Plotly Figure."""
    import plotly.graph_objects as go
    gt = {
        "seu": [1, 5, 10],
        "tid": list(range(100)),
        "gap": [(20, 30)],
        "noise": []
    }
    pred = np.zeros(100, dtype=bool)
    pred[20:35] = True
    fig = plot_anomaly_timeline(100, gt, pred)
    assert isinstance(fig, go.Figure)


def test_goes_downloader_fallback():
    """Test that get_goes_dataframe() returns DataFrame on network failure."""
    import requests
    with patch("data.goes_downloader.requests.get", side_effect=requests.exceptions.RequestException("Network error")):
        from data.goes_downloader import get_goes_dataframe
        df = get_goes_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        # Should have fallback to synthetic data
        assert "channel" in df.columns
