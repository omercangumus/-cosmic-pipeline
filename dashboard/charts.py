"""Plotly visualization charts with dark theme for cosmic-pipeline dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional


# Dark theme configuration
DARK_THEME = dict(
    paper_bgcolor="#020408",
    plot_bgcolor="#080d14",
    font=dict(color="#e8f4ff"),
    xaxis=dict(
        gridcolor="rgba(100,200,255,0.08)",
        zerolinecolor="rgba(100,200,255,0.1)"
    ),
    yaxis=dict(
        gridcolor="rgba(100,200,255,0.08)",
        zerolinecolor="rgba(100,200,255,0.1)"
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(100,200,255,0.2)",
        borderwidth=1
    ),
    margin=dict(t=50, b=50, l=60, r=30)
)


def plot_signal(
    df: pd.DataFrame,
    anomaly_mask: Optional[np.ndarray] = None,
    title: str = "Telemetry Signal",
    color: str = "#00d4ff"
) -> go.Figure:
    """
    Plot single signal with optional anomaly overlay.
    
    Args:
        df: DataFrame with columns [timestamp, value]
        anomaly_mask: Boolean array marking anomalies
        title: Plot title
        color: Line color
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Main signal line
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["value"],
        mode="lines",
        name="Signal",
        line=dict(color=color, width=1.2),
    ))
    
    # Anomaly markers
    if anomaly_mask is not None and anomaly_mask.any():
        flagged = df[anomaly_mask.astype(bool)]
        fig.add_trace(go.Scatter(
            x=flagged["timestamp"],
            y=flagged["value"],
            mode="markers",
            name="Anomaly",
            marker=dict(
                color="#ff3366",
                size=6,
                opacity=0.85,
                symbol="circle",
                line=dict(color="#ff0044", width=1)
            )
        ))
    
    fig.update_layout(title=title, **DARK_THEME)
    return fig


def plot_comparison(
    df_original: pd.DataFrame,
    df_classic: pd.DataFrame,
    df_ml: pd.DataFrame,
    classic_mask: Optional[np.ndarray] = None,
    ml_mask: Optional[np.ndarray] = None
) -> go.Figure:
    """
    Overlay original, classic-cleaned and ML-cleaned signals.
    
    Args:
        df_original: Original corrupted signal
        df_classic: Classic DSP cleaned signal
        df_ml: ML cleaned signal
        classic_mask: Anomalies detected by classic methods
        ml_mask: Anomalies detected by ML
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Original signal (faded)
    fig.add_trace(go.Scatter(
        x=df_original["timestamp"],
        y=df_original["value"],
        mode="lines",
        name="Original",
        line=dict(color="rgba(180,180,180,0.4)", width=1),
    ))
    
    # Classic DSP cleaned
    fig.add_trace(go.Scatter(
        x=df_classic["timestamp"],
        y=df_classic["value"],
        mode="lines",
        name="Classic DSP",
        line=dict(color="#f59e0b", width=1.5),
    ))
    
    # ML cleaned
    fig.add_trace(go.Scatter(
        x=df_ml["timestamp"],
        y=df_ml["value"],
        mode="lines",
        name="ML (LSTM AE)",
        line=dict(color="#00d4ff", width=1.5),
    ))
    
    # Classic detection markers
    if classic_mask is not None and classic_mask.any():
        flagged = df_original[classic_mask.astype(bool)]
        fig.add_trace(go.Scatter(
            x=flagged["timestamp"],
            y=flagged["value"],
            mode="markers",
            name="Classic Flagged",
            marker=dict(color="#f59e0b", size=5, symbol="x", opacity=0.7)
        ))
    
    # ML detection markers
    if ml_mask is not None and ml_mask.any():
        flagged = df_original[ml_mask.astype(bool)]
        fig.add_trace(go.Scatter(
            x=flagged["timestamp"],
            y=flagged["value"],
            mode="markers",
            name="ML Flagged",
            marker=dict(color="#00d4ff", size=5, symbol="x", opacity=0.7)
        ))
    
    fig.update_layout(
        title="Signal Comparison: Original vs Cleaned",
        **DARK_THEME
    )
    return fig


def plot_metrics_bar(
    metrics_classic: dict,
    metrics_ml: dict
) -> go.Figure:
    """
    Grouped bar chart: Classic DSP vs ML metrics.
    
    Args:
        metrics_classic: Dictionary with precision, recall, f1 keys
        metrics_ml: Dictionary with precision, recall, f1 keys
        
    Returns:
        Plotly Figure object
    """
    metrics_to_show = ["precision", "recall", "f1"]
    labels = ["Spike Precision", "Drift Recall", "F1 Score"]
    
    classic_vals = [metrics_classic.get(m, 0) * 100 for m in metrics_to_show]
    ml_vals = [metrics_ml.get(m, 0) * 100 for m in metrics_to_show]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="Classic DSP",
        x=labels,
        y=classic_vals,
        marker_color="#f59e0b",
        text=[f"{v:.1f}%" for v in classic_vals],
        textposition="outside"
    ))
    
    fig.add_trace(go.Bar(
        name="ML (LSTM AE)",
        x=labels,
        y=ml_vals,
        marker_color="#00d4ff",
        text=[f"{v:.1f}%" for v in ml_vals],
        textposition="outside"
    ))
    
    theme_copy = DARK_THEME.copy()
    theme_copy["yaxis"] = dict(
        range=[0, 115],
        ticksuffix="%",
        gridcolor="rgba(100,200,255,0.08)"
    )
    
    fig.update_layout(
        barmode="group",
        title="Classic DSP vs ML Performance",
        **theme_copy
    )
    return fig


def plot_anomaly_timeline(
    n_points: int,
    ground_truth_mask: dict,
    pred_mask: np.ndarray
) -> go.Figure:
    """
    Horizontal timeline: ground truth fault regions vs ensemble detections.
    
    Args:
        n_points: Total number of signal points
        ground_truth_mask: Dict with keys: seu, tid, gap, noise
        pred_mask: Boolean array of ensemble predictions
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    x_axis = list(range(n_points))
    
    fault_config = [
        ("seu", "#ff3366", "GT: SEU"),
        ("tid", "#f59e0b", "GT: TID Drift"),
        ("gap", "#a78bfa", "GT: Data Gap"),
        ("noise", "#00d4ff", "GT: Noise"),
    ]
    
    for fault_key, color, label in fault_config:
        indices = ground_truth_mask.get(fault_key, [])
        if not indices:
            continue
        
        mask = np.zeros(n_points, dtype=bool)
        
        if fault_key == "gap":
            for start, end in indices:
                mask[start:end] = True
        else:
            valid = [i for i in indices if i < n_points]
            if valid:
                mask[np.array(valid)] = True
        
        colors = [color if m else "rgba(0,0,0,0)" for m in mask]
        
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=[label] * n_points,
            mode="markers",
            marker=dict(color=colors, size=5, symbol="square"),
            name=label,
            showlegend=True
        ))
    
    # Ensemble prediction
    pred_colors = ["#00ff88" if m else "rgba(0,0,0,0)" for m in pred_mask]
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=["Ensemble Detected"] * n_points,
        mode="markers",
        marker=dict(color=pred_colors, size=5, symbol="square"),
        name="Ensemble Detected",
        showlegend=True
    ))
    
    fig.update_layout(
        title="Anomaly Detection Timeline",
        height=280,
        paper_bgcolor="#020408",
        plot_bgcolor="#080d14",
        font=dict(color="#e8f4ff"),
        xaxis=dict(title="Sample Index", gridcolor="rgba(100,200,255,0.05)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=40, b=40, l=140, r=30)
    )
    return fig
