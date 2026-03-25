""" AEGIS Dashboard — Chart Components
Dark-themed Plotly figures for satellite telemetry visualization.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Design tokens ─────────────────────────────────────
BG_PAPER   = "#020408"
BG_PLOT    = "#080d14"
COLOR_CYAN = "#00d4ff"
COLOR_AMBER= "#f59e0b"
COLOR_RED  = "#ff3366"
COLOR_GREEN= "#00ff88"
COLOR_GRAY = "rgba(180,180,180,0.35)"
GRID       = "rgba(100,200,255,0.07)"
FONT       = dict(color="#e8f4ff", family="monospace")

LAYOUT_BASE = dict(
    paper_bgcolor=BG_PAPER,
    plot_bgcolor=BG_PLOT,
    font=FONT,
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color="#8bb8d4"),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color="#8bb8d4"),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(100,200,255,0.15)",
        borderwidth=1
    ),
    margin=dict(t=50, b=50, l=60, r=30),
    hovermode="x unified",
)


def plot_signal(
    df: pd.DataFrame,
    anomaly_mask: np.ndarray = None,
    title: str = "Telemetri Sinyali",
    color: str = COLOR_CYAN,
) -> go.Figure:
    """Single signal line chart with optional anomaly overlay."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index if "timestamp" not in df.columns else df["timestamp"],
        y=df["value"],
        mode="lines",
        name="Sinyal",
        line=dict(color=color, width=1.5),
        hovertemplate="<b>Değer:</b> %{y:.2f}<extra></extra>",
    ))
    
    if anomaly_mask is not None and anomaly_mask.any():
        anomaly_idx = np.where(anomaly_mask)[0]
        fig.add_trace(go.Scatter(
            x=df.index[anomaly_idx] if "timestamp" not in df.columns else df["timestamp"].iloc[anomaly_idx],
            y=df["value"].iloc[anomaly_idx],
            mode="markers",
            name="Anomali",
            marker=dict(color=COLOR_RED, size=6, symbol="x"),
            hovertemplate="<b>Anomali:</b> %{y:.2f}<extra></extra>",
        ))
    
    fig.update_layout(**LAYOUT_BASE, title=dict(text=title, x=0.5, xanchor="center"))
    return fig


def plot_comparison(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    title: str = "Orijinal vs Temizlenmiş",
) -> go.Figure:
    """Side-by-side comparison of original and cleaned signals."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Orijinal Sinyal", "Temizlenmiş Sinyal"),
        vertical_spacing=0.12,
    )
    
    x_orig = original.index if "timestamp" not in original.columns else original["timestamp"]
    x_clean = cleaned.index if "timestamp" not in cleaned.columns else cleaned["timestamp"]
    
    fig.add_trace(
        go.Scatter(
            x=x_orig, y=original["value"],
            mode="lines", name="Orijinal",
            line=dict(color=COLOR_AMBER, width=1.2),
        ),
        row=1, col=1,
    )
    
    fig.add_trace(
        go.Scatter(
            x=x_clean, y=cleaned["value"],
            mode="lines", name="Temizlenmiş",
            line=dict(color=COLOR_GREEN, width=1.2),
        ),
        row=2, col=1,
    )
    
    fig.update_xaxes(gridcolor=GRID, color="#8bb8d4")
    fig.update_yaxes(gridcolor=GRID, color="#8bb8d4")
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, x=0.5, xanchor="center"),
        showlegend=True,
        height=600,
    )
    return fig


def plot_metrics_bar(
    metrics: dict,
    title: str = "Pipeline Metrikleri",
) -> go.Figure:
    """Horizontal bar chart for pipeline metrics."""
    labels = []
    values = []
    colors = []
    
    if "faults_detected" in metrics:
        labels.append("Tespit Edilen Hata")
        values.append(metrics["faults_detected"])
        colors.append(COLOR_RED)
    
    if "faults_corrected" in metrics:
        labels.append("Düzeltilen Hata")
        values.append(metrics["faults_corrected"])
        colors.append(COLOR_GREEN)
    
    if "processing_time" in metrics:
        labels.append("İşlem Süresi (ms)")
        values.append(metrics["processing_time"] * 1000)
        colors.append(COLOR_CYAN)
    
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
        hovertemplate="<b>%{y}:</b> %{x:.2f}<extra></extra>",
    ))
    
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Değer",
        height=300,
    )
    return fig


def plot_anomaly_timeline(
    fault_timeline: pd.DataFrame,
    title: str = "Anomali Zaman Çizelgesi",
) -> go.Figure:
    """Scatter plot showing fault types over time with severity coloring."""
    if fault_timeline.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Anomali tespit edilmedi",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLOR_GRAY),
        )
        fig.update_layout(**LAYOUT_BASE, title=dict(text=title, x=0.5, xanchor="center"))
        return fig
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fault_timeline["timestamp"],
        y=fault_timeline["fault_type"],
        mode="markers",
        marker=dict(
            size=10,
            color=fault_timeline["severity"],
            colorscale=[[0, COLOR_AMBER], [1, COLOR_RED]],
            showscale=True,
            colorbar=dict(title="Şiddet", x=1.02),
            line=dict(width=1, color="rgba(255,255,255,0.3)"),
        ),
        hovertemplate="<b>Tip:</b> %{y}<br><b>Şiddet:</b> %{marker.color:.2f}<extra></extra>",
    ))
    
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Zaman",
        yaxis_title="Hata Tipi",
        height=400,
    )
    return fig
