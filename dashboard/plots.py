"""Plot helpers for the Cosmic Pipeline dashboard."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Style constants ───────────────────────────────────────────────────────────

_TEMPLATE = "plotly_dark"
_BG = "rgba(0,0,0,0)"
_GRID = "rgba(10,14,23,0.8)"

_DETECTOR_COLORS = {
    "zscore": "#3b82f6", "sliding_window": "#8b5cf6", "gaps": "#ef4444",
    "range": "#f97316", "delta": "#eab308", "flatline": "#06b6d4",
    "duplicates": "#ec4899", "isolation_forest": "#22c55e", "lstm_ae": "#00d4ff",
}

_CHANNEL_COLORS = [
    "#f59e0b", "#00d4ff", "#22c55e", "#8b5cf6", "#ef4444",
    "#ec4899", "#06b6d4", "#3b82f6", "#eab308", "#f97316",
]


def plot_clean_vs_corrupted(clean_df, corrupted_df):
    """Two-row subplot: clean signal vs corrupted."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Temiz Sinyal (Ground Truth)", "Radyasyonla Bozulmus Sinyal"],
        vertical_spacing=0.10,
    )
    fig.add_trace(go.Scatter(
        x=clean_df["timestamp"], y=clean_df["value"],
        mode="lines", name="Temiz", line=dict(color="#00ff88", width=1),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=corrupted_df["timestamp"], y=corrupted_df["value"],
        mode="lines", name="Bozuk", line=dict(color="#f59e0b", width=1),
    ), row=2, col=1)
    fig.update_layout(
        template=_TEMPLATE, paper_bgcolor=_BG, plot_bgcolor=_GRID,
        height=450, showlegend=True,
        legend=dict(orientation="h", y=-0.08),
        margin=dict(t=40, b=50, l=60, r=30),
    )
    return fig


def plot_triple_overlay(clean_df, corrupted_df, cleaned_df, fault_mask=None):
    """Overlay: ground truth + corrupted + cleaned + anomaly markers."""
    fig = go.Figure()
    if clean_df is not None:
        fig.add_trace(go.Scatter(
            x=clean_df["timestamp"], y=clean_df["value"],
            mode="lines", name="Ground Truth",
            line=dict(color="#00ff88", width=1, dash="dot"),
        ))
    fig.add_trace(go.Scatter(
        x=corrupted_df["timestamp"], y=corrupted_df["value"],
        mode="lines", name="Bozuk (Ham)",
        line=dict(color="#f59e0b", width=1), opacity=0.4,
    ))
    fig.add_trace(go.Scatter(
        x=cleaned_df["timestamp"], y=cleaned_df["value"],
        mode="lines", name="Temizlenmis",
        line=dict(color="#00d4ff", width=2),
    ))
    if fault_mask is not None:
        mask_arr = fault_mask.values if hasattr(fault_mask, "values") else np.array(fault_mask)
        if mask_arr.any():
            idx = np.where(mask_arr)[0]
            fig.add_trace(go.Scatter(
                x=corrupted_df["timestamp"].iloc[idx],
                y=corrupted_df["value"].iloc[idx],
                mode="markers", name="Anomali",
                marker=dict(color="#ff4444", size=3, symbol="x", opacity=0.6),
            ))
    fig.update_layout(
        template=_TEMPLATE, paper_bgcolor=_BG, plot_bgcolor=_GRID,
        height=400, showlegend=True,
        legend=dict(orientation="h", y=-0.12),
        margin=dict(t=40, b=60, l=60, r=30),
    )
    return fig


def plot_detector_breakdown(detector_masks, corrupted_df):
    """Scatter plot with different color per detector."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=corrupted_df["timestamp"], y=corrupted_df["value"],
        mode="lines", name="Sinyal",
        line=dict(color="#8bb8d4", width=1), opacity=0.5,
    ))
    for name, mask in detector_masks.items():
        if mask.any():
            idx = np.where(mask.values)[0]
            fig.add_trace(go.Scatter(
                x=corrupted_df["timestamp"].iloc[idx],
                y=corrupted_df["value"].iloc[idx],
                mode="markers", name=f"{name} ({int(mask.sum())})",
                marker=dict(color=_DETECTOR_COLORS.get(name, "#ffffff"), size=4),
            ))
    fig.update_layout(
        template=_TEMPLATE, paper_bgcolor=_BG, plot_bgcolor=_GRID,
        height=400, showlegend=True,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=30, b=70, l=60, r=30),
    )
    return fig


def plot_multi_channel(multi_result):
    """Build subplot figure with one row per channel."""
    channels = multi_result["channels"]
    valid = {k: v for k, v in channels.items() if "error" not in v}
    n = len(valid)
    if n == 0:
        return go.Figure()

    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True,
        subplot_titles=[f"{col} ({v['metrics']['faults_detected']} anomali)" for col, v in valid.items()],
        vertical_spacing=0.06,
    )
    for i, (col, res) in enumerate(valid.items(), 1):
        cleaned = res["cleaned_data"]
        fig.add_trace(go.Scatter(
            x=cleaned["timestamp"], y=cleaned["value"],
            mode="lines", name=col,
            line=dict(color=_CHANNEL_COLORS[(i - 1) % len(_CHANNEL_COLORS)], width=1),
            showlegend=True,
        ), row=i, col=1)
        fm = res["fault_mask"]
        if fm.any():
            idx = np.where(fm.values)[0]
            fig.add_trace(go.Scatter(
                x=cleaned["timestamp"].iloc[idx],
                y=cleaned["value"].iloc[idx],
                mode="markers", name=f"{col} anomali",
                marker=dict(color="#ff4444", size=3, symbol="x"),
                showlegend=False,
            ), row=i, col=1)

    fig.update_layout(
        template=_TEMPLATE, paper_bgcolor=_BG, plot_bgcolor=_GRID,
        height=max(300, 200 * n), showlegend=True,
        legend=dict(orientation="h", y=-0.05),
        margin=dict(t=30, b=50, l=60, r=30),
    )
    return fig
