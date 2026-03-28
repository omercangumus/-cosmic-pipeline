"""Plotly Dash wizard dashboard for the Cosmic Signal Processing Pipeline."""

import base64
import io
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, State, dcc, html, no_update, ctx
import dash_bootstrap_components as dbc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.synthetic_generator import FaultConfig, generate_corrupted_dataset
from pipeline.detector_classic import detect_all as detect_classic
from pipeline.ensemble import ensemble_vote
from pipeline.filters_classic import apply_classic_filters, detrend_signal
from pipeline.ingestion import load_data, preprocess, validate_schema
from pipeline.validator import calculate_metrics, validate_output

# ── Constants ────────────────────────────────────────────────────────────────

PLOT_THEME = dict(
    paper_bgcolor="#0d1220",
    plot_bgcolor="#0a0e17",
    font=dict(color="#e8f4ff", family="Segoe UI, system-ui"),
    xaxis=dict(gridcolor="rgba(30,42,74,0.6)", zerolinecolor="rgba(30,42,74,0.5)"),
    yaxis=dict(gridcolor="rgba(30,42,74,0.6)", zerolinecolor="rgba(30,42,74,0.5)"),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(30,42,74,0.8)",
        borderwidth=1,
    ),
    margin=dict(t=50, b=50, l=60, r=30),
)

DETECTOR_COLORS = {
    "zscore": "#4488ff",
    "iqr": "#a855f7",
    "sliding_window": "#f59e0b",
    "gap": "#ff4444",
    "gaps": "#ff4444",
    "isolation_forest": "#00ff88",
    "lstm_ae": "#00d4ff",
}

DETECTOR_NAMES_TR = {
    "zscore": "Z-Score",
    "iqr": "IQR",
    "sliding_window": "Kayan Pencere",
    "gap": "Boşluk Tespiti",
    "gaps": "Boşluk Tespiti",
    "isolation_forest": "Isolation Forest",
    "lstm_ae": "LSTM Autoencoder",
}

CLASSIC_KEYS = {"zscore", "iqr", "sliding_window", "gap", "gaps"}
ML_KEYS = {"isolation_forest", "lstm_ae"}

STEP_LABELS = [
    "Veri Kaynağı",
    "Veri Yükleme",
    "Anomali Tespiti",
    "Ensemble Oylama",
    "Filtreleme",
    "Sonuçlar",
]

# ── Serialization helpers ─────────────────────────────────────────────────────


def _df_to_store(df: pd.DataFrame) -> dict:
    return df.to_dict(orient="split")


def _store_to_df(data: dict) -> pd.DataFrame:
    df = pd.DataFrame(data["data"], columns=data["columns"])
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _mask_to_store(mask: np.ndarray) -> list:
    return [bool(v) for v in mask]


def _store_to_mask(data: list) -> np.ndarray:
    return np.array(data, dtype=bool)


# ── Helper functions ──────────────────────────────────────────────────────────


def _gt_bool_mask(gt: dict, n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for i in gt.get("seu", []):
        if 0 <= i < n:
            mask[i] = True
    for s, e in gt.get("gap", []):
        mask[s : min(e, n)] = True
    for i in gt.get("noise", []):
        if 0 <= i < n:
            mask[i] = True
    if gt.get("tid"):
        mask[int(n * 0.4) :] = True
    return mask


def _precision_recall(det: np.ndarray, truth: np.ndarray):
    tp = np.sum(det & truth)
    fp = np.sum(det & ~truth)
    fn = np.sum(~det & truth)
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# ── UI component helpers ──────────────────────────────────────────────────────


def metric_card(label: str, value: str, color: str = "cyan", sub: str = "") -> html.Div:
    color_class = color if color in {"cyan", "green", "orange", "red", "purple"} else "cyan"
    inner = [
        html.Div(label, className="metric-label"),
        html.Div(value, className=f"metric-value {color_class}"),
    ]
    if sub:
        inner.append(html.Div(sub, className="metric-sub"))
    return html.Div(inner, className="metric-card fade-in")


def badge(text: str, color: str = "cyan") -> html.Span:
    return html.Span(text, className=f"badge badge-{color}")


def alert_box(text: str, kind: str = "info") -> html.Div:
    icons = {"info": "i", "success": "OK", "warning": "!", "error": "X"}
    icon_map = {"info": "2139", "success": "2705", "warning": "26A0", "error": "274C"}
    icon = chr(int(icon_map.get(kind, "2139"), 16))
    return html.Div(
        [
            html.Span(icon, style={"fontSize": "16px", "flexShrink": "0"}),
            html.Span(text),
        ],
        className=f"alert alert-{kind}",
    )


def check_item(label: str, detail: str = "", ok: bool = True) -> html.Div:
    icon = chr(0x2705) if ok else chr(0x274C)
    children = [
        html.Span(icon, className="check-icon"),
        html.Span(
            [
                html.Span(label, style={"color": "#e8f4ff", "fontWeight": "500"}),
            ]
            + (
                [html.Span(f" - {detail}", style={"color": "#8ab4c8"})]
                if detail
                else []
            )
        ),
    ]
    return html.Div(children, className="check-item")


# ── Stepper renderer ──────────────────────────────────────────────────────────


def render_stepper(step: int) -> html.Div:
    items = []
    for i, label in enumerate(STEP_LABELS):
        if i < step:
            circle_class = "step-circle done"
            label_class = "step-label done"
            circle_content = chr(0x2713)
        elif i == step:
            circle_class = "step-circle active"
            label_class = "step-label active"
            circle_content = str(i + 1)
        else:
            circle_class = "step-circle"
            label_class = "step-label future"
            circle_content = str(i + 1)

        cell = html.Div(
            [
                html.Div(circle_content, className=circle_class),
                html.Div(label, className=label_class),
            ],
            className="step-cell",
        )
        items.append(html.Div([cell], className="step-item", style={"flex": "0 0 auto"}))

        if i < len(STEP_LABELS) - 1:
            if i < step:
                line_class = "step-line done"
            elif i == step:
                line_class = "step-line active"
            else:
                line_class = "step-line"
            items.append(html.Div(className=line_class, style={"flex": "1"}))

    return html.Div(items, className="stepper")


# ── Sidebar ───────────────────────────────────────────────────────────────────


def render_sidebar() -> html.Div:
    return html.Div(
        [
            # Analysis method
            html.Div(
                [
                    html.Div("Analiz Yontemi", className="sidebar-label"),
                    dcc.RadioItems(
                        id="radio-method",
                        options=[
                            {"label": "Klasik DSP", "value": "classic"},
                            {"label": "ML Tabanli", "value": "ml"},
                            {"label": "Her Ikisi", "value": "both"},
                        ],
                        value="classic",
                        labelStyle={
                            "display": "block",
                            "marginBottom": "8px",
                            "cursor": "pointer",
                        },
                        inputStyle={"marginRight": "8px"},
                    ),
                ],
                className="sidebar-section",
            ),
            # Data source
            html.Div(
                [
                    html.Div("Veri Kaynagi", className="sidebar-label"),
                    dcc.RadioItems(
                        id="radio-source",
                        options=[
                            {"label": "Sentetik Uret", "value": "synthetic"},
                            {"label": "CSV Yukle", "value": "csv"},
                        ],
                        value="synthetic",
                        labelStyle={
                            "display": "block",
                            "marginBottom": "8px",
                            "cursor": "pointer",
                        },
                        inputStyle={"marginRight": "8px"},
                    ),
                ],
                className="sidebar-section",
            ),
            # Synthetic params
            html.Div(
                id="div-synthetic-params",
                children=[
                    html.Div("Sentetik Parametreler", className="sidebar-label"),
                    html.Div(
                        [
                            html.Div(
                                "Ornek Sayisi",
                                style={
                                    "fontSize": "11px",
                                    "color": "#8ab4c8",
                                    "marginBottom": "4px",
                                },
                            ),
                            dcc.Slider(
                                id="sl-n",
                                min=1000,
                                max=20000,
                                step=1000,
                                value=5000,
                                marks={1000: "1K", 10000: "10K", 20000: "20K"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            html.Div(
                                "SEU Sayisi",
                                style={
                                    "fontSize": "11px",
                                    "color": "#8ab4c8",
                                    "margin": "10px 0 4px 0",
                                },
                            ),
                            dcc.Slider(
                                id="sl-seu",
                                min=5,
                                max=30,
                                step=1,
                                value=15,
                                marks={5: "5", 15: "15", 30: "30"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            html.Div(
                                "TID Egimi",
                                style={
                                    "fontSize": "11px",
                                    "color": "#8ab4c8",
                                    "margin": "10px 0 4px 0",
                                },
                            ),
                            dcc.Slider(
                                id="sl-tid",
                                min=0.001,
                                max=0.01,
                                step=0.001,
                                value=0.003,
                                marks={0.001: "0.001", 0.005: "0.005", 0.01: "0.01"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            html.Div(
                                "Gap Sayisi",
                                style={
                                    "fontSize": "11px",
                                    "color": "#8ab4c8",
                                    "margin": "10px 0 4px 0",
                                },
                            ),
                            dcc.Slider(
                                id="sl-gaps",
                                min=2,
                                max=8,
                                step=1,
                                value=4,
                                marks={2: "2", 4: "4", 8: "8"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            html.Div(
                                "Gurultu Std",
                                style={
                                    "fontSize": "11px",
                                    "color": "#8ab4c8",
                                    "margin": "10px 0 4px 0",
                                },
                            ),
                            dcc.Slider(
                                id="sl-noise",
                                min=0.5,
                                max=5.0,
                                step=0.5,
                                value=2.0,
                                marks={0.5: "0.5", 2.0: "2", 5.0: "5"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ]
                    ),
                ],
                className="sidebar-section",
            ),
            # CSV upload
            html.Div(
                id="div-csv-upload",
                children=[
                    html.Div("CSV Dosyasi", className="sidebar-label"),
                    dcc.Upload(
                        id="upload-csv",
                        children=html.Div(
                            [
                                "Dosya Surukle veya ",
                                html.A("Sec", style={"color": "#00d4ff"}),
                            ],
                            style={"textAlign": "center"},
                        ),
                        className="upload-area",
                        accept=".csv",
                    ),
                ],
                className="sidebar-section",
                style={"display": "none"},
            ),
            # Navigation
            html.Div(
                [
                    html.Button(
                        "Geri",
                        id="btn-back",
                        n_clicks=0,
                        className="btn btn-outline",
                        style={"flex": "1"},
                    ),
                    html.Button(
                        "Ileri",
                        id="btn-next",
                        n_clicks=0,
                        className="btn btn-primary",
                        style={"flex": "1"},
                    ),
                ],
                className="sidebar-nav",
            ),
        ],
        className="sidebar",
    )


# ── App initialization ────────────────────────────────────────────────────────

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    assets_folder=str(Path(__file__).parent / "assets"),
    suppress_callback_exceptions=True,
)
app.title = "Cosmic Pipeline - Sinyal Temizleme Sihirbazi"

# ── Layout ────────────────────────────────────────────────────────────────────

app.layout = html.Div(
    [
        dcc.Store(id="store-step", data=0),
        dcc.Store(id="store-raw", data=None),
        dcc.Store(id="store-ingested", data=None),
        dcc.Store(id="store-detections", data=None),
        dcc.Store(id="store-ensemble", data=None),
        dcc.Store(id="store-filtered", data=None),
        dcc.Store(id="store-metrics", data=None),
        # Header
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("Cosmic Pipeline", className="header-title"),
                                html.Div(
                                    "TUA Astro Hackathon 2026 - Uydu Telemetri Temizleme",
                                    className="header-subtitle",
                                ),
                            ]
                        ),
                    ],
                    className="header-top",
                ),
                html.Div(id="div-stepper", children=render_stepper(0)),
            ],
            className="header",
        ),
        # Main row
        html.Div(
            [
                render_sidebar(),
                dcc.Loading(
                    type="circle",
                    color="#00d4ff",
                    children=html.Div(id="main-content", className="content-area"),
                ),
            ],
            className="main-row",
        ),
    ],
    className="app-shell",
)


# ── Chart helper ──────────────────────────────────────────────────────────────


def _signal_chart(
    x,
    y,
    title: str = "Sinyal",
    color: str = "#f59e0b",
    height: int = 300,
    extra_traces: list | None = None,
) -> dcc.Graph:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(x),
            y=list(y),
            mode="lines",
            line=dict(color=color, width=1.5),
            name="Sinyal",
        )
    )
    if extra_traces:
        for t in extra_traces:
            fig.add_trace(t)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=height,
        **PLOT_THEME,
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


# ── STEP renderers ────────────────────────────────────────────────────────────


def _render_step0(raw: dict | None) -> list:
    children = [
        html.Div(
            [
                html.Div("Veri Kaynagi", className="step-title"),
                html.Div(
                    "Soldaki panelden parametreleri ayarlayin. "
                    "Sentetik veri uretilecek veya CSV yuklenecek. "
                    "Hazir oldugunuzda 'Veri Olustur' butonuna tiklayin.",
                    className="step-desc",
                ),
            ],
            className="step-header",
        ),
        html.Button(
            "Veri Olustur",
            id="btn-generate",
            n_clicks=0,
            className="btn btn-success",
            style={"marginBottom": "16px"},
        ),
    ]

    if raw is None:
        children.append(
            alert_box(
                "Parametreleri ayarlayin ve Veri Olustur butonuna tiklayin.",
                "info",
            )
        )
    elif "error" in raw:
        children.append(alert_box(f"Hata: {raw['error']}", "error"))
    else:
        n = raw.get("n", 0)
        corrupted_data = raw.get("corrupted")
        if corrupted_data:
            df_c = _store_to_df(corrupted_data)
            vals = df_c["value"].values
            nan_count = int(pd.isna(vals).sum())
            finite_vals = vals[np.isfinite(vals)]
            val_min = float(np.min(finite_vals)) if len(finite_vals) > 0 else 0.0
            val_max = float(np.max(finite_vals)) if len(finite_vals) > 0 else 0.0

            children.append(
                html.Div(
                    [
                        metric_card("Toplam Nokta", f"{n:,}", "cyan"),
                        metric_card(
                            "NaN Sayisi",
                            str(nan_count),
                            "orange" if nan_count > 0 else "green",
                        ),
                        metric_card("Min Deger", f"{val_min:.3f}", "blue"),
                        metric_card("Max Deger", f"{val_max:.3f}", "purple"),
                    ],
                    className="grid-4",
                    style={"marginBottom": "16px"},
                )
            )

            x = (
                df_c["timestamp"]
                if "timestamp" in df_c.columns
                else np.arange(len(df_c))
            )
            children.append(
                _signal_chart(x, df_c["value"], "Ham Bozuk Sinyal", "#f59e0b", height=320)
            )
    return children


def _render_step1(raw: dict | None, ingested: dict | None) -> list:
    children = [
        html.Div(
            [
                html.Div("Veri Yukleme ve On Isleme", className="step-title"),
                html.Div(
                    "Ham veri pipeline semasina uygunlugu icin dogrulanir, "
                    "zaman damgalari ayristirilir ve tekrar eden kayitlar temizlenir.",
                    className="step-desc",
                ),
            ],
            className="step-header",
        ),
        html.Button(id="btn-generate", n_clicks=0, style={"display": "none"}),
    ]

    if ingested is None:
        children.append(
            alert_box(
                "Veri kaynagi secin ve once Veri Olustur butonuna tiklayin.", "info"
            )
        )
    elif "error" in ingested:
        children.append(alert_box(f"Hata: {ingested['error']}", "error"))
    else:
        n_rows = ingested.get("n_rows", 0)
        nan_count = ingested.get("nan_count", 0)
        validation = ingested.get("validation", {})

        children.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Sema Dogrulama", className="card-title"),
                            check_item("timestamp sutunu", "datetime64 dtype", ok=True),
                            check_item("value sutunu", "float64 dtype", ok=True),
                            check_item(
                                "Veri uzunlugu",
                                f"{n_rows:,} satir",
                                ok=n_rows >= 100,
                            ),
                            check_item(
                                "Tekillestirilme",
                                "duplicate timestamps removed",
                                ok=True,
                            ),
                        ],
                        className="card",
                    ),
                    html.Div(
                        [
                            html.Div("On Isleme Adimlari", className="card-title"),
                            check_item(
                                "Zaman damgasi ayristirma", "pd.to_datetime", ok=True
                            ),
                            check_item("Siralama", "timestamp artan sira", ok=True),
                            check_item("Float64 donusumu", "value sutunu", ok=True),
                            check_item(
                                "NaN kontrolu",
                                f"{nan_count} eksik deger",
                                ok=nan_count < n_rows * 0.5,
                            ),
                            check_item(
                                "Kalite skoru",
                                f"{validation.get('quality_score', 0):.0%}",
                                ok=validation.get("quality_score", 0) > 0.5,
                            ),
                        ],
                        className="card",
                    ),
                ],
                className="grid-2",
                style={"marginBottom": "16px"},
            )
        )

        data_store = ingested.get("data")
        if data_store:
            df_i = _store_to_df(data_store)
            x = (
                df_i["timestamp"]
                if "timestamp" in df_i.columns
                else np.arange(len(df_i))
            )
            children.append(
                _signal_chart(x, df_i["value"], "Yuklenmis Sinyal", "#f59e0b", height=280)
            )

            preview = df_i.head(20)
            rows = [
                html.Tr(
                    [
                        html.Td(str(idx)),
                        html.Td(str(row["timestamp"])),
                        html.Td(f"{row['value']:.6f}"),
                    ]
                )
                for idx, row in preview.iterrows()
            ]
            children.append(
                html.Div(
                    [
                        html.Div(
                            "Veri Onizleme (ilk 20 satir)", className="card-title"
                        ),
                        html.Table(
                            [
                                html.Thead(
                                    html.Tr(
                                        [
                                            html.Th("Index"),
                                            html.Th("Timestamp"),
                                            html.Th("Value"),
                                        ]
                                    )
                                ),
                                html.Tbody(rows),
                            ],
                            className="data-table",
                        ),
                    ],
                    className="card",
                    style={"marginTop": "16px", "overflowX": "auto"},
                )
            )

    return children


def _render_step2(
    raw: dict | None,
    ingested: dict | None,
    det: dict | None,
) -> list:
    children = [
        html.Div(
            [
                html.Div("Anomali Tespiti", className="step-title"),
                html.Div(
                    "Klasik istatistiksel (Z-Score, IQR, Kayan Pencere, Gap) ve ML tabanli "
                    "(Isolation Forest, LSTM AE) detektorler paralel calistirilir.",
                    className="step-desc",
                ),
            ],
            className="step-header",
        ),
        html.Button(
            "Detektorleri Calistir",
            id="btn-run-detect",
            n_clicks=0,
            className="btn btn-primary",
            style={"marginBottom": "16px"},
        ),
        html.Button(id="btn-generate", n_clicks=0, style={"display": "none"}),
    ]

    if det is None:
        children.append(
            alert_box("Detektorleri calistirmak icin butona tiklayin.", "info")
        )
    elif "error" in det:
        children.append(alert_box(f"Hata: {det['error']}", "error"))
    else:
        masks_store = det.get("masks", {})
        counts = det.get("counts", {})
        ml_error = det.get("ml_error")
        elapsed = det.get("elapsed", 0)

        classic_rows = []
        for key in ["zscore", "iqr", "sliding_window", "gaps"]:
            count = counts.get(key, 0)
            color = DETECTOR_COLORS.get(key, "#8ab4c8")
            name = DETECTOR_NAMES_TR.get(key, key)
            classic_rows.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    style={
                                        "width": "8px",
                                        "height": "8px",
                                        "borderRadius": "50%",
                                        "background": color,
                                        "flexShrink": "0",
                                    }
                                ),
                                html.Span(name, style={"color": "#e8f4ff"}),
                            ],
                            className="detector-name",
                        ),
                        badge(str(count), "orange" if count > 0 else "gray"),
                    ],
                    className="detector-row",
                )
            )

        ml_rows = []
        for key in ["isolation_forest", "lstm_ae"]:
            count = counts.get(key, 0)
            color = DETECTOR_COLORS.get(key, "#8ab4c8")
            name = DETECTOR_NAMES_TR.get(key, key)
            ml_rows.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    style={
                                        "width": "8px",
                                        "height": "8px",
                                        "borderRadius": "50%",
                                        "background": color,
                                        "flexShrink": "0",
                                    }
                                ),
                                html.Span(name, style={"color": "#e8f4ff"}),
                            ],
                            className="detector-name",
                        ),
                        badge(str(count), "green" if count > 0 else "gray"),
                    ],
                    className="detector-row",
                )
            )

        if ml_error:
            ml_rows.append(
                html.Div(
                    badge(f"ML Hatasi: {ml_error[:60]}", "orange"),
                    style={"marginTop": "8px", "fontSize": "11px"},
                )
            )

        children.append(
            html.Div(
                [
                    html.Div(
                        [html.Div("Klasik Detektorler", className="card-title")]
                        + classic_rows,
                        className="card",
                    ),
                    html.Div(
                        [html.Div("ML Detektorler", className="card-title")] + ml_rows,
                        className="card",
                    ),
                ],
                className="grid-2",
                style={"marginBottom": "16px"},
            )
        )

        if masks_store:
            all_masks = [np.array(v, dtype=bool) for v in masks_store.values()]
            if all_masks:
                combined = np.zeros(len(all_masks[0]), dtype=bool)
                for m in all_masks:
                    combined |= m
                total_unique = int(combined.sum())
            else:
                total_unique = 0
        else:
            total_unique = 0

        children.append(
            html.Div(
                [
                    metric_card("Toplam Benzersiz Anomali", str(total_unique), "orange"),
                    metric_card("Calisma Suresi", f"{elapsed:.2f}s", "cyan"),
                    metric_card("Aktif Detektor", str(len(masks_store)), "green"),
                ],
                className="grid-3",
                style={"marginBottom": "16px"},
            )
        )

        if ingested and "data" in ingested and masks_store:
            df_i = _store_to_df(ingested["data"])
            x = (
                df_i["timestamp"]
                if "timestamp" in df_i.columns
                else np.arange(len(df_i))
            )
            vals = df_i["value"].values

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(x),
                    y=list(vals),
                    mode="lines",
                    line=dict(color="#f59e0b", width=1.2),
                    name="Sinyal",
                )
            )
            for det_key, mask_list in masks_store.items():
                mask_arr = np.array(mask_list, dtype=bool)
                if mask_arr.sum() == 0:
                    continue
                idx = np.where(mask_arr)[0]
                color = DETECTOR_COLORS.get(det_key, "#ffffff")
                name = DETECTOR_NAMES_TR.get(det_key, det_key)
                x_vals = [
                    x.iloc[i] if hasattr(x, "iloc") else int(x[i]) for i in idx
                ]
                y_vals = [
                    float(vals[i]) if np.isfinite(vals[i]) else None for i in idx
                ]
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="markers",
                        marker=dict(color=color, size=6, opacity=0.8),
                        name=name,
                    )
                )
            fig.update_layout(
                title=dict(
                    text="Anomali Tespiti - Detektor Karsilastirmasi",
                    font=dict(size=14),
                ),
                height=360,
                **PLOT_THEME,
            )
            children.append(dcc.Graph(figure=fig, config={"displayModeBar": False}))

    return children


def _render_step3(
    raw: dict | None,
    ingested: dict | None,
    det: dict | None,
    ens: dict | None,
) -> list:
    children = [
        html.Div(
            [
                html.Div("Ensemble Oylama", className="step-title"),
                html.Div(
                    "Tum detektorlerin sonuclari birlestirilir. "
                    "OR mantigi (any stratejisi) en yuksek recall saglar.",
                    className="step-desc",
                ),
            ],
            className="step-header",
        ),
        html.Button(
            "Ensemble Oyla",
            id="btn-run-ensemble",
            n_clicks=0,
            className="btn btn-primary",
            style={"marginBottom": "16px"},
        ),
        html.Button(id="btn-generate", n_clicks=0, style={"display": "none"}),
        html.Button(id="btn-run-detect", n_clicks=0, style={"display": "none"}),
        html.Div(
            [
                html.Div("Ensemble Stratejisi: OR (any)", className="strategy-title"),
                html.P(
                    "Her detektor farkli bozulma turunu tespit eder: "
                    "Z-Score/IQR spike, Sliding Window drift, Gap NaN, "
                    "Isolation Forest genel anomali, LSTM AE zaman serisi tutarsizligi. "
                    "OR mantigi ile en az bir detektorun isaretledigi noktalar anomali sayilir - "
                    "bu yaklasim en yuksek recall degerini garanti eder.",
                    style={
                        "color": "#8ab4c8",
                        "fontSize": "13px",
                        "lineHeight": "1.6",
                    },
                ),
            ],
            className="strategy-card",
            style={"marginBottom": "16px"},
        ),
    ]

    if ens is None:
        children.append(
            alert_box("Ensemble oylama icin butona tiklayin.", "info")
        )
    elif "error" in ens:
        children.append(alert_box(f"Hata: {ens['error']}", "error"))
    else:
        classic_count = ens.get("classic_count", 0)
        ml_count = ens.get("ml_count", 0)
        total = ens.get("total", 0)
        n = ens.get("n", 1)

        children.append(
            html.Div(
                [
                    metric_card("Klasik Tespit", str(classic_count), "blue"),
                    metric_card("ML Tespit", str(ml_count), "green"),
                    metric_card(
                        "Ensemble Final",
                        str(total),
                        "orange",
                        f"{total / max(n, 1) * 100:.1f}% oran",
                    ),
                ],
                className="grid-3",
                style={"marginBottom": "16px"},
            )
        )

        if det and "masks" in det:
            masks_store = det["masks"]
            det_keys = list(masks_store.keys())
            if det_keys:
                n_total = len(list(masks_store.values())[0])
                subsample = 200
                step_size = max(1, n_total // subsample)
                sample_idx = list(range(0, n_total, step_size))[:subsample]

                z_matrix = []
                for key in det_keys:
                    mask_arr = np.array(masks_store[key], dtype=bool)
                    z_row = [int(mask_arr[i]) for i in sample_idx]
                    z_matrix.append(z_row)

                y_labels = [DETECTOR_NAMES_TR.get(k, k) for k in det_keys]

                fig_heat = go.Figure(
                    go.Heatmap(
                        z=z_matrix,
                        x=sample_idx,
                        y=y_labels,
                        colorscale=[[0, "#0a0e17"], [1, "#00d4ff"]],
                        showscale=False,
                    )
                )
                fig_heat.update_layout(
                    title=dict(
                        text="Oylama Matrisi (200 ornek)", font=dict(size=14)
                    ),
                    height=280,
                    **PLOT_THEME,
                )
                children.append(
                    dcc.Graph(figure=fig_heat, config={"displayModeBar": False})
                )

        ens_mask_store = ens.get("mask")
        if ingested and "data" in ingested and ens_mask_store:
            df_i = _store_to_df(ingested["data"])
            ens_mask = _store_to_mask(ens_mask_store)
            x = (
                df_i["timestamp"]
                if "timestamp" in df_i.columns
                else np.arange(len(df_i))
            )
            vals = df_i["value"].values

            anom_idx = np.where(ens_mask)[0]
            fig_ens = go.Figure()
            fig_ens.add_trace(
                go.Scatter(
                    x=list(x),
                    y=list(vals),
                    mode="lines",
                    line=dict(color="#f59e0b", width=1.2),
                    name="Sinyal",
                )
            )
            if len(anom_idx) > 0:
                x_a = [
                    x.iloc[i] if hasattr(x, "iloc") else int(x[i])
                    for i in anom_idx
                ]
                y_a = [
                    float(vals[i]) if np.isfinite(vals[i]) else None
                    for i in anom_idx
                ]
                fig_ens.add_trace(
                    go.Scatter(
                        x=x_a,
                        y=y_a,
                        mode="markers",
                        marker=dict(color="#ff4444", size=7, symbol="x"),
                        name="Ensemble Anomali",
                    )
                )
            fig_ens.update_layout(
                title=dict(text="Ensemble Final Maske", font=dict(size=14)),
                height=300,
                **PLOT_THEME,
            )
            children.append(
                dcc.Graph(
                    figure=fig_ens,
                    config={"displayModeBar": False},
                    style={"marginTop": "12px"},
                )
            )

    return children


def _render_step4(
    ingested: dict | None,
    ens: dict | None,
    flt: dict | None,
) -> list:
    children = [
        html.Div(
            [
                html.Div("Filtreleme", className="step-title"),
                html.Div(
                    "Tespit edilen anomaliler median filtre, Savitzky-Golay, wavelet denoising "
                    "ve interpolasyon zinciriyle temizlenir.",
                    className="step-desc",
                ),
            ],
            className="step-header",
        ),
        html.Button(
            "Filtrele",
            id="btn-run-filter",
            n_clicks=0,
            className="btn btn-success",
            style={"marginBottom": "16px"},
        ),
        html.Button(id="btn-generate", n_clicks=0, style={"display": "none"}),
        html.Button(id="btn-run-detect", n_clicks=0, style={"display": "none"}),
        html.Button(id="btn-run-ensemble", n_clicks=0, style={"display": "none"}),
        html.Div(
            [
                html.Div("Filtre Zinciri", className="card-title"),
                html.Div(
                    [
                        html.Span("Detrend", className="filter-pill"),
                        html.Span("->", className="filter-arrow"),
                        html.Span("Medyan Filtre", className="filter-pill"),
                        html.Span("->", className="filter-arrow"),
                        html.Span("Savitzky-Golay", className="filter-pill"),
                        html.Span("->", className="filter-arrow"),
                        html.Span("Wavelet", className="filter-pill"),
                        html.Span("->", className="filter-arrow"),
                        html.Span("Interpolasyon", className="filter-pill"),
                    ],
                    className="filter-chain",
                ),
            ],
            className="card",
            style={"marginBottom": "16px"},
        ),
    ]

    if flt is None:
        children.append(alert_box("Filtreleme icin butona tiklayin.", "info"))
    elif "error" in flt:
        children.append(alert_box(f"Hata: {flt['error']}", "error"))
    else:
        elapsed = flt.get("elapsed", 0)
        filter_type = flt.get("filter_type", "classic")

        children.append(
            html.Div(
                [
                    badge(f"Filtre: {filter_type}", "cyan"),
                    badge(f"Sure: {elapsed:.2f}s", "green"),
                ],
                style={"marginBottom": "16px", "display": "flex", "gap": "8px"},
            )
        )

        if ingested and "data" in ingested and ens and "mask" in ens:
            df_i = _store_to_df(ingested["data"])
            ens_mask = _store_to_mask(ens["mask"])
            x = (
                df_i["timestamp"]
                if "timestamp" in df_i.columns
                else np.arange(len(df_i))
            )
            vals = df_i["value"].values

            anom_idx = np.where(ens_mask)[0]
            fig_before = go.Figure()
            fig_before.add_trace(
                go.Scatter(
                    x=list(x),
                    y=list(vals),
                    mode="lines",
                    line=dict(color="#f59e0b", width=1.2),
                    name="Bozuk Sinyal",
                )
            )
            if len(anom_idx) > 0:
                x_a = [
                    x.iloc[i] if hasattr(x, "iloc") else int(x[i])
                    for i in anom_idx
                ]
                y_a = [
                    float(vals[i]) if np.isfinite(vals[i]) else None
                    for i in anom_idx
                ]
                fig_before.add_trace(
                    go.Scatter(
                        x=x_a,
                        y=y_a,
                        mode="markers",
                        marker=dict(color="#ff4444", size=6, symbol="x"),
                        name="Anomali",
                    )
                )
            fig_before.update_layout(
                title=dict(text="Filtreleme Oncesi", font=dict(size=14)),
                height=280,
                **PLOT_THEME,
            )
            children.append(
                dcc.Graph(figure=fig_before, config={"displayModeBar": False})
            )

        cleaned_store = flt.get("cleaned")
        if cleaned_store:
            df_clean = _store_to_df(cleaned_store)
            x_c = (
                df_clean["timestamp"]
                if "timestamp" in df_clean.columns
                else np.arange(len(df_clean))
            )
            children.append(
                _signal_chart(
                    x_c,
                    df_clean["value"],
                    "Filtreleme Sonrasi - Temizlenmis Sinyal",
                    "#00ff88",
                    height=280,
                )
            )

    return children


def _render_step5(
    raw: dict | None,
    ingested: dict | None,
    flt: dict | None,
    det: dict | None,
    ens: dict | None,
    met: dict | None,
) -> list:
    children = [
        html.Div(
            [
                html.Div("Sonuclar ve Metrikler", className="step-title"),
                html.Div(
                    "Pipeline ciktisinin kalitesini RMSE, SNR, Precision/Recall "
                    "metrikleriyle degerlendirin ve temizlenmis veriyi indirin.",
                    className="step-desc",
                ),
            ],
            className="step-header",
        ),
        html.Button(
            "Metrikleri Hesapla",
            id="btn-run-metrics",
            n_clicks=0,
            className="btn btn-primary",
            style={"marginBottom": "16px"},
        ),
        html.Button(id="btn-generate", n_clicks=0, style={"display": "none"}),
        html.Button(id="btn-run-detect", n_clicks=0, style={"display": "none"}),
        html.Button(id="btn-run-ensemble", n_clicks=0, style={"display": "none"}),
        html.Button(id="btn-run-filter", n_clicks=0, style={"display": "none"}),
    ]

    if met is None:
        children.append(
            alert_box("Metrikleri hesaplamak icin butona tiklayin.", "info")
        )
    elif "error" in met:
        children.append(alert_box(f"Hata: {met['error']}", "error"))
    else:
        rmse_before = met.get("rmse_before", 0) or 0
        rmse_after = met.get("rmse_after", 0) or 0
        snr_after = met.get("snr_after", 0) or 0
        mae = met.get("mae", 0) or 0
        r2 = met.get("r2", 0) or 0
        precision = met.get("precision", 0) or 0
        recall = met.get("recall", 0) or 0
        f1 = met.get("f1", 0) or 0
        proc_time = met.get("processing_time", 0) or 0

        rmse_improvement = 0.0
        if rmse_before > 0:
            rmse_improvement = (rmse_before - rmse_after) / rmse_before * 100

        snr_str = (
            f"{snr_after:.1f} dB"
            if np.isfinite(snr_after)
            else "inf dB"
        )

        children.append(
            html.Div(
                [
                    metric_card(
                        "RMSE Oncesi",
                        f"{rmse_before:.4f}",
                        "orange",
                        "bozuk sinyal",
                    ),
                    metric_card(
                        "RMSE Sonrasi",
                        f"{rmse_after:.4f}",
                        "green",
                        f"{rmse_improvement:.1f}% iyilesme",
                    ),
                    metric_card(
                        "SNR",
                        snr_str,
                        "cyan",
                        "temizlenmis sinyal",
                    ),
                    metric_card("Precision", f"{precision:.3f}", "blue"),
                    metric_card("Recall", f"{recall:.3f}", "purple"),
                    metric_card("F1 Skoru", f"{f1:.3f}", "green"),
                ],
                className="grid-6",
                style={"marginBottom": "16px"},
            )
        )

        children.append(
            html.Div(
                [
                    metric_card("MAE", f"{mae:.4f}", "orange"),
                    metric_card("R2 Skoru", f"{r2:.4f}", "cyan"),
                    metric_card("Isleme Suresi", f"{proc_time:.2f}s", "green"),
                ],
                className="grid-3",
                style={"marginBottom": "16px"},
            )
        )

        if raw and raw.get("gt") and raw.get("clean") and flt and "cleaned" in flt:
            gt = raw["gt"]
            df_clean_ref = _store_to_df(raw["clean"])
            df_cleaned = _store_to_df(flt["cleaned"])
            n_pts = len(df_clean_ref)
            gt_mask = _gt_bool_mask(gt, n_pts)

            ens_mask_arr = None
            if ens and "mask" in ens:
                ens_mask_arr = _store_to_mask(ens["mask"])

            x_ref = (
                df_clean_ref["timestamp"]
                if "timestamp" in df_clean_ref.columns
                else np.arange(n_pts)
            )
            x_cleaned = (
                df_cleaned["timestamp"]
                if "timestamp" in df_cleaned.columns
                else np.arange(len(df_cleaned))
            )

            fig_timeline = go.Figure()
            fig_timeline.add_trace(
                go.Scatter(
                    x=list(x_ref),
                    y=list(df_clean_ref["value"].values),
                    mode="lines",
                    line=dict(color="#4488ff", width=1.5, dash="dot"),
                    name="Ground Truth (Temiz)",
                )
            )
            fig_timeline.add_trace(
                go.Scatter(
                    x=list(x_cleaned),
                    y=list(df_cleaned["value"].values),
                    mode="lines",
                    line=dict(color="#00ff88", width=1.5),
                    name="Temizlenmis",
                )
            )

            gt_idx = np.where(gt_mask)[0]
            if len(gt_idx) > 0:
                x_gt = [
                    x_ref.iloc[i] if hasattr(x_ref, "iloc") else int(x_ref[i])
                    for i in gt_idx
                ]
                y_gt = [float(df_clean_ref["value"].values[i]) for i in gt_idx]
                fig_timeline.add_trace(
                    go.Scatter(
                        x=x_gt,
                        y=y_gt,
                        mode="markers",
                        marker=dict(color="#ff4444", size=5, opacity=0.5),
                        name="Ground Truth Anomali",
                    )
                )

            if ens_mask_arr is not None:
                det_idx = np.where(ens_mask_arr)[0]
                valid_det_idx = [i for i in det_idx if i < n_pts]
                if len(valid_det_idx) > 0:
                    x_det = [
                        x_ref.iloc[i] if hasattr(x_ref, "iloc") else int(x_ref[i])
                        for i in valid_det_idx
                    ]
                    y_det = [
                        float(df_clean_ref["value"].values[i])
                        for i in valid_det_idx
                    ]
                    fig_timeline.add_trace(
                        go.Scatter(
                            x=x_det,
                            y=y_det,
                            mode="markers",
                            marker=dict(
                                color="#f59e0b", size=6, symbol="triangle-up"
                            ),
                            name="Ensemble Tespiti",
                        )
                    )

            fig_timeline.update_layout(
                title=dict(
                    text="Ground Truth vs Tespit Karsilastirmasi",
                    font=dict(size=14),
                ),
                height=360,
                **PLOT_THEME,
            )
            children.append(
                dcc.Graph(
                    figure=fig_timeline,
                    config={"displayModeBar": False},
                    style={"marginBottom": "16px"},
                )
            )

        children.append(
            html.Div(
                [
                    dcc.Download(id="download-csv"),
                    html.Button(
                        "Temizlenmis CSV Indir",
                        id="btn-download",
                        n_clicks=0,
                        className="btn btn-cyan",
                    ),
                ],
                style={"marginTop": "8px"},
            )
        )

    return children


# ── CALLBACKS ─────────────────────────────────────────────────────────────────


@app.callback(
    Output("div-synthetic-params", "style"),
    Output("div-csv-upload", "style"),
    Input("radio-source", "value"),
)
def toggle_source(source: str):
    if source == "synthetic":
        return {}, {"display": "none"}
    return {"display": "none"}, {}


@app.callback(
    Output("store-step", "data"),
    Input("btn-next", "n_clicks"),
    Input("btn-back", "n_clicks"),
    State("store-step", "data"),
    State("store-raw", "data"),
    prevent_initial_call=True,
)
def navigate_step(n_next: int, n_back: int, step: int, raw: dict | None):
    triggered = ctx.triggered_id
    if triggered == "btn-back":
        return max(0, step - 1)
    if triggered == "btn-next":
        if step == 0 and raw is None:
            return step
        return min(5, step + 1)
    return step


@app.callback(
    Output("div-stepper", "children"),
    Input("store-step", "data"),
)
def update_stepper(step: int):
    return render_stepper(step)


@app.callback(
    Output("store-raw", "data"),
    Input("btn-generate", "n_clicks"),
    State("radio-source", "value"),
    State("sl-n", "value"),
    State("sl-seu", "value"),
    State("sl-tid", "value"),
    State("sl-gaps", "value"),
    State("sl-noise", "value"),
    State("upload-csv", "contents"),
    prevent_initial_call=True,
)
def generate_data(
    n_clicks: int,
    source: str,
    n: int,
    seu: int,
    tid: float,
    gaps: int,
    noise: float,
    csv_contents: str | None,
):
    if not n_clicks:
        return no_update
    try:
        if source == "synthetic":
            config = FaultConfig(
                seu_count=int(seu),
                tid_slope=float(tid),
                gap_count=int(gaps),
                noise_std_max=float(noise),
            )
            clean_df, corrupted_df, gt = generate_corrupted_dataset(n=int(n), config=config)
            return {
                "corrupted": _df_to_store(corrupted_df),
                "clean": _df_to_store(clean_df),
                "gt": gt,
                "source": "synthetic",
                "n": int(n),
            }
        else:
            if csv_contents is None:
                return {"error": "CSV dosyasi yuklenmedi."}
            content_type, content_string = csv_contents.split(",", 1)
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            return {
                "corrupted": _df_to_store(df),
                "clean": None,
                "gt": None,
                "source": "csv",
                "n": len(df),
            }
    except Exception as exc:
        logger.exception("generate_data error")
        return {"error": str(exc)}


@app.callback(
    Output("store-ingested", "data"),
    Input("store-raw", "data"),
)
def ingest_data(raw: dict | None):
    if raw is None:
        return None
    if "error" in raw:
        return {"error": raw["error"]}
    try:
        corrupted_store = raw.get("corrupted")
        if corrupted_store is None:
            return {"error": "Corrupted data not found in store."}
        df_raw = _store_to_df(corrupted_store)
        df_loaded = load_data(df_raw)
        validate_schema(df_loaded)
        df_prep = preprocess(df_loaded)
        validation = validate_output(df_prep)
        nan_count = int(df_prep["value"].isna().sum())
        return {
            "data": _df_to_store(df_prep),
            "validation": validation,
            "n_rows": len(df_prep),
            "nan_count": nan_count,
        }
    except Exception as exc:
        logger.exception("ingest_data error")
        return {"error": str(exc)}


@app.callback(
    Output("store-detections", "data"),
    Input("btn-run-detect", "n_clicks"),
    State("store-raw", "data"),
    State("store-ingested", "data"),
    State("radio-method", "value"),
    prevent_initial_call=True,
)
def run_detection(n_clicks: int, raw: dict | None, ingested: dict | None, method: str):
    if not n_clicks:
        return no_update
    if ingested is None or "error" in ingested:
        return {"error": "Once veri yukleme adimini tamamlayin."}
    try:
        t0 = time.time()
        df_i = _store_to_df(ingested["data"])
        df_detrended = detrend_signal(df_i)

        masks: dict[str, list] = {}
        counts: dict[str, int] = {}
        ml_error = None

        if method in ("classic", "both"):
            classic_results = detect_classic(df_detrended, df_original=df_i)
            for key, series in classic_results.items():
                arr = series.values.astype(bool)
                masks[key] = _mask_to_store(arr)
                counts[key] = int(arr.sum())

        if method in ("ml", "both"):
            try:
                from pipeline.detector_ml import detect_all_ml

                ml_results = detect_all_ml(df_detrended)
                for key, series in ml_results.items():
                    arr = series.values.astype(bool)
                    masks[key] = _mask_to_store(arr)
                    counts[key] = int(arr.sum())
            except Exception as exc:
                ml_error = str(exc)
                logger.warning("ML detection failed: %s", exc)

        elapsed = time.time() - t0
        return {
            "masks": masks,
            "counts": counts,
            "method": method,
            "elapsed": elapsed,
            "ml_error": ml_error,
        }
    except Exception as exc:
        logger.exception("run_detection error")
        return {"error": str(exc)}


@app.callback(
    Output("store-ensemble", "data"),
    Input("btn-run-ensemble", "n_clicks"),
    State("store-detections", "data"),
    prevent_initial_call=True,
)
def run_ensemble(n_clicks: int, det: dict | None):
    if not n_clicks:
        return no_update
    if det is None or "error" in det:
        return {"error": "Once anomali tespiti adimini tamamlayin."}
    try:
        masks_store = det.get("masks", {})
        if not masks_store:
            return {"error": "Hic detektor maskesi bulunamadi."}

        all_series = [pd.Series(_store_to_mask(v)) for v in masks_store.values()]
        ens_series = ensemble_vote(all_series, strategy="any")
        ens_arr = ens_series.values.astype(bool)

        classic_series = [
            pd.Series(_store_to_mask(v))
            for k, v in masks_store.items()
            if k in CLASSIC_KEYS
        ]
        ml_series = [
            pd.Series(_store_to_mask(v))
            for k, v in masks_store.items()
            if k in ML_KEYS
        ]

        classic_combined = np.zeros(len(ens_arr), dtype=bool)
        for s in classic_series:
            classic_combined |= s.values.astype(bool)

        ml_combined = np.zeros(len(ens_arr), dtype=bool)
        for s in ml_series:
            ml_combined |= s.values.astype(bool)

        return {
            "mask": _mask_to_store(ens_arr),
            "classic_count": int(classic_combined.sum()),
            "ml_count": int(ml_combined.sum()),
            "total": int(ens_arr.sum()),
            "n": len(ens_arr),
        }
    except Exception as exc:
        logger.exception("run_ensemble error")
        return {"error": str(exc)}


@app.callback(
    Output("store-filtered", "data"),
    Input("btn-run-filter", "n_clicks"),
    State("store-ingested", "data"),
    State("store-ensemble", "data"),
    State("radio-method", "value"),
    prevent_initial_call=True,
)
def run_filters(n_clicks: int, ingested: dict | None, ens: dict | None, method: str):
    if not n_clicks:
        return no_update
    if ingested is None or "error" in ingested:
        return {"error": "Once veri yukleme adimini tamamlayin."}
    if ens is None or "error" in ens:
        return {"error": "Once ensemble oylama adimini tamamlayin."}
    try:
        t0 = time.time()
        df_i = _store_to_df(ingested["data"])
        ens_mask = _store_to_mask(ens["mask"])
        mask_series = pd.Series(ens_mask, index=df_i.index)
        filter_type = "classic"
        cleaned_df = None

        if method in ("ml", "both"):
            try:
                from pipeline.filters_ml import reconstruct_with_lstm

                cleaned_df = reconstruct_with_lstm(df_i, mask_series)
                filter_type = "lstm"
            except Exception as exc:
                logger.warning(
                    "LSTM filter failed, falling back to classic: %s", exc
                )
                cleaned_df = None

        if cleaned_df is None:
            cleaned_df = apply_classic_filters(df_i, mask_series)
            filter_type = "classic"

        elapsed = time.time() - t0
        return {
            "cleaned": _df_to_store(cleaned_df),
            "method": method,
            "filter_type": filter_type,
            "elapsed": elapsed,
        }
    except Exception as exc:
        logger.exception("run_filters error")
        return {"error": str(exc)}


@app.callback(
    Output("store-metrics", "data"),
    Input("btn-run-metrics", "n_clicks"),
    State("store-raw", "data"),
    State("store-ingested", "data"),
    State("store-filtered", "data"),
    State("store-detections", "data"),
    State("store-ensemble", "data"),
    prevent_initial_call=True,
)
def compute_metrics(
    n_clicks: int,
    raw: dict | None,
    ingested: dict | None,
    flt: dict | None,
    det: dict | None,
    ens: dict | None,
):
    if not n_clicks:
        return no_update
    if ingested is None or "error" in ingested:
        return {"error": "Once veri yukleme adimini tamamlayin."}
    if flt is None or "error" in flt:
        return {"error": "Once filtreleme adimini tamamlayin."}
    try:
        t0 = time.time()
        df_corrupted = _store_to_df(ingested["data"])
        df_cleaned = _store_to_df(flt["cleaned"])

        clean_ref = None
        if raw and raw.get("clean"):
            clean_ref = _store_to_df(raw["clean"])

        metrics_before = calculate_metrics(
            df_corrupted, df_corrupted, ground_truth=clean_ref
        )
        metrics_after = calculate_metrics(
            df_corrupted, df_cleaned, ground_truth=clean_ref
        )

        precision = 0.0
        recall = 0.0
        f1 = 0.0

        if raw and raw.get("gt") and ens and "mask" in ens:
            gt = raw["gt"]
            n_pts = len(df_corrupted)
            gt_mask = _gt_bool_mask(gt, n_pts)
            ens_mask = _store_to_mask(ens["mask"])
            if len(ens_mask) == n_pts:
                precision, recall, f1 = _precision_recall(ens_mask, gt_mask)

        faults_detected = (
            int(_store_to_mask(ens["mask"]).sum()) if ens and "mask" in ens else 0
        )
        proc_time = time.time() - t0

        def _safe(v):
            if v is None:
                return 0.0
            try:
                f = float(v)
                return 0.0 if not np.isfinite(f) else f
            except Exception:
                return 0.0

        return {
            "rmse_before": _safe(metrics_before.get("rmse")),
            "rmse_after": _safe(metrics_after.get("rmse")),
            "snr_before": float(metrics_before.get("snr", 0) or 0),
            "snr_after": float(metrics_after.get("snr", 0) or 0),
            "mae": _safe(metrics_after.get("mae")),
            "r2": _safe(metrics_after.get("r2_score")),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "processing_time": proc_time,
            "faults_detected": faults_detected,
        }
    except Exception as exc:
        logger.exception("compute_metrics error")
        return {"error": str(exc)}


@app.callback(
    Output("main-content", "children"),
    Input("store-step", "data"),
    Input("store-raw", "data"),
    Input("store-ingested", "data"),
    Input("store-detections", "data"),
    Input("store-ensemble", "data"),
    Input("store-filtered", "data"),
    Input("store-metrics", "data"),
    State("radio-method", "value"),
    State("radio-source", "value"),
)
def render_main_content(
    step: int,
    raw: dict | None,
    ingested: dict | None,
    det: dict | None,
    ens: dict | None,
    flt: dict | None,
    met: dict | None,
    method: str,
    source: str,
):
    if step == 0:
        return _render_step0(raw)
    elif step == 1:
        return _render_step1(raw, ingested)
    elif step == 2:
        return _render_step2(raw, ingested, det)
    elif step == 3:
        return _render_step3(raw, ingested, det, ens)
    elif step == 4:
        return _render_step4(ingested, ens, flt)
    elif step == 5:
        return _render_step5(raw, ingested, flt, det, ens, met)
    return []


@app.callback(
    Output("download-csv", "data"),
    Input("btn-download", "n_clicks"),
    State("store-filtered", "data"),
    prevent_initial_call=True,
)
def download_csv(n_clicks: int, flt: dict | None):
    if not n_clicks:
        return no_update
    if flt is None or "error" in flt or "cleaned" not in flt:
        return no_update
    df = _store_to_df(flt["cleaned"])
    return dcc.send_data_frame(df.to_csv, "cosmic_cleaned_signal.csv", index=False)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8050)
