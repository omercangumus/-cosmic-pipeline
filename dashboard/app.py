"""Gradio dashboard for the Cosmic Signal Processing Pipeline."""

import base64
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.synthetic_generator import FaultConfig, generate_corrupted_dataset, generate_clean_signal
from pipeline.orchestrator import run_pipeline, run_pipeline_multi

logging.basicConfig(level=logging.INFO)

# ── Log capture ────────────────────────────────────────────────────────────────

_PIPELINE_LOGGERS = [
    "pipeline.orchestrator", "pipeline.detector_classic", "pipeline.detector_ml",
    "pipeline.ensemble", "pipeline.filters_classic",
    "pipeline.validator", "pipeline.ingestion",
]


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs: list[str] = []
        self.t0: float | None = None

    def emit(self, record: logging.LogRecord) -> None:
        if self.t0 is None:
            self.t0 = time.time()
        elapsed = time.time() - self.t0
        self.logs.append(f"[{elapsed:07.3f}] {self.format(record)}")


def _run_with_logs(df: pd.DataFrame, method: str) -> tuple[dict, list[str]]:
    handler = _LogCapture()
    handler.setLevel(logging.INFO)
    handler.t0 = time.time()
    for name in _PIPELINE_LOGGERS:
        logging.getLogger(name).addHandler(handler)
        logging.getLogger(name).setLevel(logging.INFO)
    try:
        result = run_pipeline(df.copy(), method=method)
    finally:
        for name in _PIPELINE_LOGGERS:
            logging.getLogger(name).removeHandler(handler)
    return result, handler.logs


# ── Plot helpers ───────────────────────────────────────────────────────────────

_TEMPLATE = "plotly_dark"
_BG = "rgba(0,0,0,0)"
_GRID = "rgba(10,14,23,0.8)"


def _plot_clean_vs_corrupted(clean_df, corrupted_df):
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


def _plot_triple_overlay(clean_df, corrupted_df, cleaned_df, fault_mask=None):
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


def _plot_detector_breakdown(detector_masks, corrupted_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=corrupted_df["timestamp"], y=corrupted_df["value"],
        mode="lines", name="Sinyal",
        line=dict(color="#8bb8d4", width=1), opacity=0.5,
    ))
    colors = {
        "zscore": "#3b82f6", "sliding_window": "#8b5cf6", "gaps": "#ef4444",
        "range": "#f97316", "delta": "#eab308", "flatline": "#06b6d4",
        "duplicates": "#ec4899", "isolation_forest": "#22c55e", "lstm_ae": "#00d4ff",
    }
    for name, mask in detector_masks.items():
        if mask.any():
            idx = np.where(mask.values)[0]
            fig.add_trace(go.Scatter(
                x=corrupted_df["timestamp"].iloc[idx],
                y=corrupted_df["value"].iloc[idx],
                mode="markers", name=f"{name} ({int(mask.sum())})",
                marker=dict(color=colors.get(name, "#ffffff"), size=4),
            ))
    fig.update_layout(
        template=_TEMPLATE, paper_bgcolor=_BG, plot_bgcolor=_GRID,
        height=400, showlegend=True,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=30, b=70, l=60, r=30),
    )
    return fig


def _plot_multi_channel(multi_result):
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
    colors = ["#f59e0b", "#00d4ff", "#22c55e", "#8b5cf6", "#ef4444",
              "#ec4899", "#06b6d4", "#3b82f6", "#eab308", "#f97316"]
    for i, (col, res) in enumerate(valid.items(), 1):
        cleaned = res["cleaned_data"]
        fig.add_trace(go.Scatter(
            x=cleaned["timestamp"], y=cleaned["value"],
            mode="lines", name=col,
            line=dict(color=colors[(i - 1) % len(colors)], width=1),
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


# ── Log formatter ──────────────────────────────────────────────────────────────

def _format_pipeline_log(logs, method, result=None):
    lines = []
    lines.append("=" * 45)
    lines.append(f"  Pipeline -- Yontem: {method.upper()}")
    lines.append("=" * 45)
    lines.append("")

    st = {
        "zscore": 0, "sliding_window": 0, "gaps": 0, "range": 0,
        "delta": 0, "flatline": 0, "duplicates": 0,
        "isolation_forest": 0, "lstm_ae": 0,
    }
    # Use detector_counts from pipeline result (reliable) instead of log parsing
    if result and "detector_counts" in result:
        for key in st:
            st[key] = result["detector_counts"].get(key, 0)

    lines.append("ADIM 1 -- Veri Alimi")
    lines.append("   Veri yuklendi ve on isleme tamamlandi.")
    lines.append("")
    lines.append("ADIM 2 -- On Isleme")
    lines.append("   Lineer trend (TID drift) kaldirildi.")
    lines.append("")
    lines.append("ADIM 3 -- Anomali Tespiti")
    lines.append("   +---------------------------------------------+")
    lines.append("   | KATMAN 1 -- Deterministik Kurallar           |")
    lines.append(f"   |   Veri bosluklari (gap):          {st['gaps']:>5}     |")
    lines.append(f"   |   Fiziksel limit ihlali:          {st['range']:>5}     |")
    lines.append(f"   |   Ani sicrama (delta):            {st['delta']:>5}     |")
    lines.append(f"   |   Sabit sinyal (flatline):        {st['flatline']:>5}     |")
    lines.append(f"   |   Tekrar eden zaman:              {st['duplicates']:>5}     |")
    lines.append("   |---------------------------------------------|")
    lines.append("   | KATMAN 2 -- Istatistiksel Analiz             |")
    lines.append(f"   |   Z-score sapmasi:                {st['zscore']:>5}     |")
    lines.append(f"   |   Kayan pencere sapmasi:          {st['sliding_window']:>5}     |")
    lines.append("   |---------------------------------------------|")
    lines.append("   | KATMAN 3 -- Makine Ogrenimi                  |")
    lines.append(f"   |   Isolation Forest:               {st['isolation_forest']:>5}     |")
    lines.append("   |---------------------------------------------|")
    lines.append("   | KATMAN 4 -- Derin Ogrenme                    |")
    lines.append(f"   |   LSTM Autoencoder:               {st['lstm_ae']:>5}     |")
    lines.append("   +---------------------------------------------+")
    lines.append("")
    lines.append("ADIM 4 -- Karar (Hybrid Majority Ensemble)")
    lines.append("   Hard rules -> otomatik anomali")
    lines.append("   Soft detectors -> en az 2 anlasma gerekli")
    if result:
        total = result.get("metrics", {}).get("faults_detected", 0)
        lines.append(f"   Toplam anomali: {total}")
    lines.append("")
    lines.append("ADIM 5 -- Temizleme (Mercek Modeli)")
    lines.append("   1. Interpolation -> bosluklar dolduruldu")
    lines.append("   2. Detrend -> lineer kayma kaldirildi")
    lines.append("   3. Median filtre -> spike'lar temizlendi")
    lines.append("")
    lines.append("ADIM 6 -- Dogrulama")
    if result:
        rv = result.get("repair_verification", {})
        status = "BASARILI" if rv.get("passed", True) else "SORUN VAR"
        elapsed = result.get("metrics", {}).get("processing_time", 0)
        lines.append(f"   Durum: {status} | Sure: {elapsed:.3f}s")

        ft = result.get("fault_timeline")
        if ft is not None and "repair_decision" in ft.columns:
            repair_n = int((ft["repair_decision"] == "repair").sum())
            flag_n = int((ft["repair_decision"] == "flag_only").sum())
            preserve_n = int((ft["repair_decision"] == "preserve").sum())
            lines.append("")
            lines.append("Onarim Karari:")
            lines.append(f"   Duzeltildi: {repair_n}")
            lines.append(f"   Isaretlendi: {flag_n}")
            lines.append(f"   Korundu: {preserve_n}")

        rc = result.get("repair_confidence")
        fm = result.get("fault_mask")
        if rc is not None and fm is not None and fm.any():
            mean_c = float(rc[fm].mean())
            min_c = float(rc[fm].min())
            lines.append(f"   Guven: ortalama={mean_c:.1%}, min={min_c:.1%}")

        si = result.get("sampling_info", {})
        if si:
            lines.append(
                f"   Ornekleme: {si.get('detected_interval', 0):.2f}s aralik, "
                f"jitter={si.get('jitter_ratio', 0):.2%}"
            )

    lines.append("=" * 45)
    return "\n".join(lines)


# ── App state ──────────────────────────────────────────────────────────────────

_state: dict = {}


# ── Callbacks ──────────────────────────────────────────────────────────────────

def generate_data(n_samples, seed, seu_count, tid_slope, gap_count, noise_max):
    try:
        config = FaultConfig(
            seu_count=int(seu_count), tid_slope=float(tid_slope),
            gap_count=int(gap_count), noise_std_max=float(noise_max),
        )
        clean, corrupted, mask = generate_corrupted_dataset(
            n=int(n_samples), config=config, seed=int(seed),
        )
        _state["clean"] = clean
        _state["corrupted"] = corrupted
        _state["gt_mask"] = mask

        fig = _plot_clean_vs_corrupted(clean, corrupted)

        clean_table = clean.head(30)
        corrupted_table = corrupted[["timestamp", "value"]].head(50)

        orbits = clean["orbit_id"].nunique() if "orbit_id" in clean.columns else "?"
        info = (
            f"OK — {int(n_samples)} satir uretildi | SEU: {int(seu_count)} | "
            f"Gap: {int(gap_count)} | Orbits: {orbits}"
        )
        return fig, clean_table, corrupted_table, info
    except Exception as e:
        return None, None, None, f"HATA: {e}"


def upload_csv(file):
    empty_update = gr.update(choices=[], value=[], visible=False)
    if file is None:
        return None, None, None, "Dosya secilmedi", empty_update
    try:
        fname = Path(file.name).name
        raw_df = None
        # Try reading raw file to detect multi-column
        try:
            suffix = fname.lower().rsplit(".", 1)[-1]
            if suffix in ("xlsx", "xls"):
                raw_df = pd.read_excel(file.name)
            elif suffix == "json":
                raw_df = pd.read_json(file.name)
            elif suffix == "tsv":
                raw_df = pd.read_csv(file.name, sep="\t")
            else:
                raw_df = pd.read_csv(file.name)
        except Exception:
            pass

        # Detect numeric columns (exclude timestamp/label)
        time_aliases = {"timestamp", "time_tag", "date", "datetime", "time", "ds"}
        numeric_cols = []
        if raw_df is not None:
            numeric_cols = [
                c for c in raw_df.select_dtypes(include=["number"]).columns
                if c.lower() not in time_aliases and c != "label"
            ]

        if len(numeric_cols) > 1:
            # Multi-column CSV — store raw df for multi-channel processing
            # Resolve timestamp
            time_col = None
            for alias in time_aliases:
                if alias in raw_df.columns:
                    time_col = alias
                    break
            if time_col:
                raw_df["timestamp"] = pd.to_datetime(raw_df[time_col], errors="coerce")
            else:
                raw_df["timestamp"] = pd.date_range("2024-01-01", periods=len(raw_df), freq="1s")

            _state["raw_multi"] = raw_df
            _state["multi_columns"] = numeric_cols
            # Also set single-column fallback (first numeric col)
            first_col = numeric_cols[0]
            df = raw_df[["timestamp", first_col]].rename(columns={first_col: "value"}).copy()
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            _state["corrupted"] = df
            _state["clean"] = None
            _state["gt_mask"] = None

            table = raw_df[["timestamp"] + numeric_cols[:5]].head(50)
            fig = go.Figure()
            for i, col in enumerate(numeric_cols[:5]):
                fig.add_trace(go.Scatter(
                    x=raw_df["timestamp"], y=raw_df[col],
                    mode="lines", name=col, opacity=0.7,
                ))
            fig.update_layout(
                template=_TEMPLATE, paper_bgcolor=_BG,
                plot_bgcolor=_GRID, height=350,
            )
            col_update = gr.update(choices=numeric_cols, value=numeric_cols[:3], visible=True)
            return (
                fig, None, table,
                f"OK — {len(raw_df)} satir, {len(numeric_cols)} kanal: {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}",
                col_update,
            )

        # Single-column: use csv_parser (handles normalization)
        from utils.csv_parser import parse_uploaded_csv
        with open(file.name, "rb") as f:
            raw = f.read()
        encoded = base64.b64encode(raw).decode()
        contents = f"data:text/csv;base64,{encoded}"
        df, error = parse_uploaded_csv(contents, fname)
        if error:
            return None, None, None, f"HATA: {error}", empty_update

        _state["corrupted"] = df
        _state["clean"] = None
        _state["gt_mask"] = None
        _state.pop("raw_multi", None)
        _state.pop("multi_columns", None)

        table = df[["timestamp", "value"]].head(50)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["value"],
            mode="lines", name="Yuklenen Veri",
            line=dict(color="#f59e0b", width=1),
        ))
        fig.update_layout(
            template=_TEMPLATE, paper_bgcolor=_BG,
            plot_bgcolor=_GRID, height=350,
        )
        return fig, None, table, f"OK — {len(df)} satir yuklendi", empty_update
    except Exception as e:
        return None, None, None, f"HATA: {e}", empty_update


def run_pipeline_ui(method, selected_columns):
    if "corrupted" not in _state:
        return None, "Once veri uretin veya yukleyin.", None, "", None, "", None, ""

    clean = _state.get("clean")
    raw_multi = _state.get("raw_multi")

    # ── Multi-channel mode ──
    if raw_multi is not None and selected_columns and len(selected_columns) > 1:
        try:
            handler = _LogCapture()
            handler.setLevel(logging.INFO)
            handler.t0 = time.time()
            for name in _PIPELINE_LOGGERS:
                logging.getLogger(name).addHandler(handler)
                logging.getLogger(name).setLevel(logging.INFO)
            try:
                multi_result = run_pipeline_multi(
                    raw_multi, method=method, columns=selected_columns,
                )
            finally:
                for name in _PIPELINE_LOGGERS:
                    logging.getLogger(name).removeHandler(handler)
        except Exception as e:
            return None, f"Pipeline hatasi: {e}", None, "", None, "", None, ""

        summary = multi_result["summary"]
        channels = multi_result["channels"]

        # Store first valid channel as primary result for other tabs
        first_valid = next((v for v in channels.values() if "error" not in v), None)
        if first_valid:
            _state["result"] = first_valid
        _state["method"] = method

        # Multi-channel plot
        fig_overlay = _plot_multi_channel(multi_result)

        # Combined log
        log_lines = [
            "=" * 45,
            f"  MULTI-CHANNEL Pipeline — {summary['total_channels']} kanal",
            "=" * 45, "",
        ]
        for col, res in channels.items():
            if "error" in res:
                log_lines.append(f"[{col}] HATA: {res['error']}")
            else:
                m = res["metrics"]
                counts = res.get("detector_counts", {})
                active = ", ".join(f"{k}:{v}" for k, v in counts.items() if v > 0)
                log_lines.append(f"[{col}] {m['faults_detected']} anomali | {active}")
        log_lines.extend(["", f"Toplam: {summary['total_faults']} anomali | Sure: {summary['processing_time']:.2f}s"])
        log_text = "\n".join(log_lines)

        metrics_text = " | ".join(
            f"{col}: {summary['per_channel'].get(col, 0)}"
            for col in selected_columns
        ) + f" | Toplam: {summary['total_faults']} | Sure: {summary['processing_time']:.2f}s"

        # Detector breakdown from first valid channel
        fig_detectors = go.Figure()
        if first_valid:
            try:
                corrupted_first = _state["corrupted"]
                from pipeline.detector_classic import detect_all
                from pipeline.filters_classic import detrend_signal
                data_dt = detrend_signal(corrupted_first)
                det_masks = detect_all(data_dt, df_original=corrupted_first)
                fig_detectors = _plot_detector_breakdown(det_masks, corrupted_first)
            except Exception:
                pass

        # Fault timeline from all channels
        ft_rows = []
        for col, res in channels.items():
            if "error" not in res:
                ft = res.get("fault_timeline", pd.DataFrame())
                if not ft.empty:
                    ft = ft.copy()
                    ft.insert(0, "channel", col)
                    ft_rows.append(ft)
        if ft_rows:
            ft_display = pd.concat(ft_rows, ignore_index=True).head(100)
        else:
            ft_display = pd.DataFrame({"Bilgi": ["Anomali bulunamadi"]})

        rv_text = "Multi-channel: " + ", ".join(
            f"{col}: {'OK' if res.get('repair_verification', {}).get('passed', True) else 'SORUN'}"
            for col, res in channels.items() if "error" not in res
        )

        tracer_table = first_valid.get("tracer_table", pd.DataFrame()) if first_valid else pd.DataFrame()
        tracer_summary = first_valid.get("tracer_summary", "") if first_valid else ""

        return fig_overlay, log_text, fig_detectors, metrics_text, ft_display, rv_text, tracer_table, tracer_summary

    # ── Single-channel mode ──
    corrupted = _state["corrupted"]

    # If a single column is selected from multi-col CSV, use it
    if raw_multi is not None and selected_columns and len(selected_columns) == 1:
        col = selected_columns[0]
        corrupted = raw_multi[["timestamp", col]].rename(columns={col: "value"}).copy()
        corrupted["value"] = pd.to_numeric(corrupted["value"], errors="coerce")
        corrupted = corrupted.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        _state["corrupted"] = corrupted

    try:
        result, logs = _run_with_logs(corrupted, method)
    except Exception as e:
        return None, f"Pipeline hatasi: {e}", None, "", None, "", None, ""

    _state["result"] = result
    _state["method"] = method

    log_text = _format_pipeline_log(logs, method, result)
    fig_overlay = _plot_triple_overlay(
        clean, corrupted, result["cleaned_data"], result["fault_mask"],
    )

    # Metrics text
    metrics = result.get("metrics", {})
    faults = metrics.get("faults_detected", 0)
    proc_time = metrics.get("processing_time", 0)

    if clean is not None:
        from pipeline.validator import calculate_metrics
        gt_metrics = calculate_metrics(corrupted, result["cleaned_data"], ground_truth=clean)
        metrics_text = (
            f"RMSE: {gt_metrics.get('rmse', 0):.4f} | "
            f"SNR: {gt_metrics.get('snr', 0):.2f} dB | "
            f"R2: {gt_metrics.get('r2_score', 0):.4f} | "
            f"Anomali: {faults} | Sure: {proc_time:.3f}s"
        )
    else:
        metrics_text = f"Anomali: {faults} | Sure: {proc_time:.3f}s"

    # Detector breakdown
    try:
        from pipeline.detector_classic import detect_all
        from pipeline.filters_classic import detrend_signal
        data_dt = detrend_signal(corrupted)
        det_masks = detect_all(data_dt, df_original=corrupted)
        fig_detectors = _plot_detector_breakdown(det_masks, corrupted)
    except Exception:
        fig_detectors = go.Figure()

    # Fault timeline table
    ft = result.get("fault_timeline", pd.DataFrame())
    ft_display = ft.head(100) if not ft.empty else pd.DataFrame({"Bilgi": ["Anomali bulunamadi"]})

    # Repair verification
    rv = result.get("repair_verification", {})
    if rv.get("passed", True):
        rv_text = "Dogrulama basarili — yeni NaN/Inf yok, varyans normal"
    else:
        rv_text = "Sorun var: " + ", ".join(rv.get("issues", []))

    # Tracer outputs
    tracer_table = result.get("tracer_table", pd.DataFrame())
    tracer_summary = result.get("tracer_summary", "")

    return fig_overlay, log_text, fig_detectors, metrics_text, ft_display, rv_text, tracer_table, tracer_summary


# ── Gradio UI ──────────────────────────────────────────────────────────────────

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="*neutral_950",
    body_background_fill_dark="*neutral_950",
    block_background_fill="*neutral_900",
    block_background_fill_dark="*neutral_900",
    block_border_color="*neutral_700",
    block_label_text_color="*neutral_200",
    block_title_text_color="*neutral_100",
    input_background_fill="*neutral_800",
    input_background_fill_dark="*neutral_800",
    button_primary_background_fill="*primary_600",
    button_primary_text_color="white",
)

with gr.Blocks(title="Cosmic Pipeline") as app:

    gr.Markdown("# 🛰️ Kozmik Veri Ayiklama ve Isleme Hatti")
    gr.Markdown("**TUA Astro Hackathon 2026** — Ahmet & Omer")

    with gr.Tabs():

        # ── TAB 1: Veri ──────────────────────────────
        with gr.Tab("📡 Veri"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Sentetik Veri Parametreleri")
                    sl_n = gr.Slider(1000, 20000, value=5000, step=1000, label="Veri Noktasi")
                    num_seed = gr.Number(value=42, label="Seed", precision=0)
                    sl_seu = gr.Slider(5, 50, value=15, step=1, label="SEU (Bit-flip) Sayisi")
                    sl_tid = gr.Slider(0.001, 0.01, value=0.003, step=0.001, label="TID Drift Egimi")
                    sl_gap = gr.Slider(1, 10, value=4, step=1, label="Veri Boslugu Sayisi")
                    sl_noise = gr.Slider(0.5, 5.0, value=2.0, step=0.1, label="Gurultu Seviyesi")
                    btn_gen = gr.Button("🔬 Sentetik Veri Uret", variant="primary", size="lg")

                    gr.Markdown("---")
                    gr.Markdown("### CSV Yukle")
                    csv_file = gr.File(label="CSV/TSV/Excel/JSON/HDF5/Parquet", file_types=[".csv", ".tsv", ".xlsx", ".xls", ".json", ".h5", ".hdf5", ".parquet"])
                    btn_csv = gr.Button("📁 CSV Yukle")
                    chk_columns = gr.CheckboxGroup(
                        choices=[], value=[], visible=False,
                        label="Kanallar (Multi-Channel)",
                    )

                    txt_status = gr.Textbox(label="Durum", interactive=False)

                with gr.Column(scale=3):
                    plot_preview = gr.Plot(label="Sinyal Onizleme")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Temiz Veri")
                            tbl_clean = gr.Dataframe(label="Ground Truth", max_height=300)
                        with gr.Column():
                            gr.Markdown("#### Bozuk Veri")
                            tbl_corrupt = gr.Dataframe(label="Radyasyonlu", max_height=300)

            btn_gen.click(
                fn=generate_data,
                inputs=[sl_n, num_seed, sl_seu, sl_tid, sl_gap, sl_noise],
                outputs=[plot_preview, tbl_clean, tbl_corrupt, txt_status],
            )
            btn_csv.click(
                fn=upload_csv,
                inputs=[csv_file],
                outputs=[plot_preview, tbl_clean, tbl_corrupt, txt_status, chk_columns],
            )

        # ── TAB 2: Pipeline ──────────────────────────
        with gr.Tab("🔧 Pipeline"):
            with gr.Row():
                radio_method = gr.Radio(
                    choices=["classic", "ml", "both"],
                    value="classic",
                    label="Pipeline Yontemi",
                )
                btn_run = gr.Button(
                    "▶️ Pipeline Calistir", variant="primary", size="lg",
                )

            txt_metrics = gr.Textbox(label="Metrikler", interactive=False)

            with gr.Tabs():
                with gr.Tab("📈 Sonuc Grafigi"):
                    plot_result = gr.Plot(label="Ground Truth vs Bozuk vs Temizlenmis")

                with gr.Tab("🔍 Detektor Detayi"):
                    plot_det = gr.Plot(label="Her Dedektorun Buldugu Anomaliler")

                with gr.Tab("📋 Islem Raporu"):
                    code_log = gr.Code(label="Pipeline Log", language=None, lines=30)

                with gr.Tab("📊 Anomali Tablosu"):
                    tbl_faults = gr.Dataframe(label="Tespit Edilen Anomaliler", max_height=400)

                with gr.Tab("🔍 Dogrulama"):
                    txt_verify = gr.Textbox(
                        label="Onarim Dogrulama", interactive=False, lines=3,
                    )

                with gr.Tab("🔬 Adim Adim Izleme"):
                    tbl_tracer = gr.Dataframe(label="Pipeline Adim Tablosu", max_height=500)
                    code_tracer = gr.Code(label="Ozet Rapor", language=None, lines=25)

            btn_run.click(
                fn=run_pipeline_ui,
                inputs=[radio_method, chk_columns],
                outputs=[plot_result, code_log, plot_det, txt_metrics, tbl_faults, txt_verify, tbl_tracer, code_tracer],
            )


if __name__ == "__main__":
    app.launch(server_port=7860, share=False, theme=theme)
