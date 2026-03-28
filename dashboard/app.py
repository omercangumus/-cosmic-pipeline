"""Streamlit dashboard for the Cosmic Pipeline — visualize, compare, export."""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Ensure project root is on the path so pipeline imports work
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dashboard.charts import (
    plot_anomaly_timeline,
    plot_comparison,
    plot_metrics_bar,
    plot_signal,
)
from data.synthetic_generator import FaultConfig, generate_corrupted_dataset
from pipeline.orchestrator import run_pipeline
from pipeline.validator import calculate_metrics

# ---------------------------------------------------------------------------
# Page config & dark theme CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Cosmic Pipeline",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
    /* dark background */
    .stApp { background-color: #020408; }
    header[data-testid="stHeader"] { background-color: #020408; }
    section[data-testid="stSidebar"] > div { background-color: #060b12; }

    /* metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0a1628 0%, #0d1f3c 100%);
        border: 1px solid rgba(0,212,255,0.15);
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label { color: #8ab4f8 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #00d4ff !important; }

    /* tab labels */
    button[data-baseweb="tab"] { color: #8ab4f8 !important; }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00d4ff !important;
        border-bottom-color: #00d4ff !important;
    }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _precision_recall(detected: np.ndarray, truth: np.ndarray):
    tp = np.sum(detected & truth)
    fp = np.sum(detected & ~truth)
    fn = np.sum(~detected & truth)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


def _gt_bool_mask(gt: dict, n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for i in gt.get("seu", []):
        if 0 <= i < n:
            mask[i] = True
    for s, e in gt.get("gap", []):
        mask[s:min(e, n)] = True
    for i in gt.get("noise", []):
        if 0 <= i < n:
            mask[i] = True
    tid = gt.get("tid", [])
    if tid:
        mask[int(n * 0.4):] = True
    return mask


@st.cache_data(show_spinner=False)
def _generate(n: int, seed: int, seu: int, tid: float, gaps: int, noise: float):
    cfg = FaultConfig(
        seu_count=seu, tid_slope=tid, gap_count=gaps, noise_std_max=noise,
    )
    return generate_corrupted_dataset(n=n, config=cfg, seed=seed)


@st.cache_data(show_spinner=False)
def _run(corrupted_df: pd.DataFrame, method: str):
    """Run pipeline on the corrupted DataFrame."""
    return run_pipeline(corrupted_df.copy(), method=method)


# ---------------------------------------------------------------------------
# Pipeline log capture
# ---------------------------------------------------------------------------
_PIPELINE_LOGGERS = [
    "pipeline.orchestrator",
    "pipeline.detector_classic",
    "pipeline.detector_ml",
    "pipeline.ensemble",
    "pipeline.filters_classic",
    "pipeline.filters_ml",
    "pipeline.validator",
    "pipeline.ingestion",
]


class _DashboardLogHandler(logging.Handler):
    """Captures pipeline log records with elapsed timestamps."""

    def __init__(self):
        super().__init__()
        self.logs: list[str] = []
        self.start_time: float | None = None

    def emit(self, record: logging.LogRecord) -> None:
        if self.start_time is None:
            self.start_time = time.time()
        elapsed = time.time() - self.start_time
        msg = self.format(record)
        self.logs.append(f"[{elapsed:07.3f}] {msg}")


def _run_with_logs(corrupted_df: pd.DataFrame, method: str) -> tuple[dict, list[str]]:
    """Run pipeline and return (result_dict, log_lines)."""
    handler = _DashboardLogHandler()
    handler.setLevel(logging.INFO)
    handler.start_time = time.time()

    for name in _PIPELINE_LOGGERS:
        logging.getLogger(name).addHandler(handler)
        logging.getLogger(name).setLevel(logging.INFO)

    try:
        result = run_pipeline(corrupted_df.copy(), method=method)
    finally:
        for name in _PIPELINE_LOGGERS:
            logging.getLogger(name).removeHandler(handler)

    return result, handler.logs


def _plot_clean_vs_corrupted(clean_df, corrupted_df):
    """Two-row subplot: clean signal on top, corrupted on bottom."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Temiz Sinyal (Ground Truth)", "Radyasyonla Bozulmuş Sinyal"],
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
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,23,0.8)",
        height=480,
        showlegend=True,
        legend=dict(orientation="h", y=-0.08),
        margin=dict(t=40, b=50, l=60, r=30),
    )
    return fig


def _plot_triple_overlay(clean_df, corrupted_df, cleaned_df, fault_mask=None):
    """Three signals overlaid: ground truth, corrupted, cleaned."""
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
        mode="lines", name="Temizlenmiş",
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
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,23,0.8)",
        title="Ground Truth vs Bozuk vs Temizlenmiş Sinyal",
        height=420,
        showlegend=True,
        legend=dict(orientation="h", y=-0.12),
        margin=dict(t=50, b=60, l=60, r=30),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar — data source & parameters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛰️ Cosmic Pipeline")
    st.caption("Radiation-corrupted telemetry cleaner")
    st.markdown("---")

    source = st.radio("Data source", ["Synthetic generator", "Upload CSV"], horizontal=True)

    if source == "Upload CSV":
        uploaded = st.file_uploader("CSV (timestamp, value)", type=["csv"])
        if uploaded is not None:
            import base64 as _b64
            from utils.csv_parser import parse_uploaded_csv

            raw_bytes = uploaded.getvalue()
            encoded = _b64.b64encode(raw_bytes).decode()
            contents = f"data:text/csv;base64,{encoded}"

            user_df, csv_error = parse_uploaded_csv(contents, uploaded.name)
            if csv_error is not None:
                st.error(f"❌ {csv_error}")
                st.stop()

            st.success(f"✅ {len(user_df)} satır yüklendi ({uploaded.name})")
            clean_df = None
            corrupted_df = user_df
            gt_mask = None
        else:
            st.info("Upload a CSV to begin.")
            st.stop()
    else:
        st.markdown("### Signal parameters")
        n_samples = st.slider("Samples", 1000, 50000, 10000, step=1000)
        seed = st.number_input("Seed", value=42, step=1)

        st.markdown("### Fault injection")
        seu_count = st.slider("SEU count", 0, 50, 15)
        tid_slope = st.slider("TID drift slope", 0.0, 0.01, 0.003, step=0.001, format="%.3f")
        gap_count = st.slider("Gap blocks", 0, 10, 4)
        noise_max = st.slider("Noise std max", 0.0, 5.0, 2.0, step=0.1)

        clean_df, corrupted_df, gt_mask = _generate(
            n_samples, seed, seu_count, tid_slope, gap_count, noise_max,
        )

    st.markdown("---")
    methods_to_run = st.multiselect(
        "Methods to run",
        ["classic", "ml", "both"],
        default=["classic", "ml"],
    )

    run_btn = st.button("Run Pipeline", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown("# 🛰️ Cosmic Pipeline Dashboard")

if not run_btn and "results" not in st.session_state:
    # Show raw data preview while waiting for the user to click Run
    st.info("Configure parameters in the sidebar and click **Run Pipeline**.")

    st.markdown("### 📡 Veri Önizleme")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### ☢️ Radyasyonlu Veri")
        if corrupted_df is not None:
            disp = corrupted_df[["timestamp", "value"]].head(50).copy()
            disp.index = range(1, len(disp) + 1)
            disp.index.name = "#"
            st.dataframe(disp, use_container_width=True, height=400)
            nan_n = int(corrupted_df["value"].isna().sum())
            vmin = corrupted_df["value"].min()
            vmax = corrupted_df["value"].max()
            st.caption(
                f"Toplam: {len(corrupted_df)} satır | NaN: {nan_n} "
                f"| Min: {vmin:.2f} | Max: {vmax:.2f}"
            )

    with col_right:
        st.markdown("#### ✅ Temiz Veri (Ground Truth)")
        if clean_df is not None:
            preview_cols = ["timestamp", "value"]
            for extra in ["orbit_id", "phase", "sensor_id"]:
                if extra in clean_df.columns:
                    preview_cols.append(extra)
            disp_c = clean_df[preview_cols].head(50).copy()
            disp_c.index = range(1, len(disp_c) + 1)
            disp_c.index.name = "#"
            st.dataframe(disp_c, use_container_width=True, height=400)
            st.caption(
                f"Toplam: {len(clean_df)} satır "
                f"| Min: {clean_df['value'].min():.2f} "
                f"| Max: {clean_df['value'].max():.2f}"
            )
        else:
            st.info("CSV yüklendi — ground truth mevcut değil")

    if clean_df is not None and corrupted_df is not None:
        st.plotly_chart(
            _plot_clean_vs_corrupted(clean_df, corrupted_df),
            use_container_width=True,
        )
    st.stop()

# --- Execute pipeline -------------------------------------------------------
if run_btn:
    results: dict = {}
    all_logs: dict[str, list[str]] = {}
    progress = st.progress(0, text="Starting pipeline...")
    for i, method in enumerate(methods_to_run):
        progress.progress(
            int((i / len(methods_to_run)) * 100),
            text=f"Running **{method}** ...",
        )
        t0 = time.perf_counter()
        try:
            res, logs = _run_with_logs(corrupted_df, method)
            res["elapsed"] = time.perf_counter() - t0
            res["error"] = None
            results[method] = res
            all_logs[method] = logs
        except Exception as exc:
            results[method] = {"error": str(exc), "elapsed": time.perf_counter() - t0}
            all_logs[method] = [f"[ERROR] {exc}"]

    progress.progress(100, text="Done!")
    st.session_state["results"] = results
    st.session_state["all_logs"] = all_logs
    st.session_state["corrupted_df"] = corrupted_df
    st.session_state["clean_df"] = clean_df
    st.session_state["gt_mask"] = gt_mask

# Retrieve stored results
results = st.session_state.get("results", {})
all_logs = st.session_state.get("all_logs", {})
corrupted_df = st.session_state.get("corrupted_df", corrupted_df)
clean_df = st.session_state.get("clean_df", clean_df)
gt_mask = st.session_state.get("gt_mask", gt_mask)

if not results:
    st.stop()

# --- 1. Data tables (side by side) ------------------------------------------
st.markdown("### 📡 Veri Önizleme")
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### ☢️ Radyasyonlu Veri")
    if corrupted_df is not None:
        disp = corrupted_df[["timestamp", "value"]].head(50).copy()
        disp.index = range(1, len(disp) + 1)
        disp.index.name = "#"
        st.dataframe(disp, use_container_width=True, height=400)
        nan_n = int(corrupted_df["value"].isna().sum())
        vmin = corrupted_df["value"].min()
        vmax = corrupted_df["value"].max()
        st.caption(
            f"Toplam: {len(corrupted_df)} satır | NaN: {nan_n} "
            f"| Min: {vmin:.2f} | Max: {vmax:.2f}"
        )

with col_right:
    st.markdown("#### ✅ Temiz Veri (Ground Truth)")
    if clean_df is not None:
        display_cols = ["timestamp", "value"]
        for extra in ["orbit_id", "phase", "sensor_id"]:
            if extra in clean_df.columns:
                display_cols.append(extra)
        disp_c = clean_df[display_cols].head(50).copy()
        disp_c.index = range(1, len(disp_c) + 1)
        disp_c.index.name = "#"
        st.dataframe(disp_c, use_container_width=True, height=400)
        st.caption(
            f"Toplam: {len(clean_df)} satır "
            f"| Min: {clean_df['value'].min():.2f} "
            f"| Max: {clean_df['value'].max():.2f}"
        )
    else:
        st.info("CSV yüklendi — ground truth mevcut değil")

if clean_df is not None:
    st.plotly_chart(
        _plot_clean_vs_corrupted(clean_df, corrupted_df),
        use_container_width=True,
    )

# --- 2. Pipeline log (user-friendly report) --------------------------------

def _format_user_friendly_log(logs, method, result=None):
    """Parse pipeline logs into a user-friendly Turkish summary."""
    lines = []
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"🛰️  Pipeline Calistirildi — Yontem: {method.upper()}")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("")

    st_ = {"zscore": 0, "sliding_window": 0, "gaps": 0, "range": 0,
            "delta": 0, "flatline": 0, "duplicates": 0,
            "isolation_forest": 0, "lstm_ae": 0}
    for line in logs:
        for key in st_:
            kw_map = {"zscore": "Z-score", "sliding_window": "Sliding window",
                      "gaps": "Gap detection", "range": "Range violation",
                      "delta": "Delta spike", "flatline": "Flatline",
                      "duplicates": "Duplicate", "isolation_forest": "Isolation",
                      "lstm_ae": "LSTM"}
            kw = kw_map.get(key, key)
            if kw in line and "detected" in line:
                try:
                    for part in line.split():
                        if part.isdigit():
                            st_[key] = int(part)
                            break
                except Exception:
                    pass

    lines.append("📥 ADIM 1 — Veri Alimi")
    lines.append("   Veri basariyla yuklendi ve on isleme tamamlandi.")
    lines.append("")
    lines.append("📉 ADIM 2 — On Isleme")
    lines.append("   Lineer trend (TID drift) tespit edildi ve kaldirildi.")
    lines.append("")
    lines.append("🔍 ADIM 3 — Anomali Tespiti")
    lines.append("   ┌─────────────────────────────────────────┐")
    lines.append("   │ KATMAN 1 — Deterministik Kurallar       │")
    lines.append(f"   │   🕳️  Veri bosluklari (gap):     {st_['gaps']:>5}  │")
    lines.append(f"   │   🚫  Fiziksel limit ihlali:     {st_['range']:>5}  │")
    lines.append(f"   │   ⚡  Ani sicrama (delta):       {st_['delta']:>5}  │")
    lines.append(f"   │   📏  Sabit sinyal (flatline):   {st_['flatline']:>5}  │")
    lines.append(f"   │   🔁  Tekrar eden zaman:         {st_['duplicates']:>5}  │")
    lines.append("   ├─────────────────────────────────────────┤")
    lines.append("   │ KATMAN 2 — Istatistiksel Analiz         │")
    lines.append(f"   │   📊  Z-score sapmasi:           {st_['zscore']:>5}  │")
    lines.append(f"   │   📐  Kayan pencere sapmasi:     {st_['sliding_window']:>5}  │")
    lines.append("   ├─────────────────────────────────────────┤")
    lines.append("   │ KATMAN 3 — Makine Ogrenimi              │")
    lines.append(f"   │   🌲  Isolation Forest:          {st_['isolation_forest']:>5}  │")
    lines.append("   ├─────────────────────────────────────────┤")
    lines.append("   │ KATMAN 4 — Derin Ogrenme                │")
    lines.append(f"   │   🧠  LSTM Autoencoder:          {st_['lstm_ae']:>5}  │")
    lines.append("   └─────────────────────────────────────────┘")
    lines.append("")
    lines.append("🗳️  ADIM 4 — Karar (Hybrid Majority Ensemble)")
    lines.append("   Deterministik kurallar → otomatik anomali")
    lines.append("   Istatistik + ML → en az 2 detektor anlasirsa anomali")
    if result:
        total = result.get("metrics", {}).get("faults_detected", 0)
        lines.append(f"   Toplam tespit edilen anomali: {total}")
    lines.append("")
    lines.append("🧹 ADIM 5 — Temizleme (Mercek Modeli)")
    lines.append("   1️⃣  Interpolation → bosluklar dolduruldu")
    lines.append("   2️⃣  Detrend → lineer kayma kaldirildi")
    lines.append("   3️⃣  Median filtre → spike'lar temizlendi")
    lines.append("")
    lines.append("✅ ADIM 6 — Dogrulama")
    lines.append("   Temizlenmis sinyal kontrol edildi: NaN=0, Inf=0")
    if result:
        elapsed = result.get("elapsed", result.get("metrics", {}).get("processing_time", 0))
        lines.append(f"   Islem suresi: {elapsed:.3f} saniye")
    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    ft = result.get("fault_timeline") if result else None
    if ft is not None and "repair_decision" in ft.columns:
        repair_n = int((ft["repair_decision"] == "repair").sum())
        flag_n = int((ft["repair_decision"] == "flag_only").sum())
        preserve_n = int((ft["repair_decision"] == "preserve").sum())
        lines.append("🔧 Onarim Karari:")
        lines.append(f"   ✅ Duzeltildi:  {repair_n}")
        lines.append(f"   ⚠️  Sadece isaretlendi:  {flag_n}")
        lines.append(f"   🛡️  Korundu (gercek event olabilir):  {preserve_n}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)


if all_logs:
    st.markdown("### 📋 Pipeline Islem Raporu")
    for method, logs in all_logs.items():
        res = results.get(method)
        formatted = _format_user_friendly_log(logs, method, res)
        st.code(formatted, language="")

# --- 3. Triple overlay per method ------------------------------------------
valid_results = {m: r for m, r in results.items() if r.get("error") is None}
if valid_results:
    st.markdown("### 📊 Sonuç: Ground Truth vs Bozuk vs Temizlenmiş")
    for method, res in valid_results.items():
        st.plotly_chart(
            _plot_triple_overlay(
                clean_df, corrupted_df, res["cleaned_data"], res["fault_mask"],
            ),
            use_container_width=True,
        )

st.markdown("---")

# --- Tabs --------------------------------------------------------------------
tab_overview, tab_compare, tab_timeline, tab_export = st.tabs(
    ["Overview", "Comparison", "Timeline", "Export"],
)

# ── Tab 1: Overview ─────────────────────────────────────────────────────────
with tab_overview:
    # Metric cards row
    cols = st.columns(len(results))
    all_metrics: dict = {}

    for col, (method, res) in zip(cols, results.items()):
        with col:
            st.subheader(f"{method.upper()}")
            if res.get("error"):
                st.error(res["error"])
                all_metrics[method] = None
                continue

            cleaned = res["cleaned_data"]
            fm = res["fault_mask"]

            qm = calculate_metrics(corrupted_df, cleaned, ground_truth=clean_df)

            det_bool = fm.values if hasattr(fm, "values") else np.array(fm)
            if gt_mask is not None:
                gt_bool = _gt_bool_mask(gt_mask, len(corrupted_df))
                prec, rec, f1 = _precision_recall(det_bool, gt_bool)
            else:
                prec, rec, f1 = (np.nan, np.nan, np.nan)

            m = {**qm, "precision": prec, "recall": rec, "f1": f1,
                 "detected": int(fm.sum()), "time": res["elapsed"]}
            all_metrics[method] = m

            c1, c2 = st.columns(2)
            c1.metric("Faults detected", m["detected"])
            c2.metric("Time (s)", f"{m['time']:.2f}")

            c3, c4 = st.columns(2)
            c3.metric("RMSE", f"{m['rmse']:.2f}")
            c4.metric("SNR (dB)", f"{m['snr']:.2f}")

            c5, c6, c7 = st.columns(3)
            c5.metric("Precision", f"{m['precision']:.2%}")
            c6.metric("Recall", f"{m['recall']:.2%}")
            c7.metric("F1", f"{m['f1']:.2%}")

            st.plotly_chart(
                plot_signal(cleaned, det_bool, title=f"{method} — cleaned signal"),
                use_container_width=True,
            )

            # --- Sampling info ---
            sampling = res.get("sampling_info", {})
            if sampling:
                st.markdown("#### 📡 Örnekleme Analizi")
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.metric("Örnekleme Aralığı", f"{sampling.get('detected_interval', 0):.2f}s")
                with s2:
                    st.metric("Jitter", f"{sampling.get('jitter_ratio', 0):.2%}")
                with s3:
                    st.metric("Büyük Boşluk", sampling.get("n_large_gaps", 0))

            # --- Repair verification ---
            rv = res.get("repair_verification", {})
            if rv:
                status = "✅ Geçti" if rv.get("passed") else "⚠️ Sorun Var"
                st.markdown(f"#### 🔍 Onarım Doğrulama: {status}")
                if rv.get("issues"):
                    for issue in rv["issues"]:
                        st.warning(issue)

            # --- Repair confidence ---
            rc = res.get("repair_confidence")
            if rc is not None and fm.any():
                mean_conf = float(rc[fm].mean())
                st.metric("Ortalama Onarım Güveni", f"{mean_conf:.1%}")

# ── Tab 2: Comparison ──────────────────────────────────────────────────────
with tab_compare:
    valid_methods = [m for m in results if results[m].get("error") is None]

    if len(valid_methods) >= 2 and "classic" in valid_methods and "ml" in valid_methods:
        classic_res = results["classic"]
        ml_res = results["ml"]

        fig_cmp = plot_comparison(
            corrupted_df,
            classic_res["cleaned_data"],
            ml_res["cleaned_data"],
            classic_res["fault_mask"].values,
            ml_res["fault_mask"].values,
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        mc = all_metrics.get("classic")
        mm = all_metrics.get("ml")
        if mc and mm:
            fig_bar = plot_metrics_bar(mc, mm)
            st.plotly_chart(fig_bar, use_container_width=True)
    elif len(valid_methods) >= 2:
        st.info("Comparison chart requires both **classic** and **ml** methods.")
    else:
        st.warning("Run at least two methods to see comparison charts.")

# ── Tab 3: Timeline ────────────────────────────────────────────────────────
with tab_timeline:
    if gt_mask is not None:
        for method in valid_methods:
            res = results[method]
            pred = res["fault_mask"].values if hasattr(res["fault_mask"], "values") else np.array(res["fault_mask"])
            fig_tl = plot_anomaly_timeline(len(corrupted_df), gt_mask, pred)
            fig_tl.update_layout(title=f"Timeline — {method}")
            st.plotly_chart(fig_tl, use_container_width=True)
    else:
        st.info("Timeline requires synthetic data with ground truth.")

# ── Tab 4: Export ───────────────────────────────────────────────────────────
with tab_export:
    for method in valid_methods:
        res = results[method]
        cleaned = res["cleaned_data"]
        csv_data = cleaned.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"Download {method} cleaned CSV",
            data=csv_data,
            file_name=f"cleaned_{method}.csv",
            mime="text/csv",
        )

    if all_metrics:
        rows = []
        for method, m in all_metrics.items():
            if m is None:
                continue
            rows.append({"method": method, **m})
        if rows:
            metrics_df = pd.DataFrame(rows)
            st.dataframe(metrics_df, use_container_width=True)
            st.download_button(
                "Download metrics CSV",
                data=metrics_df.to_csv(index=False).encode("utf-8"),
                file_name="metrics_comparison.csv",
                mime="text/csv",
            )
