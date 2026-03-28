"""Callback handlers for the Cosmic Pipeline dashboard."""

import base64
import logging
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data.synthetic_generator import FaultConfig, generate_corrupted_dataset
from pipeline.orchestrator import run_pipeline, run_pipeline_multi

# ── User-friendly error messages ──────────────────────────────────────────────

_ERROR_MESSAGES = {
    "Signal length must be": "Sinyal cok kisa. En az 100 veri noktasi gerekli.",
    "Missing required column": "CSV'de 'timestamp' ve 'value' sutunlari bulunamadi.",
    "No numeric columns": "CSV'de sayisal sutun bulunamadi.",
    "CUDA out of memory": "GPU bellegi yetersiz. 'classic' yontemi deneyin.",
    "No data": "Veri bulunamadi. Lutfen gecerli bir dosya yukleyin.",
    "Not enough finite": "Yeterli gecerli veri yok. NaN orani cok yuksek.",
}


def _user_friendly_error(e: Exception) -> str:
    msg = str(e)
    for key, friendly in _ERROR_MESSAGES.items():
        if key in msg:
            return friendly
    return f"Pipeline hatasi: {msg}"

from dashboard.plots import (
    _BG, _GRID, _TEMPLATE,
    plot_clean_vs_corrupted,
    plot_detector_breakdown,
    plot_multi_channel,
    plot_triple_overlay,
)

# ── Log capture ───────────────────────────────────────────────────────────────

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


# ── Log formatter ─────────────────────────────────────────────────────────────

def format_pipeline_log(logs, method, result=None):
    """Build a formatted pipeline log string for display."""
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


# ── App state ─────────────────────────────────────────────────────────────────

_state: dict = {}


# ── Callbacks ─────────────────────────────────────────────────────────────────

def generate_data(n_samples, seed, seu_count, tid_slope, gap_count, noise_max):
    """Generate synthetic telemetry data."""
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

        fig = plot_clean_vs_corrupted(clean, corrupted)

        clean_table = clean.head(30)
        corrupted_table = corrupted[["timestamp", "value"]].head(50)

        orbits = clean["orbit_id"].nunique() if "orbit_id" in clean.columns else "?"
        info = (
            f"OK — {int(n_samples)} satir uretildi | SEU: {int(seu_count)} | "
            f"Gap: {int(gap_count)} | Orbits: {orbits}"
        )
        return fig, clean_table, corrupted_table, info
    except Exception as e:
        return None, None, None, f"HATA: {_user_friendly_error(e)}"


def upload_csv(file):
    """Parse and load an uploaded CSV/TSV/Excel/JSON/HDF5/Parquet file."""
    empty_update = gr.update(choices=[], value=[], visible=False)
    if file is None:
        return None, None, None, "Dosya secilmedi", empty_update
    try:
        fname = Path(file.name).name
        raw_df = None
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

        time_aliases = {"timestamp", "time_tag", "date", "datetime", "time", "ds"}
        numeric_cols = []
        if raw_df is not None:
            numeric_cols = [
                c for c in raw_df.select_dtypes(include=["number"]).columns
                if c.lower() not in time_aliases and c != "label"
            ]

        if len(numeric_cols) > 1:
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
        return None, None, None, f"HATA: {_user_friendly_error(e)}", empty_update


def run_pipeline_ui(method, selected_columns):
    """Run the pipeline and return all UI outputs."""
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
            return None, _user_friendly_error(e), None, "", None, "", None, ""

        summary = multi_result["summary"]
        channels = multi_result["channels"]

        first_valid = next((v for v in channels.values() if "error" not in v), None)
        if first_valid:
            _state["result"] = first_valid
        _state["method"] = method

        fig_overlay = plot_multi_channel(multi_result)

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

        fig_detectors = go.Figure()
        if first_valid:
            try:
                corrupted_first = _state["corrupted"]
                from pipeline.detector_classic import detect_all
                from pipeline.filters_classic import detrend_signal
                data_dt = detrend_signal(corrupted_first)
                det_masks = detect_all(data_dt, df_original=corrupted_first)
                fig_detectors = plot_detector_breakdown(det_masks, corrupted_first)
            except Exception:
                pass

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

    if raw_multi is not None and selected_columns and len(selected_columns) == 1:
        col = selected_columns[0]
        corrupted = raw_multi[["timestamp", col]].rename(columns={col: "value"}).copy()
        corrupted["value"] = pd.to_numeric(corrupted["value"], errors="coerce")
        corrupted = corrupted.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        _state["corrupted"] = corrupted

    try:
        result, logs = _run_with_logs(corrupted, method)
    except Exception as e:
        return None, _user_friendly_error(e), None, "", None, "", None, ""

    _state["result"] = result
    _state["method"] = method

    log_text = format_pipeline_log(logs, method, result)
    fig_overlay = plot_triple_overlay(
        clean, corrupted, result["cleaned_data"], result["fault_mask"],
    )

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

    try:
        from pipeline.detector_classic import detect_all
        from pipeline.filters_classic import detrend_signal
        data_dt = detrend_signal(corrupted)
        det_masks = detect_all(data_dt, df_original=corrupted)
        fig_detectors = plot_detector_breakdown(det_masks, corrupted)
    except Exception:
        fig_detectors = go.Figure()

    ft = result.get("fault_timeline", pd.DataFrame())
    ft_display = ft.head(100) if not ft.empty else pd.DataFrame({"Bilgi": ["Anomali bulunamadi"]})

    rv = result.get("repair_verification", {})
    if rv.get("passed", True):
        rv_text = "Dogrulama basarili — yeni NaN/Inf yok, varyans normal"
    else:
        rv_text = "Sorun var: " + ", ".join(rv.get("issues", []))

    tracer_table = result.get("tracer_table", pd.DataFrame())
    tracer_summary = result.get("tracer_summary", "")

    return fig_overlay, log_text, fig_detectors, metrics_text, ft_display, rv_text, tracer_table, tracer_summary
