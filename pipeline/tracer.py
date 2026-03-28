"""Pipeline step tracer — records what happens to data at each step."""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    """Record of a single pipeline step."""
    step_number: int
    step_name: str
    description: str
    duration_ms: float
    input_count: int
    output_count: int
    nan_before: int
    nan_after: int
    mean_before: float
    mean_after: float
    std_before: float
    std_after: float
    min_before: float
    min_after: float
    max_before: float
    max_after: float
    anomalies_found: int = 0
    anomaly_details: str = ""
    change_summary: str = ""


def _safe_stats(values: np.ndarray) -> dict:
    """Compute stats from a value array, handling all-NaN gracefully."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(float(np.nanmean(values)), 4),
        "std": round(float(np.nanstd(values)), 4),
        "min": round(float(np.nanmin(values)), 4),
        "max": round(float(np.nanmax(values)), 4),
    }


class PipelineTracer:
    """
    Traces pipeline steps, recording what happens to the data at each stage.

    Usage::

        tracer = PipelineTracer()
        tracer.snapshot("Load", "Data loaded", df_before, df_after)
        tracer.snapshot_detection("Z-Score", "scan", mask, df)
        table = tracer.to_dataframe()
        summary = tracer.to_summary()
    """

    def __init__(self) -> None:
        self.steps: list[StepRecord] = []
        self._counter = 0
        self._last_time = time.time()

    # ── Recording methods ─────────────────────────────────────────────────

    def snapshot(
        self,
        step_name: str,
        description: str,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
    ) -> StepRecord:
        """Record a filtering / transformation step."""
        now = time.time()
        duration = (now - self._last_time) * 1000
        self._last_time = now
        self._counter += 1

        v_before = df_before["value"].values.astype(np.float64)
        v_after = df_after["value"].values.astype(np.float64)
        sb = _safe_stats(v_before)
        sa = _safe_stats(v_after)

        # Change summary
        nan_diff = int(np.isnan(v_after).sum()) - int(np.isnan(v_before).sum())
        changes: list[str] = []
        if nan_diff < 0:
            changes.append(f"{abs(nan_diff)} NaN dolduruldu")
        if nan_diff > 0:
            changes.append(f"{nan_diff} yeni NaN")
        mean_change = abs(sa["mean"] - sb["mean"])
        if mean_change > 0.01:
            changes.append(f"ortalama {mean_change:.2f} degisti")
        std_change = sa["std"] - sb["std"]
        if abs(std_change) > 0.01:
            d = "artti" if std_change > 0 else "azaldi"
            changes.append(f"varyans {abs(std_change):.2f} {d}")
        if len(v_before) == len(v_after):
            finite_both = np.isfinite(v_before) & np.isfinite(v_after)
            if finite_both.any():
                max_pt = float(np.max(np.abs(v_after[finite_both] - v_before[finite_both])))
                if max_pt > 0.1:
                    changes.append(f"max nokta degisimi: {max_pt:.2f}")
        change_summary = " | ".join(changes) if changes else "minimal degisim"

        record = StepRecord(
            step_number=self._counter,
            step_name=step_name,
            description=description,
            duration_ms=round(duration, 1),
            input_count=len(df_before),
            output_count=len(df_after),
            nan_before=int(np.isnan(v_before).sum()),
            nan_after=int(np.isnan(v_after).sum()),
            mean_before=sb["mean"], mean_after=sa["mean"],
            std_before=sb["std"], std_after=sa["std"],
            min_before=sb["min"], min_after=sa["min"],
            max_before=sb["max"], max_after=sa["max"],
            change_summary=change_summary,
        )
        self.steps.append(record)
        return record

    def snapshot_detection(
        self,
        step_name: str,
        description: str,
        mask: pd.Series,
        df: pd.DataFrame,
        detector_name: str = "",
    ) -> StepRecord:
        """Record an anomaly detection step."""
        now = time.time()
        duration = (now - self._last_time) * 1000
        self._last_time = now
        self._counter += 1

        v = df["value"].values.astype(np.float64)
        s = _safe_stats(v)
        n_anom = int(mask.sum())

        if 0 < n_anom <= 20:
            idx_arr = np.where(mask.values)[0][:10]
            parts = []
            for i in idx_arr:
                val = v[i] if np.isfinite(v[i]) else "NaN"
                parts.append(f"#{i}={val}")
            detail = ", ".join(parts)
            if n_anom > 10:
                detail += f" ... (+{n_anom - 10} daha)"
        elif n_anom > 20:
            detail = f"{n_anom} nokta isaretlendi"
        else:
            detail = "anomali yok"

        record = StepRecord(
            step_number=self._counter,
            step_name=step_name,
            description=description,
            duration_ms=round(duration, 1),
            input_count=len(df), output_count=len(df),
            nan_before=int(np.isnan(v).sum()), nan_after=int(np.isnan(v).sum()),
            mean_before=s["mean"], mean_after=s["mean"],
            std_before=s["std"], std_after=s["std"],
            min_before=s["min"], min_after=s["min"],
            max_before=s["max"], max_after=s["max"],
            anomalies_found=n_anom,
            anomaly_details=detail,
            change_summary=f"{n_anom} anomali tespit edildi ({detector_name})",
        )
        self.steps.append(record)
        return record

    def snapshot_ensemble(
        self,
        hard_count: int,
        soft_count: int,
        total_count: int,
        strategy: str = "hybrid_majority",
    ) -> StepRecord:
        """Record the ensemble voting step."""
        now = time.time()
        duration = (now - self._last_time) * 1000
        self._last_time = now
        self._counter += 1

        record = StepRecord(
            step_number=self._counter,
            step_name="Ensemble Oylama",
            description=f"Strateji: {strategy}",
            duration_ms=round(duration, 1),
            input_count=0, output_count=0,
            nan_before=0, nan_after=0,
            mean_before=0, mean_after=0,
            std_before=0, std_after=0,
            min_before=0, min_after=0,
            max_before=0, max_after=0,
            anomalies_found=total_count,
            anomaly_details=f"Hard: {hard_count} | Soft: {soft_count} | Toplam: {total_count}",
            change_summary=f"Hard: {hard_count}, Soft: {soft_count}, Toplam: {total_count}",
        )
        self.steps.append(record)
        return record

    # ── Output methods ────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Return all steps as a human-readable DataFrame."""
        if not self.steps:
            return pd.DataFrame()
        rows = []
        for s in self.steps:
            rows.append({
                "#": s.step_number,
                "Adim": s.step_name,
                "Aciklama": s.description,
                "Sure (ms)": s.duration_ms,
                "Giris": s.input_count,
                "Cikis": s.output_count,
                "NaN Once": s.nan_before,
                "NaN Sonra": s.nan_after,
                "Ort. Once": s.mean_before,
                "Ort. Sonra": s.mean_after,
                "Std Once": s.std_before,
                "Std Sonra": s.std_after,
                "Anomali": s.anomalies_found,
                "Detay": s.anomaly_details,
                "Degisim": s.change_summary,
            })
        return pd.DataFrame(rows)

    def to_summary(self) -> str:
        """Return a human-readable text summary."""
        if not self.steps:
            return "Henuz adim kaydedilmedi."
        lines: list[str] = []
        lines.append("=" * 64)
        lines.append("  PIPELINE ADIM ADIM IZLEME RAPORU")
        lines.append("=" * 64)
        for s in self.steps:
            lines.append("")
            lines.append(f"  {s.step_number}. {s.step_name}")
            lines.append(f"     {s.description}")
            lines.append(f"     Sure: {s.duration_ms:.1f}ms")
            if s.anomalies_found > 0:
                lines.append(f"     {s.anomalies_found} anomali bulundu")
                if s.anomaly_details:
                    lines.append(f"        {s.anomaly_details[:60]}")
            if s.nan_before != s.nan_after:
                lines.append(f"     NaN: {s.nan_before} -> {s.nan_after}")
            if s.change_summary and "anomali" not in s.change_summary.lower():
                lines.append(f"     {s.change_summary[:60]}")
            lines.append(f"     Aralik: [{s.min_after:.2f}, {s.max_after:.2f}]")
            lines.append("-" * 64)
        total_time = sum(s.duration_ms for s in self.steps)
        total_anom = sum(s.anomalies_found for s in self.steps)
        lines.append(f"  Toplam: {len(self.steps)} adim | {total_time:.0f}ms | {total_anom} anomali")
        lines.append("=" * 64)
        return "\n".join(lines)
