"""Modern pipeline visualization — story-driven detector cards."""

import numpy as np


def _generate_pipeline_animation(result: dict, corrupted_df) -> str:
    """Generate story-driven HTML cards for each detector."""
    detector_counts = result.get("detector_counts", {})
    fault_timeline = result.get("fault_timeline")
    cleaned = result.get("cleaned_data")
    metrics = result.get("metrics", {})

    detector_info = {
        "zscore": {"name": "Z-Score Detector", "icon": "\U0001F4CA", "problem": "Istatistiksel aykiri deger", "method": "Z-score hesaplandi"},
        "sliding_window": {"name": "Sliding Window", "icon": "\U0001F4D0", "problem": "Kayan pencere sapmasi", "method": "Pencere ortalamasi kontrolu"},
        "gaps": {"name": "Gap Detector", "icon": "\U0001F573\uFE0F", "problem": "Veri boslugu", "method": "Zaman serisi surekliligi kontrolu"},
        "range": {"name": "Range Detector", "icon": "\U0001F3AF", "problem": "Fiziksel limit disi", "method": "Sensor limit kontrolu"},
        "delta": {"name": "Delta Detector", "icon": "\u26A1", "problem": "Ani sicrama", "method": "Ardisik fark kontrolu"},
        "flatline": {"name": "Flatline Detector", "icon": "\U0001F4CF", "problem": "Sensor dondu", "method": "Sabit deger tekrar kontrolu"},
        "duplicates": {"name": "Duplicate Detector", "icon": "\u264A", "problem": "Tekrar eden zaman", "method": "Timestamp tekrar kontrolu"},
        "isolation_forest": {"name": "Isolation Forest (ML)", "icon": "\U0001F332", "problem": "ML anomali", "method": "Isolation Forest skoru"},
        "lstm_ae": {"name": "LSTM Autoencoder (DL)", "icon": "\U0001F9E0", "problem": "Temporal pattern anomali", "method": "Reconstruction error"},
    }

    cards_html = ""
    for det_name, count in detector_counts.items():
        if count == 0:
            continue
        info = detector_info.get(det_name, {"name": det_name, "icon": "\U0001F50D", "problem": "Anomali", "method": "Tespit"})

        example_ts = None
        example_idx = None
        corrupted_val = None
        cleaned_val = None

        if fault_timeline is not None and not fault_timeline.empty:
            matches = fault_timeline[fault_timeline["fault_type"].str.contains(det_name, na=False)]
            if not matches.empty:
                row = matches.iloc[0]
                example_ts = row["timestamp"]
                if example_ts is not None and corrupted_df is not None:
                    try:
                        diffs = (corrupted_df["timestamp"] - example_ts).abs()
                        example_idx = diffs.idxmin()
                        val = corrupted_df.loc[example_idx, "value"]
                        corrupted_val = float(val) if not np.isnan(val) else None
                        if cleaned is not None and example_idx in cleaned.index:
                            cval = cleaned.loc[example_idx, "value"]
                            cleaned_val = float(cval) if not np.isnan(cval) else None
                    except Exception:
                        pass

        ts_str = str(example_ts)[:19] if example_ts is not None else "\u2014"
        idx_str = f"#{example_idx}" if example_idx is not None else ""
        cor_str = f"{corrupted_val:.1f}" if corrupted_val is not None else "\u2014"
        cln_str = f"{cleaned_val:.1f}" if cleaned_val is not None else "\u2014"

        cards_html += f"""
        <div class="det-card">
          <div class="det-header">
            <span class="det-title">{info['icon']} {info['name']}</span>
            <span class="det-badge">{count} tespit</span>
          </div>
          <div class="det-body">
            <div class="det-grid">
              <div class="det-block">
                <div class="det-label">\U0001F4CD NOKTA</div>
                <div class="det-value">{ts_str} {idx_str}</div>
              </div>
              <div class="det-block">
                <div class="det-label">\u274C PROBLEM</div>
                <div class="det-value det-red">{info['problem']}</div>
              </div>
            </div>
            <div class="det-block det-full">
              <div class="det-label">\U0001F50D NASIL YAKALANDI</div>
              <div class="det-value det-blue">{info['method']}</div>
            </div>
            <div class="det-ba">
              <span class="ba-before">{cor_str}</span>
              <span class="ba-arrow">\u2192</span>
              <span class="ba-after">{cln_str}</span>
            </div>
          </div>
        </div>
        """

    total = metrics.get("faults_detected", 0)
    corrected = metrics.get("faults_corrected", total)
    elapsed = metrics.get("processing_time", 0)
    active_count = sum(1 for v in detector_counts.values() if v > 0)

    summary_html = f"""
    <div class="summary-card">
      <div class="summary-icon">\u2713</div>
      <div class="summary-title">Pipeline Tamamlandi</div>
      <div class="summary-stats">{total} anomali \u00B7 {corrected} duzeltildi \u00B7 {active_count} dedektor \u00B7 {elapsed:.2f}s</div>
    </div>
    """

    hero_html = f"""
    <div class="hero">
      <h1>\U0001F6F0\uFE0F Cosmic Data Cleaning Pipeline</h1>
      <p class="hero-sub">AI-powered anomaly detection &amp; repair</p>
      <div class="hero-stats">
        <div class="stat-card"><div class="stat-num">{active_count}</div><div class="stat-label">Dedektor</div></div>
        <div class="stat-card"><div class="stat-num">{total}</div><div class="stat-label">Anomali</div></div>
        <div class="stat-card"><div class="stat-num">{corrected}</div><div class="stat-label">Duzeltme</div></div>
        <div class="stat-card"><div class="stat-num">{elapsed:.1f}s</div><div class="stat-label">Sure</div></div>
      </div>
    </div>
    """

    css = """
    <style>
    .pipeline-viz{font-family:'Inter',system-ui,sans-serif;color:#c9d1d9;max-width:800px;margin:0 auto}
    .hero{text-align:center;padding:32px 24px;background:linear-gradient(135deg,#0d1117,#161b22);border-radius:16px;margin-bottom:24px;border:1px solid #21262d}
    .hero h1{color:#58a6ff;font-size:1.5em;margin:0 0 4px}
    .hero-sub{color:#7d8590;font-size:0.95em;margin:0 0 20px}
    .hero-stats{display:flex;gap:16px;justify-content:center;flex-wrap:wrap}
    .stat-card{background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:14px 20px;min-width:100px}
    .stat-num{color:#58a6ff;font-size:1.8em;font-weight:700;line-height:1}
    .stat-label{color:#7d8590;font-size:0.75em;text-transform:uppercase;letter-spacing:0.5px;margin-top:4px}
    .det-card{background:#0d1117;border:1px solid #21262d;border-radius:12px;margin-bottom:20px;overflow:hidden;box-shadow:0 4px 12px rgba(0,0,0,0.3)}
    .det-header{display:flex;justify-content:space-between;align-items:center;padding:14px 20px;border-bottom:1px solid #21262d;background:#161b22}
    .det-title{color:#58a6ff;font-size:1.1em;font-weight:600}
    .det-badge{background:#1f6feb22;color:#58a6ff;padding:4px 12px;border-radius:20px;font-size:0.85em;font-weight:500}
    .det-body{padding:20px}
    .det-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
    .det-block{background:#161b22;border-radius:8px;padding:12px}
    .det-full{margin-bottom:16px}
    .det-label{color:#7d8590;font-size:0.8em;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px}
    .det-value{color:#c9d1d9;font-size:0.95em}
    .det-red{color:#ef4444}
    .det-blue{color:#58a6ff}
    .det-ba{display:flex;align-items:center;justify-content:center;gap:20px;padding:20px;background:#161b22;border-radius:8px}
    .ba-before{color:#ef4444;font-size:2.8em;font-weight:700;line-height:1}
    .ba-arrow{color:#58a6ff;font-size:1.5em}
    .ba-after{color:#3fb950;font-size:2.8em;font-weight:700;line-height:1}
    .summary-card{text-align:center;padding:24px;background:linear-gradient(135deg,#0d1117,#161b22);border:1px solid #3fb950;border-radius:12px;margin-top:8px}
    .summary-icon{color:#3fb950;font-size:2em;margin-bottom:4px}
    .summary-title{color:#3fb950;font-size:1.2em;font-weight:600;margin-bottom:4px}
    .summary-stats{color:#7d8590;font-size:0.95em}
    </style>
    """

    return f"{css}<div class=\"pipeline-viz\">{hero_html}{cards_html}{summary_html}</div>"
