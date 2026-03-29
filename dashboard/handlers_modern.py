"""Modern pipeline visualization — story-driven detector cards with detailed explanations."""

import numpy as np
import pandas as pd


def _generate_pipeline_animation(result: dict, corrupted_df) -> str:
    """Generate story-driven HTML cards for each detector with full explanations."""
    detector_counts = result.get("detector_counts", {})
    fault_timeline = result.get("fault_timeline")
    cleaned = result.get("cleaned_data")
    metrics = result.get("metrics", {})

    values = corrupted_df["value"].values.astype(np.float64)
    finite = values[np.isfinite(values)]
    global_mean = float(np.nanmean(finite)) if len(finite) > 1 else 0
    global_std = float(np.nanstd(finite, ddof=1)) if len(finite) > 1 else 1

    # Deltas for delta detector
    deltas = np.abs(np.diff(values, prepend=values[0]))
    finite_deltas = deltas[np.isfinite(deltas)]
    delta_mean = float(np.nanmean(finite_deltas)) if len(finite_deltas) > 1 else 0
    delta_std = float(np.nanstd(finite_deltas)) if len(finite_deltas) > 1 else 1

    # Default thresholds (from detect_all defaults)
    THRESHOLDS = {
        "zscore": {"threshold": 2.0},
        "sliding_window": {"window": 50, "threshold": 3.0},
        "gaps": {"max_gap_seconds": 60},
        "range": {"multiplier": 10.0},
        "delta": {"multiplier": 5.0},
        "flatline": {"min_duration": 20},
    }

    detector_info = {
        "zscore": {"name": "Z-Score Detector", "icon": "\U0001F4CA"},
        "sliding_window": {"name": "Sliding Window", "icon": "\U0001F4D0"},
        "gaps": {"name": "Gap Detector", "icon": "\U0001F573\uFE0F"},
        "range": {"name": "Range Detector", "icon": "\U0001F3AF"},
        "delta": {"name": "Delta Detector", "icon": "\u26A1"},
        "flatline": {"name": "Flatline Detector", "icon": "\U0001F4CF"},
        "duplicates": {"name": "Duplicate Detector", "icon": "\u264A"},
        "isolation_forest": {"name": "Isolation Forest (ML)", "icon": "\U0001F332"},
        "lstm_ae": {"name": "LSTM Autoencoder (DL)", "icon": "\U0001F9E0"},
    }

    def _format_ts(ts_val):
        """Timestamp'i okunabilir formata cevir."""
        try:
            ts_f = float(ts_val)
            mins = int(ts_f // 60)
            secs = int(ts_f % 60)
            return f"{mins}. dakika {secs}. saniye"
        except (ValueError, TypeError):
            try:
                ts_pd = pd.Timestamp(ts_val)
                return ts_pd.strftime("%H:%M:%S")
            except Exception:
                return str(ts_val)

    def _get_example(det_name):
        """fault_timeline'dan bu dedektorun en anlamli ornek anomalisini bul."""
        if fault_timeline is None or fault_timeline.empty:
            return None
        matches = fault_timeline[fault_timeline["fault_type"].str.contains(det_name, na=False)]
        if matches.empty:
            return None

        # Severity'ye gore sirala, en yuksekten basla
        matches_sorted = matches.sort_values("severity", ascending=False)

        best = None
        best_ratio = float("inf")

        for _, row in matches_sorted.head(10).iterrows():
            ts = row["timestamp"]
            try:
                diffs = (corrupted_df["timestamp"] - ts).abs()
                idx = diffs.idxmin()
                cor_val = float(corrupted_df.loc[idx, "value"])
                cln_val = float(cleaned.loc[idx, "value"]) if cleaned is not None else None

                # NaN kontrolu
                if np.isnan(cor_val) or (cln_val is not None and np.isnan(cln_val)):
                    continue

                # Mantik kontrolu: temiz deger bozuktan cok farkli olmamali
                if cln_val is not None and cor_val != 0:
                    ratio = abs(cln_val - cor_val) / max(abs(cor_val), 1)
                    # Ideal: bozuk deger duzeltilmis ama makul aralikta
                    if ratio < best_ratio and ratio < 10:
                        best = {"ts": ts, "idx": idx, "cor": cor_val, "cln": cln_val}
                        best_ratio = ratio
                        if ratio < 2:  # Yeterince iyi, dur
                            break
                elif best is None:
                    best = {"ts": ts, "idx": idx, "cor": cor_val, "cln": cln_val}
            except Exception:
                continue

        # Hic mantikli bulamadiysa en yuksek severity olani al
        if best is None and not matches_sorted.empty:
            row = matches_sorted.iloc[0]
            ts = row["timestamp"]
            try:
                diffs = (corrupted_df["timestamp"] - ts).abs()
                idx = diffs.idxmin()
                cor_val = float(corrupted_df.loc[idx, "value"])
                cln_val = float(cleaned.loc[idx, "value"]) if cleaned is not None else None
                best = {"ts": ts, "idx": idx, "cor": cor_val, "cln": cln_val}
            except Exception:
                return None

        return best

    def _build_detail(det_name, ex):
        """Dedektore ozel detay HTML uret."""
        if ex is None:
            return (
                '<div class="det-block det-full"><div class="det-value">Ornek veri bulunamadi</div></div>',
                "",
                "",
            )

        cor = ex["cor"]
        cln = ex["cln"]
        idx = ex["idx"]

        if det_name == "zscore":
            t = THRESHOLDS["zscore"]["threshold"]
            z = abs(cor - global_mean) / global_std if global_std > 1e-12 else 0
            lo = global_mean - t * global_std
            hi = global_mean + t * global_std
            problem = f'''<div class="det-label">\u274C PROBLEM</div>
                <div class="det-value det-red">Istatistiksel aykiri deger</div>
                <div class="det-detail">
                    <span class="dt-item">Ortalama: <b>{global_mean:.1f}</b></span>
                    <span class="dt-item">Std sapma: <b>{global_std:.1f}</b></span>
                    <span class="dt-item">Beklenen aralik: <b>{lo:.0f} \u2014 {hi:.0f}</b></span>
                    <span class="dt-item det-red">Gercek deger: <b>{cor:.1f}</b> (aralik disi!)</span>
                </div>'''
            method = f'''<div class="det-label">\U0001F50D NASIL YAKALANDI</div>
                <div class="det-formula">Z = |{cor:.1f} \u2212 {global_mean:.1f}| / {global_std:.1f} = <b>{z:.2f}</b></div>
                <div class="det-value det-blue">Esik: {t} \u2192 {"Asildi!" if z > t else "Normal"}</div>'''
            repair = "Pencere ortalamasi ile degistirildi"

        elif det_name == "sliding_window":
            w = THRESHOLDS["sliding_window"]["window"]
            t = THRESHOLDS["sliding_window"]["threshold"]
            start = max(0, idx - w // 2)
            end = min(len(values), idx + w // 2)
            window_vals = values[start:end]
            window_finite = window_vals[np.isfinite(window_vals)]
            w_median = float(np.nanmedian(window_finite)) if len(window_finite) > 0 else 0
            w_std = float(np.nanstd(window_finite)) if len(window_finite) > 0 else 1
            dev = abs(cor - w_median) / w_std if w_std > 1e-12 else 0
            problem = f'''<div class="det-label">\u274C PROBLEM</div>
                <div class="det-value det-red">Lokal pencereden sapma</div>
                <div class="det-detail">
                    <span class="dt-item">Pencere boyutu: <b>{w} nokta</b></span>
                    <span class="dt-item">Pencere medyani: <b>{w_median:.1f}</b></span>
                    <span class="dt-item">Pencere std: <b>{w_std:.1f}</b></span>
                    <span class="dt-item det-red">Gercek deger: <b>{cor:.1f}</b></span>
                </div>'''
            method = f'''<div class="det-label">\U0001F50D NASIL YAKALANDI</div>
                <div class="det-formula">Sapma = |{cor:.1f} \u2212 {w_median:.1f}| / {w_std:.1f} = <b>{dev:.2f}</b></div>
                <div class="det-value det-blue">Esik: {t} \u2192 {"Asildi!" if dev > t else "Normal"}</div>'''
            repair = f"Pencere medyani ({w_median:.1f}) ile degistirildi"

        elif det_name == "delta":
            m = THRESHOLDS["delta"]["multiplier"]
            threshold = delta_mean + m * delta_std
            prev_val = float(values[idx - 1]) if idx > 0 else cor
            actual_delta = abs(cor - prev_val)
            problem = f'''<div class="det-label">\u274C PROBLEM</div>
                <div class="det-value det-red">Ani sicrama</div>
                <div class="det-detail">
                    <span class="dt-item">Onceki deger: <b>{prev_val:.1f}</b></span>
                    <span class="dt-item">Simdiki deger: <b>{cor:.1f}</b></span>
                    <span class="dt-item">Fark: <b>{actual_delta:.1f}</b></span>
                    <span class="dt-item">Ortalama fark: <b>{delta_mean:.1f}</b></span>
                </div>'''
            method = f'''<div class="det-label">\U0001F50D NASIL YAKALANDI</div>
                <div class="det-formula">Delta = |{cor:.1f} \u2212 {prev_val:.1f}| = <b>{actual_delta:.1f}</b></div>
                <div class="det-formula">Esik = {delta_mean:.1f} + {m}\xd7{delta_std:.1f} = <b>{threshold:.1f}</b></div>
                <div class="det-value det-blue">{"Asildi!" if actual_delta > threshold else "Normal"}</div>'''
            repair = "Komsularin ortalamasi ile degistirildi"

        elif det_name == "range":
            m = THRESHOLDS["range"]["multiplier"]
            lo = global_mean - m * global_std
            hi = global_mean + m * global_std
            problem = f'''<div class="det-label">\u274C PROBLEM</div>
                <div class="det-value det-red">Fiziksel limit disi</div>
                <div class="det-detail">
                    <span class="dt-item">Fiziksel aralik: <b>{lo:.0f} \u2014 {hi:.0f}</b></span>
                    <span class="dt-item">Carpan: <b>{m}\u03c3</b></span>
                    <span class="dt-item det-red">Gercek deger: <b>{cor:.1f}</b> (imkansiz!)</span>
                </div>'''
            method = f'''<div class="det-label">\U0001F50D NASIL YAKALANDI</div>
                <div class="det-formula">|{cor:.1f} \u2212 {global_mean:.1f}| = {abs(cor - global_mean):.1f} &gt; {m}\xd7{global_std:.1f} = {m * global_std:.1f}</div>
                <div class="det-value det-blue">Fiziksel limit ihlali!</div>'''
            repair = "Interpolasyon ile duzeltildi"

        elif det_name == "flatline":
            dur = THRESHOLDS["flatline"]["min_duration"]
            problem = f'''<div class="det-label">\u274C PROBLEM</div>
                <div class="det-value det-red">Sensor dondu (sabit sinyal)</div>
                <div class="det-detail">
                    <span class="dt-item">Deger: <b>{cor:.1f}</b> uzun sure sabit kaldi</span>
                    <span class="dt-item">Min tekrar esigi: <b>{dur} ardisik nokta</b></span>
                </div>'''
            method = f'''<div class="det-label">\U0001F50D NASIL YAKALANDI</div>
                <div class="det-formula">Ayni deger \u2265 {dur} kez tekrarlandi</div>
                <div class="det-value det-blue">Sensor ariza tespit edildi</div>'''
            repair = "Komsulardan interpolasyon ile duzeltildi"

        elif det_name == "gaps":
            max_gap = THRESHOLDS["gaps"]["max_gap_seconds"]
            problem = f'''<div class="det-label">\u274C PROBLEM</div>
                <div class="det-value det-red">Veri boslugu (kayip nokta)</div>
                <div class="det-detail">
                    <span class="dt-item">Max izin verilen bosluk: <b>{max_gap}s</b></span>
                    <span class="dt-item det-red">Bu noktada veri eksik</span>
                </div>'''
            method = f'''<div class="det-label">\U0001F50D NASIL YAKALANDI</div>
                <div class="det-formula">Zaman farki &gt; {max_gap} saniye</div>
                <div class="det-value det-blue">Veri surekliligi bozuldu</div>'''
            repair = "Interpolasyon ile dolduruldu"

        elif det_name == "duplicates":
            problem = '''<div class="det-label">\u274C PROBLEM</div>
                <div class="det-value det-red">Tekrar eden timestamp</div>
                <div class="det-detail">
                    <span class="dt-item det-red">Ayni zaman damgasi birden fazla</span>
                </div>'''
            method = '''<div class="det-label">\U0001F50D NASIL YAKALANDI</div>
                <div class="det-formula">Timestamp tekrar kontrolu</div>
                <div class="det-value det-blue">Duplike kayit tespit edildi</div>'''
            repair = "Duplike kayit kaldirildi"

        elif det_name == "isolation_forest":
            problem = f'''<div class="det-label">\u274C PROBLEM</div>
                <div class="det-value det-red">ML anomali (cok boyutlu aykiri)</div>
                <div class="det-detail">
                    <span class="dt-item">Model: <b>Isolation Forest</b></span>
                    <span class="dt-item det-red">Bu nokta normal veri dagilimina uymuyor</span>
                </div>'''
            method = f'''<div class="det-label">\U0001F50D NASIL YAKALANDI</div>
                <div class="det-value det-blue">Veri noktasi rastgele bolunmelerle hizla izole edilebildi \u2014 normal veriden farkli dagilim gosteriyor</div>'''
            repair = "Komsu degerlerin ortalamasi ile duzeltildi"

        elif det_name == "lstm_ae":
            problem = f'''<div class="det-label">\u274C PROBLEM</div>
                <div class="det-value det-red">Temporal pattern anomali</div>
                <div class="det-detail">
                    <span class="dt-item">Model: <b>LSTM Autoencoder</b></span>
                    <span class="dt-item det-red">Gecmis veriden ogrenen model bu noktayi tahmin edemedi</span>
                </div>'''
            method = f'''<div class="det-label">\U0001F50D NASIL YAKALANDI</div>
                <div class="det-value det-blue">Model sinyali yeniden uretmeye calisti, bu noktadaki hata (reconstruction error) diger noktalara gore cok yuksek</div>'''
            repair = "Model tahmini ile duzeltildi"

        else:
            problem = '''<div class="det-label">\u274C PROBLEM</div>
                <div class="det-value det-red">Anomali tespit edildi</div>'''
            method = '''<div class="det-label">\U0001F50D NASIL YAKALANDI</div>
                <div class="det-value det-blue">Otomatik tespit</div>'''
            repair = ""

        repair_html = f'<div class="det-repair">{repair}</div>' if repair else ""
        return problem, method, repair_html

    # Kartlari olustur
    cards_html = ""
    for det_name, count in detector_counts.items():
        if count == 0:
            continue
        info = detector_info.get(det_name, {"name": det_name, "icon": "\U0001F50D"})
        ex = _get_example(det_name)

        ts_str = _format_ts(ex["ts"]) if ex else "\u2014"
        idx_str = f'{ex["idx"]}. veri noktasi' if ex else ""
        cor_str = f'{ex["cor"]:.1f}' if ex else "\u2014"
        cln_str = f'{ex["cln"]:.1f}' if ex and ex["cln"] is not None else "\u2014"

        problem_html, method_html, repair_html = _build_detail(det_name, ex)

        # Mantik uyarisi
        ba_warning = ""
        if ex and ex["cor"] is not None and ex["cln"] is not None:
            ratio = abs(ex["cln"] - ex["cor"]) / max(abs(ex["cor"]), 1)
            if ratio > 5:
                ba_warning = '<div class="det-repair det-red" style="font-style:normal">\u26A0\uFE0F Bu ornek asiri bozulmus \u2014 tipik duzeltmeler daha kucuk farklar icerir</div>'

        cards_html += f"""
        <div class="det-card">
          <div class="det-header">
            <span class="det-title">{info["icon"]} {info["name"]}</span>
            <span class="det-badge">{count} tespit</span>
          </div>
          <div class="det-body">
            <div class="det-block">
              <div class="det-label">\U0001F4CD NOKTA</div>
              <div class="det-value">{ts_str} \u00B7 {idx_str}</div>
            </div>
            <div class="det-block">{problem_html}</div>
            <div class="det-block det-full">{method_html}</div>
            <div class="det-ba">
              <div class="ba-side">
                <div class="ba-label">BOZUK</div>
                <div class="ba-before">{cor_str}</div>
              </div>
              <span class="ba-arrow">\u2192</span>
              <div class="ba-side">
                <div class="ba-label">TEMIZ</div>
                <div class="ba-after">{cln_str}</div>
              </div>
            </div>
            {ba_warning}
            {repair_html}
          </div>
        </div>
        """

    # Hero + Summary
    total = metrics.get("faults_detected", 0)
    corrected = metrics.get("faults_corrected", total)
    elapsed = metrics.get("processing_time", 0)
    active_count = sum(1 for v in detector_counts.values() if v > 0)

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

    summary_html = f"""
    <div class="summary-card">
      <div class="summary-icon">\u2713</div>
      <div class="summary-title">Pipeline Tamamlandi</div>
      <div class="summary-stats">{total} anomali \u00B7 {corrected} duzeltildi \u00B7 {active_count} dedektor \u00B7 {elapsed:.2f}s</div>
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
    .det-body{padding:20px;display:flex;flex-direction:column;gap:12px}
    .det-block{background:#161b22;border-radius:8px;padding:12px}
    .det-full{width:100%}
    .det-label{color:#7d8590;font-size:0.8em;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px}
    .det-value{color:#c9d1d9;font-size:0.95em}
    .det-red{color:#ef4444}
    .det-blue{color:#58a6ff}
    .det-detail{display:flex;flex-direction:column;gap:3px;margin-top:6px;font-size:0.9em}
    .dt-item{color:#8b949e}
    .dt-item b{color:#c9d1d9}
    .det-formula{font-family:'JetBrains Mono',monospace;font-size:0.95em;color:#d2a8ff;background:#0d1117;padding:6px 10px;border-radius:4px;margin:4px 0}
    .det-formula b{color:#f0883e}
    .det-ba{display:flex;align-items:center;justify-content:center;gap:24px;padding:20px;background:#161b22;border-radius:8px}
    .ba-side{text-align:center}
    .ba-label{color:#7d8590;font-size:0.7em;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
    .ba-before{color:#ef4444;font-size:2.5em;font-weight:700;line-height:1}
    .ba-arrow{color:#58a6ff;font-size:1.5em}
    .ba-after{color:#3fb950;font-size:2.5em;font-weight:700;line-height:1}
    .det-repair{color:#7d8590;font-size:0.85em;font-style:italic;text-align:center;padding:4px}
    .summary-card{text-align:center;padding:24px;background:linear-gradient(135deg,#0d1117,#161b22);border:1px solid #3fb950;border-radius:12px;margin-top:8px}
    .summary-icon{color:#3fb950;font-size:2em;margin-bottom:4px}
    .summary-title{color:#3fb950;font-size:1.2em;font-weight:600;margin-bottom:4px}
    .summary-stats{color:#7d8590;font-size:0.95em}
    </style>
    """

    return f'{css}<div class="pipeline-viz">{hero_html}{cards_html}{summary_html}</div>'
