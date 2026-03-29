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

    total_points = len(corrupted_df)
    signal_range = float(np.nanmax(finite) - np.nanmin(finite)) if len(finite) > 1 else 1

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

    # Pipeline piramit katmanlari — gosterim sirasi
    LAYER_ORDER = [
        ("KATMAN 1 \u2014 Deterministik Kurallar", ["gaps", "range", "delta", "flatline", "duplicates"]),
        ("KATMAN 2 \u2014 Istatistiksel Analiz", ["zscore", "sliding_window"]),
        ("KATMAN 3 \u2014 Makine Ogrenimi", ["isolation_forest"]),
        ("KATMAN 4 \u2014 Derin Ogrenme", ["lstm_ae"]),
    ]

    quality_metrics = result.get("quality_metrics", {})

    def _get_stats(det_name):
        """Bu dedektorun istatistiklerini hesapla."""
        if fault_timeline is None or fault_timeline.empty:
            return None
        matches = fault_timeline[fault_timeline["fault_type"].str.contains(det_name, na=False)]
        if matches.empty:
            return None

        cor_vals = []
        cln_vals = []
        diffs = []

        for _, row in matches.iterrows():
            ts = row["timestamp"]
            try:
                ts_diffs = (corrupted_df["timestamp"] - ts).abs()
                idx = ts_diffs.idxmin()
                cv = float(corrupted_df.loc[idx, "value"])
                if np.isnan(cv):
                    continue
                cor_vals.append(cv)
                if cleaned is not None:
                    clv = float(cleaned.loc[idx, "value"])
                    if not np.isnan(clv):
                        cln_vals.append(clv)
                        diffs.append(abs(clv - cv))
            except Exception:
                continue

        if not cor_vals:
            return None

        stats = {
            "count": len(matches),
            "cor_mean": float(np.mean(cor_vals)),
            "cor_min": float(np.min(cor_vals)),
            "cor_max": float(np.max(cor_vals)),
        }
        if cln_vals:
            stats["cln_mean"] = float(np.mean(cln_vals))
            stats["diff_mean"] = float(np.mean(diffs))
            stats["diff_median"] = float(np.median(diffs))
            stats["diff_min"] = float(np.min(diffs))
            stats["diff_max"] = float(np.max(diffs))

        return stats

    def _build_detail(det_name, stats):
        """Dedektore ozel detay HTML uret."""
        if stats is None:
            return (
                '<div class="det-block det-full"><div class="det-value">Istatistik hesaplanamadi</div></div>',
                "",
                "",
            )

        if det_name == "zscore":
            t = THRESHOLDS["zscore"]["threshold"]
            lo = global_mean - t * global_std
            hi = global_mean + t * global_std
            problem = f'''<div class="det-label">\u274C PROBLEM</div>
                <div class="det-value det-red">Istatistiksel aykiri deger</div>
                <div class="det-detail">
                    <span class="dt-item">Sinyal ortalamasi: <b>{global_mean:.1f}</b></span>
                    <span class="dt-item">Std sapma: <b>{global_std:.1f}</b></span>
                    <span class="dt-item">Beklenen aralik: <b>{lo:.0f} \u2014 {hi:.0f}</b></span>
                </div>'''
            method = f'''<div class="det-label">\U0001F50D NASIL YAKALANDI</div>
                <div class="det-formula">Z = |deger \u2212 {global_mean:.1f}| / {global_std:.1f}</div>
                <div class="det-value det-blue">Esik: {t} (bu degeri asan noktalar anomali)</div>'''
            repair = "Pencere ortalamasi ile degistirildi"

        elif det_name == "sliding_window":
            w = THRESHOLDS["sliding_window"]["window"]
            t = THRESHOLDS["sliding_window"]["threshold"]
            problem = f'''<div class="det-label">\u274C PROBLEM</div>
                <div class="det-value det-red">Lokal pencereden sapma</div>
                <div class="det-detail">
                    <span class="dt-item">Pencere boyutu: <b>{w} nokta</b></span>
                    <span class="dt-item">Her nokta komsu {w} noktanin medyani ile karsilastirildi</span>
                </div>'''
            method = f'''<div class="det-label">\U0001F50D NASIL YAKALANDI</div>
                <div class="det-formula">Sapma = |deger \u2212 pencere_medyani| / pencere_std</div>
                <div class="det-value det-blue">Esik: {t} (bu degeri asan noktalar anomali)</div>'''
            repair = "Pencere medyani ile degistirildi"

        elif det_name == "delta":
            m = THRESHOLDS["delta"]["multiplier"]
            threshold = delta_mean + m * delta_std
            problem = f'''<div class="det-label">\u274C PROBLEM</div>
                <div class="det-value det-red">Ani sicrama</div>
                <div class="det-detail">
                    <span class="dt-item">Ortalama ardisik fark: <b>{delta_mean:.1f}</b></span>
                    <span class="dt-item">Fark std sapma: <b>{delta_std:.1f}</b></span>
                </div>'''
            method = f'''<div class="det-label">\U0001F50D NASIL YAKALANDI</div>
                <div class="det-formula">Esik = {delta_mean:.1f} + {m}\xd7{delta_std:.1f} = <b>{threshold:.1f}</b></div>
                <div class="det-value det-blue">Ardisik iki nokta arasi fark &gt; {threshold:.1f} ise anomali</div>'''
            repair = "Komsularin ortalamasi ile degistirildi"

        elif det_name == "range":
            m = THRESHOLDS["range"]["multiplier"]
            lo = global_mean - m * global_std
            hi = global_mean + m * global_std
            problem = f'''<div class="det-label">\u274C PROBLEM</div>
                <div class="det-value det-red">Fiziksel limit disi</div>
                <div class="det-detail">
                    <span class="dt-item">Fiziksel aralik: <b>{lo:.0f} \u2014 {hi:.0f}</b></span>
                    <span class="dt-item">Carpan: <b>{m}\u03c3</b> (cok agresif esik)</span>
                </div>'''
            method = f'''<div class="det-label">\U0001F50D NASIL YAKALANDI</div>
                <div class="det-formula">|deger \u2212 {global_mean:.1f}| &gt; {m} \xd7 {global_std:.1f} = {m * global_std:.1f}</div>
                <div class="det-value det-blue">Bu aralik disindaki degerler fiziksel olarak imkansiz</div>'''
            repair = "Interpolasyon ile duzeltildi"

        elif det_name == "flatline":
            dur = THRESHOLDS["flatline"]["min_duration"]
            problem = f'''<div class="det-label">\u274C PROBLEM</div>
                <div class="det-value det-red">Sensor dondu (sabit sinyal)</div>
                <div class="det-detail">
                    <span class="dt-item">Min tekrar esigi: <b>{dur} ardisik nokta</b></span>
                    <span class="dt-item">Ayni deger \u2265 {dur} kez tekrar ederse sensor arizasi</span>
                </div>'''
            method = f'''<div class="det-label">\U0001F50D NASIL YAKALANDI</div>
                <div class="det-formula">Ardisik ayni deger sayisi \u2265 {dur}</div>
                <div class="det-value det-blue">Sensor donmus \u2014 gercek sinyal interpolasyon ile kurtarildi</div>'''
            repair = "Komsulardan interpolasyon ile duzeltildi"

        elif det_name == "gaps":
            max_gap = THRESHOLDS["gaps"]["max_gap_seconds"]
            problem = f'''<div class="det-label">\u274C PROBLEM</div>
                <div class="det-value det-red">Veri boslugu (kayip nokta)</div>
                <div class="det-detail">
                    <span class="dt-item">Max izin verilen bosluk: <b>{max_gap} saniye</b></span>
                    <span class="dt-item">Bundan buyuk bosluklar anomali</span>
                </div>'''
            method = f'''<div class="det-label">\U0001F50D NASIL YAKALANDI</div>
                <div class="det-formula">Ardisik timestamp farki &gt; {max_gap}s</div>
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
            problem = '''<div class="det-label">\u274C PROBLEM</div>
                <div class="det-value det-red">ML anomali (cok boyutlu aykiri)</div>
                <div class="det-detail">
                    <span class="dt-item">Model: <b>Isolation Forest</b></span>
                    <span class="dt-item det-red">Bu nokta normal veri dagilimina uymuyor</span>
                </div>'''
            method = '''<div class="det-label">\U0001F50D NASIL YAKALANDI</div>
                <div class="det-value det-blue">Veri noktasi rastgele bolunmelerle hizla izole edilebildi \u2014 normal veriden farkli dagilim gosteriyor</div>'''
            repair = "Komsu degerlerin ortalamasi ile duzeltildi"

        elif det_name == "lstm_ae":
            problem = '''<div class="det-label">\u274C PROBLEM</div>
                <div class="det-value det-red">Temporal pattern anomali</div>
                <div class="det-detail">
                    <span class="dt-item">Model: <b>LSTM Autoencoder</b></span>
                    <span class="dt-item det-red">Gecmis veriden ogrenen model bu noktayi tahmin edemedi</span>
                </div>'''
            method = '''<div class="det-label">\U0001F50D NASIL YAKALANDI</div>
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

    # Kartlari katman sirasina gore olustur
    cards_html = ""
    for layer_title, layer_dets in LAYER_ORDER:
        # Bu katmanda aktif dedektor var mi?
        layer_active = [d for d in layer_dets if detector_counts.get(d, 0) > 0]
        if not layer_active:
            continue

        cards_html += f"""
        <div class="layer-header">
          <div class="layer-title">{layer_title}</div>
        </div>
        """

        for det_name in layer_active:
            count = detector_counts[det_name]
            info = detector_info.get(det_name, {"name": det_name, "icon": "\U0001F50D"})
            stats = _get_stats(det_name)

            problem_html, method_html, repair_html = _build_detail(det_name, stats)

            # Duzeltme metrikleri
            if stats and "diff_mean" in stats:
                pct_affected = stats["count"] / max(total_points, 1) * 100
                pct_correction = stats["diff_mean"] / max(signal_range, 1) * 100
                ba_html = f'''
                <div class="det-metrics">
                  <div class="dm-item">
                    <div class="dm-num">{stats["count"]}</div>
                    <div class="dm-label">anomali<br>({pct_affected:.1f}%)</div>
                  </div>
                  <div class="dm-item">
                    <div class="dm-num">\u00B1{stats["diff_median"]:.0f}</div>
                    <div class="dm-label">medyan<br>duzeltme</div>
                  </div>
                  <div class="dm-item">
                    <div class="dm-num">{pct_correction:.1f}%</div>
                    <div class="dm-label">sinyal araligina<br>gore duzeltme</div>
                  </div>
                </div>
                <div class="det-stats-row">
                  <span class="dt-item">Min: <b>\u00B1{stats["diff_min"]:.0f}</b></span>
                  <span class="dt-item">Medyan: <b>\u00B1{stats["diff_median"]:.0f}</b></span>
                  <span class="dt-item">Max: <b>\u00B1{stats["diff_max"]:.0f}</b></span>
                </div>'''
            elif stats:
                pct_affected = stats["count"] / max(total_points, 1) * 100
                ba_html = f'''
                <div class="det-metrics">
                  <div class="dm-item">
                    <div class="dm-num">{stats["count"]}</div>
                    <div class="dm-label">anomali<br>({pct_affected:.1f}%)</div>
                  </div>
                </div>'''
            else:
                ba_html = ''

            cards_html += f"""
            <div class="det-card">
              <div class="det-header">
                <span class="det-title">{info["icon"]} {info["name"]}</span>
                <span class="det-badge">{count} tespit</span>
              </div>
              <div class="det-body">
                <div class="det-block">{problem_html}</div>
                <div class="det-block det-full">{method_html}</div>
                {ba_html}
                {repair_html}
              </div>
            </div>
            """

    # Hero + Summary
    total = metrics.get("faults_detected", 0)
    corrected = metrics.get("faults_corrected", total)
    elapsed = metrics.get("processing_time", 0)
    active_count = sum(1 for v in detector_counts.values() if v > 0)

    # Kalite metrikleri (varsa)
    quality_cards = ""
    if quality_metrics:
        rmse = quality_metrics.get("rmse")
        snr = quality_metrics.get("snr")
        r2 = quality_metrics.get("r2_score")
        if rmse is not None:
            quality_cards += f'<div class="stat-card stat-quality"><div class="stat-num">{rmse:.2f}</div><div class="stat-label">RMSE</div></div>'
        if snr is not None:
            quality_cards += f'<div class="stat-card stat-quality"><div class="stat-num">{snr:.1f}</div><div class="stat-label">SNR (dB)</div></div>'
        if r2 is not None:
            quality_cards += f'<div class="stat-card stat-quality"><div class="stat-num">{r2:.3f}</div><div class="stat-label">R\u00B2</div></div>'

    quality_row = ""
    if quality_cards:
        quality_row = f"""
        <div class="hero-quality-label">Sinyal Kalitesi (Ground Truth)</div>
        <div class="hero-stats">{quality_cards}</div>
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
      {quality_row}
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
    .layer-header{margin:28px 0 12px;padding:0 4px}
    .layer-title{color:#7d8590;font-size:0.8em;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;border-bottom:1px solid #21262d;padding-bottom:8px}
    .hero-quality-label{color:#7d8590;font-size:0.75em;text-transform:uppercase;letter-spacing:1px;margin-top:16px;margin-bottom:8px}
    .stat-quality .stat-num{color:#3fb950}
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
    .det-metrics{display:flex;justify-content:center;gap:32px;padding:16px;background:#161b22;border-radius:8px}
    .dm-item{text-align:center}
    .dm-num{color:#58a6ff;font-size:1.8em;font-weight:700;line-height:1}
    .dm-label{color:#7d8590;font-size:0.7em;text-transform:uppercase;letter-spacing:0.3px;margin-top:4px;line-height:1.3}
    .det-repair{color:#7d8590;font-size:0.85em;font-style:italic;text-align:center;padding:4px}
    .det-stats-row{display:flex;gap:16px;justify-content:center;color:#8b949e;font-size:0.85em;padding:4px 0}
    .det-stats-row b{color:#c9d1d9}
    .summary-card{text-align:center;padding:24px;background:linear-gradient(135deg,#0d1117,#161b22);border:1px solid #3fb950;border-radius:12px;margin-top:8px}
    .summary-icon{color:#3fb950;font-size:2em;margin-bottom:4px}
    .summary-title{color:#3fb950;font-size:1.2em;font-weight:600;margin-bottom:4px}
    .summary-stats{color:#7d8590;font-size:0.95em}
    </style>
    """

    return f'{css}<div class="pipeline-viz">{hero_html}{cards_html}{summary_html}</div>'
