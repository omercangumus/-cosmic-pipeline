# CLAUDE.md — Kozmik Veri Ayiklama Pipeline'i

## Proje
TUA Astro Hackathon 2026 (28-29 Mart, Elazig).
Radyasyonla bozulmus uydu telemetri verilerini klasik DSP + ML hibrit yaklasimla filtreleme/temizleme pipeline'i.

## Takim
- **Ahmet**: Python / AI / Algoritma
- **Omer**: Infra / Dashboard / DevOps

## Tech Stack
- Python 3.11+, NumPy, SciPy, PyWavelets, Pandas
- ML: scikit-learn (Isolation Forest), PyTorch (LSTM Autoencoder)
- Dashboard: Gradio
- Gorsellestirme: Plotly
- Test: pytest
- Deploy: Docker

## Dizin Yapisi
```
cosmic_pipeline/
├── pipeline/
│   ├── orchestrator.py       # Pipeline akis yoneticisi
│   ├── ingestion.py          # CSV/TSV/Excel/JSON veri okuma + sema kontrolu
│   ├── detector_classic.py   # 7 detektor: zscore, sliding_window, gaps, range, delta, flatline, duplicates
│   ├── detector_ml.py        # Isolation Forest + LSTM Autoencoder
│   ├── ensemble.py           # hybrid_majority_vote (hard rules + soft agreement)
│   ├── filters_classic.py    # Mercek modeli: interpolation -> detrend -> median (sadece anomali noktalarina)
│   ├── validator.py          # Quality check, repair eligibility/confidence/verification, sampling rate
│   └── tracer.py             # Adim adim izleme tablosu (NaN/mean/std/min/max her adimda)
├── data/
│   └── synthetic_generator.py  # 20 sutunlu gercekci uydu termal telemetrisi
├── models/
│   ├── lstm_autoencoder.py   # LSTM AE (Encoder 64->32, Decoder 32->64)
│   ├── train.py              # Egitim scripti
│   └── lstm_ae.pt            # Egitilmis model (~480KB, .gitignore'da)
├── dashboard/
│   └── app.py                # Gradio dashboard (veri uretimi + pipeline calistirma)
├── config/
│   ├── config.py             # Config dataclass'lari
│   ├── parser.py             # YAML parser
│   └── default.yaml          # Varsayilan ayarlar
├── utils/
│   └── csv_parser.py         # CSV/TSV/Excel/JSON destegi
├── tests/
│   ├── unit/                 # Birim testleri
│   └── integration/          # Entegrasyon testleri
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

## Pipeline Adimlari (7 adim)
1. **Ingestion**: CSV/TSV/Excel/JSON -> DataFrame, sema kontrolu, on isleme
2. **Detrend**: Lineer trend kaldirma (detection kopyasi uzerinde, orijinal degismez)
3. **Detection**: 7 klasik + 2 ML detektor -> her biri ayri mask uretir
4. **Ensemble**: hybrid_majority_vote — hard rules (gap/range/delta/flatline/duplicates -> otomatik), soft (zscore/sliding/IF/LSTM -> min 2 anlasma)
5. **Repair Decision**: assess_repair_eligibility() -> "repair" / "flag_only" / "preserve" karari. flag_only ve preserve noktalari filtrelenmez.
6. **Filtering**: Mercek modeli — interpolation -> detrend -> median. SADECE fault_mask==True VE repair_decision=="repair" olan noktalara uygulanir. Temiz noktalar ASLA degismez.
7. **Validation + Tracer**: Quality check, repair confidence, repair verification, adim adim izleme tablosu

## Ensemble Detaylari
- **Hard detectors** (any-True -> anomali): gaps, range, delta, flatline, duplicates
- **Soft detectors** (>=2 agreement): zscore, sliding_window, isolation_forest, lstm_ae
- **Pyramid logging**: L1(hard) -> L2(statistical) -> L3(ML) -> L4(temporal)

## ML Detaylari
- **Isolation Forest**: Features -> value, first_derivative, rolling_std_10, rolling_std_50, deviation_from_rolling_median
- **LSTM Autoencoder**: Encoder LSTM(64->32) -> Decoder LSTM(32->64). Reconstruction error threshold ile anomali.

## Sik Kullanilan Komutlar
```bash
pip install -r requirements.txt
pytest tests/ -v
python dashboard/app.py              # Gradio dashboard (localhost:7860)
python models/train.py               # LSTM AE egitimi
docker compose up -d                 # Docker ile calistirma
```

## Kod Kurallari
- Ingilizce comment/docstring
- Type hint kullan
- Moduler tasarim — her modul bagimsiz test edilebilir
- print() yerine logging
- Config degerleri fonksiyon parametresi olsun, hardcode etme

## Bilinen Kisitlar
- **Tek kanal**: Pipeline tek `value` kolonu isliyor. Multi-column CSV'de sadece ilk numeric kolon alinir.
- Multi-channel support roadmap'te — her kolon icin ayri pipeline calistirma + dashboard multi-channel plot gerekli.
