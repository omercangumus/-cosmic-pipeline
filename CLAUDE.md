# CLAUDE.md — Kozmik Veri Ayıklama Pipeline'ı

## Proje
TUA Astro Hackathon 2026 (28-29 Mart, Elazığ). Hedef: Ankara finalleri.
Radyasyonla bozulmuş uydu telemetri verilerini klasik DSP + ML hibrit yaklaşımıyla filtreleme/temizleme pipeline'ı.

## Takım
- **Ahmet**: Python / AI / Algoritma (tüm pipeline core, ML modelleri, sinyal işleme)
- **Ömer**: Infra / Dashboard / DevOps (Streamlit, sentetik veri üretici, GOES indirme)

## Tech Stack
- Python 3.11+
- Sinyal işleme: NumPy, SciPy, PyWavelets
- ML: scikit-learn (Isolation Forest), PyTorch (LSTM Autoencoder)
- Veri: Pandas, netCDF4
- Görselleştirme: Matplotlib, Plotly
- Dashboard: Streamlit
- Test: pytest

## Dizin Yapısı
```
cosmic_pipeline/
├── pipeline/
│   ├── ingestion.py          # CSV + netCDF4 veri okuma/parse
│   ├── detector_classic.py   # Z-score, IQR, sliding window
│   ├── detector_ml.py        # Isolation Forest + LSTM Autoencoder tespit
│   ├── ensemble.py           # Majority voting + confidence weighting
│   ├── filters_classic.py    # Median, Savitzky-Golay, wavelet, interpolasyon
│   ├── filters_ml.py         # LSTM reconstruction temizleme
│   ├── validator.py          # SNR, RMSE, precision, recall
│   └── orchestrator.py       # Pipeline akış yöneticisi
├── data/
│   ├── synthetic_generator.py  # Bit-flip (float32 binary), drift, gap, noise
│   └── goes_downloader.py      # NOAA GOES netCDF4 indirme + DataFrame çevirici
├── models/
│   ├── lstm_ae.py            # LSTM Autoencoder (Encoder 64→32, Decoder 32→64)
│   └── train.py              # Eğitim + model kaydetme
├── dashboard/
│   ├── app.py                # Streamlit ana uygulama
│   └── charts.py             # Plotly comparison grafikleri
├── tests/
├── requirements.txt
└── README.md
```

## Veri Formatı
Ortak DataFrame formatı: `timestamp`, `channel`, `value` kolonları.
- Sentetik: CSV, ground truth mask ile (bozulma pozisyonları kayıtlı)
- Gerçek: NOAA GOES netCDF4 (SEISS proton/elektron flux, MAG, XRS)

## 4 Bozulma Türü
1. **SEU (Single Event Upset)**: Anlık spike — float32 binary'de 1-3 bit flip
2. **TID Drift**: Yavaş monoton kalibrasyon kayması
3. **Data Dropout**: Ardışık NaN/sıfır blokları (latch-up)
4. **Noise Floor Artışı**: Genel gürültü seviyesi yükselir, SNR düşer

## Pipeline Adımları (6 adım)
1. **Ingestion**: CSV/netCDF4 → ortak DataFrame
2. **Anomali Tespit (DUAL TRACK)**: Klasik (Z-score, IQR, sliding window) vs ML (Isolation Forest, LSTM AE). Her biri ayrı anomaly_flag üretir.
3. **Filtreleme (DUAL TRACK)**: Spike→median/LSTM, Drift→detrend/AE baseline, Noise→SG+wavelet/learned denoising, Gap→interpolasyon
4. **Karşılaştırma**: Klasik vs ML yan yana (SNR, RMSE, precision/recall, işleme süresi)
5. **Validasyon**: Temizlenmiş veri kalitesi, before/after overlay
6. **Dashboard**: Streamlit — veri yükle, method seç, comparison view, metrik kartları, export

## ML Detayları
- **Isolation Forest**: Feature'lar → value, first_derivative, rolling_std_10, rolling_std_50, deviation_from_rolling_median
- **LSTM Autoencoder**: Encoder LSTM(64→32) → Decoder LSTM(32→64). Reconstruction error yüksekse anomali. Temizleme: bozuk bölgeleri reconstruct değerle değiştir. Sentetik temiz veriyle eğitim, 50-100 epoch, PyTorch.
- **Ensemble**: Z-score + IQR + IF + LSTM AE → majority voting (configurable threshold)

## Metrikler
- SNR İyileştirmesi (dB)
- Spike Precision / Drift Recall
- RMSE (temiz vs ground truth)
- İşleme süresi (10K/100K/1M nokta)
- Ensemble kazancı (tek yönteme göre artış)

## Kod Kuralları
- Türkçe comment/docstring YAZMA, İngilizce yaz
- Type hint kullan
- Her modül bağımsız çalışabilmeli (modular design)
- Fonksiyonlar tek sorumluluk: bir fonksiyon bir iş
- Pipeline adımları birbirinden bağımsız test edilebilmeli
- print() yerine logging kullan
- Config değerleri (threshold, window size vb.) fonksiyon parametresi olsun, hardcode etme

## Sık Kullanılan Komutlar
```bash
# Sanal ortam
python -m venv venv && source venv/bin/activate  # Linux
python -m venv venv && venv\Scripts\activate      # Windows

# Bağımlılıklar
pip install -r requirements.txt

# Test
pytest tests/ -v

# Dashboard
streamlit run dashboard/app.py

# Pipeline CLI (varsa)
python -m pipeline.orchestrator --input data/sample.csv --method ensemble
```

## Dikkat Edilecekler
- GOES verisi büyük olabilir, indirirken tarih aralığı dar tut (1-2 günlük event)
- LSTM eğitimi RTX 3050 4GB VRAM'de yapılacak, batch size küçük tut (32-64)
- Sentetik veri üreticide ground truth mask MUTLAKA üret — precision/recall hesabı buna bağlı
- Hackathon = 2 gün, MVP öncelikli: önce klasik pipeline çalışsın, sonra ML ekle
- Ömer'in dashboard branch'i ayrı, git conflict'e dikkat
