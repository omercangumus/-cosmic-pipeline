# 🛰️ Kozmik Veri Ayıklama ve İşleme Hattı

AI-Powered Satellite Telemetry Anomaly Detection & Repair

**TUA Astro Hackathon 2026** — Fırat Üniversitesi, Elazığ

**Takım:** Ahmed Hüsrev Sayın · Ömer Can Gümüş · Yunus Emre Dertli

---

## 📋 Proje Özeti

Uzay ortamında uydu sensörleri kozmik radyasyona maruz kalır. Bu radyasyon telemetri verilerinde SEU (bit-flip), TID drift, veri boşlukları ve sensör donmaları oluşturur. Bu proje, bozulan uydu telemetri verilerini otomatik olarak tespit edip onaran bir veri işleme pipeline'ıdır.

Sistem dual-track mimarisiyle çalışır: klasik sinyal işleme (DSP) teknikleri ile makine öğrenimi/derin öğrenme modellerini bir arada kullanarak yüksek doğruluklu anomali tespiti ve onarım gerçekleştirir.

![Veri Önizleme](<img width="1474" height="1149" alt="image" src="https://github.com/user-attachments/assets/1b3ada30-29e9-4b6f-abb1-b9c9005aeaed" />
)

---

## 🏗️ Çözüm Mimarisi

### 4 Katmanlı Piramit

| Katman | Yöntem | Dedektörler | Karar Mekanizması |
|--------|--------|-------------|-------------------|
| K1: Deterministik | Fiziksel kurallar | Gap · Range · Delta · Flatline · Duplicate | Herhangi biri → otomatik anomali |
| K2: İstatistiksel | İstatistiksel analiz | Z-Score (eşik=2.0) · Sliding Window (pencere=50, eşik=3.0) | ≥2 dedektör anlaşmalı |
| K3: Makine Öğrenimi | Unsupervised ML | Isolation Forest | ≥2 dedektör anlaşmalı |
| K4: Derin Öğrenme | Temporal DL | LSTM Autoencoder | ≥2 dedektör anlaşmalı |

### Pipeline Akışı

```
Veri Alımı → Ön İşleme (Detrend) → Anomali Tespiti (7 Dedektör) → Ensemble Voting → Temizleme (Interpolation + Median) → Doğrulama
```

> **Tasarım kararı:** Temiz noktalar ASLA değiştirilmez — sadece anomali olarak işaretlenen noktalar düzeltilir.

---

## 🔍 Dedektör Detayları

### Katman 1 — Deterministik Kurallar
- **Gap Detector:** Ardışık timestamp farkı > adaptif eşik veya NaN değeri
- **Range Detector:** |değer − μ| > 10σ → fiziksel olarak imkansız değerler
- **Delta Detector:** Ardışık fark > μΔ + 5σΔ → ani sıçrama
- **Flatline Detector:** Aynı değer ≥ 20 ardışık noktada → sensör donması
- **Duplicate Detector:** Aynı timestamp'e sahip birden fazla kayıt

### Katman 2 — İstatistiksel Analiz
- **Z-Score Detector:** Z = |değer − μ| / σ, eşik = 2.0
- **Sliding Window:** 50 noktalık pencere medyanından sapma / pencere std, eşik = 3.0

### Katman 3 — Makine Öğrenimi
- **Isolation Forest:** Rastgele bölünmelerle izole edilen noktalar anomali

### Katman 4 — Derin Öğrenme
- **LSTM Autoencoder:** Zaman serisi paternlerini öğrenir, reconstruction error yüksek → anomali

![Dedektör Detayı](<img width="2211" height="613" alt="image" src="https://github.com/user-attachments/assets/2dc62669-a2cd-442e-9193-2ff4ce08de44" />

)

---

## 📊 Dashboard

Gradio tabanlı interaktif web arayüzü:

### Veri Sekmesi
Sentetik veri üretimi (NOAA GOES-16 profilleri) veya gerçek CSV/TSV/Excel/JSON/HDF5/Parquet yükleme. Multi-channel desteği.

### Pipeline Sekmesi
- 🎯 **Pipeline Görselleştirme:** Story-driven dedektör kartları — problem, formül, düzeltme metrikleri
- 📈 **Sonuç Grafiği:** Ground Truth vs Bozuk vs Temizlenmiş overlay
- 🔍 **Dedektör Detayı:** Her dedektörün anomali haritası
- 📋 **İşlem Raporu:** Katmanlı pipeline logu
- 📊 **Anomali Tablosu:** Tespit edilen anomalilerin listesi
- 🔍 **Doğrulama:** Onarım doğrulama sonuçları
- 🔬 **Adım Adım İzleme:** Pipeline tracer tablosu


![Pipeline Görselleştirme](<img width="302" height="1097" alt="image" src="https://github.com/user-attachments/assets/7c8438cc-4083-496f-a775-73c3affd56ce" />
)

---

## 🎮 DataCraft v5

Three.js tabanlı 3D eğitici oyun:

- Oyuncu "Veri Muhafızı" olarak anomalileri toplar
- 8 görev, 2 aşama (Temel + Gelişmiş)
- 5 tematik bölge
- Gerçek pipeline entegrasyonu
- WASD + mouse kontrol, görev briefing paneli
- 3-2-1 countdown otomatik geçiş

![DataCraft v5](<img width="2519" height="1679" alt="image" src="https://github.com/user-attachments/assets/27199b42-08b7-425d-9f1e-9b4e693c6bde" />
)

---

## 🚀 Kurulum

```bash
git clone https://github.com/omercangumus/-cosmic-pipeline.git
cd cosmic-pipeline
pip install -r requirements.txt
python dashboard/app.py
```

Tarayıcıda `http://127.0.0.1:7860` adresine gidin.

---

## 🧪 Test

```bash
python -m pytest tests/ -q
# 220 passed
```

---

## 📈 Sonuçlar

| Metrik | Değer |
|--------|-------|
| RMSE Azalma | %86 |
| F1 Artışı | 110x |
| İşleme Süresi | 0.2s (2000 nokta, classic) |
| Test | 220/220 |
| Dedektör | 7 (4 katman) |
| Veri Formatı | 6 format |

---

## 🛠️ Teknoloji Yığını

| Kategori | Teknoloji |
|----------|-----------|
| Dil | Python 3.11 |
| Web Arayüzü | Gradio |
| Görselleştirme | Plotly, HTML/CSS |
| ML | Scikit-learn (Isolation Forest) |
| DL | PyTorch (LSTM Autoencoder) |
| 3D Oyun | Three.js |
| Test | pytest |
| Veri | NumPy, Pandas, SciPy |

---

## 📁 Proje Yapısı

```
cosmic-pipeline/
├── pipeline/          # Orchestrator, dedektörler, ensemble, filtreler, validator, tracer
├── models/            # LSTM Autoencoder model ve eğitim scripti
├── data/              # Sentetik veri üretici, test örnekleri
├── dashboard/         # Gradio UI, handlers, modern görselleştirme, DataCraft oyunu
├── tests/             # 220 birim + entegrasyon testi
├── config/            # Pipeline konfigürasyonu
└── sunum_assets/      # Ekran görüntüleri
```

---

## 📄 Lisans

MIT
