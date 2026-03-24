# 🚀 COSMIC PIPELINE - AHMET HANDOFF DOCUMENT

**TUA Astro Hackathon 2026 - Satellite Telemetry Radiation-Fault Cleaning Pipeline**

---

## 📋 PROJECT OVERVIEW

Bu proje, uydu telemetri verilerindeki radyasyon kaynaklı hataları (SEU, TID, veri boşlukları, gürültü) tespit edip temizleyen bir ML pipeline'ı. Ömer altyapı ve dashboard'u tamamladı, Ahmet pipeline ve ML modellerini implement edecek.

**Repository**: `https://github.com/omercangumus/-cosmic-pipeline.git`

**Tech Stack**:
- Python 3.11
- PyTorch (LSTM Autoencoder)
- Streamlit (Dashboard)
- Docker (Deployment)
- GOES-16 Satellite Data (NOAA SWPC API)

---

## 🎯 ÖMER'İN TAMAMLADIĞI İŞLER (DAY 1 + DAY 2)

### ✅ 1. Project Infrastructure (Day 1)
**Branch**: `feature/omer-day1-infra`

#### Dosyalar:
- **`data/synthetic_generator.py`** → Sentetik veri üretimi + fault injection
  - `generate_clean_signal()` → Temiz sinyal üretir
  - `inject_faults()` → SEU, TID, gaps, noise ekler
  - Gerçekçi fault patterns (Poisson distribution, exponential drift)
  
- **`data/goes_downloader.py`** → GOES-16 gerçek veri indirme
  - NOAA SWPC API entegrasyonu
  - Cache mekanizması (JSON)
  - Fallback: API fail olursa sentetik veri döner
  
- **`dashboard/charts.py`** → Tüm görselleştirme fonksiyonları
  - `plot_comparison()` → Orijinal vs Temizlenmiş karşılaştırma
  - `plot_fault_timeline()` → Hata zaman çizelgesi
  - `plot_metrics()` → Performans metrikleri (RMSE, MAE, R²)
  - Dark theme, responsive design

- **`config/`** → YAML konfigürasyon sistemi
  - `default.yaml` → Dengeli ayarlar
  - `fast.yaml` → Hızlı test için
  - `accurate.yaml` → Yüksek doğruluk için
  - `config.py` + `parser.py` → Config yükleme

- **`tests/test_synthetic_generator.py`** → 10 test (hepsi passing)
  - Veri boyutu, fault injection, istatistiksel doğruluk testleri

- **`Makefile`** → Tüm komutlar (install, run, test, train, generate, lint, docker-*)

- **`README.md`** → Proje dokümantasyonu

- **`.gitignore`** → Model weights, cache, venv hariç


### ✅ 2. Dashboard Implementation (Day 2)
**Branch**: `feature/omer-day2-dashboard`

#### Dosyalar:
- **`dashboard/app.py`** → Streamlit dashboard (3 tab)
  - **Tab 1**: Sentetik veri üretimi + CSV upload
  - **Tab 2**: GOES gerçek veri indirme
  - **Tab 3**: Pipeline çalıştırma + karşılaştırma + export
  - Session state yönetimi (`.get()` ile safe access)
  - Pipeline entegrasyonu hazır (Ahmet'in `run_pipeline()` fonksiyonunu bekliyor)

- **`tests/test_dashboard.py`** → 6 test (hepsi passing)
  - Session state, tab rendering, data flow testleri

**Test Durumu**: 16/16 passing (10 synthetic + 6 dashboard)

---

### ✅ 3. Docker Deployment (Day 2)
**Branch**: `feature/docker-deploy`

#### Dosyalar:
- **`Dockerfile`** → Python 3.11 + netCDF4 + Streamlit
- **`docker-compose.yml`** → Volume mounts (models, data/cache, data/raw)
- **`.dockerignore`** → Model weights ve cache hariç
- **`run.bat`** → Windows için tek tıkla başlatma
- **`stop.bat`** → Container durdurma
- **`Makefile`** → Docker targets (docker-build, docker-run, docker-deploy)

**Deployment**:
```bash
# Windows
run.bat

# Linux/Mac
make docker-deploy
```

---

## 🔧 AHMET'İN YAPACAĞI İŞLER

### 📂 Pipeline Modülleri (FLAT STRUCTURE)

**ÖNEMLI**: Pipeline dosyaları `pipeline/` altında FLAT yapıda, subdirectory YOK!

```
pipeline/
├── orchestrator.py       ← MAIN ENTRY POINT (Ahmet implement edecek)
├── ingestion.py          ← Veri yükleme (Ahmet implement edecek)
├── detector_classic.py   ← Klasik anomali tespiti (Ahmet implement edecek)
├── detector_ml.py        ← ML anomali tespiti (Ahmet implement edecek)
├── ensemble.py           ← Detector birleştirme (Ahmet implement edecek)
├── filters_classic.py    ← Klasik filtreleme (Ahmet implement edecek)
├── filters_ml.py         ← ML reconstruction (Ahmet implement edecek)
├── validator.py          ← Sonuç validasyonu (Ahmet implement edecek)
└── __init__.py
```

**NOT**: `pipeline/detectors/` ve `pipeline/filters/` subdirectory'leri VAR ama bunlar KULLANILMIYOR. Ömer yanlışlıkla oluşturmuş, ignore et.


---

## 🎯 AHMET'İN TASK LİSTESİ

### **Task 11**: Implement `pipeline/orchestrator.py`
**Branch**: `feature/ahmet-day1-core` (zaten oluşturuldu)

**Beklenen Interface**:
```python
def run_pipeline(df: pd.DataFrame, config: dict) -> dict:
    """
    Ana pipeline entry point.
    
    Args:
        df: Kirli telemetri verisi (columns: timestamp, value)
        config: Konfigürasyon dict (config/default.yaml'dan gelir)
    
    Returns:
        {
            'cleaned_data': pd.DataFrame,  # Temizlenmiş veri
            'fault_mask': pd.Series,       # Boolean mask (True = fault detected)
            'metrics': {
                'faults_detected': int,
                'faults_corrected': int,
                'processing_time': float
            },
            'fault_timeline': pd.DataFrame  # columns: timestamp, fault_type, severity
        }
    """
```

**Dashboard Entegrasyonu**:
`dashboard/app.py` şu şekilde çağırıyor:
```python
from pipeline.orchestrator import run_pipeline

result = run_pipeline(corrupted_df, config)
cleaned_df = result['cleaned_data']
fault_mask = result['fault_mask']
metrics = result['metrics']
```

**Orchestrator İçinde Yapılacaklar**:
1. `ingestion.py` ile veri yükle ve validate et
2. `detector_classic.py` + `detector_ml.py` ile anomali tespit et
3. `ensemble.py` ile detector sonuçlarını birleştir
4. `filters_classic.py` + `filters_ml.py` ile hataları düzelt
5. `validator.py` ile sonucu validate et
6. Metrikleri hesapla ve döndür

---

### **Task 12**: Implement `pipeline/ingestion.py`
**Branch**: `feature/ahmet-day1-core`

**Fonksiyonlar**:
```python
def load_data(source: str | pd.DataFrame) -> pd.DataFrame:
    """CSV path veya DataFrame al, standardize et."""
    
def validate_schema(df: pd.DataFrame) -> bool:
    """Columns: timestamp, value kontrolü."""
    
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Timestamp parse, NaN handle, normalization."""
```

---

### **Task 13**: Implement `pipeline/detector_classic.py`
**Branch**: `feature/ahmet-day1-core`

**Fonksiyonlar**:
```python
def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    """Z-score ile outlier detection."""
    
def detect_outliers_iqr(df: pd.DataFrame) -> pd.Series:
    """IQR ile outlier detection."""
    
def detect_gaps(df: pd.DataFrame, max_gap_seconds: int = 60) -> pd.Series:
    """Timestamp gap detection."""
```

**Return**: Boolean mask (True = anomaly detected)


---

### **Task 14**: Implement `pipeline/detector_ml.py`
**Branch**: `feature/ahmet-day1-core`

**Fonksiyonlar**:
```python
def detect_with_lstm(df: pd.DataFrame, model_path: str, threshold: float) -> pd.Series:
    """
    LSTM Autoencoder ile anomali tespiti.
    
    Args:
        df: Telemetri verisi
        model_path: 'models/lstm_ae.pt' path
        threshold: Reconstruction error threshold
    
    Returns:
        Boolean mask (True = anomaly)
    """
```

**Kullanım**:
- `models/lstm_autoencoder.py` içindeki `LSTMAutoencoder` sınıfını kullan
- Model weights: `models/lstm_ae.pt` (Ahmet train edecek)
- Reconstruction error > threshold ise anomaly

---

### **Task 15**: Implement `pipeline/ensemble.py`
**Branch**: `feature/ahmet-day1-core`

**Fonksiyonlar**:
```python
def ensemble_vote(masks: list[pd.Series], strategy: str = 'majority') -> pd.Series:
    """
    Birden fazla detector sonucunu birleştir.
    
    Args:
        masks: [classic_mask, ml_mask, ...]
        strategy: 'majority', 'any', 'all'
    
    Returns:
        Final boolean mask
    """
```

**Stratejiler**:
- `majority`: Çoğunluk oyu (2/3 detector True derse True)
- `any`: Herhangi biri True derse True (hassas)
- `all`: Hepsi True derse True (muhafazakar)

---

### **Task 16**: Implement `pipeline/filters_classic.py`
**Branch**: `feature/ahmet-day1-core`

**Fonksiyonlar**:
```python
def median_filter(df: pd.DataFrame, mask: pd.Series, window: int = 5) -> pd.DataFrame:
    """Median filter ile anomali düzeltme."""
    
def interpolate_gaps(df: pd.DataFrame, mask: pd.Series, method: str = 'linear') -> pd.DataFrame:
    """Gap interpolation."""
    
def savgol_filter(df: pd.DataFrame, mask: pd.Series, window: int = 11, polyorder: int = 3) -> pd.DataFrame:
    """Savitzky-Golay filter."""
```

**Return**: Düzeltilmiş DataFrame

---

### **Task 17**: Implement `pipeline/filters_ml.py`
**Branch**: `feature/ahmet-day1-core`

**Fonksiyonlar**:
```python
def reconstruct_with_lstm(df: pd.DataFrame, mask: pd.Series, model_path: str) -> pd.DataFrame:
    """
    LSTM Autoencoder ile anomali reconstruction.
    
    Args:
        df: Kirli veri
        mask: Anomaly mask
        model_path: 'models/lstm_ae.pt'
    
    Returns:
        Düzeltilmiş DataFrame (mask=True olan yerleri reconstruct et)
    """
```


---

### **Task 18**: Implement `pipeline/validator.py`
**Branch**: `feature/ahmet-day1-core`

**Fonksiyonlar**:
```python
def validate_output(df: pd.DataFrame) -> dict:
    """
    Temizlenmiş verinin kalitesini kontrol et.
    
    Returns:
        {
            'is_valid': bool,
            'issues': list[str],
            'quality_score': float  # 0-1 arası
        }
    """
    
def calculate_metrics(original: pd.DataFrame, cleaned: pd.DataFrame, ground_truth: pd.DataFrame = None) -> dict:
    """
    RMSE, MAE, R² hesapla.
    
    Returns:
        {
            'rmse': float,
            'mae': float,
            'r2_score': float,
            'snr': float  # Signal-to-Noise Ratio
        }
    """
```

---

### **Task 19**: Train LSTM Autoencoder
**Branch**: `feature/ahmet-day2-ml`

**Dosya**: `models/train.py`

**Yapılacaklar**:
1. `data/synthetic_generator.py` ile training data üret (100k samples)
2. `models/lstm_autoencoder.py` içindeki `LSTMAutoencoder` sınıfını kullan
3. Model train et (PyTorch)
4. `models/lstm_ae.pt` olarak kaydet
5. Training logs ve metrics kaydet

**Model Architecture** (zaten `models/lstm_autoencoder.py` içinde var):
```python
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        # Encoder: LSTM
        # Decoder: LSTM
        # Output: Reconstructed signal
```

**Training Config**:
- Epochs: 50-100
- Batch size: 64
- Learning rate: 0.001
- Loss: MSE
- Optimizer: Adam

**Komut**:
```bash
make train
# veya
python models/train.py
```

---

### **Task 20**: Write Tests
**Branch**: `feature/ahmet-day2-ml`

**Dosyalar**:
- `tests/unit/test_orchestrator.py`
- `tests/unit/test_detectors.py`
- `tests/unit/test_filters.py`
- `tests/integration/test_pipeline_e2e.py`

**Test Coverage**:
- Unit tests: Her modül için ayrı
- Integration test: End-to-end pipeline testi
- Sentetik veri ile doğruluk testi

**Komut**:
```bash
make test
# veya
pytest tests/ -v --cov
```

**Hedef**: 80%+ code coverage


---

## 🌳 GIT BRANCH STRATEGY

### Mevcut Branchler:
```
main                          ← Stable release (v1.0-hackathon-final)
├── develop                   ← Integration branch
│   ├── feature/omer-day1-infra          ✅ MERGED
│   ├── feature/omer-day2-dashboard      ✅ MERGED
│   ├── feature/docker-deploy            ✅ MERGED
│   ├── feature/ahmet-day1-core          🔄 AHMET ÇALIŞACAK
│   └── feature/ahmet-day2-ml            🔄 AHMET ÇALIŞACAK
```

### Ahmet'in Workflow'u:

#### 1. Day 1 Core Pipeline (Task 11-18)
```bash
# Clone repo
git clone https://github.com/omercangumus/-cosmic-pipeline.git
cd -cosmic-pipeline

# Ahmet'in branch'ine geç
git checkout feature/ahmet-day1-core
git pull origin feature/ahmet-day1-core

# Implement pipeline modülleri
# ... kod yaz ...

# Commit
git add pipeline/orchestrator.py
git commit -m "feat: implement orchestrator with full pipeline flow"

git add pipeline/ingestion.py
git commit -m "feat: implement data ingestion and validation"

git add pipeline/detector_classic.py
git commit -m "feat: implement classic anomaly detectors (z-score, IQR, gaps)"

git add pipeline/detector_ml.py
git commit -m "feat: implement LSTM-based anomaly detection"

git add pipeline/ensemble.py
git commit -m "feat: implement ensemble voting for detectors"

git add pipeline/filters_classic.py
git commit -m "feat: implement classic filters (median, interpolation, savgol)"

git add pipeline/filters_ml.py
git commit -m "feat: implement LSTM reconstruction filter"

git add pipeline/validator.py
git commit -m "feat: implement output validation and metrics"

# Push
git push origin feature/ahmet-day1-core
```

#### 2. Day 2 ML Training (Task 19-20)
```bash
# ML branch'ine geç
git checkout feature/ahmet-day2-ml
git pull origin feature/ahmet-day2-ml

# Model train et
make train
# veya
python models/train.py

# Test yaz
# ... test kod yaz ...

# Commit
git add models/train.py
git commit -m "feat: train LSTM autoencoder with 100k synthetic samples"

git add tests/unit/test_orchestrator.py
git commit -m "test: add orchestrator unit tests"

git add tests/unit/test_detectors.py
git commit -m "test: add detector unit tests"

git add tests/unit/test_filters.py
git commit -m "test: add filter unit tests"

git add tests/integration/test_pipeline_e2e.py
git commit -m "test: add end-to-end pipeline integration test"

# Push
git push origin feature/ahmet-day2-ml
```

#### 3. Merge to Develop
```bash
# Day 1 core merge
git checkout develop
git merge --no-ff feature/ahmet-day1-core -m "merge: Ahmet Day 1 core pipeline complete"
git push origin develop

# Day 2 ML merge
git merge --no-ff feature/ahmet-day2-ml -m "merge: Ahmet Day 2 ML training and tests complete"
git push origin develop

# Tag
git tag v2.0-ahmet-complete
git push origin v2.0-ahmet-complete
```


---

## 📦 DEPENDENCIES (requirements.txt)

Ömer zaten ekledi, Ahmet ekstra bir şey eklemeye gerek yok:

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
torch>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
netCDF4>=1.6.0
requests>=2.31.0
pyyaml>=6.0.0
pytest>=7.4.0
pytest-cov>=4.1.0
```

---

## 🔍 ÖMER'İN OLUŞTURDUĞU YARDIMCI DOSYALAR

### 1. `IMPORT_REFERENCE.md`
Pipeline modüllerinin nasıl import edileceğini gösterir:
```python
from pipeline.orchestrator import run_pipeline
from pipeline.ingestion import load_data, validate_schema
from pipeline.detector_classic import detect_outliers_zscore
from pipeline.detector_ml import detect_with_lstm
from pipeline.ensemble import ensemble_vote
from pipeline.filters_classic import median_filter
from pipeline.filters_ml import reconstruct_with_lstm
from pipeline.validator import validate_output
```

### 2. `PROJECT_STATUS.md`
Proje durumu ve task breakdown.

### 3. `FINAL_SUMMARY.md`
Ömer'in tamamladığı işlerin özeti.

### 4. `DAY2_COMPLETE.md`
Day 2 dashboard implementation detayları.

### 5. `SETUP.md`
Proje kurulum adımları.

---

## 🧪 TEST DURUMU

### Ömer'in Testleri (16/16 PASSING):
- `tests/test_synthetic_generator.py` → 10/10 ✅
- `tests/test_dashboard.py` → 6/6 ✅

### Ahmet'in Yazacağı Testler:
- `tests/unit/test_orchestrator.py`
- `tests/unit/test_detectors.py`
- `tests/unit/test_filters.py`
- `tests/integration/test_pipeline_e2e.py`

**Hedef**: 30+ test, 80%+ coverage

---

## 🐳 DOCKER DEPLOYMENT

### Ömer'in Hazırladığı Docker Setup:

**Dockerfile**:
- Base: Python 3.11-slim
- Dependencies: netCDF4, scipy, torch, streamlit
- Entry point: `streamlit run dashboard/app.py`
- Healthcheck: Streamlit health endpoint

**docker-compose.yml**:
- Port: 8501
- Volumes:
  - `./models:/app/models` → Model weights (commit edilmez)
  - `./data/cache:/app/cache` → GOES cache
  - `./data/raw:/app/data/raw` → Raw data

**Kullanım**:
```bash
# Windows (Ahmet için)
run.bat

# Linux/Mac
make docker-deploy

# Manuel
docker compose build
docker compose up -d

# Logs
docker compose logs -f cosmic-pipeline

# Stop
docker compose down
```

**NOT**: `models/lstm_ae.pt` Docker image'a dahil DEĞİL, volume mount ile gelir. Ahmet train ettikten sonra `models/` klasörüne kaydetmeli.


---

## 📊 DATA FLOW

### 1. Sentetik Veri (Test için)
```python
from data.synthetic_generator import generate_clean_signal, inject_faults

# Temiz sinyal üret
clean_df = generate_clean_signal(n_samples=10000)

# Fault inject et
corrupted_df, fault_mask = inject_faults(clean_df)

# Fault types:
# - SEU (Single Event Upset): Bit-flip, spike
# - TID (Total Ionizing Dose): Exponential drift
# - Gaps: Missing data
# - Noise: Gaussian noise
```

### 2. Gerçek Veri (GOES-16)
```python
from data.goes_downloader import download_goes_data

# GOES-16 XRS data indir
df = download_goes_data(
    start_date='2024-01-01',
    end_date='2024-01-02',
    satellite='goes16',
    instrument='xrs'
)

# Cache: data/cache/*.json
# Fallback: API fail olursa sentetik veri döner
```

### 3. Pipeline Flow
```python
from pipeline.orchestrator import run_pipeline
from config.parser import load_config

# Config yükle
config = load_config('config/default.yaml')

# Pipeline çalıştır
result = run_pipeline(corrupted_df, config)

# Result structure:
{
    'cleaned_data': pd.DataFrame,      # Temizlenmiş veri
    'fault_mask': pd.Series,           # Boolean mask
    'metrics': {
        'faults_detected': 150,
        'faults_corrected': 145,
        'processing_time': 2.3
    },
    'fault_timeline': pd.DataFrame     # timestamp, fault_type, severity
}
```

### 4. Dashboard Visualization
```python
from dashboard.charts import plot_comparison, plot_metrics, plot_fault_timeline

# Karşılaştırma grafiği
fig1 = plot_comparison(original_df, cleaned_df, fault_mask)

# Metrik grafiği
fig2 = plot_metrics(metrics)

# Fault timeline
fig3 = plot_fault_timeline(fault_timeline)

# Streamlit'te göster
st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)
```

---

## 🎨 CONFIG SYSTEM

Ömer 3 farklı config hazırladı:

### 1. `config/default.yaml` (Dengeli)
```yaml
pipeline:
  detectors:
    classic:
      zscore_threshold: 3.0
      iqr_multiplier: 1.5
      gap_threshold_seconds: 60
    ml:
      model_path: "models/lstm_ae.pt"
      reconstruction_threshold: 0.1
  ensemble:
    strategy: "majority"  # majority, any, all
  filters:
    classic:
      median_window: 5
      interpolation_method: "linear"
      savgol_window: 11
      savgol_polyorder: 3
    ml:
      use_reconstruction: true
```

### 2. `config/fast.yaml` (Hızlı Test)
```yaml
pipeline:
  detectors:
    classic:
      zscore_threshold: 2.5  # Daha hassas
    ml:
      reconstruction_threshold: 0.15  # Daha toleranslı
  ensemble:
    strategy: "any"  # Herhangi biri True derse True
```

### 3. `config/accurate.yaml` (Yüksek Doğruluk)
```yaml
pipeline:
  detectors:
    classic:
      zscore_threshold: 3.5  # Daha muhafazakar
    ml:
      reconstruction_threshold: 0.05  # Daha hassas
  ensemble:
    strategy: "all"  # Hepsi True derse True
```

**Kullanım**:
```python
from config.parser import load_config

config = load_config('config/accurate.yaml')
result = run_pipeline(df, config)
```


---

## 🚨 ÖNEMLI NOTLAR (AHMET İÇİN)

### 1. Pipeline Dosya Yapısı
**FLAT STRUCTURE KULLAN**:
```
✅ DOĞRU:
pipeline/orchestrator.py
pipeline/detector_classic.py
pipeline/detector_ml.py

❌ YANLIŞ:
pipeline/detectors/dsp_detector.py
pipeline/detectors/lstm_detector.py
```

Ömer yanlışlıkla `pipeline/detectors/` ve `pipeline/filters/` subdirectory'leri oluşturmuş ama bunlar KULLANILMIYOR. Ignore et.

### 2. Model Dosya İsimleri
- Model class: `LSTMAutoencoder` (zaten `models/lstm_autoencoder.py` içinde var)
- Model weights: `models/lstm_ae.pt` (Ahmet train edip kaydedecek)
- Training script: `models/train.py` (Ahmet implement edecek)

### 3. Dashboard Entegrasyonu
Dashboard `pipeline.orchestrator.run_pipeline()` fonksiyonunu bekliyor:
```python
# dashboard/app.py içinde
from pipeline.orchestrator import run_pipeline

result = run_pipeline(corrupted_df, config)
```

**Return format MUTLAKA şu şekilde olmalı**:
```python
{
    'cleaned_data': pd.DataFrame,
    'fault_mask': pd.Series,
    'metrics': dict,
    'fault_timeline': pd.DataFrame
}
```

### 4. Session State Safety
Dashboard `.get()` ile safe access yapıyor, KeyError atmaz:
```python
# dashboard/app.py içinde
cleaned_df = st.session_state.get('cleaned_data')
if cleaned_df is not None:
    # ... işlem yap
```

### 5. Model Weights Git'e Commit Edilmez
`.gitignore` içinde:
```
models/*.pt
models/*.pth
```

Model weights Docker volume mount ile gelir:
```yaml
volumes:
  - ./models:/app/models
```

### 6. Test Coverage
Ömer 16 test yazdı (hepsi passing). Ahmet en az 15-20 test daha yazmalı:
- Unit tests: Her modül için
- Integration test: End-to-end pipeline
- Hedef: 80%+ coverage

### 7. Commit Convention
```bash
feat: yeni özellik
fix: bug fix
test: test ekleme
chore: dependency, config değişikliği
docs: dokümantasyon
```

### 8. Branch Merge Sırası
```
feature/ahmet-day1-core → develop
feature/ahmet-day2-ml → develop
develop → main (final release)
```


---

## 🎯 AHMET'İN ÖNCELIK SIRASI

### Day 1 (Core Pipeline) - 6-8 saat
1. **`pipeline/ingestion.py`** (1 saat)
   - Veri yükleme, validation, preprocessing
   - En basit modül, buradan başla

2. **`pipeline/detector_classic.py`** (1.5 saat)
   - Z-score, IQR, gap detection
   - Scipy/numpy kullan, kolay

3. **`pipeline/detector_ml.py`** (1.5 saat)
   - LSTM Autoencoder ile anomaly detection
   - `models/lstm_autoencoder.py` sınıfını kullan
   - Model henüz train edilmemiş, dummy model ile test et

4. **`pipeline/ensemble.py`** (1 saat)
   - Detector sonuçlarını birleştir
   - Basit voting logic

5. **`pipeline/filters_classic.py`** (1.5 saat)
   - Median, interpolation, Savitzky-Golay
   - Scipy kullan

6. **`pipeline/filters_ml.py`** (1 saat)
   - LSTM reconstruction
   - `detector_ml.py` ile benzer logic

7. **`pipeline/validator.py`** (1 saat)
   - Output validation, metrics
   - RMSE, MAE, R² hesapla

8. **`pipeline/orchestrator.py`** (1.5 saat)
   - Tüm modülleri birleştir
   - Main entry point
   - Dashboard entegrasyonu test et

### Day 2 (ML Training + Tests) - 4-6 saat
1. **`models/train.py`** (2-3 saat)
   - LSTM Autoencoder train et
   - 100k sentetik sample kullan
   - `models/lstm_ae.pt` kaydet

2. **Unit Tests** (2 saat)
   - `tests/unit/test_orchestrator.py`
   - `tests/unit/test_detectors.py`
   - `tests/unit/test_filters.py`

3. **Integration Test** (1 saat)
   - `tests/integration/test_pipeline_e2e.py`
   - End-to-end pipeline testi

4. **Dashboard Test** (30 dakika)
   - Dashboard'u çalıştır
   - Tüm tabları test et
   - Export fonksiyonlarını test et

---

## 🔗 USEFUL LINKS

- **Repository**: https://github.com/omercangumus/-cosmic-pipeline.git
- **NOAA SWPC API**: https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json
- **GOES-16 Docs**: https://www.ncei.noaa.gov/products/goes-r-series-advanced-baseline-imager
- **PyTorch LSTM**: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- **Streamlit Docs**: https://docs.streamlit.io/

---

## 📞 CONTACT

**Ömer**: Altyapı, dashboard, Docker (TAMAMLANDI ✅)
**Ahmet**: Pipeline, ML modelleri (DEVAM EDİYOR 🔄)

**Sorular için**:
- `IMPORT_REFERENCE.md` → Import örnekleri
- `PROJECT_STATUS.md` → Proje durumu
- `FINAL_SUMMARY.md` → Ömer'in özeti
- `DAY2_COMPLETE.md` → Dashboard detayları

---

## ✅ CHECKLIST (AHMET İÇİN)

### Day 1 Core Pipeline
- [ ] `pipeline/ingestion.py` implement edildi
- [ ] `pipeline/detector_classic.py` implement edildi
- [ ] `pipeline/detector_ml.py` implement edildi
- [ ] `pipeline/ensemble.py` implement edildi
- [ ] `pipeline/filters_classic.py` implement edildi
- [ ] `pipeline/filters_ml.py` implement edildi
- [ ] `pipeline/validator.py` implement edildi
- [ ] `pipeline/orchestrator.py` implement edildi
- [ ] Dashboard entegrasyonu test edildi
- [ ] `feature/ahmet-day1-core` branch'i push edildi

### Day 2 ML Training + Tests
- [ ] `models/train.py` implement edildi
- [ ] LSTM Autoencoder train edildi
- [ ] `models/lstm_ae.pt` kaydedildi
- [ ] Unit testler yazıldı (15+ test)
- [ ] Integration test yazıldı
- [ ] Tüm testler passing (30+ test)
- [ ] Code coverage 80%+
- [ ] `feature/ahmet-day2-ml` branch'i push edildi

### Final Merge
- [ ] `feature/ahmet-day1-core` → `develop` merge edildi
- [ ] `feature/ahmet-day2-ml` → `develop` merge edildi
- [ ] Tag oluşturuldu: `v2.0-ahmet-complete`
- [ ] Docker deployment test edildi
- [ ] Demo hazır 🚀

---

## 🎉 FINAL NOTES

Ömer altyapıyı, dashboard'u ve Docker deployment'ı tamamladı. Ahmet pipeline ve ML modellerini implement edecek. İki kişinin işi birbirinden bağımsız, conflict olmaz.

**Ahmet'in yapması gerekenler**:
1. Pipeline modüllerini implement et (Task 11-18)
2. LSTM Autoencoder'ı train et (Task 19)
3. Testleri yaz (Task 20)
4. Branch'leri merge et
5. Demo'ya hazır ol 🚀

**Ömer'in hazırladıkları**:
- ✅ Sentetik veri üretimi
- ✅ GOES gerçek veri indirme
- ✅ Dashboard (3 tab)
- ✅ Görselleştirme fonksiyonları
- ✅ Config sistemi
- ✅ Docker deployment
- ✅ Git branch yapısı
- ✅ 16 test (hepsi passing)

**Başarılar Ahmet! 🚀**

