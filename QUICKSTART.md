# 🚀 AEGIS Cosmic Pipeline - Quickstart

## Tek Komut ile Başlat

### Windows
```bash
run.bat
```

### Linux/Mac
```bash
make docker-deploy
```

Dashboard otomatik açılır: **http://localhost:8501**

---

## İlk Kullanım (5 Dakika)

### 1️⃣ Sistemi Başlat
```bash
# Windows
run.bat

# Linux/Mac
make docker-deploy
```

Beklenen çıktı:
```
✅ Model hazir
🔨 Building Docker image...
🚀 Starting containers...
✅ Dashboard: http://localhost:8501
```

### 2️⃣ Dashboard'u Aç
Browser otomatik açılır veya manuel: http://localhost:8501

### 3️⃣ Veri Oluştur
1. Sol sidebar → "Sentetik Veri" seçili
2. "🔄 Veri Oluştur" butonuna tıkla
3. 5000 sample veri oluşturulur (SEU, TID, gaps, noise)

### 4️⃣ Pipeline'ı Çalıştır
1. Tab 2: "🔧 Pipeline & Temizleme"
2. "▶️ Pipeline'ı Çalıştır" butonuna tıkla
3. Orijinal vs Temizlenmiş karşılaştırma görüntülenir

### 5️⃣ Sonuçları İncele
1. Tab 3: "📈 Sonuçlar & Metrikler"
2. Metrics: Tespit edilen hata, düzeltilen hata, işlem süresi
3. Anomali zaman çizelgesi
4. "📥 Temizlenmiş Veriyi İndir" ile CSV export

---

## Durdurma

### Windows
```bash
stop.bat
```

### Linux/Mac
```bash
make docker-down
```

---

## Sorun Giderme

### Docker yüklü değil
```bash
# Windows/Mac: Docker Desktop indir
https://www.docker.com/products/docker-desktop

# Linux
sudo apt-get install docker.io docker-compose
```

### Port 8501 kullanımda
```bash
# docker-compose.yml dosyasını düzenle
ports:
  - "8502:8501"  # 8502 kullan

# Sonra tekrar başlat
docker compose down
docker compose up -d
```

### Model bulunamadı
```bash
# Model eğit
python models/train.py

# Container'ı yeniden başlat
docker compose restart
```

### Container başlamıyor
```bash
# Logları kontrol et
docker compose logs cosmic-pipeline

# Temiz build
docker compose down
docker compose build --no-cache
docker compose up -d
```

---

## Gelişmiş Kullanım

### GOES CSV Yükleme
1. Sidebar → "GOES CSV Yükle" seç
2. CSV dosyası yükle (columns: timestamp, value)
3. Pipeline'ı çalıştır

### Pipeline Method Seçimi
- **classic**: DSP tabanlı (hızlı, model gerektirmez)
- **ml**: LSTM + IForest (yavaş, model gerektirir)
- **both**: Ensemble (en doğru, en yavaş)

### Gelişmiş Ayarlar
Sidebar → "🔧 Gelişmiş Ayarlar"
- Z-Score Eşiği: 2.0-5.0 (default: 3.5)
- IQR Çarpanı: 1.0-3.0 (default: 1.5)
- Pencere Boyutu: 20-100 (default: 50)

---

## Komutlar

### Docker Management
```bash
make docker-up        # Başlat
make docker-down      # Durdur
make docker-restart   # Yeniden başlat
make docker-logs      # Logları göster
make docker-ps        # Status
make docker-shell     # Container'a gir
make docker-clean     # Temizle
```

### Local Development
```bash
make install          # Dependencies
make run              # Local dashboard
make test             # Tests
make train            # Train model
```

---

## Dokümantasyon

- **README.md** - Genel bakış
- **DOCKER_GUIDE.md** - Docker detayları (400+ satır)
- **DEPLOYMENT_SUMMARY.md** - Deployment özeti
- **AHMET_HANDOFF.md** - Proje handoff (2312 satır)

---

## Destek

**GitHub**: https://github.com/omercangumus/-cosmic-pipeline

**Issues**: https://github.com/omercangumus/-cosmic-pipeline/issues

---

**TUA Astro Hackathon 2026** | Ömer & Ahmet

**Status**: ✅ PRODUCTION READY
