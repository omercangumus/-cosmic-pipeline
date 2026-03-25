@echo off
echo.
echo ========================================
echo   AEGIS COSMIC PIPELINE - LAUNCHER
echo ========================================
echo.

:: Gerekli dizinleri olustur
IF NOT EXIST "data\cache" mkdir data\cache
IF NOT EXIST "data\raw" mkdir data\raw
IF NOT EXIST "models" mkdir models

:: Model var mi kontrol et
IF NOT EXIST "models\lstm_ae.pt" (
    echo [!] Model bulunamadi. Once egitim yapiliyor...
    python models/train.py
    IF ERRORLEVEL 1 (
        echo [UYARI] Model egitimi basarisiz. Dashboard yine de baslatiliyor...
        echo [UYARI] ML ozellikleri calismayabilir, classic mode kullanin.
    ) ELSE (
        echo [OK] Model hazir.
    )
) ELSE (
    echo [OK] Model zaten mevcut.
)

echo.
echo [*] Docker image build ediliyor...
docker compose build
IF ERRORLEVEL 1 (
    echo [HATA] Docker build basarisiz!
    pause
    exit /b 1
)

echo.
echo [*] Container baslatiliyor...
docker compose up -d
IF ERRORLEVEL 1 (
    echo [HATA] Container baslatma basarisiz!
    pause
    exit /b 1
)

echo.
echo [*] Container saglik kontrolu bekleniyor...
timeout /t 5 /nobreak >nul

docker compose ps

echo.
echo ========================================
echo   AEGIS Dashboard HAZIR!
echo   URL: http://localhost:8501
echo ========================================
echo.
echo Loglar icin: docker compose logs -f
echo Durdurmak icin: stop.bat veya docker compose down
echo.

start http://localhost:8501
