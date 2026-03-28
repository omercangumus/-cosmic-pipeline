@echo off
echo.
echo ========================================
echo   COSMIC PIPELINE - LAUNCHER
echo ========================================
echo.

:: 1. Model var mi kontrol et
IF NOT EXIST "models\lstm_ae.pt" (
    echo [!] Model bulunamadi. Once egitim yapiliyor...
    python models/train.py
    echo [OK] Model hazir.
) ELSE (
    echo [OK] Model zaten mevcut.
)

echo.
echo [*] Docker image build ediliyor...
docker compose build

echo.
echo [*] Dashboard baslatiliyor...
docker compose up -d

echo.
echo ========================================
echo   Dashboard aciliyor: http://localhost:7860
echo ========================================
start http://localhost:7860
