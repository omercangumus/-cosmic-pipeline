@echo off
echo.
echo ========================================
echo   AEGIS COSMIC PIPELINE - SHUTDOWN
echo ========================================
echo.

echo [*] Container durduruluyor...
docker compose down

IF ERRORLEVEL 1 (
    echo [HATA] Container durdurma basarisiz!
    pause
    exit /b 1
)

echo.
echo [OK] Container basariyla durduruldu.
echo.

:: Temizlik isteniyor mu?
set /p cleanup="Tum Docker kaynaklarini temizle? (y/N): "
IF /I "%cleanup%"=="y" (
    echo [*] Docker kaynaklari temizleniyor...
    docker compose down --rmi all --volumes --remove-orphans
    echo [OK] Temizlik tamamlandi.
) ELSE (
    echo [*] Kaynaklariniz korundu. Tekrar baslatmak icin: run.bat
)

echo.
