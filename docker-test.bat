@echo off
echo.
echo ========================================
echo   AEGIS DOCKER TEST
echo ========================================
echo.

echo [1/5] Checking Docker installation...
docker --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [HATA] Docker bulunamadi! Lutfen Docker Desktop yukleyin.
    echo https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)
echo [OK] Docker kurulu

echo.
echo [2/5] Checking Docker Compose...
docker compose version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [HATA] Docker Compose bulunamadi!
    pause
    exit /b 1
)
echo [OK] Docker Compose kurulu

echo.
echo [3/5] Checking Docker daemon...
docker ps >nul 2>&1
IF ERRORLEVEL 1 (
    echo [HATA] Docker daemon calismyor! Docker Desktop'i baslatin.
    pause
    exit /b 1
)
echo [OK] Docker daemon aktif

echo.
echo [4/5] Checking required directories...
IF NOT EXIST "data\cache" mkdir data\cache
IF NOT EXIST "data\raw" mkdir data\raw
IF NOT EXIST "models" mkdir models
echo [OK] Dizinler hazir

echo.
echo [5/5] Testing Docker build...
docker compose build >nul 2>&1
IF ERRORLEVEL 1 (
    echo [HATA] Docker build basarisiz!
    echo Detayli log icin: docker compose build
    pause
    exit /b 1
)
echo [OK] Docker build basarili

echo.
echo ========================================
echo   TUM TESTLER BASARILI!
echo ========================================
echo.
echo Sisteminiz hazir. Baslatmak icin:
echo   run.bat
echo.
pause
