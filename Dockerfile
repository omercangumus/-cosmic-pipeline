FROM python:3.11-slim

WORKDIR /app

# netCDF4 ve scipy için sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libnetcdf-dev \
    libnetcdf-c++4-dev \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Önce requirements — layer cache için
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Projeyi kopyala
COPY . .

# lstm_ae.pt image'a dahil edilmez — volume ile gelir
# .dockerignore bunu zaten engeller

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "dashboard/app.py"]
