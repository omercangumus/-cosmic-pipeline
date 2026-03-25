.PHONY: install run test train generate lint

install:
	pip install -r requirements.txt

run:
	streamlit run dashboard/app.py

test:
	pytest tests/ -v --cov=pipeline --cov=data --cov=models --cov=utils --cov-report=term-missing

train:
	python models/train.py

generate:
	python -c "\
from data.synthetic_generator import generate_clean_signal, inject_faults; \
import pandas as pd, os; \
os.makedirs('data/raw', exist_ok=True); \
df = generate_clean_signal(10000); \
corrupted, mask = inject_faults(df); \
corrupted.to_csv('data/raw/synthetic_corrupted.csv', index=False); \
print('Generated: data/raw/synthetic_corrupted.csv')"

lint:
	python -m pylint pipeline/ data/ models/ utils/ dashboard/ || true

# ─── Docker ───────────────────────────────────────────
docker-build:
	@echo "🔨 Building Docker image..."
	docker compose build --no-cache

docker-up:
	@echo "🚀 Starting containers..."
	docker compose up -d
	@echo "✅ Dashboard: http://localhost:8501"

docker-down:
	@echo "🛑 Stopping containers..."
	docker compose down

docker-restart:
	@echo "🔄 Restarting containers..."
	docker compose restart

docker-logs:
	@echo "📋 Showing logs (Ctrl+C to exit)..."
	docker compose logs -f cosmic-pipeline

docker-shell:
	@echo "🐚 Opening shell in container..."
	docker compose exec cosmic-pipeline bash

docker-ps:
	@echo "📊 Container status:"
	docker compose ps

docker-clean:
	@echo "🧹 Cleaning up Docker resources..."
	docker compose down --rmi all --volumes --remove-orphans
	@echo "✅ Cleanup complete"

# Tek komutla build + başlat
docker-deploy: docker-build docker-up
	@echo ""
	@echo "=========================================="
	@echo "  AEGIS Dashboard READY!"
	@echo "  URL: http://localhost:8501"
	@echo "=========================================="
	@echo ""
	@echo "Commands:"
	@echo "  make docker-logs    - View logs"
	@echo "  make docker-down    - Stop containers"
	@echo "  make docker-restart - Restart containers"

# Hızlı başlatma (cache kullanarak)
docker-quick:
	@echo "⚡ Quick start (using cache)..."
	docker compose build
	docker compose up -d
	@echo "✅ Dashboard: http://localhost:8501"
