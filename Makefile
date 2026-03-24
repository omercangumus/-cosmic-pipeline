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
	docker compose build --no-cache

docker-run:
	docker compose up -d

docker-stop:
	docker compose down

docker-logs:
	docker compose logs -f cosmic-pipeline

docker-shell:
	docker compose exec cosmic-pipeline bash

docker-clean:
	docker compose down --rmi all --volumes --remove-orphans

# Tek komutla build + başlat
docker-deploy: docker-build docker-run
	@echo "✅ Dashboard: http://localhost:8501"
