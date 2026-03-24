# 🌌 Cosmic Pipeline

> TUA Astro Hackathon 2026 | Radyasyonla bozulmuş uydu telemetrisini temizleyen hibrit DSP + ML pipeline

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Train LSTM AE model (run once before hackathon)
make train

# Launch dashboard
make run
```

## Architecture

```
Ingestion (CSV / GOES JSON)
       ↓
  ┌────┴────┐
Classic    ML
  DSP    (IF + LSTM AE)
  ↓          ↓
  └────┬────┘
  Ensemble Vote
       ↓
  Streamlit Dashboard
```

## Data Sources

| Source | Type | Details |
|--------|------|---------|
| Synthetic Generator | Controlled | SEU bit-flip, TID drift, data gaps, noise floor rise |
| NOAA SWPC GOES-16 | Real-time | Proton flux JSON API — live solar particle data |

## Fault Types

| Fault | Physical Cause | Signature |
|-------|----------------|-----------|
| SEU | High-energy particle bit-flip | Instant spike to physically impossible value |
| TID Drift | Cumulative ionizing dose | Monotonic calibration bias |
| Data Gap | Latch-up / transmission error | Consecutive NaN blocks |
| Noise Floor | Radiation background rise | Increasing signal variance over time |

## Team

| Role | Person |
|------|--------|
| 🟠 Python / AI / Algo | Ahmet Hüsrev Sayın |
| 🔵 Infra / Dashboard  | Ömer Can Gümüş |

## Branch Strategy

```
main ← develop ← feature/omer-day1-infra
              ← feature/omer-day2-dashboard
              ← feature/ahmet-day1-core
              ← feature/ahmet-day2-ml
```

Merge flow: `feature/*` → `develop` → `main` (checkpoint only)

Tags: `v0.1-day1-checkpoint` · `v1.0-hackathon-final`

## Commands

```bash
make install   # install dependencies
make run       # launch Streamlit dashboard
make test      # run pytest with coverage
make train     # train LSTM AE model
make generate  # generate synthetic test data
make lint      # run linting checks
```

## Metrics

> ⚠️ Metrics are computed live from the pipeline.
> Run `make run` → Tab 1 → Generate Signal → Run Pipeline → Comparison tab.

## Project Structure

```
cosmic-pipeline/
├── .github/
│   └── PULL_REQUEST_TEMPLATE.md
├── pipeline/
│   ├── detectors/
│   │   ├── dsp_detector.py
│   │   └── lstm_detector.py
│   ├── filters/
│   │   ├── classic_filter.py
│   │   └── ml_reconstructor.py
│   ├── ensemble_voter.py
│   └── pipeline.py
├── data/
│   ├── synthetic_generator.py
│   ├── goes_downloader.py
│   └── raw/
├── models/
│   ├── lstm_autoencoder.py
│   └── train.py
├── dashboard/
│   ├── app.py
│   └── charts.py
├── config/
│   ├── config.py
│   ├── parser.py
│   ├── default.yaml
│   ├── fast.yaml
│   └── accurate.yaml
├── utils/
│   ├── validation.py
│   ├── metrics.py
│   └── logging.py
├── tests/
│   └── ...
├── notebooks/
│   └── goes_exploration.ipynb
├── .gitignore
├── requirements.txt
├── Makefile
└── README.md
```

## License

MIT License - TUA Astro Hackathon 2026
