# Cosmic Pipeline

> TUA Astro Hackathon 2026 | Radyasyonla bozulmus uydu telemetrisini temizleyen hibrit DSP + ML pipeline

## Quickstart

```bash
pip install -r requirements.txt
python dashboard/app.py          # -> http://localhost:7860
```

## Architecture

```
CSV / Excel / JSON / Synthetic Generator
       |
   Ingestion (parse + schema validation)
       |
   Detrend (linear trend removal -- detection copy only)
       |
  +----+----+
Classic      ML
(7 det.)   (IF + LSTM AE)
  |           |
  +----+-----+
  Hybrid Ensemble
  (hard rules: ANY  +  soft: MAJORITY >=2)
       |
  Repair Decision (repair / flag_only / preserve)
       |
  Filtering -- ONLY on repairable anomalies
  (interpolation -> detrend -> median)
       |
  Validation + Repair Verification
       |
  Tracer (step-by-step report)
       |
  Gradio Dashboard
```

## Fault Types

| Fault | Physical Cause | Signature |
|-------|----------------|-----------|
| SEU | High-energy particle bit-flip | Instant spike to impossible value |
| TID Drift | Cumulative ionizing dose | Monotonic calibration bias |
| Data Gap | Latch-up / transmission error | Consecutive NaN blocks |
| Noise Floor | Radiation background rise | Increasing signal variance |

## Detectors (9 total)

**Hard rules** (any-True -> anomaly): `gaps`, `range`, `delta`, `flatline`, `duplicates`

**Soft** (>=2 agreement): `zscore`, `sliding_window`, `isolation_forest`, `lstm_ae`

## Team

| Role | Person |
|------|--------|
| Python / AI / Algo | Ahmet Husrev Sayin |
| Infra / Dashboard  | Omer Can Gumus |

## Commands

```bash
make install   # install dependencies
make run       # launch Gradio dashboard
make test      # pytest with coverage
make train     # train LSTM AE model
```

## Docker

```bash
docker compose up -d    # -> http://localhost:7860
```

## Project Structure

```
cosmic_pipeline/
+-- pipeline/           # Core: orchestrator, detectors, ensemble, filters, validator, tracer
+-- data/               # Synthetic telemetry generator (20 satellite channels)
+-- models/             # LSTM Autoencoder (train + inference + lstm_ae.pt)
+-- dashboard/          # Gradio app
+-- config/             # YAML config system
+-- utils/              # CSV/TSV/Excel/JSON parser
+-- tests/              # Unit + integration tests
```

## Current Limitations

- **No real-time streaming**: Batch processing only.

## Roadmap

- [x] **Multi-channel pipeline**: Process all numeric columns independently — per-channel detection and filtering
- [x] **HDF5/Parquet format support**: Optional (`pip install h5py pyarrow`)
- [ ] Real-time streaming support
- [ ] Adaptive threshold tuning per sensor type

## License

MIT License -- TUA Astro Hackathon 2026
