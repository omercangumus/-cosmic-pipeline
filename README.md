# Cosmic Pipeline

A hybrid DSP/ML system for detecting and correcting radiation-induced faults in satellite telemetry data.

Built for **TUA Astro Hackathon 2026** by Team Cosmic.

## Overview

The Cosmic Pipeline processes time-series telemetry signals corrupted by space radiation effects:
- **Single Event Upsets (SEU)**: Bit-flips from high-energy particles
- **Total Ionizing Dose (TID)**: Cumulative radiation causing signal drift
- **Data Gaps**: Missing data segments
- **Noise**: Random signal corruption

The system uses a multi-stage pipeline:
1. **Data Ingestion**: Synthetic generation or GOES satellite data download
2. **Detection**: Parallel DSP detectors (Z-score, IQR, Isolation Forest) + LSTM Autoencoder
3. **Ensemble Voting**: Majority voting across detector outputs
4. **Correction**: Classic filters (median, Savitzky-Golay, wavelet) + ML reconstruction
5. **Visualization**: Interactive Streamlit dashboard

## Team

- **Ömer (omercangumus)**: Infrastructure, data layer, dashboard
- **Ahmet (ahmetsn702)**: Pipeline core, ML models, detection/filtering algorithms

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd cosmic-pipeline

# Install dependencies
make install
# or: pip install -r requirements.txt
```

### Train the LSTM Model

```bash
python models/train.py --epochs 50 --hidden-dim 64
```

### Run the Dashboard

```bash
make run-dashboard
# or: streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

### Run Tests

```bash
make test
# or: pytest tests/ -v
```

## Project Structure

```
cosmic-pipeline/
├── data/                    # Data generation and retrieval
│   ├── synthetic_generator.py
│   └── goes_downloader.py
├── pipeline/                # Core processing pipeline
│   ├── detectors/          # Anomaly detection
│   │   ├── dsp_detector.py
│   │   └── lstm_detector.py
│   ├── filters/            # Signal correction
│   │   ├── classic_filter.py
│   │   └── ml_reconstructor.py
│   ├── ensemble_voter.py
│   └── pipeline.py
├── models/                  # ML models
│   ├── lstm_autoencoder.py
│   └── train.py
├── dashboard/               # Streamlit UI
│   └── app.py
├── config/                  # Configuration management
│   ├── config.py
│   ├── parser.py
│   ├── default.yaml
│   ├── fast.yaml
│   └── accurate.yaml
├── utils/                   # Utilities
│   ├── validation.py
│   ├── metrics.py
│   └── logging.py
└── tests/                   # Test suite
    ├── unit/
    ├── integration/
    └── property/
```

## Configuration

The pipeline supports YAML/JSON configuration files. Example configurations:

- `config/default.yaml`: Balanced performance and accuracy
- `config/fast.yaml`: Fast processing (DSP only)
- `config/accurate.yaml`: Maximum accuracy (all detectors + ML)

Load a configuration:

```python
from config.parser import ConfigParser

config = ConfigParser.parse("config/default.yaml")
```

## Usage Examples

### Process Synthetic Data

```python
from data.synthetic_generator import SyntheticGenerator, FaultConfig
from pipeline.pipeline import Pipeline
from config.parser import ConfigParser

# Generate synthetic telemetry
generator = SyntheticGenerator()
fault_config = FaultConfig(seu_probability=0.01, noise_snr_db=20.0)
timestamps, signal, ground_truth = generator.generate(
    duration=100.0,
    sampling_rate=10.0,
    fault_config=fault_config
)

# Run pipeline
config = ConfigParser.parse("config/default.yaml")
pipeline = Pipeline(config)
result = pipeline.process(signal, timestamps, ground_truth)

print(f"SNR Improvement: {result.metrics['snr_improvement']:.2f} dB")
print(f"Detection F1 Score: {result.metrics['f1_score']:.3f}")
```

### Download GOES Data

```python
from data.goes_downloader import GOESDownloader, GOESConfig
from datetime import datetime, timedelta

downloader = GOESDownloader()
config = GOESConfig(cache_enabled=True)

end_time = datetime.now()
start_time = end_time - timedelta(hours=24)

df = downloader.download(start_time, end_time, config)
print(df.head())
```

## Development

### Git Workflow

- `main`: Stable releases
- `develop`: Integration branch
- `feature/*`: Feature development branches

### Commit Convention

- `feat:` New features
- `fix:` Bug fixes
- `chore:` Maintenance/refactoring
- `docs:` Documentation updates

### Running Tests

```bash
# All tests
make test

# Specific test file
pytest tests/unit/test_synthetic_generator.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Code Quality

```bash
# Run linter
make lint

# Clean cache files
make clean
```

## Performance Targets

- **Throughput**: Process 1000+ samples/second (DSP detectors)
- **Latency**: Complete 10,000-sample signals within 5 seconds
- **Test Suite**: All tests complete within 60 seconds
- **Coverage**: Minimum 80% code coverage

## API Documentation

Full API documentation is available in the `docs/` directory (generated with Sphinx).

To generate documentation:

```bash
cd docs/
make html
```

## License

MIT License - TUA Astro Hackathon 2026

## Acknowledgments

- NOAA Space Weather Prediction Center for GOES data API
- TUA Astro Hackathon organizers
