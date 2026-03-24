# Import Reference for Cosmic Pipeline

## Correct Import Structure

### Data Layer (Ömer - Complete ✅)
```python
from data.synthetic_generator import generate_clean_signal, inject_faults
from data.goes_downloader import get_goes_dataframe, SOLAR_FLARE_EVENTS
```

### Dashboard (Ömer - Complete ✅)
```python
from dashboard.charts import (
    plot_signal,
    plot_comparison,
    plot_metrics_bar,
    plot_anomaly_timeline
)
```

### Pipeline (Ahmet - To Be Implemented)
```python
# Main orchestrator - this is what dashboard imports
from pipeline.orchestrator import run_pipeline

# Individual modules (for Ahmet's implementation)
from pipeline.detector_classic import detect_classic
from pipeline.detector_ml import detect_ml
from pipeline.ensemble import ensemble_vote
from pipeline.filters_classic import filter_classic
from pipeline.filters_ml import filter_ml
from pipeline.ingestion import ingest_data
from pipeline.validator import validate_signal
```

### Models (Ahmet - To Be Implemented)
```python
from models.lstm_autoencoder import LSTMAutoencoder
from models.train import train_model
```

### Utils (Ahmet - To Be Implemented)
```python
from utils.validation import DataValidator
from utils.metrics import compute_metrics
from utils.logging import setup_logging
```

### Config (Ahmet - To Be Implemented)
```python
from config.config import PipelineConfig
from config.parser import parse_config
```

## File Structure

```
pipeline/
├── __init__.py
├── orchestrator.py          ← Main entry point (run_pipeline function)
├── detector_classic.py      ← Z-score, IQR, Isolation Forest
├── detector_ml.py           ← LSTM Autoencoder detection
├── ensemble.py              ← Ensemble voting
├── filters_classic.py       ← Median, Savitzky-Golay, wavelet
├── filters_ml.py            ← ML reconstruction
├── ingestion.py             ← Data preprocessing
└── validator.py             ← Data validation
```

## Expected Interface for Dashboard Integration

### run_pipeline() Function

```python
def run_pipeline(
    df: pd.DataFrame,
    methods: list = ["classic", "ml"],
    ground_truth_mask: dict = None
) -> dict:
    """
    Run the complete pipeline on telemetry data.
    
    Args:
        df: DataFrame with columns [timestamp, value]
        methods: List of methods to use (e.g., ["classic", "ml"])
        ground_truth_mask: Optional dict with keys: seu, tid, gap, noise
        
    Returns:
        {
            "classic": {
                "cleaned_df": pd.DataFrame,
                "anomaly_mask": np.ndarray (bool),
                "metrics": {
                    "snr": float,
                    "rmse": float,
                    "precision": float,
                    "recall": float,
                    "f1": float
                }
            },
            "ml": {
                "cleaned_df": pd.DataFrame,
                "anomaly_mask": np.ndarray (bool),
                "metrics": {
                    "snr": float,
                    "rmse": float,
                    "precision": float,
                    "recall": float,
                    "f1": float
                }
            },
            "ensemble": {
                "anomaly_mask": np.ndarray (bool),
                "metrics": dict
            }
        }
    """
```

## Dashboard Usage

The dashboard (`dashboard/app.py`) imports and uses the pipeline like this:

```python
try:
    from pipeline.orchestrator import run_pipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

# When user clicks "Run Pipeline" button:
if PIPELINE_AVAILABLE:
    results = run_pipeline(
        df,
        methods=["classic", "ml"],
        ground_truth_mask=gt_mask  # optional, only for synthetic data
    )
    st.session_state["results"] = results
```

## Notes

- All pipeline files are flat under `pipeline/` directory (no subdirectories)
- The `orchestrator.py` is the main entry point that coordinates all other modules
- Dashboard expects specific return structure from `run_pipeline()`
- All metrics must include: snr, rmse, precision, recall, f1
- DataFrames must have columns: [timestamp, value]
- Anomaly masks are boolean numpy arrays
