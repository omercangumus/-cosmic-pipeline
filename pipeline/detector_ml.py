"""ML-based anomaly detection: Isolation Forest + LSTM Autoencoder."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


def detect_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.05,
    seed: int = 42,
) -> pd.Series:
    """
    Detect anomalies using Isolation Forest on engineered features.

    Features: value, first_derivative, rolling_std_10, rolling_std_50,
    deviation_from_rolling_median.

    Args:
        df: DataFrame with a 'value' column.
        contamination: Expected fraction of anomalies.
        seed: Random seed.

    Returns:
        Boolean Series (True = anomaly).
    """
    features = _build_features(df["value"])

    clf = IsolationForest(
        contamination=contamination,
        random_state=seed,
        n_estimators=100,
        n_jobs=-1,
    )
    preds = clf.fit_predict(features)

    mask = pd.Series(preds == -1, index=df.index)
    logger.info("Isolation Forest (contamination=%.2f): %d anomalies", contamination, mask.sum())
    return mask


def _build_features(values: pd.Series) -> np.ndarray:
    """Build feature matrix for Isolation Forest."""
    v = values.copy().astype(np.float64)

    first_derivative = v.diff().fillna(0)
    rolling_std_10 = v.rolling(10, min_periods=1).std().fillna(0)
    rolling_std_50 = v.rolling(50, min_periods=1).std().fillna(0)
    rolling_median = v.rolling(50, min_periods=1).median()
    deviation = (v - rolling_median).fillna(0)

    features = np.column_stack([
        v.fillna(0).values,
        first_derivative.values,
        rolling_std_10.values,
        rolling_std_50.values,
        deviation.values,
    ])
    return features


def detect_with_lstm(
    df: pd.DataFrame,
    model_path: str | Path = "models/lstm_ae.pt",
    threshold_percentile: float = 95.0,
    window_size: int | None = None,
) -> pd.Series:
    """
    Detect anomalies using LSTM Autoencoder reconstruction error.

    Points with reconstruction error above the threshold percentile
    are flagged as anomalies.

    Args:
        df: DataFrame with a 'value' column.
        model_path: Path to trained model checkpoint.
        threshold_percentile: Percentile of reconstruction error for threshold.
        window_size: Override window size (uses checkpoint value if None).

    Returns:
        Boolean Series (True = anomaly).
    """
    import torch
    from models.train import create_sequences, load_model

    model_path = Path(model_path)
    if not model_path.exists():
        logger.warning("Model not found at %s, returning empty mask", model_path)
        return pd.Series(np.zeros(len(df), dtype=bool), index=df.index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(model_path, device=device)
    norm = checkpoint["normalization"]
    cfg = checkpoint["config"]

    ws = window_size or cfg["window_size"]

    values = df["value"].values.astype(np.float64)

    # Normalize with training stats
    values_norm = (values - norm["mean"]) / norm["std"]

    # Replace NaN with 0 for LSTM input
    values_norm = np.nan_to_num(values_norm, nan=0.0)

    if len(values_norm) < ws:
        logger.warning("Signal too short (%d) for LSTM window (%d)", len(values_norm), ws)
        return pd.Series(np.zeros(len(df), dtype=bool), index=df.index)

    sequences = create_sequences(values_norm, ws)
    tensor = torch.tensor(sequences, dtype=torch.float32).to(device)

    # Compute reconstruction error per window
    errors_per_window = model.reconstruction_error(tensor)  # (n_windows, ws)
    errors_np = errors_per_window.cpu().numpy()

    # Map window-level errors back to per-point errors (average overlapping windows)
    n = len(values)
    point_errors = np.zeros(n, dtype=np.float64)
    point_counts = np.zeros(n, dtype=np.float64)

    for i in range(len(errors_np)):
        point_errors[i : i + ws] += errors_np[i]
        point_counts[i : i + ws] += 1

    point_counts[point_counts == 0] = 1
    avg_errors = point_errors / point_counts

    threshold = np.percentile(avg_errors, threshold_percentile)
    mask = pd.Series(avg_errors > threshold, index=df.index)

    logger.info(
        "LSTM detector (percentile=%.0f, threshold=%.4f): %d anomalies",
        threshold_percentile, threshold, mask.sum(),
    )
    return mask


def detect_all_ml(
    df: pd.DataFrame,
    model_path: str | Path = "models/lstm_ae.pt",
    contamination: float = 0.05,
    threshold_percentile: float = 95.0,
    seed: int = 42,
) -> dict[str, pd.Series]:
    """
    Run all ML detectors and return individual masks.

    Args:
        df: DataFrame with 'value' column.
        model_path: Path to LSTM checkpoint.
        contamination: Isolation Forest contamination param.
        threshold_percentile: LSTM reconstruction error percentile.
        seed: Random seed.

    Returns:
        Dict of detector_name → boolean mask.
    """
    results = {
        "isolation_forest": detect_isolation_forest(df, contamination=contamination, seed=seed),
        "lstm_ae": detect_with_lstm(df, model_path=model_path, threshold_percentile=threshold_percentile),
    }

    total = sum(m.sum() for m in results.values())
    logger.info("ML detection complete: %d total flags across %d detectors", total, len(results))
    return results
