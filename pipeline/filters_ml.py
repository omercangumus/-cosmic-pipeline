"""ML-based signal reconstruction using LSTM Autoencoder."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def reconstruct_with_lstm(
    df: pd.DataFrame,
    mask: pd.Series,
    model_path: str | Path = "models/lstm_ae.pt",
    blend_window: int = 10,
) -> pd.DataFrame:
    """
    Replace anomalous points with LSTM Autoencoder reconstructed values.

    Only points where mask=True are replaced. A blending window smooths
    the transition between original and reconstructed values.

    Args:
        df: DataFrame with a 'value' column.
        mask: Boolean mask (True = anomaly to fix).
        model_path: Path to trained LSTM checkpoint.
        blend_window: Half-width of blending zone at anomaly boundaries.

    Returns:
        Corrected DataFrame.
    """
    import torch
    from models.train import create_sequences, load_model

    model_path = Path(model_path)
    if not model_path.exists():
        logger.warning("Model not found at %s, falling back to interpolation", model_path)
        return _fallback_interpolation(df, mask)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(model_path, device=device)
    norm = checkpoint["normalization"]
    cfg = checkpoint["config"]
    ws = cfg["window_size"]

    from pipeline.filters_classic import detrend_signal

    out = df.copy()
    values = out["value"].values.astype(np.float64)

    if len(values) < ws:
        logger.warning("Signal too short for LSTM reconstruction, using interpolation")
        return _fallback_interpolation(df, mask)

    # Detrend before LSTM: remove linear drift so the model sees stationary data
    df_detrended = detrend_signal(out)
    values_dt = df_detrended["value"].values.astype(np.float64)

    # Normalize
    values_clean = values_dt.copy()
    # Replace anomalous points with local median before feeding to LSTM
    mask_arr = mask.values.astype(bool)
    values_clean[mask_arr] = np.nan
    filled = pd.Series(values_clean).interpolate(method="linear", limit_direction="both").ffill().bfill().values
    values_norm = (filled - norm["mean"]) / norm["std"]

    sequences = create_sequences(values_norm, ws)
    tensor = torch.tensor(sequences, dtype=torch.float32).to(device)

    # Reconstruct
    with torch.no_grad():
        reconstructed_seq = model(tensor)
    recon_np = reconstructed_seq.cpu().numpy().squeeze(-1)  # (n_windows, ws)

    # Average overlapping reconstructions
    n = len(values)
    recon_sum = np.zeros(n, dtype=np.float64)
    recon_count = np.zeros(n, dtype=np.float64)

    for i in range(len(recon_np)):
        recon_sum[i : i + ws] += recon_np[i]
        recon_count[i : i + ws] += 1

    recon_count[recon_count == 0] = 1
    reconstructed = recon_sum / recon_count

    # Denormalize
    reconstructed = reconstructed * norm["std"] + norm["mean"]

    # Replace only masked points, with blending at boundaries
    # Work on detrended values — drift (TID) is removed intentionally
    result = values_dt.copy()
    if blend_window > 0:
        result = _blend_replace(result, reconstructed, mask_arr, blend_window)
    else:
        result[mask_arr] = reconstructed[mask_arr]

    out["value"] = result
    n_fixed = mask_arr.sum()
    logger.info("LSTM reconstruction: corrected %d points (blend_window=%d)", n_fixed, blend_window)
    return out


def _blend_replace(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
    half_width: int,
) -> np.ndarray:
    """
    Replace masked values with blended transition at boundaries.

    Creates a smooth alpha ramp at the edges of each anomalous region
    to avoid discontinuity artifacts.
    """
    n = len(original)
    result = original.copy()

    # Build a smooth alpha mask: 1.0 at anomaly center, ramp at edges
    alpha = np.zeros(n, dtype=np.float64)
    alpha[mask] = 1.0

    # Smooth the binary mask edges with a linear ramp
    for _ in range(half_width):
        smoothed = alpha.copy()
        for i in range(1, n - 1):
            if alpha[i] == 0 and (alpha[i - 1] > 0 or alpha[i + 1] > 0):
                smoothed[i] = max(alpha[i - 1], alpha[i + 1]) * 0.5
        alpha = smoothed

    # Blend: result = (1 - alpha) * original + alpha * reconstructed
    blend_mask = alpha > 0
    orig_vals = original[blend_mask].copy()
    recon_vals = reconstructed[blend_mask]
    a = alpha[blend_mask]

    # Where original is NaN, use reconstructed directly
    nan_mask = np.isnan(orig_vals)
    orig_vals[nan_mask] = recon_vals[nan_mask]

    result[blend_mask] = (1 - a) * orig_vals + a * recon_vals
    return result


def _fallback_interpolation(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Simple interpolation fallback when LSTM model is not available."""
    from pipeline.filters_classic import interpolate_gaps

    logger.info("Using interpolation fallback for ML filter")
    return interpolate_gaps(df, mask)
