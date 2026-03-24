"""Train LSTM Autoencoder on clean synthetic telemetry data."""

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data.synthetic_generator import generate_clean_signal
from models.lstm_autoencoder import LSTMAutoencoder

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/lstm_ae.pt")


def create_sequences(values: np.ndarray, window_size: int = 50) -> np.ndarray:
    """
    Slice 1D signal into overlapping windows for LSTM input.

    Args:
        values: 1D array of signal values.
        window_size: Length of each subsequence.

    Returns:
        (n_windows, window_size, 1) array.
    """
    n = len(values)
    if n < window_size:
        raise ValueError(f"Signal length ({n}) < window_size ({window_size})")

    sequences = []
    for i in range(n - window_size + 1):
        sequences.append(values[i : i + window_size])

    arr = np.array(sequences, dtype=np.float32)
    return arr.reshape(-1, window_size, 1)


def normalize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Z-score normalize. Returns (normalized, mean, std)."""
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if std < 1e-12:
        std = 1.0
    return (values - mean) / std, mean, std


def train_model(
    n_samples: int = 100_000,
    window_size: int = 50,
    hidden_dim: int = 64,
    latent_dim: int = 32,
    num_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    model_path: Path = DEFAULT_MODEL_PATH,
    seed: int = 42,
) -> dict:
    """
    Train LSTM Autoencoder on clean synthetic data and save weights.

    Args:
        n_samples: Number of clean signal samples to generate.
        window_size: Subsequence length for training.
        hidden_dim: LSTM hidden dimension.
        latent_dim: Latent space dimension.
        num_layers: Number of LSTM layers.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate.
        model_path: Where to save trained weights.
        seed: Random seed.

    Returns:
        Dict with training history and model info.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # --- Generate clean training data ---
    logger.info("Generating %d clean samples...", n_samples)
    df = generate_clean_signal(n=n_samples, seed=seed)
    values = df["value"].values.astype(np.float64)

    # Normalize
    values_norm, mean, std = normalize(values)

    # Create sequences
    sequences = create_sequences(values_norm, window_size)
    logger.info("Created %d training sequences (window=%d)", len(sequences), window_size)

    # Split train/val (90/10)
    n_train = int(len(sequences) * 0.9)
    train_data = torch.tensor(sequences[:n_train], dtype=torch.float32)
    val_data = torch.tensor(sequences[n_train:], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_data), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_data), batch_size=batch_size, shuffle=False
    )

    # --- Build model ---
    model = LSTMAutoencoder(
        input_dim=1,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %d", param_count)

    # --- Training loop ---
    history = {"train_loss": [], "val_loss": []}
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                val_loss += loss.item() * batch.size(0)
        val_loss /= len(val_data)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "Epoch %d/%d — train_loss: %.6f, val_loss: %.6f",
                epoch, epochs, train_loss, val_loss,
            )

    elapsed = time.perf_counter() - t_start
    logger.info("Training complete in %.1fs", elapsed)

    # --- Save model + normalization params ---
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "input_dim": 1,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "num_layers": num_layers,
            "window_size": window_size,
        },
        "normalization": {"mean": mean, "std": std},
        "history": history,
    }
    torch.save(checkpoint, model_path)
    logger.info("Model saved to %s", model_path)

    return {
        "model_path": str(model_path),
        "epochs": epochs,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "training_time": elapsed,
        "param_count": param_count,
    }


def load_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    device: torch.device | None = None,
) -> tuple[LSTMAutoencoder, dict]:
    """
    Load trained LSTM Autoencoder from checkpoint.

    Args:
        model_path: Path to saved .pt file.
        device: Target device. Auto-detects if None.

    Returns:
        Tuple of (model, checkpoint_dict).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    model = LSTMAutoencoder(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        num_layers=cfg["num_layers"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info("Loaded model from %s (device=%s)", model_path, device)
    return model, checkpoint


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    result = train_model()
    print(f"\nTraining complete: {result}")
