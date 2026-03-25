"""LSTM Autoencoder for telemetry anomaly detection and signal reconstruction."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """LSTM encoder: compresses input sequence into a latent representation."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, latent_dim: int = 32, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_dim) — take last layer
        latent = self.fc(h_n[-1])  # (batch, latent_dim)
        return latent


class Decoder(nn.Module):
    """LSTM decoder: reconstructs sequence from latent representation."""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64, output_dim: int = 1, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        # z: (batch, latent_dim)
        h = self.fc(z)  # (batch, hidden_dim)
        # Repeat latent across time steps
        h_repeated = h.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_dim)
        out, _ = self.lstm(h_repeated)  # (batch, seq_len, hidden_dim)
        reconstructed = self.output_fc(out)  # (batch, seq_len, output_dim)
        return reconstructed


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for telemetry signal anomaly detection.

    Architecture: Encoder LSTM(input→64→32) → Decoder LSTM(32→64→output).
    Trained on clean signals; high reconstruction error indicates anomaly.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, num_layers)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode then decode.

        Args:
            x: (batch, seq_len, input_dim) input tensor.

        Returns:
            Reconstructed tensor with same shape as input.
        """
        seq_len = x.size(1)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent, seq_len)
        return reconstructed

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-timestep MSE reconstruction error.

        Args:
            x: (batch, seq_len, input_dim) input tensor.

        Returns:
            (batch, seq_len) tensor of per-timestep errors.
        """
        with torch.no_grad():
            x_hat = self.forward(x)
            error = ((x - x_hat) ** 2).mean(dim=-1)  # mean over input_dim
        return error
