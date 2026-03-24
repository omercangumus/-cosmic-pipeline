"""Synthetic telemetry signal generator with realistic radiation fault injection."""

import numpy as np
import pandas as pd
import struct
import random
from typing import Tuple
from dataclasses import dataclass


@dataclass
class FaultConfig:
    """Configuration for fault injection parameters."""
    seu_count: int = 15
    tid_slope: float = 0.003
    gap_count: int = 4
    gap_size_range: Tuple[int, int] = (10, 40)
    noise_std_max: float = 2.0


def _flip_bits(value: float, n_bits: int = 2) -> float:
    """
    Realistic SEU simulation: flip n_bits random bits in float32 binary representation.
    
    Args:
        value: Original float value
        n_bits: Number of bits to flip
        
    Returns:
        Float value with flipped bits
    """
    packed = struct.pack('f', np.float32(value))
    as_int = int.from_bytes(packed, byteorder='little')
    
    for _ in range(n_bits):
        bit_position = random.randint(0, 31)
        as_int ^= (1 << bit_position)
    
    result_bytes = as_int.to_bytes(4, byteorder='little')
    return struct.unpack('f', result_bytes)[0]


def generate_clean_signal(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a clean synthetic telemetry signal.
    
    Args:
        n: Number of samples
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns [timestamp, value]
    """
    np.random.seed(seed)
    
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1s")
    t = np.arange(n)
    
    # Composite signal: sine + linear trend + slow oscillation + noise
    value = (
        np.sin(2 * np.pi * 0.01 * t) * 10 +  # base sine wave
        t * 0.001 +                           # linear trend
        np.sin(2 * np.pi * 0.005 * t) * 2 +  # slow oscillation
        np.random.normal(0, 0.1, n)           # tiny background noise
    )
    
    return pd.DataFrame({"timestamp": timestamps, "value": value})


def inject_faults(
    df: pd.DataFrame,
    seu_count: int = 15,
    tid_slope: float = 0.003,
    gap_count: int = 4,
    noise_std_max: float = 2.0,
    seed: int = 42
) -> Tuple[pd.DataFrame, dict]:
    """
    Inject realistic radiation faults into clean telemetry signal.
    
    Args:
        df: Clean signal DataFrame
        seu_count: Number of SEU bit-flips to inject
        tid_slope: TID drift slope coefficient
        gap_count: Number of data gap blocks
        noise_std_max: Maximum noise standard deviation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (corrupted_df, ground_truth_mask)
        ground_truth_mask contains keys: seu, tid, gap, noise
    """
    ground_truth_mask = {"seu": [], "tid": [], "gap": [], "noise": []}
    
    # Create copy to avoid modifying input
    corrupted = df.copy()
    values = corrupted["value"].values.copy().astype(np.float64)
    n = len(values)
    
    rng = np.random.default_rng(seed)
    random.seed(seed)
    
    # SEU — realistic bit-flips
    seu_indices = rng.choice(n, size=seu_count, replace=False).tolist()
    for idx in seu_indices:
        values[idx] = _flip_bits(float(values[idx]), n_bits=rng.integers(1, 4))
    ground_truth_mask["seu"] = sorted(seu_indices)
    
    # TID drift — monotonic polynomial bias
    drift = np.polyval([tid_slope, tid_slope * 0.5, 0], np.arange(n))
    values += drift
    ground_truth_mask["tid"] = list(range(n))
    
    # Data gaps — NaN blocks (latch-up simulation)
    gap_starts = sorted(rng.choice(n - 50, size=gap_count, replace=False).tolist())
    used = set()
    gaps = []
    
    for start in gap_starts:
        # Avoid overlapping gaps
        if any(abs(start - s) < 50 for s in used):
            continue
        
        length = int(rng.integers(10, 41))
        end = min(start + length, n)
        values[start:end] = np.nan
        gaps.append((start, end))
        used.add(start)
    
    ground_truth_mask["gap"] = gaps
    
    # Noise floor rise — variable std increasing over time
    noise_std = np.linspace(0.1, noise_std_max, n)
    noise = rng.normal(0, noise_std)
    values += noise
    ground_truth_mask["noise"] = [i for i in range(n) if noise_std[i] > 1.0]
    
    corrupted["value"] = values
    return corrupted, ground_truth_mask
