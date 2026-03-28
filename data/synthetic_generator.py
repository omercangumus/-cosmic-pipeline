"""Synthetic telemetry signal generator with realistic radiation fault injection."""

import logging
import struct
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FaultConfig:
    """Configuration for fault injection parameters."""

    seu_count: int = 15
    tid_slope: float = 0.003
    gap_count: int = 4
    gap_size_range: tuple[int, int] = (10, 40)
    noise_std_max: float = 2.0


def _flip_bits(value: float, n_bits: int, rng: np.random.Generator) -> float:
    """
    Simulate SEU by flipping random bits in float32 binary representation.

    Args:
        value: Original float value.
        n_bits: Number of bits to flip (1-3).
        rng: NumPy random generator for reproducibility.

    Returns:
        Float value with flipped bits.
    """
    packed = struct.pack("f", np.float32(value))
    as_int = int.from_bytes(packed, byteorder="little")

    positions = rng.choice(32, size=n_bits, replace=False)
    for bit_pos in positions:
        as_int ^= 1 << int(bit_pos)

    result_bytes = as_int.to_bytes(4, byteorder="little")
    result = struct.unpack("f", result_bytes)[0]

    # Bit flips in exponent bits can produce NaN/Inf — cap to finite spike
    if not np.isfinite(result):
        sign = 1.0 if rng.random() > 0.5 else -1.0
        return abs(value) * rng.uniform(10, 100) * sign

    # Clamp: keep result within ±100x of original magnitude
    magnitude = max(abs(value), 1e-6)
    limit = magnitude * 100.0
    result = float(np.clip(result, -limit, limit))

    return result


def _inject_seu(
    values: np.ndarray, count: int, rng: np.random.Generator
) -> list[int]:
    """
    Inject Single Event Upset (bit-flip spikes) into signal.

    Args:
        values: Signal array (modified in-place).
        count: Number of SEU events to inject.
        rng: NumPy random generator.

    Returns:
        Sorted list of affected indices.
    """
    n = len(values)
    count = min(count, n)
    indices = sorted(rng.choice(n, size=count, replace=False).tolist())

    for idx in indices:
        n_bits = int(rng.integers(1, 4))
        values[idx] = _flip_bits(float(values[idx]), n_bits, rng)

    logger.debug("Injected %d SEU bit-flips", len(indices))
    return indices


def _inject_tid_drift(values: np.ndarray, slope: float) -> list[int]:
    """
    Inject TID-style monotonic calibration drift across entire signal.

    Args:
        values: Signal array (modified in-place).
        slope: Drift rate coefficient.

    Returns:
        List of all affected indices (entire signal).
    """
    n = len(values)
    t = np.arange(n)
    drift = np.polyval([slope, slope * 0.5, 0], t)
    values += drift

    logger.debug("Injected TID drift with slope=%.4f", slope)
    return list(range(n))


def _inject_gaps(
    values: np.ndarray,
    count: int,
    size_range: tuple[int, int],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """
    Inject data dropout blocks (NaN) simulating latch-up events.

    Args:
        values: Signal array (modified in-place).
        count: Number of gap blocks.
        size_range: (min_length, max_length) of each gap.
        rng: NumPy random generator.

    Returns:
        List of (start, end) tuples for each gap.
    """
    n = len(values)
    min_separation = max(size_range[1], 50)

    if n <= min_separation:
        logger.warning("Signal too short (%d) for gap injection", n)
        return []

    pool_size = n - min_separation
    n_candidates = min(count * 3, pool_size)
    candidates = sorted(
        rng.choice(pool_size, size=n_candidates, replace=False).tolist()
    )

    gaps: list[tuple[int, int]] = []
    used: set[int] = set()

    for start in candidates:
        if len(gaps) >= count:
            break
        if any(abs(start - s) < min_separation for s in used):
            continue

        length = int(rng.integers(size_range[0], size_range[1] + 1))
        end = min(start + length, n)
        values[start:end] = np.nan
        gaps.append((start, end))
        used.add(start)

    logger.debug("Injected %d data gaps", len(gaps))
    return gaps


def _inject_noise_floor(
    values: np.ndarray, std_max: float, rng: np.random.Generator
) -> list[int]:
    """
    Inject rising noise floor across the signal.

    Args:
        values: Signal array (modified in-place).
        std_max: Maximum noise standard deviation at signal end.
        rng: NumPy random generator.

    Returns:
        List of indices where noise std exceeds 1.0.
    """
    n = len(values)
    noise_std = np.linspace(0.1, std_max, n)
    noise = rng.normal(0, noise_std)
    values += noise

    threshold = 1.0
    affected = [i for i in range(n) if noise_std[i] > threshold]

    logger.debug(
        "Injected noise floor rise, %d points above threshold", len(affected)
    )
    return affected


def generate_clean_signal(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic satellite telemetry signal.

    Simulates a thermal sensor on a LEO satellite:
    - Base temperature ~20 C (spacecraft internal)
    - Orbital period ~90 min (sinusoidal day/night cycle)
    - Slow thermal drift (sun angle change over orbits)
    - Eclipse cooling events (periodic dips)
    - Minor sensor noise

    Args:
        n: Number of samples.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns [timestamp, value, orbit_id, phase, sensor_id].
        Pipeline only uses timestamp + value; extras are metadata.
    """
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range("2024-01-01", periods=n, freq="1s")
    t = np.arange(n, dtype=np.float64)

    # Orbital period: ~5400 seconds (90 minutes)
    orbital_period = 5400.0
    orbit_phase = (t % orbital_period) / orbital_period  # 0 to 1

    # Base temperature
    base_temp = 20.0

    # Day/night orbital cycle: +/-8 C sinusoidal
    orbital_cycle = 8.0 * np.sin(2 * np.pi * t / orbital_period)

    # Slow thermal drift over ~10 orbits
    long_period = orbital_period * 10
    thermal_drift = 2.0 * np.sin(2 * np.pi * t / long_period)

    # Eclipse cooling: sharp dip in phase 0.6-0.95
    eclipse_mask = (orbit_phase > 0.6) & (orbit_phase < 0.95)
    eclipse_cooling = np.where(
        eclipse_mask,
        -3.0 * np.sin(np.pi * (orbit_phase - 0.6) / 0.35),
        0.0,
    )

    # Minor sensor noise: +/-0.1 C Gaussian
    noise = rng.normal(0, 0.1, n)

    value = base_temp + orbital_cycle + thermal_drift + eclipse_cooling + noise

    # Metadata
    orbit_id = (t // orbital_period).astype(int)
    phase_labels = np.where(
        orbit_phase < 0.3, "ascending",
        np.where(orbit_phase < 0.5, "peak",
                 np.where(orbit_phase < 0.6, "descending",
                          np.where(orbit_phase < 0.95, "eclipse", "recovery"))),
    )

    logger.info(
        "Generated realistic thermal telemetry: %d samples, %d orbits",
        n, int(orbit_id[-1]) + 1,
    )

    return pd.DataFrame({
        "timestamp": timestamps,
        "value": value,
        "orbit_id": orbit_id,
        "phase": phase_labels,
        "sensor_id": "THERM-001",
    })


def inject_faults(
    df: pd.DataFrame,
    seu_count: int = 15,
    tid_slope: float = 0.003,
    gap_count: int = 4,
    noise_std_max: float = 2.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Inject realistic radiation faults into clean telemetry signal.

    Applies four fault types in order: SEU bit-flips, TID drift,
    data gaps (NaN blocks), and rising noise floor.

    Args:
        df: Clean signal DataFrame with columns [timestamp, value].
        seu_count: Number of SEU bit-flip events.
        tid_slope: TID drift slope coefficient.
        gap_count: Number of data gap blocks.
        noise_std_max: Maximum noise standard deviation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (corrupted_df, ground_truth_mask).
        ground_truth_mask keys: seu (list[int]), tid (list[int]),
        gap (list[tuple[int,int]]), noise (list[int]).
    """
    rng = np.random.default_rng(seed)

    # Only work with timestamp + value for injection
    corrupted = df[["timestamp", "value"]].copy()
    values = corrupted["value"].values.copy().astype(np.float64)

    mask = {
        "seu": _inject_seu(values, seu_count, rng),
        "tid": _inject_tid_drift(values, tid_slope),
        "gap": _inject_gaps(values, gap_count, (10, 40), rng),
        "noise": _inject_noise_floor(values, noise_std_max, rng),
    }

    corrupted["value"] = values

    logger.info(
        "Injected faults: %d SEU, %d drift, %d gaps, %d noisy",
        len(mask["seu"]),
        len(mask["tid"]),
        len(mask["gap"]),
        len(mask["noise"]),
    )

    return corrupted, mask


def generate_corrupted_dataset(
    n: int = 10000,
    config: FaultConfig | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Convenience function: generate clean signal and inject faults.

    Args:
        n: Number of samples.
        config: Fault injection configuration. Uses defaults if None.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (clean_df, corrupted_df, ground_truth_mask).
    """
    if config is None:
        config = FaultConfig()

    clean_df = generate_clean_signal(n, seed=seed)
    corrupted_df, mask = inject_faults(
        clean_df,
        seu_count=config.seu_count,
        tid_slope=config.tid_slope,
        gap_count=config.gap_count,
        noise_std_max=config.noise_std_max,
        seed=seed,
    )

    logger.info("Generated corrupted dataset with %d samples", n)
    return clean_df, corrupted_df, mask
