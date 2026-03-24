"""
Synthetic telemetry generator with fault injection.

This module will be implemented by Ömer (omercangumus) in Task 2.
Generates synthetic telemetry signals with radiation faults.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class FaultConfig:
    """Configuration for fault injection."""
    seu_probability: float = 0.01
    tid_drift_rate: float = 0.001
    gap_probability: float = 0.005
    gap_size_range: Tuple[int, int] = (5, 20)
    noise_snr_db: float = 20.0


class SyntheticGenerator:
    """Generates synthetic telemetry with radiation faults."""
    
    def generate(
        self,
        duration: float,
        sampling_rate: float,
        fault_config: FaultConfig
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic telemetry signal.
        
        Args:
            duration: Signal duration in seconds
            sampling_rate: Samples per second
            fault_config: Fault injection parameters
            
        Returns:
            Tuple of (timestamps, corrupted_signal, ground_truth_labels)
        """
        # TODO: Implement in Task 2
        raise NotImplementedError("To be implemented in Task 2")
