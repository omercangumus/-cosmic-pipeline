"""
DSP-based anomaly detection module.

This module will be implemented by Ahmet (ahmetsn702).
Implements Z-score, IQR, and Isolation Forest detection methods.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple
import numpy as np


class DSPMethod(Enum):
    """Available DSP detection methods."""
    ZSCORE = "zscore"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"


@dataclass
class DSPConfig:
    """Configuration for DSP detectors."""
    method: DSPMethod
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    iforest_contamination: float = 0.1


class DSPDetector:
    """Classic signal processing anomaly detectors."""
    
    def detect(
        self,
        signal: np.ndarray,
        config: DSPConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using DSP methods.
        
        Args:
            signal: Input time-series signal
            config: Detection configuration
            
        Returns:
            Tuple of (binary_labels, anomaly_scores)
        """
        # TODO: Implement by Ahmet
        raise NotImplementedError("To be implemented by Ahmet")
