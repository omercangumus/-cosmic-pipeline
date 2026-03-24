"""
Classic signal filtering module.

This module will be implemented by Ahmet (ahmetsn702).
Implements median, Savitzky-Golay, and wavelet filtering.
"""

from enum import Enum
from dataclasses import dataclass
import numpy as np


class FilterMethod(Enum):
    """Available filtering methods."""
    MEDIAN = "median"
    SAVITZKY_GOLAY = "savitzky_golay"
    WAVELET = "wavelet"


@dataclass
class FilterConfig:
    """Configuration for classic filters."""
    method: FilterMethod
    median_window: int = 5
    sg_window: int = 11
    sg_polyorder: int = 3
    wavelet_family: str = "db4"
    wavelet_level: int = 3


class ClassicFilter:
    """Traditional signal filtering methods."""
    
    def filter(
        self,
        signal: np.ndarray,
        anomaly_mask: np.ndarray,
        config: FilterConfig
    ) -> np.ndarray:
        """
        Apply filtering to anomalous regions.
        
        Args:
            signal: Input signal
            anomaly_mask: Binary mask of anomalous regions
            config: Filter configuration
            
        Returns:
            Filtered signal
        """
        # TODO: Implement by Ahmet
        raise NotImplementedError("To be implemented by Ahmet")
