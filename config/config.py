"""Configuration dataclasses and utilities."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectorConfig:
    """Configuration for anomaly detectors."""
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    gap_threshold_seconds: int = 60
    sliding_window_size: int = 50
    sliding_window_threshold: float = 3.0


@dataclass
class MLConfig:
    """Configuration for ML models."""
    model_path: str = "models/lstm_ae.pt"
    reconstruction_threshold: float = 0.1
    isolation_forest_contamination: float = 0.1


@dataclass
class EnsembleConfig:
    """Configuration for ensemble voting."""
    strategy: str = "majority"  # majority, any, all, weighted
    min_agreement: Optional[int] = None


@dataclass
class FilterConfig:
    """Configuration for signal filters."""
    median_window: int = 5
    interpolation_method: str = "linear"
    savgol_window: int = 11
    savgol_polyorder: int = 3
    wavelet: str = "db4"
    wavelet_level: int = 3


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    detectors: DetectorConfig
    ml: MLConfig
    ensemble: EnsembleConfig
    filters: FilterConfig
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'PipelineConfig':
        """Create PipelineConfig from dictionary."""
        pipeline_cfg = config_dict.get('pipeline', {})
        
        detectors_cfg = pipeline_cfg.get('detectors', {}).get('classic', {})
        ml_cfg = pipeline_cfg.get('detectors', {}).get('ml', {})
        ensemble_cfg = pipeline_cfg.get('ensemble', {})
        filters_cfg = pipeline_cfg.get('filters', {}).get('classic', {})
        
        return cls(
            detectors=DetectorConfig(**detectors_cfg),
            ml=MLConfig(**ml_cfg),
            ensemble=EnsembleConfig(**ensemble_cfg),
            filters=FilterConfig(**filters_cfg)
        )
