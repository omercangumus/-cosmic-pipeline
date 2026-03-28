"""Configuration dataclasses for the cosmic pipeline."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DetectorConfig:
    """Configuration for anomaly detectors."""
    zscore_threshold: float = 2.0
    gap_threshold_seconds: int = 60
    sliding_window_size: int = 50
    sliding_window_threshold: float = 3.0
    range_std_multiplier: float = 10.0


@dataclass
class MLConfig:
    """Configuration for ML models."""
    model_path: str = "models/lstm_ae.pt"
    contamination: float = 0.05
    threshold_percentile: float = 95.0


@dataclass
class EnsembleConfig:
    """Configuration for ensemble voting."""
    strategy: str = "hybrid"
    min_agreement: int = 2


@dataclass
class FilterConfig:
    """Configuration for signal filters."""
    median_window: int = 5


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    detectors: DetectorConfig = field(default_factory=DetectorConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PipelineConfig":
        """Create PipelineConfig from a flat or nested dictionary."""
        det_cfg = config_dict.get("dsp_detector", {})
        ml_cfg = config_dict.get("lstm_detector", {})
        ens_cfg = config_dict.get("ensemble", {})
        flt_cfg = config_dict.get("classic_filter", {})

        return cls(
            detectors=DetectorConfig(**{
                k: v for k, v in det_cfg.items()
                if k in DetectorConfig.__dataclass_fields__
            }),
            ml=MLConfig(**{
                k: v for k, v in ml_cfg.items()
                if k in MLConfig.__dataclass_fields__
            }),
            ensemble=EnsembleConfig(**{
                k: v for k, v in ens_cfg.items()
                if k in EnsembleConfig.__dataclass_fields__
            }),
            filters=FilterConfig(**{
                k: v for k, v in flt_cfg.items()
                if k in FilterConfig.__dataclass_fields__
            }),
        )


def validate_config(config: dict) -> list[str]:
    """Validate pipeline config dict and return warnings for risky values."""
    warnings = []
    dsp = config.get("dsp_detector", {})
    ens = config.get("ensemble", {})
    flt = config.get("classic_filter", {})

    zt = dsp.get("zscore_threshold", 2.0)
    if zt < 1.0:
        warnings.append("zscore_threshold < 1.0: cok hassas, false positive artabilir")
    if zt > 5.0:
        warnings.append("zscore_threshold > 5.0: cok genis, anomaliler kacirilabilir")

    ma = ens.get("min_agreement", 2)
    if ma < 1:
        warnings.append("min_agreement < 1: tum soft dedektorler gecerli olur")
    if ma > 4:
        warnings.append("min_agreement > 4: hic anomali bulunamaz")

    mw = flt.get("median_window", 5)
    if mw < 3:
        warnings.append("median_window < 3: etkin filtreleme yapamaz")
    if mw > 51:
        warnings.append("median_window > 51: sinyal detayini kaybedebilir")

    return warnings
