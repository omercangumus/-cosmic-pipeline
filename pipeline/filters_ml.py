"""ML-based signal filtering — stub module.

LSTM is used for *detection* only (pipeline.detector_ml).
Filtering is handled entirely by the classic layered pipeline
(interpolation → detrend → median).  This module is kept as a
no-op entry point so existing imports do not break.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def reconstruct_with_lstm(
    df: pd.DataFrame,
    mask: pd.Series,
    **kwargs,
) -> pd.DataFrame:
    """Fallback: delegate to classic interpolation.

    LSTM reconstruction has been removed from the filtering path.
    This stub exists for backward compatibility.
    """
    from pipeline.filters_classic import interpolate_gaps

    logger.info("ML filter stub: delegating to interpolation fallback")
    return interpolate_gaps(df, mask)
