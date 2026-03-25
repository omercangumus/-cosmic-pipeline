"""Ensemble voting logic for combining multiple detector results."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def ensemble_vote(
    masks: list[pd.Series],
    strategy: str = "majority",
    min_agreement: int | None = None,
) -> pd.Series:
    """
    Combine multiple detector boolean masks into a single consensus mask.

    Args:
        masks: List of boolean Series from individual detectors.
        strategy: Voting strategy — 'majority', 'any', or 'all'.
        min_agreement: Minimum number of detectors that must agree
                       (overrides strategy if provided).

    Returns:
        Final boolean mask (True = anomaly by consensus).

    Raises:
        ValueError: If masks list is empty or lengths mismatch.
    """
    if not masks:
        raise ValueError("At least one detector mask is required")

    n = len(masks[0])
    for i, m in enumerate(masks):
        if len(m) != n:
            raise ValueError(
                f"Mask length mismatch: mask[0]={n}, mask[{i}]={len(m)}"
            )

    vote_matrix = np.column_stack([m.values.astype(int) for m in masks])
    vote_sum = vote_matrix.sum(axis=1)
    n_detectors = len(masks)

    if min_agreement is not None:
        threshold = min_agreement
    elif strategy == "majority":
        threshold = (n_detectors // 2) + 1
    elif strategy == "any":
        threshold = 1
    elif strategy == "all":
        threshold = n_detectors
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'majority', 'any', or 'all'.")

    result = pd.Series(vote_sum >= threshold, index=masks[0].index)

    logger.info(
        "Ensemble vote (%s, threshold=%d/%d): %d anomalies from %d total points",
        strategy if min_agreement is None else f"min_agreement={min_agreement}",
        threshold,
        n_detectors,
        result.sum(),
        n,
    )
    return result


def weighted_vote(
    masks: list[pd.Series],
    weights: list[float],
    threshold: float = 0.5,
) -> pd.Series:
    """
    Combine detector masks using confidence-weighted voting.

    Each detector's vote is multiplied by its weight. A point is flagged
    if the weighted sum exceeds the threshold.

    Args:
        masks: List of boolean Series from individual detectors.
        weights: Weight for each detector (higher = more trusted).
        threshold: Weighted vote threshold (0-1 range recommended).

    Returns:
        Final boolean mask.

    Raises:
        ValueError: If masks and weights lengths mismatch.
    """
    if len(masks) != len(weights):
        raise ValueError(
            f"Masks ({len(masks)}) and weights ({len(weights)}) length mismatch"
        )
    if not masks:
        raise ValueError("At least one detector mask is required")

    weight_arr = np.array(weights, dtype=np.float64)
    total_weight = weight_arr.sum()
    if total_weight < 1e-12:
        raise ValueError("Total weight must be positive")

    normalized = weight_arr / total_weight

    vote_matrix = np.column_stack([m.values.astype(np.float64) for m in masks])
    weighted_sum = vote_matrix @ normalized

    result = pd.Series(weighted_sum >= threshold, index=masks[0].index)

    logger.info(
        "Weighted vote (threshold=%.2f): %d anomalies detected",
        threshold,
        result.sum(),
    )
    return result
