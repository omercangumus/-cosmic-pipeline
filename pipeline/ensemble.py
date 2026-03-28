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


def hybrid_majority_vote(
    hard_masks: list[pd.Series],
    soft_masks: list[pd.Series],
    min_agreement: int = 2,
) -> pd.Series:
    """
    Hybrid majority voting.

    Hard masks (gap, range violation): any True → automatic anomaly.
    Soft masks (z-score, IF, LSTM): at least min_agreement must agree.

    Args:
        hard_masks: Masks from indisputable detectors (gaps, range).
        soft_masks: Masks from statistical/ML detectors.
        min_agreement: Minimum soft detectors that must agree.

    Returns:
        Combined boolean mask.

    Raises:
        ValueError: If both lists are empty.
    """
    if not hard_masks and not soft_masks:
        raise ValueError("At least one mask (hard or soft) is required")

    ref = hard_masks[0] if hard_masks else soft_masks[0]
    n = len(ref)

    # Hard anomalies — any strategy (indisputable corruptions)
    hard_result = pd.Series(np.zeros(n, dtype=bool), index=ref.index)
    for m in hard_masks:
        hard_result = hard_result | m

    # Soft anomalies — majority voting
    soft_result = pd.Series(np.zeros(n, dtype=bool), index=ref.index)
    if soft_masks:
        vote_matrix = np.column_stack([m.values.astype(int) for m in soft_masks])
        vote_sum = vote_matrix.sum(axis=1)
        soft_result = pd.Series(vote_sum >= min_agreement, index=ref.index)

    final = hard_result | soft_result

    logger.info(
        "Hybrid majority: %d hard anomalies, %d soft anomalies (min_agreement=%d), %d total",
        int(hard_result.sum()),
        int((soft_result & ~hard_result).sum()),
        min_agreement,
        int(final.sum()),
    )
    return final


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
