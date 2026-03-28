"""Unit tests for pipeline.ensemble module."""

import numpy as np
import pandas as pd
import pytest

from pipeline.ensemble import ensemble_vote, hybrid_majority_vote, weighted_vote


def _mask(values: list[bool]) -> pd.Series:
    return pd.Series(values, dtype=bool)


# --- ensemble_vote ---

def test_majority_basic():
    m1 = _mask([True, True, False, False])
    m2 = _mask([True, False, True, False])
    m3 = _mask([True, True, True, False])
    result = ensemble_vote([m1, m2, m3], strategy="majority")
    assert list(result) == [True, True, True, False]


def test_any_strategy():
    m1 = _mask([True, False, False])
    m2 = _mask([False, False, True])
    result = ensemble_vote([m1, m2], strategy="any")
    assert list(result) == [True, False, True]


def test_all_strategy():
    m1 = _mask([True, True, False])
    m2 = _mask([True, False, False])
    result = ensemble_vote([m1, m2], strategy="all")
    assert list(result) == [True, False, False]


def test_min_agreement_override():
    m1 = _mask([True, True, False, False])
    m2 = _mask([True, False, True, False])
    m3 = _mask([False, False, True, False])
    result = ensemble_vote([m1, m2, m3], min_agreement=1)
    assert list(result) == [True, True, True, False]


def test_single_detector():
    m = _mask([True, False, True])
    result = ensemble_vote([m])
    assert list(result) == [True, False, True]


def test_empty_masks_raises():
    with pytest.raises(ValueError, match="At least one"):
        ensemble_vote([])


def test_length_mismatch_raises():
    m1 = _mask([True, False])
    m2 = _mask([True, False, True])
    with pytest.raises(ValueError, match="length mismatch"):
        ensemble_vote([m1, m2])


def test_invalid_strategy_raises():
    m = _mask([True])
    with pytest.raises(ValueError, match="Unknown strategy"):
        ensemble_vote([m], strategy="invalid")


def test_returns_bool_series():
    m1 = _mask([True, False])
    m2 = _mask([False, True])
    result = ensemble_vote([m1, m2], strategy="any")
    assert isinstance(result, pd.Series)
    assert result.dtype == bool


# --- hybrid_majority_vote ---

def test_hybrid_hard_mask_always_wins():
    hard = [_mask([True, False, False, True])]
    soft = [_mask([False, False, False, False])]
    result = hybrid_majority_vote(hard, soft, min_agreement=1)
    # Hard True at 0,3 regardless of soft
    assert list(result) == [True, False, False, True]


def test_hybrid_soft_needs_agreement():
    hard = [_mask([False, False, False, False])]
    s1 = _mask([True, True, False, False])
    s2 = _mask([True, False, True, False])
    s3 = _mask([False, False, True, False])
    result = hybrid_majority_vote(hard, [s1, s2, s3], min_agreement=2)
    # Point 0: s1+s2 agree → True
    # Point 1: only s1 → False
    # Point 2: s2+s3 agree → True
    # Point 3: none → False
    assert list(result) == [True, False, True, False]


def test_hybrid_combined():
    hard = [_mask([False, False, True, False])]  # gap at 2
    s1 = _mask([True, False, False, False])
    s2 = _mask([True, False, False, True])
    result = hybrid_majority_vote(hard, [s1, s2], min_agreement=2)
    # 0: both soft agree → True
    # 1: none → False
    # 2: hard → True
    # 3: only s2 → False
    assert list(result) == [True, False, True, False]


def test_hybrid_empty_raises():
    with pytest.raises(ValueError, match="At least one"):
        hybrid_majority_vote([], [])


def test_hybrid_only_hard():
    h1 = _mask([True, False])
    h2 = _mask([False, True])
    result = hybrid_majority_vote([h1, h2], [])
    assert list(result) == [True, True]


def test_hybrid_only_soft():
    s1 = _mask([True, True, False])
    s2 = _mask([True, False, False])
    result = hybrid_majority_vote([], [s1, s2], min_agreement=2)
    assert list(result) == [True, False, False]


# --- weighted_vote ---

def test_weighted_basic():
    m1 = _mask([True, False, False])
    m2 = _mask([True, True, False])
    result = weighted_vote([m1, m2], weights=[0.8, 0.2], threshold=0.5)
    assert list(result) == [True, False, False]


def test_weighted_equal_weights():
    m1 = _mask([True, False])
    m2 = _mask([False, True])
    result = weighted_vote([m1, m2], weights=[1.0, 1.0], threshold=0.5)
    assert list(result) == [True, True]


def test_weighted_mismatch_raises():
    m1 = _mask([True])
    with pytest.raises(ValueError, match="length mismatch"):
        weighted_vote([m1], weights=[1.0, 2.0])
