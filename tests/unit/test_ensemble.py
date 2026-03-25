"""Unit tests for pipeline.ensemble module."""

import numpy as np
import pandas as pd
import pytest

from pipeline.ensemble import ensemble_vote, weighted_vote


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
    # min_agreement=1 should behave like "any"
    result = ensemble_vote([m1, m2, m3], min_agreement=1)
    assert list(result) == [True, True, True, False]


def test_single_detector():
    m = _mask([True, False, True])
    result = ensemble_vote([m], strategy="majority")
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


# --- weighted_vote ---

def test_weighted_basic():
    m1 = _mask([True, False, False])
    m2 = _mask([True, True, False])
    # m1 weight 0.8, m2 weight 0.2 → point 0: 1.0, point 1: 0.2, point 2: 0
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
