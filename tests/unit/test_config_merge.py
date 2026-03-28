"""Unit tests for config deep merge."""

from pipeline.orchestrator import DEFAULT_CONFIG, _deep_merge


def test_partial_config_merge():
    """Partial config override should not lose other defaults."""
    partial = {"ensemble": {"min_agreement": 3}}
    merged = _deep_merge(DEFAULT_CONFIG, partial)
    assert merged["ensemble"]["min_agreement"] == 3
    assert merged["dsp_detector"]["zscore_threshold"] == 2.0
    assert merged["classic_filter"]["median_window"] == 5


def test_empty_config_uses_defaults():
    """Empty override returns a copy of defaults."""
    merged = _deep_merge(DEFAULT_CONFIG, {})
    assert merged == DEFAULT_CONFIG


def test_deep_merge_nested_override():
    """Override a nested key without losing siblings."""
    partial = {"dsp_detector": {"zscore_threshold": 3.5}}
    merged = _deep_merge(DEFAULT_CONFIG, partial)
    assert merged["dsp_detector"]["zscore_threshold"] == 3.5
    assert merged["dsp_detector"]["window"] == 50
    assert merged["dsp_detector"]["range_std_multiplier"] == 10.0


def test_deep_merge_does_not_mutate_base():
    """Deep merge should not modify the original base dict."""
    import copy
    original = copy.deepcopy(DEFAULT_CONFIG)
    _deep_merge(DEFAULT_CONFIG, {"ensemble": {"min_agreement": 99}})
    assert DEFAULT_CONFIG == original


def test_pipeline_uses_merged_config():
    """Pipeline with partial config still uses all defaults."""
    import numpy as np
    import pandas as pd
    from pipeline.orchestrator import run_pipeline

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=200, freq="1s"),
        "value": np.random.default_rng(42).normal(0, 1, 200),
    })
    # Only override ensemble — other sections should still work
    result = run_pipeline(df, config={"ensemble": {"min_agreement": 3}}, method="classic")
    assert "cleaned_data" in result
    assert "fault_mask" in result


def test_config_validation_warnings():
    """Extreme values produce warnings."""
    from config.config import validate_config
    w = validate_config({"dsp_detector": {"zscore_threshold": 0.5}})
    assert any("hassas" in x for x in w)


def test_config_validation_no_warnings():
    """Default config produces no warnings."""
    from config.config import validate_config
    from pipeline.orchestrator import DEFAULT_CONFIG
    w = validate_config(DEFAULT_CONFIG)
    assert len(w) == 0
