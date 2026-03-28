"""Tests for metrics export."""

import json

import pytest

from utils.metrics_export import export_metrics


def test_export_json():
    result = {
        "metrics": {"faults_detected": 10, "processing_time": 1.5},
        "detector_counts": {"zscore": 5, "gaps": 3},
    }
    out = export_metrics(result, "json")
    data = json.loads(out)
    assert data["faults_detected"] == 10
    assert data["detector_counts"]["zscore"] == 5


def test_export_csv():
    result = {
        "metrics": {"faults_detected": 10},
        "detector_counts": {"zscore": 5},
    }
    out = export_metrics(result, "csv")
    assert "faults_detected" in out
    assert "detector_zscore" in out


def test_export_invalid_format():
    with pytest.raises(ValueError, match="Desteklenmeyen"):
        export_metrics({"metrics": {}}, "xml")


def test_export_empty_result():
    out = export_metrics({}, "json")
    data = json.loads(out)
    assert data == {"detector_counts": {}}
