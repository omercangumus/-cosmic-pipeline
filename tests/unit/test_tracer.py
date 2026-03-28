"""Unit tests for pipeline.tracer module."""

import numpy as np
import pandas as pd
import pytest

from pipeline.tracer import PipelineTracer, StepRecord


def _make_df(n=100, base=10.0):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1s"),
        "value": np.ones(n) * base + rng.normal(0, 0.1, n),
    })


class TestPipelineTracer:

    def test_snapshot_records_step(self):
        tracer = PipelineTracer()
        df1 = _make_df()
        df2 = _make_df(base=12.0)
        tracer.snapshot("Test Step", "Description", df1, df2)
        assert len(tracer.steps) == 1
        assert tracer.steps[0].step_name == "Test Step"
        assert tracer.steps[0].step_number == 1

    def test_snapshot_detection_counts_anomalies(self):
        tracer = PipelineTracer()
        df = _make_df()
        mask = pd.Series(np.zeros(100, dtype=bool))
        mask.iloc[10] = True
        mask.iloc[50] = True
        tracer.snapshot_detection("Z-Score", "test", mask, df, "zscore")
        assert tracer.steps[0].anomalies_found == 2

    def test_snapshot_ensemble(self):
        tracer = PipelineTracer()
        tracer.snapshot_ensemble(hard_count=5, soft_count=10, total_count=15)
        assert tracer.steps[0].anomalies_found == 15

    def test_to_dataframe_has_correct_columns(self):
        tracer = PipelineTracer()
        df = _make_df()
        tracer.snapshot("Step 1", "desc", df, df)
        table = tracer.to_dataframe()
        assert "#" in table.columns
        assert "Adim" in table.columns
        assert "NaN Once" in table.columns
        assert "Degisim" in table.columns

    def test_to_dataframe_empty(self):
        tracer = PipelineTracer()
        table = tracer.to_dataframe()
        assert table.empty

    def test_to_summary_not_empty(self):
        tracer = PipelineTracer()
        df = _make_df()
        tracer.snapshot("Step 1", "desc", df, df)
        summary = tracer.to_summary()
        assert "Step 1" in summary
        assert "RAPORU" in summary

    def test_multiple_steps_sequential(self):
        tracer = PipelineTracer()
        df1 = _make_df()
        df2 = _make_df(base=15.0)
        df3 = _make_df(base=8.0)
        tracer.snapshot("Adim 1", "ilk", df1, df2)
        tracer.snapshot("Adim 2", "ikinci", df2, df3)
        assert len(tracer.steps) == 2
        assert tracer.steps[0].step_number == 1
        assert tracer.steps[1].step_number == 2

    def test_nan_tracking(self):
        tracer = PipelineTracer()
        df1 = _make_df()
        df1.loc[10:14, "value"] = np.nan
        df2 = _make_df()
        tracer.snapshot("Gap Fill", "NaN doldurma", df1, df2)
        assert tracer.steps[0].nan_before == 5
        assert tracer.steps[0].nan_after == 0

    def test_pipeline_returns_tracer(self):
        from data.synthetic_generator import generate_corrupted_dataset
        from pipeline.orchestrator import run_pipeline

        _, corrupted, _ = generate_corrupted_dataset(n=2000, seed=42)
        result = run_pipeline(corrupted, method="classic")
        assert "tracer_table" in result
        assert "tracer_summary" in result
        assert len(result["tracer_table"]) > 0
        assert "RAPORU" in result["tracer_summary"]
