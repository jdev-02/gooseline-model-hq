"""Plumbing tests for src/eval.py. Synthetic data only; never used to train models."""

import numpy as np
import pytest

from src.core.eval import (
    regression_report,
    assert_no_leakage,
    brier_score,
    reliability_diagram,
)


def test_regression_report_returns_metrics_and_fig():
    rng = np.random.default_rng(0)
    y_true = rng.normal(100, 5, size=200)
    y_pred = y_true + rng.normal(0, 1, size=200)
    metrics, fig = regression_report(y_true, y_pred, label="test", show=False)
    assert metrics["rmse"] > 0
    assert metrics["n"] == 200
    assert fig is not None


def test_assert_no_leakage_raises_on_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(AssertionError, match="suspiciously close to zero"):
        assert_no_leakage(y, y.copy())


def test_assert_no_leakage_passes_on_noisy_predictions():
    rng = np.random.default_rng(0)
    y = rng.normal(100, 5, size=100)
    pred = y + rng.normal(0, 2, size=100)
    assert_no_leakage(y, pred)  # should not raise


def test_brier_score_perfect_and_worst():
    y = np.array([0, 1, 0, 1])
    assert brier_score(y, y.astype(float)) == pytest.approx(0.0)
    assert brier_score(y, 1 - y.astype(float)) == pytest.approx(1.0)


def test_brier_score_constant_baserate():
    # Constant prediction at base rate should yield p(1-p) for binary outcomes.
    y = np.array([0, 0, 1, 1, 1])  # base rate 0.6
    p = np.full_like(y, 0.6, dtype=float)
    assert brier_score(y, p) == pytest.approx(0.6 * 0.4, abs=1e-3)


def test_reliability_diagram_runs():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=500)
    y = (rng.uniform(0, 1, size=500) < p).astype(int)
    fig, table = reliability_diagram(y, p, n_bins=10, label="synthetic")
    assert fig is not None
    assert len(table) == 10
