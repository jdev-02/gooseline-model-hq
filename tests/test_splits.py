"""Plumbing tests for src/splits.py walk-forward CV."""

import numpy as np
import pandas as pd
import pytest

from src.core.splits import walk_forward_splits


def test_walk_forward_basic_shape():
    ts = pd.date_range("2024-01-01", periods=100, freq="h")
    splits = list(walk_forward_splits(ts, n_splits=4, embargo_hours=0))
    assert len(splits) == 4
    for s in splits:
        assert s.train_idx.max() < s.test_idx.min()  # no overlap, no future leakage
        assert len(s.train_idx) > 0
        assert len(s.test_idx) > 0


def test_walk_forward_train_window_expands():
    ts = pd.date_range("2024-01-01", periods=100, freq="h")
    splits = list(walk_forward_splits(ts, n_splits=4, embargo_hours=0))
    train_sizes = [len(s.train_idx) for s in splits]
    assert train_sizes == sorted(train_sizes)  # monotonically expanding


def test_embargo_creates_gap():
    ts = pd.date_range("2024-01-01", periods=100, freq="h")
    splits = list(walk_forward_splits(ts, n_splits=4, embargo_hours=5))
    for s in splits:
        gap = (s.test_start - s.train_end).total_seconds() / 3600
        assert gap >= 5


def test_rejects_insufficient_data():
    ts = pd.date_range("2024-01-01", periods=5, freq="h")
    with pytest.raises(ValueError):
        list(walk_forward_splits(ts, n_splits=10, embargo_hours=0, min_train_size=2))
