"""Walk-forward cross-validation with embargo.

The NPS coursework treats data as IID and uses train_test_split. For
time-series this leaks future information into training and overstates
performance. Walk-forward fits the model the same way it will actually be
used in production: train on everything up to time T, predict T+gap.

Embargo: a gap (in hours) between the end of the training window and the
start of the test window. Prevents intraday autocorrelation around the
boundary from inflating performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
import numpy as np
import pandas as pd


@dataclass
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_splits(
    timestamps: pd.DatetimeIndex | pd.Series,
    n_splits: int = 5,
    embargo_hours: int = 24,
    min_train_size: int | None = None,
) -> Iterator[Split]:
    """Yield expanding-window walk-forward splits.

    Args:
        timestamps: ordered timestamps of every row in the dataset.
        n_splits: number of test folds.
        embargo_hours: gap between train end and test start.
        min_train_size: minimum #rows in the first fold's training set.
            Defaults to total / (n_splits + 1).

    Yields:
        Split with integer index arrays into the original data.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if not ts.is_monotonic_increasing:
        order = np.argsort(ts.values)
        ts = ts[order]
        # Caller is responsible for re-ordering their own arrays to match.

    n = len(ts)
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if min_train_size is None:
        min_train_size = n // (n_splits + 1)
    if min_train_size < 1:
        raise ValueError("min_train_size must be >= 1")

    test_size = (n - min_train_size) // n_splits
    if test_size < 1:
        raise ValueError(
            f"Not enough data for {n_splits} splits "
            f"with min_train_size={min_train_size} (n={n})."
        )

    embargo = pd.Timedelta(hours=embargo_hours)

    for k in range(n_splits):
        train_end_pos = min_train_size + k * test_size
        test_start_pos = train_end_pos
        test_end_pos = min(test_start_pos + test_size, n)

        train_end_ts = ts[train_end_pos - 1]
        # Apply embargo: shift test start forward until it's >= train_end + embargo
        cutoff = train_end_ts + embargo
        while test_start_pos < n and ts[test_start_pos] < cutoff:
            test_start_pos += 1

        if test_start_pos >= test_end_pos:
            # Embargo consumed the entire test window. Skip this fold.
            continue

        train_idx = np.arange(0, train_end_pos)
        test_idx = np.arange(test_start_pos, test_end_pos)

        yield Split(
            train_idx=train_idx,
            test_idx=test_idx,
            train_end=ts[train_end_pos - 1],
            test_start=ts[test_start_pos],
            test_end=ts[test_end_pos - 1],
        )
