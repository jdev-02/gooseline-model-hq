"""Generalized Kalman must be bit-identical to upstream on NFL weekly data,
and must scale process noise by elapsed days on a daily step column.
Synthetic data only."""
import numpy as np
import pandas as pd
import pytest

from src.core.kalman import TeamKalman
from tests._upstream_kalman import TeamKalman as UpstreamKalman


def _synthetic_games(seed=0, n_teams=8, seasons=(2020, 2021, 2022), weeks=10):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    rows = []
    for s in seasons:
        for w in range(1, weeks + 1):
            order = rng.permutation(n_teams)
            for k in range(0, n_teams, 2):
                h, a = teams[order[k]], teams[order[k + 1]]
                rows.append({"season": s, "week": w, "home_team": h, "away_team": a,
                             "result": float(rng.normal(2, 12)), "location": "Home"})
    return pd.DataFrame(rows)


def test_nfl_weekly_parity_with_upstream():
    df = _synthetic_games()
    params = dict(obs_var=150.0, season_inflate=8.0, season_revert=0.7)
    new = TeamKalman(weekly_q=0.8, **params).run(df)
    old = UpstreamKalman(weekly_q=0.8, **params).run(df)
    np.testing.assert_allclose(new["kalman_diff"], old["kalman_diff"], atol=1e-12)
    np.testing.assert_allclose(new["kalman_var"], old["kalman_var"], atol=1e-12)
    np.testing.assert_allclose(new["kalman_hfa"], old["kalman_hfa"], atol=1e-12)


def test_daily_step_scales_process_noise_by_elapsed_days():
    df = pd.DataFrame({
        "season": [2025] * 3, "day_index": [0, 1, 4],
        "home_team": ["A", "A", "A"], "away_team": ["B", "B", "B"],
        "result": [np.nan, np.nan, np.nan]})
    kf = TeamKalman(obs_var=20.0, step_q=0.01, init_var=1.0, init_hfa=0.2,
                    init_hfa_var=0.25, step_col="day_index")
    out = kf.run(df)
    v = out["kalman_var"].values
    # gap of 1 day adds 2*q (both teams), gap of 3 days adds 6*q
    assert v[1] - v[0] == pytest.approx(2 * 0.01)
    assert v[2] - v[1] == pytest.approx(6 * 0.01)


def test_doubleheader_same_day_injects_no_noise():
    df = pd.DataFrame({
        "season": [2025] * 2, "day_index": [3, 3],
        "home_team": ["A", "A"], "away_team": ["B", "B"],
        "result": [np.nan, np.nan]})
    out = TeamKalman(step_q=0.5, step_col="day_index").run(df)
    assert out["kalman_var"].iloc[0] == out["kalman_var"].iloc[1]
