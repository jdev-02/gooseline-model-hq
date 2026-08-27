"""Causality tests for MLB features and park factors. Synthetic data only."""
import numpy as np
import pandas as pd
import pytest

from src.mlb.features import build_features, MLB_FEATURE_COLS, TIER_A_COLS
from src.mlb.park import build_park_factors, park_lookup


def _games(seed=0, seasons=(2021, 2022, 2023), n_teams=6, days=40):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    rows, pk = [], 1000
    for s in seasons:
        for d in range(days):
            order = rng.permutation(n_teams)
            for k in range(0, n_teams, 2):
                h, a = teams[order[k]], teams[order[k + 1]]
                hs, as_ = rng.poisson(4.6), rng.poisson(4.3)
                pk += 1
                rows.append(dict(
                    game_id=str(pk), game_pk=pk, season=s, gameday=pd.Timestamp(f"{s}-04-01") + pd.Timedelta(days=d),
                    day_index=d, game_number=1, home_team=h, away_team=a, venue_id=int(h[1:]) + 1,
                    home_score=hs, away_score=as_, result=float(hs - as_), played=True,
                    day_night="night", home_rest=1.0, away_rest=1.0, div_game=0,
                    home_hits=hs + 4, away_hits=as_ + 4, home_lob=5, away_lob=6,
                    home_runs_late=hs // 3, away_runs_late=as_ // 3, innings_played=9,
                    home_sp_id=100 + int(h[1:]), away_sp_id=100 + int(a[1:])))
    return pd.DataFrame(rows)


def test_feature_columns_present_and_finite():
    df = build_features(_games())
    for c in MLB_FEATURE_COLS:
        if c.startswith("kalman"):
            continue  # added by TeamKalman.run, not build_features
        assert c in df.columns and np.isfinite(df[c]).all(), c


def test_features_are_causal():
    """Changing a game's outcome must not change any feature of that game or
    any earlier game."""
    g = _games()
    base = build_features(g)
    g2 = g.copy()
    j = 150
    g2.loc[j, ["home_score", "away_score", "result"]] = [15, 0, 15.0]
    alt = build_features(g2)
    for c in TIER_A_COLS[2:]:  # kalman cols are added elsewhere
        np.testing.assert_allclose(base[c].values[:j + 1], alt[c].values[:j + 1], err_msg=c)
    assert not np.allclose(base["rd_ewma_diff"].values[j + 1:], alt["rd_ewma_diff"].values[j + 1:])


def test_unplayed_games_get_features_but_no_state_update():
    g = _games()
    g.loc[g.index[-3:], ["home_score", "away_score", "result", "played"]] = [np.nan, np.nan, np.nan, False]
    df = build_features(g)
    assert np.isfinite(df["rd_ewma_diff"].iloc[-1])
    assert np.isnan(df["y"].iloc[-1])


def test_park_factor_uses_only_prior_seasons():
    g = _games()
    pf = build_park_factors(g, window_seasons=3)
    first = pf[pf.season == g.season.min()]
    assert (first.pf_shrunk == 1.0).all() and (first.n_games == 0).all()
    # Blowing up run totals in 2023 must not change the 2023 factor (only 2024+)
    g2 = g.copy()
    g2.loc[g2.season == 2023, "home_score"] += 20
    pf2 = build_park_factors(g2, window_seasons=3)
    pd.testing.assert_frame_equal(pf[pf.season <= 2023].reset_index(drop=True),
                                  pf2[pf2.season <= 2023].reset_index(drop=True))
    look = park_lookup(pf)
    assert look(999, 2023) == 0.0
