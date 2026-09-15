"""Skill block (offensive K%, ISO, staff K%): orientation and causality.
Synthetic data only."""
from collections import namedtuple

import numpy as np

from src.mlb.features import build_features, SKILL_COLS, MLB_FEATURE_COLS
from tests.test_mlb_features import _games

Obs = namedtuple("Obs", "brpi_off brpi_def lob_rate_off so pa ab doubles triples home_runs p_so p_bf")


def _lookup(games, star="T0"):
    """T0 strikes out rarely, hits for power, and its staff strikes everyone
    out. Everyone else is league average."""
    out = {}
    for r in games.itertuples(index=False):
        for team in (r.home_team, r.away_team):
            if team == star:
                obs = Obs(1.2, 1.0, 0.7, so=4, pa=38, ab=34, doubles=3, triples=1, home_runs=2, p_so=12, p_bf=36)
            else:
                obs = Obs(1.2, 1.0, 0.7, so=9, pa=38, ab=34, doubles=1, triples=0, home_runs=1, p_so=8, p_bf=36)
            out[(r.game_id, team)] = obs
    return out


def test_skill_cols_exist_finite_and_zero_without_boxscores():
    df = build_features(_games())
    for c in SKILL_COLS:
        assert c in df.columns
        assert np.isfinite(df[c]).all()
        assert (df[c] == 0).all(), f"{c} should be neutral with no team stats"


def test_skill_cols_oriented_positive_favors_home():
    g = _games()
    df = build_features(g, team_lookup=_lookup(g))
    # first game of the star is neutral: nothing observed yet
    first = df[(df.home_team == "T0") | (df.away_team == "T0")].iloc[0]
    for c in SKILL_COLS:
        assert first[c] == 0, c
    later = df[df.season == 2023]
    home = later[later.home_team == "T0"]
    away = later[later.away_team == "T0"]
    assert len(home) and len(away)
    for c in SKILL_COLS:
        assert (home[c] > 0).all(), f"{c} should favor the star at home"
        assert (away[c] < 0).all(), f"{c} should count against the star's opponent"


def test_skill_cols_are_causal():
    g = _games()
    lk = _lookup(g)
    base = build_features(g, team_lookup=lk)
    # flip the last game's stats: nothing earlier may move
    last = g.iloc[-1]
    lk2 = dict(lk)
    lk2[(last.game_id, last.home_team)] = Obs(1.2, 1.0, 0.7, 20, 38, 34, 0, 0, 0, 0, 36)
    alt = build_features(g, team_lookup=lk2)
    for c in SKILL_COLS:
        np.testing.assert_allclose(base[c].values, alt[c].values)


def test_skill_cols_not_shipped_unless_gated():
    """They enter MLB_FEATURE_COLS only by an explicit edit after the
    walk-forward gate passes; this pins the current state either way so a
    silent change is caught."""
    shipped = [c for c in SKILL_COLS if c in MLB_FEATURE_COLS]
    assert shipped in ([], SKILL_COLS)
