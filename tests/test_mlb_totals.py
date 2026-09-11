"""Totals-feature tests: umpire and weather inputs. Synthetic data only."""
import numpy as np
import pandas as pd

from src.mlb.totals import build_total_features, TOTAL_FEATURE_COLS, STUDY_COLS, _wind_signed
from tests.test_mlb_features import _games


def _wx_games():
    g = _games()
    rng = np.random.default_rng(1)
    g["hp_umpire_id"] = rng.integers(1, 5, len(g))
    g["temp_f"] = rng.integers(45, 95, len(g)).astype(float)
    g["wind_mph"] = rng.integers(0, 20, len(g)).astype(float)
    g["wind_dir"] = rng.choice(["Out To CF", "In From LF", "L To R", "Calm"], len(g))
    g["condition"] = rng.choice(["Clear", "Cloudy", "Roof Closed", "Dome"], len(g))
    return g


def test_wind_sign_convention():
    assert _wind_signed(10, "Out To CF") == 10.0
    assert _wind_signed(10, "In From RF") == -10.0
    assert _wind_signed(10, "L To R") == 0.0
    assert _wind_signed(None, "Out To CF") == 0.0
    assert _wind_signed(float("nan"), "Out To CF") == 0.0


def test_columns_present_and_finite_with_and_without_weather():
    for g in (_games(), _wx_games()):
        df = build_total_features(g)
        for c in TOTAL_FEATURE_COLS + STUDY_COLS:
            assert c in df and np.isfinite(df[c]).all(), c
    # Without the hydrated columns the new terms are silently zero, never NaN.
    df = build_total_features(_games())
    assert (df[["ump_runs", "temp_c70", "wind_out", "roof_closed"]] == 0).all().all()


def test_unplayed_game_falls_back_to_venue_climatology_and_roof_history():
    g = _wx_games()
    # Venue 1 is always open and hot; venue 2 is always under a roof.
    g.loc[g.venue_id == 1, ["condition", "temp_f"]] = ["Clear", 90.0]
    g.loc[g.venue_id == 2, "condition"] = "Roof Closed"
    last = g.index[-2:]
    g.loc[last, ["home_score", "away_score", "result", "played"]] = [np.nan, np.nan, np.nan, False]
    g.loc[last, ["condition", "temp_f", "wind_mph", "wind_dir"]] = np.nan
    g.loc[last[0], "venue_id"] = 1
    g.loc[last[1], "venue_id"] = 2
    df = build_total_features(g)
    assert df.loc[last[0], "roof_closed"] == 0 and abs(df.loc[last[0], "temp_c70"] - 2.0) < 1e-9
    assert df.loc[last[1], "roof_closed"] == 1 and df.loc[last[1], "temp_c70"] == 0


def test_roof_zeroes_weather_terms():
    df = build_total_features(_wx_games())
    closed = df[df.roof_closed == 1]
    assert len(closed) > 0
    assert (closed.temp_c70 == 0).all() and (closed.wind_out == 0).all()
    open_ = df[df.roof_closed == 0]
    assert (open_.temp_c70 != 0).any() and (open_.wind_out != 0).any()


def test_umpire_effect_is_causal_and_shrunk():
    g = _wx_games()
    df = build_total_features(g)
    # First appearance of every umpire carries no information.
    first_idx = g.groupby("hp_umpire_id").head(1).index
    assert (df.loc[first_idx, "ump_runs"] == 0).all()
    # Changing an outcome must not move any earlier or same-game umpire value.
    g2 = g.copy()
    g2.loc[g2.index[200], ["home_score", "away_score"]] = [25, 20]
    g2.loc[g2.index[200], "result"] = 5.0
    df2 = build_total_features(g2)
    pd.testing.assert_series_equal(df.ump_runs.iloc[:201], df2.ump_runs.iloc[:201])
    assert (df2.ump_runs.iloc[201:] != df.ump_runs.iloc[201:]).any()
    # A single blow-out cannot move the umpire number by more than its own
    # shrunk share of the residual.
    assert df2.ump_runs.abs().max() < 5.0
