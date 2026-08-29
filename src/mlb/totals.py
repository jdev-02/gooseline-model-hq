"""Total-runs model: same data and same machinery as the margin model, with
every feature re-oriented from a difference to a sum.

The margin model asks "who is better and by how much." Totals ask "how much
scoring will this environment produce," which is a different question with a
different answer set: park and the two starters dominate, and team strength
barely matters at all. So the features here are sums (both offenses, both
staffs, the yard) rather than home-minus-away differences.

Target: y_total = home_score + away_score.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from src.mlb.features import _Ewma, _decay, _fip_num, FIP_LEAGUE_DEFAULT, STRIKE_PCT_DEFAULT

TOTAL_FEATURE_COLS = [
    "off_sum",          # both offenses: EWMA runs scored per game, summed
    "def_sum",          # both run-prevention units: EWMA runs allowed, summed
    "pressure_sum",     # baserunners per inning created by both clubs
    "sp_fip_sum",       # both listed starters, shrunk to the league mean
    "sp_k9_sum",        # strikeouts suppress balls in play, so they suppress runs
    "bp_fip_sum",       # both bullpens
    "bp_workload_sum",  # tired pens give up runs
    "log_park_factor",  # the single largest environmental term for totals
    "day_night",
    "roof_or_dh",       # placeholder for future weather; DH-game-2 fatigue
]

LEAGUE_TOTAL_DEFAULT = 8.8
K9_DEFAULT = 8.5


def build_total_features(df, team_lookup=None, pitcher_stats=None, park_fn=None,
                         form_half_life_games=25, pressure_half_life=15,
                         pitch_half_life_starts=6, bp_half_life=30,
                         fip_shrink_k=5.0):
    """Causal single pass, identical discipline to the margin features:
    read state, write the row, then update state from that row's outcome."""
    df = df.copy()
    n = len(df)
    d_form = _decay(form_half_life_games)
    d_press = _decay(pressure_half_life)
    d_sp = _decay(pitch_half_life_starts)
    d_bp = _decay(bp_half_life)

    rs, ra = {}, {}          # team -> EWMA runs scored / allowed
    press_off = {}           # team -> EWMA baserunners per inning created
    sp_fip, sp_k9, sp_last = {}, {}, {}
    bp_fip, bp_ip_log = {}, {}
    lg_num = lg_ip = lg_er = lg_k = 0.0

    pit_by_game = {}
    if pitcher_stats is not None and len(pitcher_stats):
        for key, grp in pitcher_stats.groupby("game_id"):
            pit_by_game[key] = grp

    cols = {c: np.zeros(n) for c in TOTAL_FEATURE_COLS}

    for i, r in enumerate(df.itertuples(index=False)):
        h, a = r.home_team, r.away_team
        lg_c = (lg_er / lg_ip * 9.0 - lg_num / lg_ip) if lg_ip > 100 else 3.10
        lg_fip = FIP_LEAGUE_DEFAULT if lg_ip <= 100 else (lg_num / lg_ip + lg_c)
        lg_k9 = (lg_k / lg_ip * 9.0) if lg_ip > 100 else K9_DEFAULT

        half = LEAGUE_TOTAL_DEFAULT / 2.0
        cols["off_sum"][i] = rs.get(h, half) + rs.get(a, half)
        cols["def_sum"][i] = ra.get(h, half) + ra.get(a, half)
        cols["pressure_sum"][i] = press_off.get(h, 1.4) + press_off.get(a, 1.4)

        def sp(pid):
            e = sp_fip.get(pid)
            fip = (lg_fip if e is None else
                   (e.n * (e.v + lg_c) + fip_shrink_k * lg_fip) / (e.n + fip_shrink_k))
            k = sp_k9.get(pid)
            k9 = (lg_k9 if k is None else
                  (k.n * k.v + fip_shrink_k * lg_k9) / (k.n + fip_shrink_k))
            return fip, k9

        hf, hk = sp(r.home_sp_id)
        af, ak = sp(r.away_sp_id)
        cols["sp_fip_sum"][i] = hf + af
        cols["sp_k9_sum"][i] = hk + ak
        bh, ba = bp_fip.get(h), bp_fip.get(a)
        cols["bp_fip_sum"][i] = ((lg_fip if bh is None else bh.v + lg_c) +
                                 (lg_fip if ba is None else ba.v + lg_c))

        def bp_load(team):
            q = bp_ip_log.get(team, ())
            return sum(ip for gd, ip in q if 0 < (r.gameday - gd).days <= 3)
        cols["bp_workload_sum"][i] = bp_load(h) + bp_load(a)

        cols["log_park_factor"][i] = park_fn(r.venue_id, r.season) if park_fn else 0.0
        cols["day_night"][i] = 1.0 if r.day_night == "night" else 0.0
        cols["roof_or_dh"][i] = 1.0 if int(getattr(r, "game_number", 1)) > 1 else 0.0

        if pd.isna(r.result):
            continue
        # ---- update state from this game ----
        hs, as_ = float(r.home_score), float(r.away_score)
        rs[h] = d_form * rs.get(h, half) + (1 - d_form) * hs
        rs[a] = d_form * rs.get(a, half) + (1 - d_form) * as_
        ra[h] = d_form * ra.get(h, half) + (1 - d_form) * as_
        ra[a] = d_form * ra.get(a, half) + (1 - d_form) * hs

        if team_lookup is not None:
            for team in (h, a):
                obs = team_lookup.get((r.game_id, team))
                if obs is not None:
                    press_off[team] = (d_press * press_off.get(team, 1.4)
                                       + (1 - d_press) * float(obs.brpi_off))

        grp = pit_by_game.get(r.game_id)
        if grp is not None:
            for p in grp.itertuples(index=False):
                ip = float(p.ip or 0)
                if ip <= 0:
                    continue
                num = _fip_num(p.hr or 0, p.bb or 0, p.hbp or 0, p.so or 0)
                lg_num += num
                lg_ip += ip
                lg_er += float(p.er or 0)
                lg_k += float(p.so or 0)
                if p.is_starter:
                    sp_fip.setdefault(p.pitcher_id, _Ewma(d_sp, 0.0)).push(num / ip)
                    sp_k9.setdefault(p.pitcher_id, _Ewma(d_sp, K9_DEFAULT)).push(
                        float(p.so or 0) / ip * 9.0)
                    sp_last[p.pitcher_id] = r.gameday
                else:
                    bp_fip.setdefault(p.team, _Ewma(d_bp, 0.0)).push(num / ip)
                    bp_ip_log.setdefault(p.team, deque(maxlen=60)).append((r.gameday, ip))

    for c, v in cols.items():
        df[c] = v
    df["y"] = (df["home_score"] + df["away_score"]).astype(float)
    return df


# --------------------------------------------------------------------------
# Baselines. A totals model has to beat the yard and the league average before
# anyone should believe it knows anything about pitching.
# --------------------------------------------------------------------------

class LeagueMeanTotal:
    name = "league_mean_total"

    def fit(self, df):
        y = df["y"].dropna()
        self.mu_, self.sigma_ = float(y.mean()), float(y.std())
        return self

    def predict_dist(self, df):
        n = len(df)
        return np.full(n, self.mu_), np.full(n, self.sigma_)


class ParkAdjustedMeanTotal:
    """League mean scaled by the venue's expanding park factor. This is the
    honest floor: it uses no pitching or batting information at all."""
    name = "park_adjusted_mean"

    def fit(self, df):
        y = df["y"].dropna()
        self.mu_ = float(y.mean())
        resid = y - self.mu_ * np.exp(df.loc[y.index, "log_park_factor"])
        self.sigma_ = float(resid.std())
        return self

    def predict_dist(self, df):
        mu = self.mu_ * np.exp(df["log_park_factor"].values)
        return mu, np.full(len(df), self.sigma_)


def prob_over(mu, sigma, strike):
    """P(total > strike). Kalshi's floor_strike is already the .5 line, so no
    continuity correction is needed: floor_strike 8.5 means 9 runs or more."""
    from scipy.stats import norm
    return 1.0 - norm.cdf((strike - np.asarray(mu)) / np.asarray(sigma))
