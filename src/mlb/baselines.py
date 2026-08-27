"""The named baselines every MLB model must beat (docs/baselines.md).

HomeAlways      : mu = mean train margin, p_home = train home-win rate.
EloLite         : plain Elo on wins, K=4, home edge 24 Elo points, with one
                  OLS scale from Elo gap to run margin fit on train.
MarketImplied   : de-vigged Kalshi mid, only where a logged price exists.

All expose fit(df) / predict_dist(df) / predict_proba_home(df) with the same
(mu, sigma) contract as src.core.models so the walk-forward harness and the
Brier/reliability gates treat them identically to real models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


class HomeAlwaysBaseline:
    name = "home_always"

    def fit(self, df):
        y = df["y"].dropna()
        self.mu_ = float(y.mean())
        self.sigma_ = float(y.std())
        self.p_ = float((y > 0).mean())
        return self

    def predict_dist(self, df):
        n = len(df)
        return np.full(n, self.mu_), np.full(n, self.sigma_)

    def predict_proba_home(self, df):
        return np.full(len(df), self.p_)


class EloLiteBaseline:
    name = "elo_lite"

    def __init__(self, k=4.0, hfa=24.0):
        self.k, self.hfa = k, hfa

    def _walk(self, df):
        """Elo gap (home+hfa - away) *before* each game, updated causally."""
        r = {}
        gaps = np.empty(len(df))
        for i, g in enumerate(df.itertuples(index=False)):
            rh, ra = r.get(g.home_team, 1500.0), r.get(g.away_team, 1500.0)
            gaps[i] = rh + self.hfa - ra
            if pd.isna(g.result):
                continue
            e = 1.0 / (1.0 + 10 ** (-gaps[i] / 400.0))
            s = 1.0 if g.result > 0 else 0.0
            r[g.home_team] = rh + self.k * (s - e)
            r[g.away_team] = ra - self.k * (s - e)
        self.ratings_ = r
        return gaps

    def fit(self, df):
        gaps = self._walk(df)
        m = df["y"].notna().values
        x, y = gaps[m], df["y"].values[m]
        self.beta_ = float((x * y).sum() / (x * x).sum())
        resid = y - self.beta_ * x
        self.sigma_ = float(resid.std())
        self._train_gaps = gaps
        return self

    def gaps_for(self, df_all):
        return self._walk(df_all)

    def predict_dist_from_gaps(self, gaps):
        return self.beta_ * gaps, np.full(len(gaps), self.sigma_)

    def predict_proba_from_gaps(self, gaps):
        return 1.0 / (1.0 + 10 ** (-gaps / 400.0))


def market_implied_home(ask_home, ask_away):
    """De-vigged home probability from the two yes-asks."""
    if ask_home is None or ask_away is None:
        return np.nan
    return ask_home / (ask_home + ask_away)


def p_home_from_dist(mu, sigma, recal=1.0):
    return norm.cdf(np.asarray(mu) / (recal * np.asarray(sigma)))
