"""Expanding-window park factors. Strictly causal: the factor used for a game
on date t only consults games with gameday < t. A full-sample park factor is
the easiest leak in this dataset, so the lookup is built as a per-venue time
series and the test suite asserts the causality."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_park_factors(games, window_seasons=3, shrink_k=150.0):
    """-> DataFrame[venue_id, season, asof_date, n_games, pf_raw, pf_shrunk, log_pf]

    One row per (venue, season): the factor a game in that season sees is
    computed from the previous `window_seasons` completed seasons only. Using
    completed seasons (rather than a rolling daily window) keeps the factor
    stable within a season and makes causality trivially auditable.
    """
    g = games[games["played"]].copy()
    g["total"] = g["home_score"] + g["away_score"]
    per = g.groupby(["season", "venue_id"]).agg(runs=("total", "sum"), n=("total", "size"))
    league = g.groupby("season").agg(runs=("total", "sum"), n=("total", "size"))
    rows = []
    seasons = sorted(games["season"].unique())
    venues = games["venue_id"].dropna().unique()
    for s in seasons:
        prior = [p for p in seasons if s - window_seasons <= p < s]
        lg_r = league.loc[league.index.isin(prior), "runs"].sum()
        lg_n = league.loc[league.index.isin(prior), "n"].sum()
        lg_rate = lg_r / lg_n if lg_n else np.nan
        for v in venues:
            sub = per[per.index.get_level_values("season").isin(prior)
                      & (per.index.get_level_values("venue_id") == v)]
            n = int(sub["n"].sum())
            if n and lg_rate:
                pf_raw = (sub["runs"].sum() / n) / lg_rate
            else:
                pf_raw = np.nan
            pf_shr = 1.0 if not n or np.isnan(pf_raw) else (n * pf_raw + shrink_k) / (n + shrink_k)
            rows.append({"venue_id": v, "season": s, "asof_date": f"{s}-01-01",
                         "n_games": n, "pf_raw": pf_raw, "pf_shrunk": pf_shr,
                         "log_pf": float(np.log(pf_shr))})
    return pd.DataFrame(rows)


def park_lookup(table):
    d = {(r.venue_id, r.season): r.log_pf for r in table.itertuples(index=False)}
    return lambda venue_id, season: d.get((venue_id, season), 0.0)
