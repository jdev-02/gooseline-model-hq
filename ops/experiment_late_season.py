"""Does the model get worse once teams are locked into playoff seeding?

Late September is full of "dead rubber" games: a division winner rests
starters, an eliminated team calls up September rookies, a bullpen gets
run into the ground because the standings no longer matter. None of that
is a feature this model sees. This checks, on held-out data, whether the
walk-forward gate the site's numbers already passed actually holds up in
the last two weeks of the season, using exactly the frozen configs the
site runs today -- not a new fit.

  uv run python ops/experiment_late_season.py

"Late" = the last 14 calendar days of each season's regular-season slate
(by day_index, per season, so postseason games -- not in games.csv --
never leak in). This is a proxy for "seeding is mostly decided," not the
real thing (an exact per-day clinch/eliminate reconstruction from
standings), but it is the cheap, honest first check: if the model is
broken by low-stakes baseball, this window is where it would show.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.eval import brier_score  # noqa: E402
from src.core.kalman import TeamKalman  # noqa: E402
from src.core.walkforward import walk_forward, evaluate  # noqa: E402
from src.mlb.baselines import p_home_from_dist  # noqa: E402
from src.mlb.compile import DATA, load_games, load_team_game_stats, load_pitcher_game_stats  # noqa: E402
from src.mlb.features import build_features  # noqa: E402
from src.mlb.park import build_park_factors, park_lookup  # noqa: E402
from src.mlb.totals import build_total_features, TOTAL_FEATURE_COLS, NegBinomTotal  # noqa: E402

TEST_SEASONS = (2023, 2024, 2025)
LATE_DAYS = 14   # last two weeks of each season's own regular-season slate


def tag_late(df):
    end = df.groupby("season")["day_index"].transform("max")
    df = df.copy()
    df["days_before_end"] = end - df["day_index"]
    df["late"] = df["days_before_end"] < LATE_DAYS
    return df


def split_metrics(y, p, late, label):
    rows = []
    for name, mask in (("early", ~late), ("late (last 14 days)", late)):
        m = mask.values if hasattr(mask, "values") else mask
        yy, pp = y[m], p[m]
        if len(yy) < 30:
            continue
        b = brier_score(yy, pp)
        nll = -(yy * np.log(np.clip(pp, 1e-6, 1 - 1e-6))
                + (1 - yy) * np.log(np.clip(1 - pp, 1e-6, 1 - 1e-6))).mean()
        rows.append({"where": label, "window": name, "n": int(m.sum()),
                     "brier": b, "nll": nll, "actual_rate": float(yy.mean()),
                     "model_rate": float(pp.mean())})
    return rows


def main():
    mc = json.loads((DATA / "model_config.json").read_text())
    tc = json.loads((DATA / "totals_config.json").read_text())

    games = load_games(keep_unplayed=True, first_season=tc.get("first_season", 2008))
    park = build_park_factors(games)
    stats, pitch = load_team_game_stats(), load_pitcher_game_stats()

    # ---- margin model (moneyline) ----
    mdf = build_features(games, stats, pitch, park_lookup(park))
    mdf = TeamKalman(**mc["kalman"]).run(mdf)
    mp = pd.concat(walk_forward(mdf, mc["feature_cols"], s, lam=mc["lam"],
                                half_life_seasons=mc["half_life_seasons"],
                                step_col="day_index", refit_every=7, min_train=2000)
                   for s in TEST_SEASONS)
    mp = tag_late(mp)
    p_home = p_home_from_dist(mp["mu"].values, mp["sigma"].values, mc["recal_scale"])
    y_home = (mp["y"].values > 0).astype(float)
    rows = split_metrics(y_home, p_home, mp["late"], "moneyline")
    for name, mask in (("early", ~mp["late"]), ("late", mp["late"])):
        r = (mp["y"] - mp["mu"])[mask.values]
        rows[-1 if name == "late" else -2]["rmse_margin"] = float(np.sqrt((r ** 2).mean()))

    # ---- totals model ----
    tdf = build_total_features(games, stats, pitch, park_lookup(park))
    tp = pd.concat(walk_forward(tdf, TOTAL_FEATURE_COLS, s, step_col="day_index",
                               refit_every=7, min_train=2000,
                               half_life_seasons=tc["half_life_seasons"],
                               model_factory=lambda: NegBinomTotal())
                  for s in TEST_SEASONS)
    tp = tag_late(tp)
    # Need the fitted NegBinomTotal per season to call .prob_over on held-out
    # rows; walk_forward doesn't return the model, so refit once per season
    # on everything strictly before it, same causal boundary the site uses.
    total_rows = []
    for line in (7.5, 8.5, 9.5):
        po, y_over, late = [], [], []
        for s in TEST_SEASONS:
            tr = tdf[(tdf["season"] < s) & tdf["y"].notna()]
            te = tp[tp["season"] == s]
            te_full = tdf.loc[te.index]
            nb = NegBinomTotal().fit(tr[TOTAL_FEATURE_COLS].values, tr["y"].values)
            po.append(nb.prob_over(te_full[TOTAL_FEATURE_COLS].values, line))
            y_over.append((te["y"].values > line).astype(float))
            late.append(te["late"].values)
        po, y_over, late = np.concatenate(po), np.concatenate(y_over), np.concatenate(late)
        for name, mask in (("early", ~late), ("late (last 14 days)", late)):
            if mask.sum() < 30:
                continue
            b = brier_score(y_over[mask], po[mask])
            total_rows.append({"where": f"total O/U {line}", "window": name, "n": int(mask.sum()),
                              "brier": b, "actual_rate": float(y_over[mask].mean()),
                              "model_rate": float(po[mask].mean())})

    out = pd.DataFrame(rows + total_rows)
    pd.set_option("display.width", 140)
    print(out.round(4).to_string(index=False))

    print(f"\n{LATE_DAYS}-day 'late' window sizes by season (days_before_end < {LATE_DAYS}):")
    print(mp.groupby("season")["late"].sum().to_string())

    print("\nHow to read this: brier/nll going UP and actual_rate departing from "
          "model_rate in the 'late' row versus 'early' is the signal the concern "
          "is real. A few thousandths of Brier is sampling noise at this n; "
          "0.02+ or a visible actual-vs-model gap is not.")


if __name__ == "__main__":
    main()
