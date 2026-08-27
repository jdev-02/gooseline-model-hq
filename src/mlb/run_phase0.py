"""Phase 0 for MLB: Kalman tuning, baselines, linear walk-forward, calibration.

  uv run python -m src.mlb.run_phase0 [--tier-a] [--quick]

Writes docs/baselines.md rows and figures/mlb/phase0/*. Tune on 2022 only;
headline test seasons 2023-2025.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.core.eval import brier_score, reliability_diagram, assert_no_leakage, regression_report
from src.core.kalman import TeamKalman, tune_kalman
from src.core.walkforward import walk_forward, evaluate, tune
from src.mlb.baselines import HomeAlwaysBaseline, EloLiteBaseline, p_home_from_dist
from src.mlb.compile import load_games, load_team_game_stats, load_pitcher_game_stats, DATA
from src.mlb.features import build_features, TIER_A_COLS, MLB_FEATURE_COLS
from src.mlb.park import build_park_factors, park_lookup

FIG = Path("figures/mlb/phase0")
VAL_SEASON = 2022
TEST_SEASONS = (2023, 2024, 2025)
MLB_KALMAN_FIXED = dict(step_col="day_index", init_hfa=0.2, init_hfa_var=0.25,
                        init_var=1.0, hfa_q=1e-5)


def frame(tier_a_only=False):
    games = load_games(keep_unplayed=True)
    park = build_park_factors(games)
    park.to_csv(DATA / "park_factors.csv", index=False)
    team_lookup = None if tier_a_only else load_team_game_stats()
    pitchers = None if tier_a_only else load_pitcher_game_stats()
    return build_features(games, team_lookup, pitchers, park_lookup(park)), games


def calib_report(preds, label, recal=1.0):
    p = p_home_from_dist(preds["mu"].values, preds["sigma"].values, recal)
    y = (preds["y"].values > 0).astype(float)
    b = brier_score(y, p)
    fig, rows = reliability_diagram(y, p, n_bins=10, label=label)
    fig.savefig(FIG / f"reliability_{label}.png", dpi=120)
    dev = max(abs(r[2] - r[3]) for r in rows if r[4] >= 50)
    return b, dev, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-a", action="store_true", help="schedule-only features")
    ap.add_argument("--quick", action="store_true", help="small Kalman grid")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: None

    df, games = frame(args.tier_a)
    cols = TIER_A_COLS if args.tier_a else MLB_FEATURE_COLS

    grid = ({"obs_var": [16.0, 20.0], "step_q": [0.005, 0.02],
             "season_inflate": [1.0], "season_revert": [0.75]} if args.quick else
            {"obs_var": [17.0, 19.0, 21.0], "step_q": [0.0005, 0.001, 0.002, 0.005],
             "season_inflate": [0.1, 0.25, 0.5, 1.0], "season_revert": [0.3, 0.45, 0.6, 0.75]})
    best_kf, kf_table = tune_kalman(df, grid=grid, train_end_season=VAL_SEASON, **MLB_KALMAN_FIXED)
    print("Kalman (empirical Bayes on <=2022):", best_kf)
    kf = TeamKalman(**best_kf, **MLB_KALMAN_FIXED)
    df = kf.run(df)
    print(f"Estimated HFA (runs): {kf.final_hfa_:.3f}")
    print("Top 8 ratings entering today:\n", kf.final_ratings_.head(8).round(3).to_string())
    kf.final_ratings_.head(15).plot.barh(figsize=(6, 5), title="MLB Kalman ratings (runs vs avg)")
    plt.gca().invert_yaxis(); plt.tight_layout(); plt.savefig(FIG / "kalman_ratings_2026.png", dpi=120); plt.close()
    (DATA / "kalman_params.json").write_text(json.dumps({**best_kf, **MLB_KALMAN_FIXED}, indent=2))

    played = df[df["y"].notna()]
    results = []

    # --- baselines on 2023-2025 (fit on strictly prior seasons each year) ---
    ha_rows, elo_rows = [], []
    for s in TEST_SEASONS:
        train, test = played[played["season"] < s], played[played["season"] == s]
        ha = HomeAlwaysBaseline().fit(train)
        mu, sg = ha.predict_dist(test)
        ha_rows.append(test[["game_id", "season", "y"]].assign(mu=mu, sigma=sg))
        elo = EloLiteBaseline().fit(train)
        gaps = elo.gaps_for(played[played["season"] <= s])[-len(test):]
        mu, sg = elo.predict_dist_from_gaps(gaps)
        elo_rows.append(test[["game_id", "season", "y"]].assign(
            mu=mu, sigma=sg, p_elo=elo.predict_proba_from_gaps(gaps)))
    ha_p, elo_p = pd.concat(ha_rows), pd.concat(elo_rows)
    for name, pr in (("home_always", ha_p), ("elo_lite", elo_p)):
        m = evaluate(pr)
        if name == "elo_lite":
            b = brier_score((pr["y"] > 0).astype(float), pr["p_elo"])
            dev = np.nan
        else:
            b, dev, _ = calib_report(pr, name)
        results.append({"model": name, **m, "brier": b, "max_calib_dev": dev})
        print(name, m, "brier", round(b, 4))

    # --- linear model: tune on 2022, test 2023-2025 ---
    best, table = tune(df, cols, VAL_SEASON, lam_grid=[10.0, 100.0, 1000.0],
                       half_life_grid=[2.0, 4.0, np.inf], step_col="day_index",
                       refit_every=7, min_train=2000)
    print("linear tuned on 2022:", best)
    val = walk_forward(df, cols, VAL_SEASON, lam=best["lam"], half_life_seasons=best["half_life_seasons"],
                       step_col="day_index", refit_every=7, min_train=2000)
    # RECAL_SCALE: minimize binary NLL on validation
    yv = (val["y"] > 0).astype(float).values
    scales = np.linspace(0.7, 1.5, 33)
    nll = [-(yv * np.log(p := np.clip(p_home_from_dist(val["mu"], val["sigma"], c), 1e-6, 1 - 1e-6))
             + (1 - yv) * np.log(1 - p)).mean() for c in scales]
    recal = float(scales[int(np.argmin(nll))])
    print("RECAL_SCALE (fit on 2022):", recal)

    lin_rows = []
    for s in TEST_SEASONS:
        pr = walk_forward(df, cols, s, lam=best["lam"], half_life_seasons=best["half_life_seasons"],
                          step_col="day_index", refit_every=7, min_train=2000)
        assert_no_leakage(pr["y"], pr["mu"], r2_ceiling=0.60)
        lin_rows.append(pr)
    lin_p = pd.concat(lin_rows)
    m = evaluate(lin_p)
    rep, fig = regression_report(lin_p["y"], lin_p["mu"], label="linear_2023_2025", units="runs", show=False)
    fig.savefig(FIG / "residuals_2023_2025.png", dpi=120)
    b, dev, rows = calib_report(lin_p, "linear", recal)
    results.append({"model": "linear" + ("_tierA" if args.tier_a else ""), **m, "brier": b, "max_calib_dev": dev})
    print("linear", m, "brier", round(b, 4), "max calib dev", round(dev, 3))
    print("reliability bins (stated, empirical, n):", [(round(r[2], 2), round(r[3], 2), r[4]) for r in rows if r[4]])

    # embargo robustness
    emb = pd.concat(walk_forward(df, cols, s, lam=best["lam"], half_life_seasons=best["half_life_seasons"],
                                 step_col="day_index", refit_every=7, min_train=2000, embargo_steps=1)
                    for s in TEST_SEASONS)
    print("embargo=1 nll:", round(evaluate(emb)["nll"], 4), "vs", round(m["nll"], 4))

    out = pd.DataFrame(results)
    out.to_csv(FIG / "phase0_results.csv", index=False)
    print("\n", out.round(4).to_string(index=False))
    cfg = {"kalman": {**best_kf, **MLB_KALMAN_FIXED}, "lam": best["lam"],
           "half_life_seasons": best["half_life_seasons"], "recal_scale": recal,
           "feature_cols": cols}
    (DATA / "model_config.json").write_text(json.dumps(cfg, indent=2, default=float))
    gate = (m["nll"] < min(r["nll"] for r in results[:2]) and
            b < min(r["brier"] for r in results[:2]) and dev < 0.10)
    print("\nGATE (beats HomeAlways & EloLite on NLL+Brier, calib<0.10):", "PASS" if gate else "FAIL")


if __name__ == "__main__":
    main()
