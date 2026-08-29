"""Phase 0 for the totals model: baselines, walk-forward, calibration on the
actual O/U lines the market trades.

  uv run python ops/run_totals.py [--first-season 2008]

Gate (same discipline as the margin model): beat league_mean_total and
park_adjusted_mean on NLL and on Brier at the 8.5 line across 2023-2025, with
every populated calibration decile inside 0.10. Totals are counts, so the
Gaussian fit is checked at the lines that matter rather than assumed.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.eval import brier_score, reliability_diagram, regression_report, assert_no_leakage  # noqa: E402
from src.core.walkforward import walk_forward, evaluate, tune  # noqa: E402
from src.mlb.compile import DATA, load_games, load_team_game_stats, load_pitcher_game_stats  # noqa: E402
from src.mlb.park import build_park_factors, park_lookup  # noqa: E402
from src.mlb.totals import (build_total_features, TOTAL_FEATURE_COLS, prob_over,
                            LeagueMeanTotal, ParkAdjustedMeanTotal,
                            NegBinomTotal)  # noqa: E402

FIG = Path("figures/mlb/totals")
VAL_SEASON = 2022
TEST_SEASONS = (2023, 2024, 2025)
LINES = (7.5, 8.5, 9.5)

ap = argparse.ArgumentParser()
ap.add_argument("--first-season", type=int, default=2008)
args = ap.parse_args()
FIG.mkdir(parents=True, exist_ok=True)
plt.show = lambda *a, **k: None

games = load_games(keep_unplayed=True, first_season=args.first_season)
park = build_park_factors(games)
df = build_total_features(games, load_team_game_stats(), load_pitcher_game_stats(),
                          park_lookup(park))
played = df[df["y"].notna()]
print(f"{len(played)} played games, mean total {played['y'].mean():.2f}, "
      f"sd {played['y'].std():.2f}")

results = []

# ---- baselines ----
base_preds = {}
for cls in (LeagueMeanTotal, ParkAdjustedMeanTotal):
    rows = []
    for s in TEST_SEASONS:
        tr, te = played[played["season"] < s], played[played["season"] == s]
        m = cls().fit(tr)
        mu, sg = m.predict_dist(te)
        rows.append(te[["game_id", "season", "y"]].assign(mu=mu, sigma=sg))
    p = pd.concat(rows)
    base_preds[cls.name] = p
    met = evaluate(p)
    b = brier_score((p["y"] > 8.5).astype(float), prob_over(p["mu"], p["sigma"], 8.5))
    results.append({"model": cls.name, **met, "brier_8.5": b})
    print(cls.name, met, "brier@8.5", round(b, 4))

# ---- model ----
best, _ = tune(df, TOTAL_FEATURE_COLS, VAL_SEASON, lam_grid=[10.0, 100.0, 1000.0],
               half_life_grid=[2.0, 4.0, np.inf], step_col="day_index",
               refit_every=7, min_train=2000)
print("tuned on 2022:", best)

val = walk_forward(df, TOTAL_FEATURE_COLS, VAL_SEASON, lam=best["lam"],
                   half_life_seasons=best["half_life_seasons"], step_col="day_index",
                   refit_every=7, min_train=2000)
yv = (val["y"] > 8.5).astype(float).values
scales = np.linspace(0.7, 1.5, 33)
nll = [-(yv * np.log(np.clip(prob_over(val["mu"], val["sigma"] * c, 8.5), 1e-6, 1 - 1e-6))
         + (1 - yv) * np.log(np.clip(1 - prob_over(val["mu"], val["sigma"] * c, 8.5), 1e-6, 1 - 1e-6))).mean()
       for c in scales]
recal = float(scales[int(np.argmin(nll))])
print("RECAL_SCALE (2022):", recal)

rows = []
for s in TEST_SEASONS:
    p = walk_forward(df, TOTAL_FEATURE_COLS, s, lam=best["lam"],
                     half_life_seasons=best["half_life_seasons"], step_col="day_index",
                     refit_every=7, min_train=2000)
    assert_no_leakage(p["y"], p["mu"], r2_ceiling=0.60)
    rows.append(p)
mp = pd.concat(rows)
mp["sigma"] = mp["sigma"] * recal
met = evaluate(mp)
rep, fig = regression_report(mp["y"], mp["mu"], label="totals_2023_2025",
                             units="runs", show=False)
fig.savefig(FIG / "residuals.png", dpi=120)
b85 = brier_score((mp["y"] > 8.5).astype(float), prob_over(mp["mu"], mp["sigma"], 8.5))
results.append({"model": "linear_totals", **met, "brier_8.5": b85})
print("linear_totals", met, "brier@8.5", round(b85, 4))

print("\nCalibration at the lines the market actually trades:")
for line in LINES:
    p_over = prob_over(mp["mu"], mp["sigma"], line)
    y_over = (mp["y"] > line).astype(float)
    b = brier_score(y_over, p_over)
    base = float(y_over.mean())
    print(f"  O/U {line}: model brier {b:.4f} vs base-rate floor "
          f"{base*(1-base):.4f}  (overs actually {base*100:.1f}%, "
          f"model said {p_over.mean()*100:.1f}%)")
    fig, rel = reliability_diagram(y_over, p_over, n_bins=10, label=f"over_{line}")
    fig.savefig(FIG / f"reliability_{line}.png", dpi=120)
    dev = [abs(r[2] - r[3]) for r in rel if r[4] >= 50]
    print(f"    max calibration deviation: {max(dev):.3f}" if dev else "    (thin)")

# ---- negative binomial: the count-aware version ----
print("\n=== negative binomial (log link, fitted overdispersion) ===")
nb_rows = []
for s in TEST_SEASONS:
    p = walk_forward(df, TOTAL_FEATURE_COLS, s, step_col="day_index",
                     refit_every=7, min_train=2000,
                     half_life_seasons=best["half_life_seasons"],
                     model_factory=lambda: NegBinomTotal())
    nb_rows.append(p)
nbp = pd.concat(nb_rows)
nb_met = evaluate(nbp)

# Refit once on everything before the test window to score the count NLL and
# the O/U ladder with the actual negative-binomial law, not a Gaussian proxy.
tr = df[(df["season"] < TEST_SEASONS[0]) & df["y"].notna()]
te = df[df["season"].isin(TEST_SEASONS) & df["y"].notna()]
nb = NegBinomTotal().fit(tr[TOTAL_FEATURE_COLS].values, tr["y"].values)
Xte, yte = te[TOTAL_FEATURE_COLS].values, te["y"].values
nb_nll = float(nb.nll(Xte, yte).mean())
print(f"dispersion k = {nb.k_:.1f} (lower = fatter tail; Poisson is k -> inf)")
print(f"count NLL {nb_nll:.4f}  rmse {nb_met['rmse']:.4f}")
b85_nb = brier_score((yte > 8.5).astype(float), nb.prob_over(Xte, 8.5))
results.append({"model": "negbinom_totals", **nb_met, "brier_8.5": b85_nb,
                "count_nll": nb_nll})
print(f"brier@8.5 {b85_nb:.4f}")
print("\nNegative-binomial calibration at the traded lines:")
for line in LINES:
    p_over = nb.prob_over(Xte, line)
    y_over = (yte > line).astype(float)
    base = float(y_over.mean())
    print(f"  O/U {line}: brier {brier_score(y_over, p_over):.4f} vs floor "
          f"{base*(1-base):.4f}  (overs {base*100:.1f}%, model {p_over.mean()*100:.1f}%)")
    fig, rel = reliability_diagram(y_over, p_over, n_bins=10, label=f"nb_over_{line}")
    fig.savefig(FIG / f"nb_reliability_{line}.png", dpi=120)
    dev = [abs(r[2] - r[3]) for r in rel if r[4] >= 50]
    if dev:
        print(f"    max calibration deviation: {max(dev):.3f}")

out = pd.DataFrame(results)
out.to_csv(FIG / "totals_results.csv", index=False)
print("\n", out.round(4).to_string(index=False))
cfg = {"lam": best["lam"], "half_life_seasons": best["half_life_seasons"],
       "recal_scale": recal, "feature_cols": TOTAL_FEATURE_COLS,
       "first_season": args.first_season}
(DATA / "totals_config.json").write_text(json.dumps(cfg, indent=2, default=float))

base_brier = min(r["brier_8.5"] for r in results[:2])
base_rmse = min(r["rmse"] for r in results[:2])
gauss = met["nll"] < min(r["nll"] for r in results[:2]) and b85 < base_brier
nbeat = b85_nb < base_brier and nb_met["rmse"] < base_rmse
print("\nGATE gaussian (NLL + Brier@8.5 vs baselines):", "PASS" if gauss else "FAIL")
print("GATE negbinom (Brier@8.5 + RMSE vs baselines):", "PASS" if nbeat else "FAIL")
print("Ship the negative binomial only if its gate passes AND every populated "
      "decile above is inside 0.10.")
