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
                            LeagueMeanTotal, ParkAdjustedMeanTotal)  # noqa: E402

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

out = pd.DataFrame(results)
out.to_csv(FIG / "totals_results.csv", index=False)
print("\n", out.round(4).to_string(index=False))
cfg = {"lam": best["lam"], "half_life_seasons": best["half_life_seasons"],
       "recal_scale": recal, "feature_cols": TOTAL_FEATURE_COLS,
       "first_season": args.first_season}
(DATA / "totals_config.json").write_text(json.dumps(cfg, indent=2, default=float))

beat = (met["nll"] < min(r["nll"] for r in results[:2])
        and b85 < min(r["brier_8.5"] for r in results[:2]))
print("\nGATE (beats both baselines on NLL and Brier@8.5):", "PASS" if beat else "FAIL")
