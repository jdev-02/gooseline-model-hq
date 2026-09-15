"""Does the skill block (offensive K%, ISO, staff K%) earn a place?

Three arms on the same frame, same frozen Kalman, same walk-forward
(2023-2025, refit weekly, tune lam/half-life on 2022 only):

  A. the shipped feature set (MLB_FEATURE_COLS), ridge
  B. A + SKILL_COLS, ridge
  C. A + SKILL_COLS, lasso on the margin (L1 prunes what ridge keeps)

The features came from a lasso on World Series winners (season-level,
15 positives in 450) -- a different target and a weak evaluation. This is
the test that matters for us: game-level NLL, Brier at p_home and
calibration, against the set the site runs today. Ship only if B or C
beats A on NLL and Brier.

  uv run python ops/experiment_skill_feats.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.eval import brier_score, reliability_diagram  # noqa: E402
from src.core.kalman import TeamKalman  # noqa: E402
from src.core.walkforward import walk_forward, evaluate, tune  # noqa: E402
from src.mlb.baselines import p_home_from_dist  # noqa: E402
from src.mlb.compile import load_games, load_team_game_stats, load_pitcher_game_stats, DATA  # noqa: E402
from src.mlb.features import build_features, MLB_FEATURE_COLS, SKILL_COLS  # noqa: E402
from src.mlb.park import build_park_factors, park_lookup  # noqa: E402

VAL, TEST = 2022, (2023, 2024, 2025)


class LassoGaussian:
    """Same interface as LinearGaussianModel; L1 on the margin, sigma from
    weighted residuals. lam here is sklearn's alpha."""
    def __init__(self, lam=0.01):
        self.lam = lam

    def fit(self, X, y, sample_weight=None):
        from sklearn.linear_model import Lasso
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mu_, self.sd_ = X.mean(0), X.std(0); self.sd_[self.sd_ == 0] = 1
        Z = (X - self.mu_) / self.sd_
        self.m_ = Lasso(alpha=self.lam, max_iter=5000).fit(Z, y, sample_weight=sample_weight)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
        res = y - self.m_.predict(Z)
        self.sigma_ = float(np.sqrt(np.average(res ** 2, weights=w)))
        return self

    def predict_dist(self, X):
        Z = (np.asarray(X, float) - self.mu_) / self.sd_
        return self.m_.predict(Z), np.full(len(Z), self.sigma_)


def run_arm(df, cols, label, model_factory=None, lam_grid=(10.0, 100.0, 1000.0)):
    if model_factory is None:
        best, _ = tune(df, cols, VAL, lam_grid=list(lam_grid), half_life_grid=[2.0, 4.0, np.inf],
                       step_col="day_index", refit_every=7, min_train=2000)
        kw = dict(lam=best["lam"], half_life_seasons=best["half_life_seasons"])
    else:
        # lasso: pick alpha on 2022 by NLL, half-life fixed at the ridge choice
        best_a, best_nll = None, 1e9
        for a in (0.002, 0.005, 0.01, 0.02, 0.05):
            v = walk_forward(df, cols, VAL, step_col="day_index", refit_every=7, min_train=2000,
                             half_life_seasons=4.0, model_factory=lambda a=a: LassoGaussian(a))
            nll = evaluate(v)["nll"]
            if nll < best_nll:
                best_a, best_nll = a, nll
        kw = dict(half_life_seasons=4.0, model_factory=lambda: LassoGaussian(best_a))
        best = {"alpha": best_a}
    val = walk_forward(df, cols, VAL, step_col="day_index", refit_every=7, min_train=2000, **kw)
    yv = (val["y"] > 0).astype(float).values
    scales = np.linspace(0.7, 1.5, 33)
    nll = [-(yv * np.log(np.clip(p_home_from_dist(val["mu"], val["sigma"], c), 1e-6, 1 - 1e-6))
             + (1 - yv) * np.log(np.clip(1 - p_home_from_dist(val["mu"], val["sigma"], c), 1e-6, 1 - 1e-6))).mean()
           for c in scales]
    recal = float(scales[int(np.argmin(nll))])
    pr = pd.concat(walk_forward(df, cols, s, step_col="day_index", refit_every=7, min_train=2000, **kw)
                   for s in TEST)
    m = evaluate(pr)
    p = p_home_from_dist(pr["mu"].values, pr["sigma"].values, recal)
    y = (pr["y"].values > 0).astype(float)
    b = brier_score(y, p)
    fig, rows = reliability_diagram(y, p, n_bins=10, label=label)
    plt.close(fig)
    dev = max(abs(r[2] - r[3]) for r in rows if r[4] >= 50)
    print(f"{label:<28} tuned={best}  NLL {m['nll']:.4f}  RMSE {m['rmse']:.3f}  "
          f"Brier {b:.4f}  max calib dev {dev:.3f}  recal {recal:.3f}")
    return {"arm": label, "nll": m["nll"], "rmse": m["rmse"], "brier": b, "calib": dev, "tuned": str(best)}


def main():
    games = load_games(keep_unplayed=True, first_season=2015)
    park = build_park_factors(games)
    df = build_features(games, load_team_game_stats(), load_pitcher_game_stats(), park_lookup(park))
    kp = json.loads((DATA / "kalman_params.json").read_text())
    df = TeamKalman(**kp).run(df)
    played = df[df["y"].notna()]
    print(f"{len(played)} played games; skill cols non-zero share: "
          + ", ".join(f"{c} {(played[c] != 0).mean():.2f}" for c in SKILL_COLS))
    print("corr with margin (2015+):",
          {c: round(float(played[played.season >= 2015][c].corr(played[played.season >= 2015]["y"])), 4) for c in SKILL_COLS})
    out = [run_arm(df, MLB_FEATURE_COLS, "A: shipped, ridge"),
           run_arm(df, MLB_FEATURE_COLS + SKILL_COLS, "B: +skill, ridge"),
           run_arm(df, MLB_FEATURE_COLS + SKILL_COLS, "C: +skill, lasso", model_factory=True)]
    t = pd.DataFrame(out)
    print("\n", t.round(4).to_string(index=False))
    a = out[0]
    for arm in out[1:]:
        better = arm["nll"] < a["nll"] and arm["brier"] < a["brier"] and arm["calib"] < 0.10
        print(f"{arm['arm']}: {'BEATS' if better else 'does not beat'} A on NLL+Brier with calib<0.10")
    Path("figures/mlb/phase0").mkdir(parents=True, exist_ok=True)
    t.to_csv("figures/mlb/phase0/skill_experiment.csv", index=False)


if __name__ == "__main__":
    main()
