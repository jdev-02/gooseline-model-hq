"""Export per-game walk-forward predictions for the site's Track Record tab.

Mirrors src/site/nfl_site.py:history_tables, but for MLB: every prediction was
made before the model had seen that game, refit weekly on games already played.

  uv run python ops/export_history.py --seasons 2023,2024,2025
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.walkforward import walk_forward  # noqa: E402
from src.mlb.compile import DATA  # noqa: E402
from src.mlb.rundown import build_frame, load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seasons", default="2023,2024,2025")
args = ap.parse_args()
seasons = [int(s) for s in args.seasons.split(",")]

cfg = load_config()
cols = cfg["feature_cols"]
recal = float(cfg.get("recal_scale", 1.0))
df = build_frame(cfg)

frames = []
for s in seasons:
    p = walk_forward(df, cols, s, lam=cfg["lam"],
                     half_life_seasons=cfg.get("half_life_seasons") or np.inf,
                     step_col="day_index", refit_every=7, min_train=2000)
    p["season"] = s
    frames.append(p)
hist = pd.concat(frames, ignore_index=True)

meta = df[["game_id", "gameday", "home_score", "away_score"]]
hist = hist.merge(meta, on="game_id", how="left")
hist["p_home"] = norm.cdf(hist["mu"] / (recal * hist["sigma"]))
hist["su_pick"] = np.where(hist["mu"] > 0, hist["home_team"], hist["away_team"])
hist["su_win"] = np.where(hist["y"] > 0, hist["home_team"], hist["away_team"])
hist["su_correct"] = hist["su_pick"] == hist["su_win"]
# Run line: the model's side at the standard +/-1.5
hist["rl_pick"] = np.where(hist["mu"] > 0, hist["home_team"], hist["away_team"])
home_cover = hist["y"] > 1.5
away_cover = hist["y"] < -1.5
hist["rl_correct"] = np.where(hist["rl_pick"] == hist["home_team"], home_cover, away_cover)

out = DATA / "mlb_history.csv"
hist.to_csv(out, index=False)

by_season = hist.groupby("season").apply(lambda g: pd.Series({
    "games": len(g),
    "winner_pct": 100 * g["su_correct"].mean(),
    "runline_pct": 100 * g["rl_correct"].mean(),
    "avg_miss": (g["y"] - g["mu"]).abs().mean(),
}), include_groups=False).round(1)
print(by_season.to_string())
print(f"\nwrote {out} ({len(hist)} rows)")
