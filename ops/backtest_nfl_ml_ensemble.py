"""Same test as backtest_nfl_ml.py, but walking forward the deep ensemble the
live site actually uses, not the linear proxy history_tables uses because the
ensemble is slow. One season is enough to say whether the two disagree.

  uv run python ops/backtest_nfl_ml_ensemble.py --season 2025
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.nfl.rundown as rd  # noqa: E402
from src.core.walkforward import walk_forward
from src.core.ensemble import DeepEnsemble
from ops.backtest_nfl_ml import implied, decimal, paper, line

ap = argparse.ArgumentParser(); ap.add_argument("--season", type=int, default=2025)
a = ap.parse_args()
df = rd.build_frame("data/nfl/games.csv", "data/nfl/team_game_stats.csv")
fac = lambda: DeepEnsemble(n_members=5, hidden=16, weight_decay=1e-2, epochs=200, seed=0)
p = walk_forward(df, rd.V3_COLS, a.season, half_life_seasons=rd.DECAY_HL,
                 model_factory=fac, refit_every=1, min_train=100)
p["p_home"] = norm.cdf(p["mu"] / (rd.RECAL_SCALE * p["sigma"]))
g = pd.read_csv("data/nfl/games.csv", low_memory=False)[["game_id","away_moneyline","home_moneyline"]]
h = p.merge(g, on="game_id").dropna(subset=["away_moneyline","home_moneyline"])
ih, ia = h["home_moneyline"].map(implied), h["away_moneyline"].map(implied)
h["mkt_home"] = ih/(ih+ia); h["dec_home"] = h["home_moneyline"].map(decimal); h["dec_away"] = h["away_moneyline"].map(decimal)
h["week"] = p.set_index("game_id").loc[h["game_id"], "week"].values if "week" in p else 0
print(f"ENSEMBLE walk-forward, {a.season}, {len(h)} games, vs de-vigged Vegas ML, $15 flat")
print(f"winner pick: {((h.mu>0)==(h.y>0)).mean()*100:.1f}%")
for thr in (0.04, 0.08, 0.12):
    line(f"edge >= {thr:.0%}", paper(h, thr))
line("backs favorite (>=4%)", paper(h, 0.04, only="fav"))
line("backs underdog (>=4%)", paper(h, 0.04, only="dog"))
