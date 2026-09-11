"""Does the NFL model add ANY information on top of the market?

The decisive test. Take the de-vigged Vegas closing probability p_vegas and
the model's walk-forward probability p_model. Blend: p = a*p_model +
(1-a)*p_vegas. If the Brier score is minimised at a=0, the model carries
nothing the market does not already hold and no betting strategy on this
data can exist. If it is minimised at a>0, chosen on prior seasons and
tested on the next, there is information, and the exploitation strategy is
to bet where the blend disagrees with the market.

Also reports calibration by market-probability bucket, which is where the
underdog problem should show up: the model saying 45% on teams that win 28%.

  uv run python ops/experiment_nfl_market_blend.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.backtest_nfl_ml import load, paper, line  # noqa: E402

h = load()
y = (h["y"] > 0).astype(float).values
pv, pm = h["mkt_home"].values, h["p_home"].values

def brier(p): return float(np.mean((p - y) ** 2))
print(f"{len(h)} games 2021-2025\n")
print("Brier score (lower is better):")
print(f"  market alone (a=0):   {brier(pv):.4f}")
print(f"  model alone  (a=1):   {brier(pm):.4f}")
print(f"  coin flip (0.5):      {brier(np.full_like(y,0.5)):.4f}")
print("\nBlend a*model + (1-a)*market, all seasons pooled:")
best_a, best_b = 0, 9
for a in np.arange(0, 1.01, 0.1):
    b = brier(a*pm + (1-a)*pv)
    if b < best_b: best_a, best_b = a, b
    print(f"  a={a:.1f}  brier={b:.4f}" + ("  <- best" if b == best_b and a == best_a else ""))
print(f"\nbest a pooled = {best_a:.1f}")

print("\nHonest version: choose a on seasons before, test on the season itself:")
for s in sorted(h.season.unique())[1:]:
    tr, te = h[h.season < s], h[h.season == s]
    ytr = (tr.y > 0).astype(float).values
    a_star = min(np.arange(0,1.01,0.05), key=lambda a: np.mean((a*tr.p_home.values+(1-a)*tr.mkt_home.values-ytr)**2))
    yte = (te.y > 0).astype(float).values
    bm = np.mean((te.mkt_home.values-yte)**2); bb = np.mean((a_star*te.p_home.values+(1-a_star)*te.mkt_home.values-yte)**2)
    print(f"  {s}: a*={a_star:.2f}  market brier {bm:.4f}  blend brier {bb:.4f}  {'blend better' if bb<bm else 'market better'}")

print("\nCalibration by MARKET probability bucket (does the model over-rate dogs?):")
h["bucket"] = pd.cut(h.mkt_home, [0,.3,.4,.5,.6,.7,1.0])
for b, g in h.groupby("bucket", observed=True):
    yy = (g.y>0).mean()
    print(f"  market {str(b):<12} n={len(g):>4}  home actually won {yy*100:5.1f}%   market said {g.mkt_home.mean()*100:5.1f}%   model said {g.p_home.mean()*100:5.1f}%")
