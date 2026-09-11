"""MLB model vs the market it actually bets: Kalshi, from our own daily log.

The MLB gate in docs/baselines.md compared against Elo and home-always. That
proves the model knows something; it does not prove it knows something the
MARKET does not. This is that test, on every game the daily run has priced
since 2026-08-27 (not just the ones it flagged), using the live Kalshi ask
recorded at run time and the result filled in by settle_log.py.

Grows by ~15 games a day. Rerun any time:  uv run python ops/backtest_mlb_market.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.kalshi import kalshi_fee  # noqa: E402

log = pd.read_csv("data/mlb/narrative/log.csv")
log = log.sort_values("run_ts").drop_duplicates(["game_pk", "date"], keep="last")
g = pd.read_csv("data/mlb/games.csv", low_memory=False)[["game_pk","home_score","away_score","result","played"]]
h = log.merge(g, on="game_pk", suffixes=("", "_g"))
h = h[h["played"] & h["mkt_home"].notna() & h["mkt_away"].notna()].copy()
h["y"] = (h["result_g"] > 0).astype(float)
h["pv"] = h["mkt_home"] / (h["mkt_home"] + h["mkt_away"])      # de-vigged Kalshi
h["pm"] = h["p_home"]
y, pv, pm = h["y"].values, h["pv"].values, h["pm"].values
brier = lambda p: float(np.mean((p - y) ** 2))

print(f"MLB, {len(h)} priced+settled games, {h.date.min()} -> {h.date.max()}\n")
print("Brier (lower is better):")
print(f"  Kalshi alone (a=0):   {brier(pv):.4f}")
print(f"  model alone  (a=1):   {brier(pm):.4f}")
print(f"  coin flip:            {brier(np.full_like(y, .5)):.4f}")
print("\nBlend a*model + (1-a)*Kalshi:")
for a in np.arange(0, 1.01, 0.25):
    print(f"  a={a:.2f}  brier={brier(a*pm+(1-a)*pv):.4f}")
best = min(np.arange(0,1.01,0.05), key=lambda a: brier(a*pm+(1-a)*pv))
print(f"  best a = {best:.2f}")

dates = sorted(h.date.unique()); cut = dates[len(dates)//2]
tr, te = h[h.date < cut], h[h.date >= cut]
if len(tr) > 30 and len(te) > 30:
    ytr = tr.y.values
    a_star = min(np.arange(0,1.01,0.05), key=lambda a: np.mean((a*tr.pm.values+(1-a)*tr.pv.values-ytr)**2))
    yte = te.y.values
    bm = np.mean((te.pv.values-yte)**2); bb = np.mean((a_star*te.pm.values+(1-a_star)*te.pv.values-yte)**2)
    print(f"\nHonest split: choose a on first half ({len(tr)} games) -> a*={a_star:.2f}; "
          f"test on second half ({len(te)} games): Kalshi {bm:.4f} vs blend {bb:.4f} -> "
          f"{'BLEND better' if bb < bm else 'Kalshi better'}")

def paper(sub, thr, only=None, stake=15.0):
    rows = []
    for r in sub.itertuples():
        eh = r.pm - r.mkt_home - kalshi_fee(r.mkt_home)
        ea = (1-r.pm) - r.mkt_away - kalshi_fee(r.mkt_away)
        if eh >= thr:   fav, won, ask = r.mkt_home >= .5, r.y == 1, r.mkt_home
        elif ea >= thr: fav, won, ask = r.mkt_away >= .5, r.y == 0, r.mkt_away
        else: continue
        if only == "fav" and not fav: continue
        if only == "dog" and fav: continue
        cost = ask + kalshi_fee(ask); n = stake / cost
        rows.append((won, n*1.0 - stake if won else -stake))
    if not rows: return 0,0,0.0,0.0
    w = sum(1 for x,_ in rows if x); pnl = sum(p for _,p in rows)
    return len(rows), w, pnl, 100*pnl/(len(rows)*stake)
def line(label, res):
    n,w,pnl,roi = res; print(f"  {label:<26} bets {n:>4}  win {(100*w/n if n else 0):>5.1f}%  P&L {pnl:>8.2f}  ROI {roi:>6.1f}%")

print("\nPaper trade vs live Kalshi ask, after 7% fee, $15 flat:")
for thr in (0.0, 0.02, 0.04, 0.06, 0.08):
    line(f"edge >= {thr:.0%}", paper(h, thr))
line("backs favorite (>=4%)", paper(h, 0.04, only="fav"))
line("backs underdog (>=4%)", paper(h, 0.04, only="dog"))

print("\nCalibration by KALSHI probability bucket:")
h["bucket"] = pd.cut(h.pv, [0,.35,.45,.55,.65,1.0])
for b, gg in h.groupby("bucket", observed=True):
    print(f"  Kalshi {str(b):<13} n={len(gg):>3}  home won {gg.y.mean()*100:5.1f}%   Kalshi said {gg.pv.mean()*100:5.1f}%   model said {gg.pm.mean()*100:5.1f}%")

# ---- totals, where the log has both a model and a market ----
t = h.dropna(subset=["p_over_8.5", "mkt_over_8.5"]).copy() if "mkt_over_8.5" in h else pd.DataFrame()
if len(t) >= 20:
    t["tot"] = t.home_score + t.away_score; yo = (t.tot > 8.5).astype(float).values
    po, ko = t["p_over_8.5"].values, t["mkt_over_8.5"].values
    print(f"\nTOTALS (over 8.5), {len(t)} games:")
    print(f"  Brier Kalshi {np.mean((ko-yo)**2):.4f}   model {np.mean((po-yo)**2):.4f}   "
          f"best blend a={min(np.arange(0,1.01,.05), key=lambda a: np.mean((a*po+(1-a)*ko-yo)**2)):.2f}")
else:
    print(f"\nTOTALS: only {len(t)} games with both a model and a market price; too few yet.")
