"""The totals model against Kalshi's over/under ladder, from our own log.

backtest_mlb_market.py found the totals model's Brier BEATS Kalshi on the 8.5
line (0.2398 vs 0.2439, best blend weight 1.0 on the model), the only place in
either sport where the model carried information the market did not. That is
a Brier score on 85 games. This asks the bettor's question: paper-traded
against the live ask after the fee, does it make money, and does it hold up
on the half of the sample it was not looked at on?

Under-side price is approximated as 1 - over_ask (Kalshi's NO ask is 1 minus
the YES bid; we only logged the ask, so this is a shade optimistic).

  uv run python ops/backtest_mlb_totals_market.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.kalshi import kalshi_fee  # noqa: E402

log = pd.read_csv("data/mlb/narrative/log.csv").sort_values("run_ts").drop_duplicates(["game_pk","date"], keep="last")
g = pd.read_csv("data/mlb/games.csv", low_memory=False)[["game_pk","home_score","away_score","played"]]
h = log.merge(g, on="game_pk"); h = h[h.played].copy()
h["tot"] = h.home_score + h.away_score
h = h.rename(columns={c: c.replace(".", "_") for c in h.columns if "p_over_" in c or "mkt_over_" in c})

def paper(sub, line, thr, stake=15.0):
    pm, mk = f"p_over_{line:g}".replace(".", "_"), f"mkt_over_{line:g}".replace(".", "_")
    rows = []
    for r in sub.dropna(subset=[pm, mk]).itertuples():
        po, ask_o = getattr(r, pm), getattr(r, mk); ask_u = 1 - ask_o
        eo = po - ask_o - kalshi_fee(ask_o); eu = (1-po) - ask_u - kalshi_fee(ask_u)
        if eo >= thr:   won, ask = r.tot > line, ask_o
        elif eu >= thr: won, ask = r.tot < line, ask_u
        else: continue
        n = stake / (ask + kalshi_fee(ask)); rows.append((won, n - stake if won else -stake))
    if not rows: return 0, 0, 0.0, 0.0
    w = sum(1 for x,_ in rows if x); pnl = sum(p for _,p in rows)
    return len(rows), w, pnl, 100*pnl/(len(rows)*stake)
def line_(label, res):
    n,w,pnl,roi = res; print(f"  {label:<28} bets {n:>3}  win {(100*w/n if n else 0):>5.1f}%  P&L {pnl:>8.2f}  ROI {roi:>6.1f}%")

for ln in (7.5, 8.5, 9.5):
    pm, mk = f"p_over_{ln:g}".replace(".", "_"), f"mkt_over_{ln:g}".replace(".", "_")
    if mk not in h: continue
    t = h.dropna(subset=[pm, mk]); 
    if len(t) < 20: print(f"\nO/U {ln}: only {len(t)} games, skipping"); continue
    yo = (t.tot > ln).astype(float).values; po, ko = t[pm].values, t[mk].values
    bm, bmod = np.mean((ko-yo)**2), np.mean((po-yo)**2)
    print(f"\n=== O/U {ln}: {len(t)} games ===")
    print(f"  Brier  Kalshi {bm:.4f}   model {bmod:.4f}   {'MODEL better' if bmod<bm else 'Kalshi better'}   (overs hit {yo.mean()*100:.0f}%, Kalshi said {ko.mean()*100:.0f}%, model {po.mean()*100:.0f}%)")
    for thr in (0.0, 0.02, 0.04, 0.06):
        line_(f"edge >= {thr:.0%}", paper(t, ln, thr))
    dates = sorted(t.date.unique()); cut = dates[len(dates)//2]
    a, b = t[t.date < cut], t[t.date >= cut]
    if len(a) > 15 and len(b) > 15:
        print(f"  split by date at {cut}:")
        line_(f"    first half ({len(a)} g), >=4%", paper(a, ln, .04))
        line_(f"    second half ({len(b)} g), >=4%", paper(b, ln, .04))
        ya, yb = (a.tot>ln).astype(float).values, (b.tot>ln).astype(float).values
        print(f"    Brier first half: Kalshi {np.mean((a[mk].values-ya)**2):.4f} model {np.mean((a[pm].values-ya)**2):.4f}"
              f"   second half: Kalshi {np.mean((b[mk].values-yb)**2):.4f} model {np.mean((b[pm].values-yb)**2):.4f}")
    t = t.copy(); t["side"] = np.where(t[pm] > t[mk], "model says OVER", "model says UNDER")
    for s, gg in t.groupby("side"):
        print(f"  {s}: n={len(gg)}, over actually hit {(gg.tot>ln).mean()*100:.0f}%, Kalshi said {gg[mk].mean()*100:.0f}%, model {gg[pm].mean()*100:.0f}%")
