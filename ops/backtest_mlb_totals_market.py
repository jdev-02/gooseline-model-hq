"""The totals model against Kalshi's over/under ladder, from our own log,
with the UNDER priced the way a NO contract is actually bought: at
1 - (YES bid), not 1 - (YES ask).

Until 2026-09-15 every UNDER edge here used the ask, overstating it by the
bid-ask spread. The log only carried the over ask, so the bid is joined from
data/kalshi_prices.db: for each logged rung, the latest KXMLBTOTAL snapshot
at or before that row's run_ts. Rows logged after the fix carry mkt_under_X
directly. Rows with no matching snapshot fall back to the ask and are
counted so the caveat is visible.

  uv run python ops/backtest_mlb_totals_market.py
"""
import sqlite3
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.kalshi import kalshi_fee  # noqa: E402

DB = "data/kalshi_prices.db"
log = pd.read_csv("data/mlb/narrative/log.csv", low_memory=False)
log = log.sort_values("run_ts").drop_duplicates(["game_pk", "date"], keep="last")
g = pd.read_csv("data/mlb/games.csv", low_memory=False)[["game_pk", "home_score", "away_score", "played"]]
h = log.merge(g, on="game_pk"); h = h[h.played].copy()
h["tot"] = h.home_score + h.away_score
h = h.rename(columns={c: c.replace(".", "_") for c in h.columns
                      if "p_over_" in c or "mkt_over_" in c or "mkt_under_" in c})

_snap = pd.DataFrame(columns=["ticker", "yes_bid", "ts"])
if Path(DB).exists():
    con = sqlite3.connect(DB)
    _snap = pd.read_sql("SELECT ticker, yes_bid, ts_utc FROM snapshots WHERE series_ticker='KXMLBTOTAL'", con)
    _snap["ts"] = pd.to_datetime(_snap["ts_utc"], utc=True)


def over_bid_at(date, away, home, strike, run_ts):
    """Latest logged YES bid for this rung at or before the run that priced it."""
    d = pd.Timestamp(date).strftime("%y%b%d").upper()
    rung = int(round(strike + 0.5))
    pat = rf"^KXMLBTOTAL-{d}\d{{4}}{away}{home}(G\d)?-{rung}$"
    s = _snap[_snap["ticker"].str.match(pat)]
    if not len(s):
        return None
    t = pd.Timestamp(run_ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    s = s[s["ts"] <= t]
    if not len(s):
        return None
    return float(s.sort_values("ts").iloc[-1]["yes_bid"])


def paper(sub, line, thr, stake=15.0, use_bid=True):
    pm, mk = f"p_over_{line:g}".replace(".", "_"), f"mkt_over_{line:g}".replace(".", "_")
    mu_ = f"mkt_under_{line:g}".replace(".", "_")
    rows, no_bid = [], 0
    for r in sub.dropna(subset=[pm, mk]).itertuples():
        po, ask_o = getattr(r, pm), getattr(r, mk)
        if use_bid:
            logged_under = getattr(r, mu_, np.nan) if mu_ in sub.columns else np.nan
            if pd.notna(logged_under):
                ask_u = float(logged_under)
            else:
                b = over_bid_at(r.date, r.away, r.home, line, r.run_ts)
                if b is None:
                    no_bid += 1
                    ask_u = 1 - ask_o
                else:
                    ask_u = 1 - b
        else:
            ask_u = 1 - ask_o
        eo = po - ask_o - kalshi_fee(ask_o); eu = (1 - po) - ask_u - kalshi_fee(ask_u)
        if eo >= thr:   won, ask = r.tot > line, ask_o
        elif eu >= thr: won, ask = r.tot < line, ask_u
        else: continue
        n = stake / (ask + kalshi_fee(ask)); rows.append((won, n - stake if won else -stake))
    if not rows: return 0, 0, 0.0, 0.0, no_bid
    w = sum(1 for x, _ in rows if x); pnl = sum(p for _, p in rows)
    return len(rows), w, pnl, 100 * pnl / (len(rows) * stake), no_bid


def line_(label, res):
    n, w, pnl, roi, nb = res
    print(f"  {label:<34} bets {n:>3}  win {(100*w/n if n else 0):>5.1f}%  P&L {pnl:>8.2f}  ROI {roi:>6.1f}%"
          + (f"  ({nb} unders: no bid logged, used ask)" if nb else ""))


for ln in (7.5, 8.5, 9.5):
    pm, mk = f"p_over_{ln:g}".replace(".", "_"), f"mkt_over_{ln:g}".replace(".", "_")
    if mk not in h: continue
    t = h.dropna(subset=[pm, mk])
    if len(t) < 20: print(f"\nO/U {ln}: only {len(t)} games, skipping"); continue
    yo = (t.tot > ln).astype(float).values; po, ko = t[pm].values, t[mk].values
    bm, bmod = np.mean((ko - yo) ** 2), np.mean((po - yo) ** 2)
    print(f"\n=== O/U {ln}: {len(t)} games ===")
    print(f"  Brier  Kalshi {bm:.4f}   model {bmod:.4f}   {'MODEL better' if bmod < bm else 'Kalshi better'}"
          f"   (overs hit {yo.mean()*100:.0f}%, Kalshi said {ko.mean()*100:.0f}%, model {po.mean()*100:.0f}%)")
    for thr in (0.0, 0.02, 0.04, 0.06):
        line_(f"edge >= {thr:.0%}, under off BID", paper(t, ln, thr, use_bid=True))
    line_("edge >= 4%, under off ASK (old)", paper(t, ln, .04, use_bid=False))
    dates = sorted(t.date.unique()); cut = dates[len(dates) // 2]
    a, b = t[t.date < cut], t[t.date >= cut]
    if len(a) > 15 and len(b) > 15:
        print(f"  split by date at {cut}:")
        line_(f"    first half ({len(a)} g), >=4%", paper(a, ln, .04))
        line_(f"    second half ({len(b)} g), >=4%", paper(b, ln, .04))
        ya, yb = (a.tot > ln).astype(float).values, (b.tot > ln).astype(float).values
        print(f"    Brier first half: Kalshi {np.mean((a[mk].values-ya)**2):.4f} model {np.mean((a[pm].values-ya)**2):.4f}"
              f"   second half: Kalshi {np.mean((b[mk].values-yb)**2):.4f} model {np.mean((b[pm].values-yb)**2):.4f}")
    t = t.copy(); t["side"] = np.where(t[pm] > t[mk], "model says OVER", "model says UNDER")
    for s, gg in t.groupby("side"):
        print(f"  {s}: n={len(gg)}, over actually hit {(gg.tot>ln).mean()*100:.0f}%, Kalshi said {gg[mk].mean()*100:.0f}%, model {gg[pm].mean()*100:.0f}%")
