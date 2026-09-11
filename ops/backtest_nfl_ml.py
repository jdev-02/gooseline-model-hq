"""NFL moneyline edges, paper-traded against Vegas closing lines, 2021-2025.

The Track Record tab shows the model picks winners 63.9% of the time. That is
not the question a bettor needs answered. The question is whether the model's
disagreements with the market's PRICE make money, and this is the walk-forward
answer: every game where the model's win probability beat the de-vigged Vegas
moneyline by at least the threshold, staked flat, paid at the actual odds.

  uv run python ops/backtest_nfl_ml.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.nfl.rundown as rd  # noqa: E402
from src.site.nfl_site import history_tables  # noqa: E402


def implied(ml):
    ml = float(ml)
    return (-ml / (-ml + 100)) if ml < 0 else 100 / (ml + 100)


def decimal(ml):
    ml = float(ml)
    return (1 + 100 / -ml) if ml < 0 else (1 + ml / 100)


def load():
    df = rd.build_frame("data/nfl/games.csv", "data/nfl/team_game_stats.csv")
    hist, _, _ = history_tables(df)
    g = pd.read_csv("data/nfl/games.csv", low_memory=False)[
        ["game_id", "away_moneyline", "home_moneyline"]]
    h = hist.merge(g, on="game_id").dropna(subset=["away_moneyline", "home_moneyline"])
    ih, ia = h["home_moneyline"].map(implied), h["away_moneyline"].map(implied)
    h["mkt_home"] = ih / (ih + ia)
    h["dec_home"] = h["home_moneyline"].map(decimal)
    h["dec_away"] = h["away_moneyline"].map(decimal)
    return h


def paper(sub, thr, stake=15.0, only=None):
    rows = []
    for r in sub.itertuples():
        eh = r.p_home - r.mkt_home
        ea = (1 - r.p_home) - (1 - r.mkt_home)
        if eh >= thr:
            side_fav = r.mkt_home >= 0.5
            won, d = r.y > 0, r.dec_home
        elif ea >= thr:
            side_fav = r.mkt_home < 0.5
            won, d = r.y < 0, r.dec_away
        else:
            continue
        if only == "fav" and not side_fav:
            continue
        if only == "dog" and side_fav:
            continue
        rows.append((won, stake * (d - 1) if won else -stake))
    if not rows:
        return 0, 0, 0.0, 0.0
    w = sum(1 for x, _ in rows if x)
    pnl = sum(p for _, p in rows)
    return len(rows), w, pnl, 100 * pnl / (len(rows) * stake)


def line(label, res):
    n, w, pnl, roi = res
    wp = 100 * w / n if n else 0
    print(f"  {label:<26} bets {n:>4}  win {wp:>5.1f}%  P&L {pnl:>9.2f}  ROI {roi:>6.1f}%")


if __name__ == "__main__":
    h = load()
    print(f"NFL moneyline vs de-vigged Vegas close, {int(h.season.min())}-{int(h.season.max())}, "
          f"{len(h)} games, $15 flat\n")
    print("By season, edge >= 4%:")
    for s, sub in h.groupby("season"):
        line(str(s), paper(sub, 0.04))
    line("ALL", paper(h, 0.04))
    print("\nBy edge threshold (does a bigger disagreement mean a better bet?):")
    for thr in (0.04, 0.08, 0.12, 0.16, 0.20):
        line(f">= {thr:.0%}", paper(h, thr))
    print("\nFavorites vs underdogs, edge >= 4%:")
    line("model backs the favorite", paper(h, 0.04, only="fav"))
    line("model backs the underdog", paper(h, 0.04, only="dog"))
    print("\nBy week of season, edge >= 4%:")
    for lo, hi, lab in [(1, 1, "week 1"), (2, 4, "weeks 2-4"), (5, 18, "weeks 5+")]:
        line(lab, paper(h[(h.week >= lo) & (h.week <= hi)], 0.04))
    print("\nBreakeven is 0% ROI. Vegas vig alone costs roughly -4.5% at standard "
          "juice, so a model with no information would sit near there.")
