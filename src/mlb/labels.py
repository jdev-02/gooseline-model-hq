"""The word on the card, decided by the record, not by prose.

BET is earned, per market, from the live paper trade in
data/mlb/paper_trades.csv: at least MIN_BETS settled, positive return
overall, and positive over the most recent half. Anything the model flags in
a market that has not earned it is RISKY, and the card quotes that market's
record in plain words. When a market's record turns, the word turns with it;
nobody edits copy. Used by the rundown (so the logged row and the public feed
carry the label) and by the site (so the page shows the same one).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.mlb.compile import DATA

MIN_BETS = 100
THRESH = 0.04


def _trades():
    p = DATA / "paper_trades.csv"
    if not p.exists():
        return None
    t = pd.read_csv(p)
    if not len(t):
        return None
    t["kind"] = t["market"].astype(str).str.upper().where(t["market"].astype(str).str.upper() == "ML", "TOTAL")
    return t


def market_record(kind):
    """kind: 'ML' or 'TOTAL'. Returns dict(n, w, roi, recent_n, recent_roi) or None."""
    t = _trades()
    if t is None:
        return None
    g = t[t["kind"] == kind]
    if not len(g) or not g["stake"].sum():
        return None
    dates = sorted(g["date"].unique())
    cut = dates[len(dates) // 2]
    rec = g[g["date"] >= cut]
    return {"n": int(len(g)), "w": int(g["won"].sum()),
            "roi": 100 * g["pnl"].sum() / g["stake"].sum(),
            "recent_n": int(len(rec)),
            "recent_roi": (100 * rec["pnl"].sum() / rec["stake"].sum()) if rec["stake"].sum() else 0.0,
            "since": str(g["date"].min())}


def bet_allowed(kind):
    r = market_record(kind)
    return bool(r and r["n"] >= MIN_BETS and r["roi"] > 0 and r["recent_roi"] > 0)


def record_phrase(kind):
    """'has returned -22.4% over 56 settled bets (19-37) since 2026-08-27'."""
    r = market_record(kind)
    if not r:
        return "has no settled bets yet"
    return (f"has returned {r['roi']:+.1f}% over {r['n']} settled bets "
            f"({r['w']}-{r['n'] - r['w']}) since {r['since']}, and {r['recent_roi']:+.1f}% "
            f"over the most recent {r['recent_n']}")


def why_not_bet(kind):
    """One clause explaining why a flagged market is RISKY rather than BET."""
    r = market_record(kind)
    if not r:
        return "no settled record yet"
    if r["n"] < MIN_BETS:
        return f"fewer than {MIN_BETS} settled bets so far ({r['n']})"
    if r["roi"] <= 0:
        return "the record is negative"
    return "the most recent half is negative"


def _num(x):
    return x is not None and not pd.isna(x)


def totals_flag(row):
    """(direction, line) if the row carries an actionable totals call."""
    te = row.get("total_edge")
    tcall = str(row.get("total_call") or "")
    if not (tcall and not tcall.startswith("no edge") and _num(te) and float(te) > THRESH):
        return None
    parts = tcall.split()
    if len(parts) != 2:
        return None
    return parts[0], float(parts[1])


def ml_flag(row):
    v = str(row.get("verdict", ""))
    if v.startswith("HIGH VALUE") and "&mdash;" in v and _num(row.get("mkt_home")):
        return v.split("&mdash;")[-1].strip()
    return None


def tier_for(row):
    """'bet' | 'risky' | 'stale' | 'none'."""
    v = str(row.get("verdict", ""))
    if totals_flag(row):
        return "bet" if bet_allowed("TOTAL") else "risky"
    if v.startswith("STALE"):
        return "stale"
    if ml_flag(row):
        return "bet" if bet_allowed("ML") else "risky"
    return "none"


def label_for(row, nick=lambda c: c):
    """The exact words on the badge, for the log and the public feed."""
    t = tier_for(row)
    tf = totals_flag(row)
    if tf:
        what = f"{tf[0].title()} {tf[1]:g} runs"
        return f"{'BET' if t == 'bet' else 'RISKY'} · {what}"
    if t == "stale":
        return "PRICE STALE · re-check"
    side = ml_flag(row)
    if side:
        return f"{'BET' if t == 'bet' else 'RISKY'} · {nick(side)} to win"
    return "NO BET"
