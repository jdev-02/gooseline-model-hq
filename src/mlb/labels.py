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
    mk = t["market"].astype(str).str.upper()
    t["kind"] = "TOTAL"
    t.loc[mk == "ML", "kind"] = "ML"
    t.loc[mk == "SPREAD", "kind"] = "SPREAD"
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


def spread_flag(row):
    """(team, signed line) if the row carries an actionable spread call."""
    se = row.get("spread_edge")
    call = str(row.get("spread_call") or "")
    if not (call and not call.startswith("no edge") and _num(se) and float(se) > THRESH):
        return None
    parts = call.split()
    if len(parts) != 2:
        return None
    try:
        return parts[0], float(parts[1])
    except ValueError:
        return None


def market_flags(row):
    """Every market the model flags on this row, with its word and edge:
    [(kind, tier, what, edge)]. One row per market, always in the same
    order, so the card can show three lines that never move."""
    out = []
    side = ml_flag(row)
    if side:
        out.append(("ML", "bet" if bet_allowed("ML") else "risky", f"{side} to win",
                    float(row.get("edge") or 0)))
    sf = spread_flag(row)
    if sf:
        out.append(("SPREAD", "bet" if bet_allowed("SPREAD") else "risky",
                    f"{sf[0]} {sf[1]:+g}", float(row.get("spread_edge") or 0)))
    tf = totals_flag(row)
    if tf:
        out.append(("TOTAL", "bet" if bet_allowed("TOTAL") else "risky",
                    f"{tf[0].title()} {tf[1]:g} runs", float(row.get("total_edge") or 0)))
    return out


_RANK = {"bet": 2, "risky": 1}
_ORDER = {"ML": 0, "SPREAD": 1, "TOTAL": 2}


def headline(row):
    """The one flag that leads the card: the highest word, and among equals
    the first line in the card's fixed order (Moneyline, Run line, Total).
    It used to be the largest edge, so the badge could say "Over 9.5" while
    the first line under it read "Moneyline"; the badge now always matches
    the first line that carries its word. The edges are on the lines."""
    fl = market_flags(row)
    if not fl:
        return None
    return max(fl, key=lambda f: (_RANK.get(f[1], 0), -_ORDER.get(f[0], 9)))


def tier_for(row):
    """'bet' | 'risky' | 'stale' | 'none'."""
    v = str(row.get("verdict", ""))
    h = headline(row)
    if h:
        return h[1]
    if v.startswith("STALE"):
        return "stale"
    return "none"


def label_for(row, nick=lambda c: c):
    """The exact words on the badge, for the log and the public feed."""
    h = headline(row)
    if h:
        kind, tier, what, _ = h
        if kind == "ML":
            what = f"{nick(what.split(' to win')[0])} to win"
        elif kind == "SPREAD":
            team, line = what.split()
            what = f"{nick(team)} {line}"
        return f"{'BET' if tier == 'bet' else 'RISKY'} · {what}"
    if str(row.get("verdict", "")).startswith("STALE"):
        return "PRICE STALE · re-check"
    return "NO BET"
