"""The word on the card, decided by the record, not by prose.

Three words, each an instruction:

  BET        one unit. The market has earned it in the live paper trade
             (data/mlb/paper_trades.csv): at least MIN_BETS settled,
             profitable overall and over the most recent half.
  SMALL BET  half a unit. Same conditions on at least SMALL_MIN bets: the
             market is up, but the sample is still short.
  PASS       do nothing. The market's live record is negative or too short,
             or the price is fair. The line still shows what the model
             likes and by how much, so the disagreement is visible; the
             record says not to act on it.

When a market's record turns, the word turns with it; nobody edits copy.
Used by the rundown (so the logged row and the public feed carry the label)
and by the site (so the page shows the same one).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.mlb.compile import DATA

MIN_BETS = 100
SMALL_MIN = 20
THRESH = 0.04
UNIT = {"bet": "one unit", "small": "half a unit"}


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
    t.loc[mk == "FAVNO", "kind"] = "FAVNO"
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


def action(kind):
    """'bet' | 'small' | 'pass' for a market, from its live record alone."""
    r = market_record(kind)
    if not r or r["roi"] <= 0 or r["recent_roi"] <= 0:
        return "pass"
    if r["n"] >= MIN_BETS:
        return "bet"
    if r["n"] >= SMALL_MIN:
        return "small"
    return "pass"


def bet_allowed(kind):
    return action(kind) == "bet"


def record_phrase(kind):
    """'has returned -22.4% over 56 settled bets (19-37) since 2026-08-27'."""
    r = market_record(kind)
    if not r:
        return "has no settled bets yet"
    return (f"has returned {r['roi']:+.1f}% over {r['n']} settled bets "
            f"({r['w']}-{r['n'] - r['w']}) since {r['since']}, and {r['recent_roi']:+.1f}% "
            f"over the most recent {r['recent_n']}")


def why_not_bet(kind):
    """One clause explaining why a flagged market is not a full BET."""
    r = market_record(kind)
    if not r:
        return "no settled record yet"
    if r["roi"] <= 0:
        return "the record is negative"
    if r["recent_roi"] <= 0:
        return "the most recent half is negative"
    if r["n"] < SMALL_MIN:
        return f"fewer than {SMALL_MIN} settled bets so far ({r['n']})"
    return f"profitable, but on fewer than {MIN_BETS} settled bets"


def _num(x):
    return x is not None and not pd.isna(x)


# The last two weeks of the regular season score fewer runs than earlier --
# division winners rest starters, eliminated teams run out call-ups, bullpens
# get emptied for evaluation rather than the win. Checked on 2023-2025
# held-out data (ops/experiment_late_season.py, 2026-09-24) against the exact
# frozen totals model the site runs: Brier at 7.5/8.5 gets meaningfully worse
# in this window, and the failure has a direction -- the model's own stated
# P(over) goes UP late in the season while the real over-rate goes DOWN by
# 5-9 points. Unders were not shown to be harmed; only Over calls are capped.
LATE_SEASON_DAYS = 14


def _is_over(direction):
    return str(direction).strip().lower().startswith("over")


def late_season_caution(row):
    d = row.get("days_before_season_end")
    return _num(d) and int(d) < LATE_SEASON_DAYS


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
        out.append(("ML", action("ML"), f"{side} to win", float(row.get("edge") or 0)))
    sf = spread_flag(row)
    if sf:
        out.append(("SPREAD", action("SPREAD"), f"{sf[0]} {sf[1]:+g}",
                    float(row.get("spread_edge") or 0)))
    tf = totals_flag(row)
    if tf:
        tier = action("TOTAL")
        if _is_over(tf[0]) and late_season_caution(row):
            tier = {"bet": "small", "small": "pass", "pass": "pass"}[tier]
        out.append(("TOTAL", tier, f"{tf[0].title()} {tf[1]:g} runs",
                    float(row.get("total_edge") or 0)))
    return out


def totals_why(row, what):
    """Why this specific totals pick isn't the market's own full word --
    the late-season caution takes priority over the generic record reason,
    since it is true regardless of how good the aggregate record looks."""
    if _is_over(what) and late_season_caution(row):
        d = int(row["days_before_season_end"])
        return (f"the last {LATE_SEASON_DAYS} days of the season score fewer runs than "
                f"earlier, on held-out data, and the model does not adjust for it -- Over "
                f"calls get overconfident here specifically ({d} days left)")
    return why_not_bet("TOTAL")


_RANK = {"bet": 3, "small": 2, "pass": 1}
_ORDER = {"ML": 0, "SPREAD": 1, "TOTAL": 2}
WORD = {"bet": "BET", "small": "SMALL BET", "pass": "PASS"}


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
    """'bet' | 'small' | 'pass' | 'stale'. A flagged market whose record says
    not to act is 'pass', the same word as a fair price: the instruction to
    the reader is the same."""
    v = str(row.get("verdict", ""))
    h = headline(row)
    if h:
        return h[1]
    if v.startswith("STALE"):
        return "stale"
    return "pass"


def pretty(kind, what, nick=lambda c: c):
    if kind == "ML":
        return f"{nick(what.split(' to win')[0])} to win"
    if kind == "SPREAD":
        team, line = what.split()
        return f"{nick(team)} {line}"
    return what


def label_for(row, nick=lambda c: c):
    """The exact words on the badge, for the log and the public feed."""
    h = headline(row)
    if h and h[1] in ("bet", "small"):
        kind, tier, what, _ = h
        return f"{WORD[tier]} · {pretty(kind, what, nick)}"
    if not h and str(row.get("verdict", "")).startswith("STALE"):
        return "PRICE STALE · re-check"
    return "PASS"


def ranked(rows):
    """Every BET and SMALL BET on the slate, best first: by word, then by
    the market's live return, then by edge. This is the list a reader acts
    on top to bottom; a slate with nothing in it means sit the day out.
    Returns [(row, (kind, tier, what, edge))]."""
    out = []
    for r in rows:
        for f in market_flags(r):
            if f[1] in ("bet", "small"):
                out.append((r, f))
    roi = {k: (market_record(k) or {}).get("roi", 0.0) for k in ("ML", "SPREAD", "TOTAL")}
    out.sort(key=lambda rf: (-_RANK[rf[1][1]], -float(roi[rf[1][0]]), -rf[1][3]))
    return out
