"""The MLB half of Model HQ: the same four tabs, card grammar, badges and
copy register as David's NFL page (src/site/nfl_site.py), in run units.

This Week / Parlay Lab / Track Record / Bayesian 101.
"""
from __future__ import annotations

import base64
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.core.kalshi import kalshi_fee
from src.mlb.compile import DATA
from src.site.names import mlb as nick
from src.mlb import labels

FIG = Path("figures/mlb/phase0")
RUNLINE_JUICE = 1.87   # a typical MLB run-line price, decimal


def american(p):
    if p <= 0 or p >= 1:
        return "n/a"
    return f"{-round(100 * p / (1 - p))}" if p >= 0.5 else f"+{round(100 * (1 - p) / p)}"


def fav_line(mu, home, away):
    """State the model's own predicted margin, not the book's run line.

    Printing "-1.5" here regardless of mu implied the model favored a side by
    a run and a half when it often predicts a tenth of a run, and it
    contradicted the run-line row on the same card.
    """
    if abs(mu) < 0.05:
        return "pick'em"
    return f"{home if mu > 0 else away} by {abs(mu):.1f}"


def img_tag(path):
    p = Path(path)
    if not p.exists():
        return ""
    b = base64.b64encode(p.read_bytes()).decode()
    return f'<img class="fig" src="data:image/png;base64,{b}" alt="{p.stem}">'


def verdict_badge(v, edge=None):
    """Display label for the moneyline verdict. The stored strings
    ("HIGH VALUE", "CAUTIOUS", "NO VALUE") are unchanged -- the log, the
    paper trade and the health gate parse them -- only what a reader sees
    changes. "HIGH VALUE" in green claimed something the record contradicts:
    moneyline flags have lost about 20% against Kalshi since 2026-08-27
    (docs/baselines.md). What the model actually produces is a
    *disagreement* with the price, so that is the word, in a neutral colour.
    Green is reserved for the totals call, the one market Kalshi has not
    been shown to beat."""
    v = str(v)
    e = "" if edge is None or pd.isna(edge) else f" &middot; {float(edge)*100:+.1f}%"
    if v.startswith("STALE"):
        return f'<span class="verdict v-caut">{v}</span>'
    if v.startswith("HIGH VALUE"):
        side = v.split("&mdash;")[-1].strip()
        return f'<span class="verdict v-dis">Model disagrees &middot; {side}{e}</span>'
    if v.startswith("CAUTIOUS"):
        side = v.split("&mdash;")[-1].strip().replace("small edge on ", "")
        return f'<span class="verdict v-none">Small disagreement &middot; {side}{e}</span>'
    if v.startswith("NO VALUE"):
        return '<span class="verdict v-none">Price is fair &middot; no bet</span>'
    return '<span class="verdict v-none">No price yet</span>'


SQUARE3_SIGMAS = 0.40   # model-minus-market gap, in predictive sigmas, past which
                        # the likelier explanation is news the model has not seen


def square3_gap(p_model, p_market):
    lo, hi = 1e-6, 1 - 1e-6
    return float(norm.ppf(min(max(p_model, lo), hi)) - norm.ppf(min(max(p_market, lo), hi)))


def verdict_tier(v):
    v = str(v)
    if v.startswith("HIGH VALUE"):
        return "high"
    if v.startswith("CAUTIOUS"):
        return "small"
    if v.startswith("NO VALUE") or v.startswith("STALE"):
        return "none"
    return "nopr"


def _num(x):
    return x is not None and not pd.isna(x)


def _cents(p):
    return f"{float(p)*100:.0f}&cent;"


_WORD = {"bet": ("BET", "v-high"), "small": ("SMALL BET", "v-small"), "pass": ("PASS", "v-none")}
_KIND_LABEL = {"ML": "Moneyline", "SPREAD": "Run line", "TOTAL": "Total runs"}
_KIND_VERB = {"ML": "Picking a team to win", "SPREAD": "Betting the run line",
              "TOTAL": "Betting total runs"}


def _market_price_and_wins(r, kind, what):
    """(cost, 'what makes it win' sentence) for one flagged market."""
    home, away = r["home"], r["away"]
    if kind == "ML":
        side = what.split(" to win")[0]
        cost = r.get("mkt_home") if side == home else r.get("mkt_away")
        return cost, f"the {nick(side)} win"
    if kind == "SPREAD":
        team, line = what.split()
        line = float(line)
        cost = r.get("spread_price")
        if line < 0:
            return cost, f"the {nick(team)} win by <b>{int(-line) + 1} runs or more</b>"
        return cost, f"the {nick(team)} win, or lose by <b>{int(line)} run{'s' if int(line) != 1 else ''} or fewer</b>"
    direction, ln, _ = what.split()
    ln = float(ln)
    if direction == "Under":
        cost = r.get(f"mkt_under_{ln:g}")
        if not _num(cost) and _num(r.get(f"mkt_over_{ln:g}")):
            cost = 1 - float(r[f"mkt_over_{ln:g}"])
        return cost, f"the two teams score <b>{int(ln)} runs or fewer</b>"
    return r.get(f"mkt_over_{ln:g}"), f"the two teams score <b>{int(ln) + 1} runs or more</b>"


def _rungs(r):
    """Every totals rung this row was priced at, from its own columns."""
    out = []
    for k in r.keys() if hasattr(r, "keys") else []:
        if str(k).startswith("p_over_"):
            try:
                out.append(float(str(k)[7:]))
            except ValueError:
                pass
    return sorted(out)


def _main_line(r):
    """The rung Kalshi is really trading: the one whose over price is
    nearest 50 cents. A reader who opens the app sees this number first."""
    best = None
    for ln in _rungs(r):
        mo = r.get(f"mkt_over_{ln:g}")
        if _num(mo) and (best is None or abs(float(mo) - 0.5) < best[0]):
            best = (abs(float(mo) - 0.5), ln)
    return None if best is None else best[1]


def _total_context(r, what):
    """For a totals pick: how often our model says it wins, what the price
    implies, and where the market's main line sits when the pick is on a
    different rung. Keeps a long shot from reading like a favourite."""
    direction, ln, _ = what.split()
    ln = float(ln)
    po = r.get(f"p_over_{ln:g}")
    if not _num(po):
        return ""
    p = float(po) if direction == "Over" else 1 - float(po)
    cost, _ = _market_price_and_wins(r, "TOTAL", what)
    s = f'Our model says it wins <b>{p*100:.0f}%</b> of the time'
    if _num(cost):
        s += f'; the price implies {float(cost)*100:.0f}%'
    ml_ = _main_line(r)
    if ml_ is not None and ml_ != ln:
        s += (f'. Kalshi&rsquo;s main line is <b>{ml_:g}</b>; this pick is on the {ln:g} rung '
              f'because that is the price the model disagrees with')
    s += "."
    alt = _safer_ticket(r, direction, ln)
    if alt:
        a_ln, a_cost, a_p, a_edge = alt
        s += (f' <b>Same opinion, safer ticket:</b> {direction} {a_ln:g} at {_cents(a_cost)} '
              f'wins {a_p*100:.0f}% of the time by our model, edge {a_edge*100:+.1f}%; '
              f'it pays less but hits more often.')
    return s


def _safer_ticket(r, direction, ln):
    """The same over/under opinion on the rung nearest the market's main
    line that still carries a positive edge after fees, if it is a
    different rung from the pick. (rung, cost, p, edge) or None."""
    from src.core.kalshi import kalshi_fee
    ml_ = _main_line(r)
    if ml_ is None:
        return None
    best = None
    for rung in _rungs(r):
        if rung == ln:
            continue
        po = r.get(f"p_over_{rung:g}")
        if not _num(po):
            continue
        if direction == "Over":
            cost, p = r.get(f"mkt_over_{rung:g}"), float(po)
        else:
            cost, p = r.get(f"mkt_under_{rung:g}"), 1 - float(po)
            if not _num(cost) and _num(r.get(f"mkt_over_{rung:g}")):
                cost = 1 - float(r[f"mkt_over_{rung:g}"])
        if not _num(cost):
            continue
        cost = float(cost)
        edge = p - cost - kalshi_fee(cost)
        if edge <= 0.02 or p <= 0.45:
            continue          # must still be a real edge and near a coin flip or better
        key = abs(rung - ml_)
        if best is None or key < best[0]:
            best = (key, rung, cost, p, edge)
    return None if best is None else best[1:]


def _pretty(kind, what):
    if kind == "ML":
        return f"{nick(what.split(' to win')[0])} to win"
    if kind == "SPREAD":
        team, line = what.split()
        return f"{nick(team)} {line}"
    return what


def _how(r, kind, tier, what, edge):
    """One sentence that is an instruction. BET / SMALL BET: what to buy, the
    price, the size, what makes it win. PASS: what the model likes, and the
    record that says not to act on it."""
    cost, wins = _market_price_and_wins(r, kind, what)
    ptxt = f" at <b>{_cents(cost)}</b>" if _num(cost) else ""
    if tier == "pass":
        s = (f'The model likes <b>{_pretty(kind, what)}</b>{ptxt} (edge {edge*100:+.1f}% after fees), '
             f'but {_KIND_VERB[kind].lower()} {labels.record_phrase(kind)}. '
             f'<b>Sit this one out</b>: {labels.why_not_bet(kind)}.')
        return f'<div class="how pass">{s}</div>'
    s = (f'Buy <b>{_pretty(kind, what)}</b>{ptxt} on Kalshi, <b>{labels.UNIT[tier]}</b>. '
         f'Edge {edge*100:+.1f}% after fees.')
    if kind == "TOTAL":
        s += " " + _total_context(r, what)
    if tier == "small":
        s += (f' <span class="warn">Half a unit because {_KIND_VERB[kind].lower()} '
              f'{labels.record_phrase(kind)}: {labels.why_not_bet(kind)}.</span>')
    return f'<div class="how">{s}</div>'


# The daily cron slots (.github/workflows/mlb-daily.yml), for the pill's
# "next slate in ..." countdown once every bet has started.
NEXT_RUNS_UTC = "15:00,16:30,20:00"


def start_here(slate):
    """The ranked list at the top of the slate: every BET and SMALL BET,
    best first, each one an instruction. Empty means sit the day out, and
    the box says so instead of hiding."""
    ranked = labels.ranked(slate)
    # Price age for the live pill. Rows carry run_ts once logged; the site is
    # built in the same job minutes after pricing, so build time is the
    # fallback and is honest to within that window.
    run = str(next((r.get("run_ts") for r in slate if r.get("run_ts")), "") or
              pd.Timestamp.utcnow().isoformat(timespec="seconds"))
    if not ranked:
        n = sum(1 for r in slate for _ in labels.market_flags(r))
        why = (f' The model likes {n} side{"s" if n != 1 else ""} today, all in markets '
               f'that have lost money in live betting; they are marked PASS on the cards '
               f'so you can see them, and the record says not to buy them.') if n else ""
        return (f'<div class="start" id="start-mlb" data-run="{run}" data-next="{NEXT_RUNS_UTC}">'
                f'<b>Start here:</b> nothing to bet today.{why}</div>')
    items = []
    for i, (r, (kind, tier, what, edge)) in enumerate(ranked, 1):
        cost, wins = _market_price_and_wins(r, kind, what)
        ptxt = f" at {_cents(cost)}" if _num(cost) else ""
        kick = str(r.get("kick_iso") or "")
        items.append(
            f'<li data-kick="{kick}" data-what="{_pretty(kind, what)}" '
            f'data-game="{nick(r["away"])} at {nick(r["home"])}">'
            f'<span class="verdict {_WORD[tier][1]} mini">{_WORD[tier][0]}</span> '
            f'<b>{_pretty(kind, what)}</b>{ptxt}, {labels.UNIT[tier]}: '
            f'{nick(r["away"])} at {nick(r["home"])}. Edge {edge*100:+.1f}%. '
            + (_total_context(r, what) + " " if kind == "TOTAL" else "")
            + f'<span class="closes" data-kick="{kick}"></span></li>')
    return (f'<div class="start" id="start-mlb" data-run="{run}" data-next="{NEXT_RUNS_UTC}">'
            f'<b>Start here</b>: {len(items)} '
            f'bet{"s" if len(items) != 1 else ""} today, best first. A unit is whatever you '
            f'decided a unit is before you opened this page. Each one has to be bought before first pitch.'
            f'<ol>{"".join(items)}</ol></div>')


def game_card(r):
    """One card, for someone who has never placed a bet.

    Three markets, three lines, always in the same order: Moneyline, Run
    line, Total runs. Each carries its own word -- BET, SMALL BET, PASS --
    decided by that market's own live record (src/mlb/labels.py). The
    headline is the strongest of the three, with one sentence saying what
    to buy, the price, the size, and what makes it win; or, on a PASS, what
    the model likes and why the record says leave it. Every number is under
    Details."""
    mu, sg, pm = r["mu"], r["sigma"], r["p_home"]
    home, away = r["home"], r["away"]
    H, A = nick(home), nick(away)
    fav0 = home if pm >= 0.5 else away
    fav_p = pm if pm >= 0.5 else 1 - pm
    v = str(r.get("verdict", ""))
    mk_h, mk_a = r.get("mkt_home"), r.get("mkt_away")
    has_price = _num(mk_h)

    flags = {f[0]: f for f in labels.market_flags(r)}     # kind -> (kind, tier, what, edge)
    head = labels.headline(r)

    # ---- the three lines ----
    mkt_rows = []
    for kind in ("ML", "SPREAD", "TOTAL"):
        f = flags.get(kind)
        if f:
            word, cls = _WORD[f[1]]
            val = (f'<span class="verdict {cls} mini">{word}</span> {_pretty(kind, f[2])} '
                   f'<span class="lnote">{f[3]*100:+.1f}%'
                   f'{" &middot; market unproven" if f[1] == "pass" else ""}</span>')
        else:
            word, cls = _WORD["pass"]
            val = f'<span class="verdict {cls} mini">{word}</span> <span class="lnote">price is fair</span>'
        mkt_rows.append(f'<div class="mrow"><span class="mlab">{_KIND_LABEL[kind]}</span>{val}</div>')
    mkts = f'<div class="mkts">{"".join(mkt_rows)}</div>'

    # ---- headline ----
    if head:
        kind, tier, what, edge = head
        word, cls = _WORD[tier]
        badge = (f'<span class="verdict {cls}">{word} &middot; {_pretty(kind, what)}</span>'
                 if tier != "pass" else f'<span class="verdict {cls}">{word}</span>')
        body = _how(r, kind, tier, what, edge)
        side = what.split(" to win")[0] if kind == "ML" else (what.split()[0] if kind == "SPREAD" else None)
        if tier != "pass" and side and side != fav0:
            body += (f'<div class="also">We still expect the <b>{nick(fav0)}</b> to win '
                     f'({fav_p*100:.0f}%); this is a price bet on the {nick(side)}, not a pick.</div>')
    elif v.startswith("STALE"):
        tier = "stale"
        badge = '<span class="verdict v-caut">PRICE STALE &middot; re-check</span>'
        body = ('<div class="how">The price we saw is too old to act on. Open Kalshi and look '
                'before you buy anything here.</div>')
    else:
        tier = "pass"
        badge = '<span class="verdict v-none">PASS</span>'
        body = ('<div class="how">Every price is fair. Nothing here is worth buying today.</div>'
                if has_price else '<div class="how">No price on Kalshi yet.</div>')
    if abs(fav_p - 0.5) < 0.005:
        who = '<div class="who">Who we think wins: <b>too close to call</b></div>'
    else:
        who = f'<div class="who">Who we think wins: <b>{nick(fav0)}</b> ({fav_p*100:.0f}% chance)</div>'

    # ---- details ----
    d = []
    for kind, f in flags.items():
        if head and kind == head[0]:
            continue
        d.append(_how(r, kind, f[1], f[2], f[3]))
    d.append(f'<div class="gap">Model line: <b>{fav_line(mu, home, away)}</b> &plusmn;{sg:.1f} runs '
             f'&middot; fair moneyline {american(fav_p)}</div>')
    d.append(f'<div class="gap">Starters: {r.get("away_sp") or "TBD"} ({A}) vs '
             f'{r.get("home_sp") or "TBD"} ({H})'
             + (' &middot; <span style="color:var(--yellow)">a starter is unlisted, range widened</span>'
                if r.get("sp_unknown") else '') + '</div>')
    focus = fav0
    pf = pm if focus == home else 1 - pm
    bars = (f'<div class="brow"><span class="blab">Our model</span><div class="btrack">'
            f'<div class="bfill model" style="width:{pf*100:.1f}%"></div></div>'
            f'<span class="bval">{pf*100:.0f}%</span></div>')
    pn = r.get("p_home_narrative")
    if r.get("narrative_shift") and _num(pn):
        pnf = pn if focus == home else 1 - pn
        bars += (f'<div class="brow"><span class="blab">+ narrative</span><div class="btrack">'
                 f'<div class="bfill model" style="width:{pnf*100:.1f}%;opacity:.5"></div></div>'
                 f'<span class="bval">{pnf*100:.0f}%</span></div>')
    if has_price:
        mk = float(mk_h)
        mk_focus = mk if focus == home else (float(mk_a) if _num(mk_a) else 1 - mk)
        bars += (f'<div class="brow"><span class="blab">Kalshi price</span><div class="btrack">'
                 f'<div class="bfill mkt" style="width:{mk_focus*100:.1f}%"></div></div>'
                 f'<span class="bval">{_cents(mk_focus)}</span></div>')
        age = r.get("price_age_min")
        agetxt = ("" if not _num(age) else " &middot; price fetched live" if age <= 2
                  else f" &middot; price {age:.0f} min old" if age <= 15
                  else f' &middot; <span style="color:var(--yellow)">price {age:.0f} min old, the market has probably moved</span>')
        d.append(f'<div class="bars">{bars}</div>'
                 f'<div class="gap">Both bars: chance the <b>{nick(focus)}</b> win. '
                 f'Gap: <b>{(pf-mk_focus)*100:+.0f}</b> points{agetxt}</div>')
        mlf = flags.get("ML")
        if mlf:
            side = mlf[2].split(" to win")[0]
            v_model = pm if side == home else 1 - pm
            v_mkt = mk if side == home else (float(mk_a) if _num(mk_a) else 1 - mk)
            z = square3_gap(v_model, v_mkt)
            if z > SQUARE3_SIGMAS:
                d.append(f'<div class="gap sq3">Model and market are about <b>{z*sg:.1f}</b> runs apart '
                         f'here, against a typical miss of &plusmn;{sg:.1f}. A gap that size usually '
                         f'means the model has not seen some news; check lineups before acting.</div>')
    else:
        d.append(f'<div class="bars">{bars}</div><div class="gap">Kalshi has not opened this game yet.</div>')
    if _num(r.get("p_home_cover")):
        ph_c, pa_c = float(r["p_home_cover"]), float(r["p_away_cover"])
        q = []
        if _num(r.get("mkt_sp_home")):
            q.append(f"{H} -1.5 at {_cents(r['mkt_sp_home'])}")
        if _num(r.get("mkt_sp_away")):
            q.append(f"{A} -1.5 at {_cents(r['mkt_sp_away'])}")
        d.append(f'<div class="gap">Run line: model has {H} -1.5 covering {ph_c*100:.0f}%, '
                 f'{A} -1.5 covering {pa_c*100:.0f}%'
                 + (f' &middot; Kalshi: {", ".join(q)}' if q else " &middot; no Kalshi quote") + '</div>')
    mt = r.get("mu_total")
    if _num(mt):
        wx = (" &middot; roof closed" if r.get("roof_closed") in (True, 1, 1.0, "True")
              else (f' &middot; {int(float(r["temp_f"]))}&deg;F at first pitch' if _num(r.get("temp_f")) else ""))
        ladder = []
        for ln in _rungs(r):
            po = r.get(f"p_over_{ln:g}")
            mo = r.get(f"mkt_over_{ln:g}")
            if not _num(po) or not _num(mo):
                continue        # only rungs Kalshi actually lists
            po = float(po)
            lean, p = ("Over", po) if po >= 0.5 else ("Under", 1 - po)
            mtxt = f" (Kalshi {(float(mo) if lean == 'Over' else 1 - float(mo))*100:.0f}%)"
            ladder.append(f'{lean} {ln:g}: {p*100:.0f}%{mtxt}')
        ml_ = _main_line(r)
        d.append(f'<div class="gap">Total runs: model expects <b>{float(mt):.1f}</b>{wx}'
                 + (f' &middot; Kalshi&rsquo;s main line is {ml_:g}' if ml_ is not None else "")
                 + (f' &middot; {" &middot; ".join(ladder)}' if ladder else "") + '</div>')
        tf = flags.get("TOTAL")
        if tf:
            direction, ln, _ = tf[2].split()
            model_side = "Over" if float(mt) > float(ln) else "Under"
            if model_side != direction:
                d.append(f'<div class="gap">Note: the model\'s own runs number ({float(mt):.1f}) leans '
                         f'{model_side}; the call is {direction} because that side is the one '
                         f'Kalshi has mispriced.</div>')
    if r.get("note"):
        d.append(f'<div class="gap"><i>Narrative: {r["note"]}</i></div>')
    details = f'<details class="why"><summary>Details</summary>{"".join(d)}</details>'

    kick = str(r.get("kick_iso") or "")
    return (f'<div class="card tier-{tier}" data-teams="{away}@{home}"><div class="match"><span class="teams">{A} at {H}</span>'
            f'<span class="date" data-kick="{kick}">{r["date"]}</span></div>'
            f'<div class="leadv">{badge}</div>{body}{mkts}{who}{details}</div>')


def build_parlays(rows, top_n=10):
    """Same banding and EV grammar as the NFL Parlay Lab."""
    legs = []
    for r in rows:
        pm = r["p_home"]
        mk = r.get("mkt_home")
        if mk is not None and not pd.isna(mk):
            mk = float(mk)
            side, p = (r["home"], pm) if pm >= 0.5 else (r["away"], 1 - pm)
            price = mk if side == r["home"] else 1 - mk
            if 0.02 < price < 0.98:
                legs.append({"game": f'{r["away"]}@{r["home"]}', "desc": f"{side} ML",
                             "p": p, "dec": 1 / (price + kalshi_fee(price)),
                             "wild": (p - price) > 0.15})
        phc = r.get("p_home_cover")
        if phc is not None and not pd.isna(phc):
            phc, pac = float(phc), float(r["p_away_cover"])
            side, p = ((r["home"], phc) if phc >= pac else (r["away"], pac))
            legs.append({"game": f'{r["away"]}@{r["home"]}', "desc": f"{side} -1.5",
                         "p": p, "dec": RUNLINE_JUICE, "wild": False})
    parlays = []
    for k in (2, 3):
        for combo in itertools.combinations(legs, k):
            if len({c["game"] for c in combo}) < k:
                continue
            p = float(np.prod([c["p"] for c in combo]))
            dec = float(np.prod([c["dec"] for c in combo]))
            wild = any(c["wild"] for c in combo)
            band = ("moon" if (wild or dec > 11) else "safe" if dec <= 2.2
                    else "balanced" if dec <= 4.0 else "long")
            parlays.append({"legs": ", ".join(c["desc"] for c in combo), "n": k,
                            "p": p, "payout_dec": dec, "ev": p * dec - 1, "band": band})
    out = []
    for band, key in (("safe", lambda x: x["p"]), ("balanced", lambda x: x["ev"]),
                      ("long", lambda x: x["ev"]), ("moon", lambda x: x["ev"])):
        out.extend(sorted([x for x in parlays if x["band"] == band],
                          key=key, reverse=True)[:top_n])
    return out


def _history():
    p = DATA / "mlb_history.csv"
    return pd.read_csv(p) if p.exists() else None


def ml_record():
    """The live moneyline record phrase (src/mlb/labels.py), from the same
    file Live Bet Performance renders, so prose and table cannot disagree."""
    return labels.record_phrase("ML")


def live_performance_html():
    """Every flagged bet the daily cron has settled, sourced straight from
    data/mlb/paper_trades.csv. That file was already regenerated every day by
    mlb-daily.yml; this is the piece that was missing — the numbers went into
    a CSV in git and nowhere a person would ever see them."""
    p = DATA / "paper_trades.csv"
    if not p.exists() or not len(pd.read_csv(p)):
        return ""
    t = pd.read_csv(p).sort_values("date")
    n, wins = len(t), int(t["won"].sum())
    staked, pnl = t["stake"].sum(), t["pnl"].sum()
    roi = 100 * pnl / staked if staked else 0.0
    # Moneyline and totals are different markets with different records;
    # one blended number hid that. Totals is the market worth watching.
    mk = t["market"].astype(str).str.upper()
    t["kind"] = np.select([mk == "ML", mk == "SPREAD"], ["Moneyline", "Spread"], "Totals")
    by_kind = "".join(
        f'<tr><td>{k}</td><td>{len(g)}</td><td>{int(g.won.sum())}-{len(g)-int(g.won.sum())}</td>'
        f'<td>{"+" if g.pnl.sum() >= 0 else ""}${g.pnl.sum():.2f}</td>'
        f'<td>{(100*g.pnl.sum()/g.stake.sum() if g.stake.sum() else 0):+.1f}%</td></tr>'
        for k, g in t.groupby("kind"))

    rows = "".join(
        f'<tr><td>{x.date}</td><td>{x.market}</td><td>{x.pick}</td>'
        f'<td>{x.ask*100:.0f}&cent;</td><td>{x.edge*100:+.1f}%</td>'
        f'<td class="{"hit" if x.won else "miss"}">{"WON" if x.won else "lost"}</td>'
        f'<td class="{"hit" if x.pnl >= 0 else "miss"}">{"+" if x.pnl >= 0 else ""}${x.pnl:.2f}</td>'
        f'<td>{"+" if x.cum_pnl >= 0 else ""}${x.cum_pnl:.2f}</td></tr>'
        for x in t.tail(40).itertuples())

    return f"""
<h2>Live Bet Performance</h2>
<p class="sub">Every game the model flagged HIGH VALUE, staked at a flat $15 and settled
against the real final score, updated automatically every morning by the same run that
prices tomorrow's slate. This is a record of what following the site would have earned,
not the walk-forward backtest below. It started 2026-08-27 and is still a small
sample. Treat the win rate honestly and the dollar figure as a rough slippage estimate
until this has run for a full season.</p>
<div class="metrics-row">
<div class="mrow"><span class="mn">{n}</span><span class="ml">bets settled</span></div>
<div class="mrow"><span class="mn">{wins}-{n-wins}</span><span class="ml">record</span></div>
<div class="mrow"><span class="mn">${pnl:+.2f}</span><span class="ml">P&amp;L on ${staked:.0f} staked</span></div>
<div class="mrow"><span class="mn">{roi:+.1f}%</span><span class="ml">ROI</span></div>
</div>
<table><tr><th>By market</th><th>Bets</th><th>Record</th><th>P&amp;L</th><th>ROI</th></tr>{by_kind}</table>
<table><tr><th>Date</th><th>Market</th><th>Pick</th><th>Price</th><th>Edge</th>
<th>Result</th><th>P&amp;L</th><th>Running total</th></tr>{rows}</table>
"""


def track_record_html():
    hist = _history()
    if hist is None:
        res = FIG / "phase0_results.csv"
        if not res.exists():
            return '<p class="sub">Run ops/export_history.py to populate this tab.</p>'
        t = pd.read_csv(res)
        rows = "".join(
            f'<tr><td>{r.model}</td><td>{int(r.n)}</td><td>{r.nll:.4f}</td>'
            f'<td>{r.rmse:.3f}</td><td>{r.brier:.4f}</td>'
            f'<td>{"" if pd.isna(r.max_calib_dev) else f"{r.max_calib_dev:.3f}"}</td></tr>'
            for r in t.itertuples())
        return (f'<table><tr><th>Model</th><th>Games</th><th>NLL</th>'
                f'<th>RMSE (runs)</th><th>Brier</th><th>Max calib. dev.</th></tr>{rows}</table>')

    by_season = hist.groupby("season").apply(lambda g: pd.Series({
        "games": len(g),
        "winner_pct": 100 * g["su_correct"].mean(),
        "rl_pct": 100 * g["rl_correct"].mean(),
        "avg_miss": (g["y"] - g["mu"]).abs().mean(),
    }), include_groups=False).round(1)
    srows = "".join(
        f'<tr><td>{s}</td><td>{int(r.games)}</td><td>{r.winner_pct:.1f}%</td>'
        f'<td>{r.rl_pct:.1f}%</td><td>{r.avg_miss:.2f}</td></tr>'
        for s, r in by_season.iterrows())

    edges = [0, .35, .45, .55, .65, 1.0]
    labels = ["0-35%", "35-45%", "45-55%", "55-65%", "65-100%"]
    bins = pd.cut(hist["p_home"], edges, labels=labels)
    calib = hist.groupby(bins, observed=True).apply(lambda g: pd.Series({
        "games": len(g),
        "model_said": 100 * g["p_home"].mean(),
        "home_actually_won": 100 * (g["y"] > 0).mean(),
    }), include_groups=False).round(1)
    crows = "".join(
        f'<tr><td>{i}</td><td>{int(r.games)}</td><td>{r.model_said:.0f}%</td>'
        f'<td>{r.home_actually_won:.0f}%</td></tr>' for i, r in calib.iterrows())

    blocks = []
    PER_SEASON = 150   # full seasons are ~2400 games; all three inlined made
                       # the page 1.7 MB, which is unusable on a phone.
    for season, g in hist.groupby("season"):
        shown = g.tail(PER_SEASON)
        dropped = len(g) - len(shown)
        g = shown
        rows = "".join(
            f'<tr><td>{x.gameday}</td><td>{x.away_team} @ {x.home_team}</td>'
            f'<td>{fav_line(x.mu, x.home_team, x.away_team)}</td>'
            f'<td>{x.p_home*100:.0f}%</td>'
            f'<td>{"+" if x.y > 0 else ""}{x.y:.0f} ({x.home_team} {x.home_score:.0f} - '
            f'{x.away_team} {x.away_score:.0f})</td>'
            f'<td class="{"hit" if x.su_correct else "miss"}">{x.su_pick} &middot; '
            f'{"HIT" if x.su_correct else "miss"}</td>'
            f'<td class="{"hit" if x.rl_correct else "miss"}">{x.rl_pick} -1.5 &middot; '
            f'{"HIT" if x.rl_correct else "miss"}</td></tr>'
            for x in g.itertuples(index=False))
        cap = (f'<p class="sub">Showing the last {len(g)} games of {len(g) + dropped}. '
               f'The season percentages above are computed over all '
               f'{len(g) + dropped}; only this listing is truncated, to keep '
               f'the page loadable on a phone.</p>' if dropped else "")
        blocks.append(
            f'<details><summary>{season} season &mdash; last {len(g)} games</summary>'
            f'{cap}<table><tr><th>Date</th><th>Game</th><th>Bayesian Model line</th>'
            f'<th>Model: home team wins</th><th>Final margin (home first)</th>'
            f'<th>Winner pick</th><th>Run-line pick</th></tr>{rows}</table></details>')

    return f"""
<p class="sub">Everything on this tab is stated from the home team's perspective.
Every prediction below was made by the Bayesian Model before it had seen the game:
it refits weekly on games already played, exactly as it runs live. Two report cards:
"winner pick" is the model picking the game outright, and "run-line pick" is its side
at the standard &plusmn;1.5. <b>Read these against the market, not against a coin
flip.</b> Picking the winner 55% of the time is real skill and is not a betting edge:
the market picks winners at least as well, and its prices already contain that. The
test that matters is the paper trade above, against Kalshi's actual price. On the run
line the model covers 40&ndash;44% over 2023&ndash;2025 against a 53.5% breakeven, so
that column is a report card, not a bet.</p>
<table><tr><th>Season</th><th>Games</th><th>Model picks winner</th>
<th>Model on the run line</th><th>Avg miss (runs)</th></tr>{srows}</table>
<p class="sub">Honesty check. Take all the games where the model gave the home team a
certain range of winning chances, then check how often the home team really won. A
single game cannot test a probability; only a pile of games can. When the last two
columns roughly match in every row, the model's stated confidence is honest.</p>
<table><tr><th>Games grouped by prediction</th><th>Games</th>
<th>The group's predictions averaged</th><th>Home teams actually won</th></tr>{crows}</table>
<div class="two">{img_tag(FIG / "reliability_linear.png")}{img_tag(FIG / "kalman_ratings_2026.png")}</div>
{"".join(blocks)}"""


B101 = """
<div class="b101">
<p><b>The one idea behind everything here.</b> A prediction is not a number, it is a
range of belief. This model never says "the Yankees will win by 1." It says "our best
guess is Yankees by 1, and here is exactly how sure we are." Bayesian modeling is the
math of keeping honest track of that sureness: start with a reasonable belief, let each
game's evidence pull it, and never claim more certainty than the evidence paid for.</p>
<p><b>Why every prediction says plus-or-minus four and a half runs.</b> That number is
measured, not assumed. Take every game since 2008, compare the final run differential to
the best pre-game prediction anyone can make, and the typical miss is about 4.4 runs.
Vegas misses by about the same. Baseball is decided by a seeing-eye single, a checked
swing, and a reliever who did not have it that night. The skill is not shrinking the 4.4;
it is knowing your 4.4 honestly.</p>
<p><b>What the model has and has not shown.</b> On outcomes it is calibrated: when it
says 60%, the home team wins about 60% of the time (the Honesty check on Track Record).
Against the market it has not won: every time it thinks a team is cheaper than it
should be, the reason is "this game is closer than you think", and buying every such
team {ML_RECORD}. The market is calibrated too, and it has information the model does not
(lineups, injuries, weather at first pitch, the sharp money). The one market where the
model has not been beaten is total runs, where temperature and park are inputs the model
reads well. That is why the total gets a green badge and the moneyline gets a grey one.</p>
<p><b>The Kalman filter: the model's memory.</b> Every club carries a rating, runs better
or worse than average on a neutral field, updated after every game by a Kalman filter,
the same math that navigated Apollo to the moon. The filter's genius is knowing how far
to move: a 10-run blowout by a club it already trusts barely moves the rating; the same
blowout by a mystery club moves it a lot. The rating drifts a little each day off, and
between seasons it shrinks toward average and its uncertainty balloons. Home field is
learned, not assumed, and it converged to about four hundredths of a run: in baseball,
home field is nearly nothing.</p>
<p><b>The stat sheet.</b> What the model is fed, for every game, always computed only
from games played before it:</p>
<table>
<tr><th>Input</th><th>What it is</th></tr>
<tr><td>Kalman rating gap</td><td>Club strength difference, runs on a neutral field</td></tr>
<tr><td>Rating uncertainty</td><td>How well the filter currently knows both clubs</td></tr>
<tr><td>Scoring form and its slope</td><td>Recent run differential, recent games weighted more, plus the trajectory of the last ten: two clubs at the same level separate here if one is rising and one is falling</td></tr>
<tr><td>Baserunner pressure</td><td>Runners put on per inning, for and against. A club getting runners on every few innings is bound to break through</td></tr>
<tr><td>Stranding rate</td><td>Share of baserunners left on base: the difference between traffic and runs</td></tr>
<tr><td>Late-inning margin</td><td>Runs scored and allowed in innings seven and later: bullpen and nerve, without needing a separate bullpen model</td></tr>
<tr><td>Starter FIP</td><td>Strikeouts, walks and homers per inning for tonight's listed starter, the part of run prevention a pitcher actually controls</td></tr>
<tr><td>Starter command and consistency</td><td>Strike rate, and how steady that strike rate is start to start. A pitcher who spots it every time out is worth more than one who averages the same but scatters</td></tr>
<tr><td>Starter rest</td><td>Days since the last outing</td></tr>
<tr><td>Bullpen quality and workload</td><td>Relief FIP, and relief innings thrown in the last three days. A gassed pen is a real edge for the other side</td></tr>
<tr><td>Park factor</td><td>How much this yard inflates runs, computed only from prior seasons</td></tr>
<tr><td>Day or night, rest, division</td><td>The small stuff, included because it is free</td></tr>
</table>
<p>Deliberately absent: betting lines. A model fed the market's answer can only agree
with the market; this one has to form its own opinion so the two can genuinely disagree.</p>
<p><b>The narrative edge.</b> A human can enter a story the model cannot see: a club in a
playoff push, a starter fresh off the injured list, a clubhouse that has come together.
It moves the predicted margin by at most one run, and it always widens the uncertainty,
so an opinion can shift the number but can never buy confidence. Both the model-only and
the model-plus-narrative calls are logged and scored separately, so whether the human
helps is a number at the end of the season, not a feeling.</p>
<p><b>What this model honestly cannot see.</b> A late scratch, three regulars resting in
a day game after a night game, wind blowing out. Every flag gets a human news
check before anything happens. When the market disagrees with the model, the market is
usually right; this system exists to find the exceptions and to know the difference.</p>
</div>"""


def health_strip(health):
    """One line a reader can trust before any number: when the run was made,
    from what, and whether every facet checked out. Built from
    data/mlb/health.json (ops/healthcheck.py); absent when no gate has run."""
    if not health:
        return ('<div class="health warn">No health record for this build. '
                'Treat every number below as unverified.</div>')
    checks = [c for c in health.get("checks", []) if c.get("stage") == "data"]
    fails = [c for c in checks if not c["ok"] and c["hard"]]
    warns = [c for c in checks if not c["ok"] and not c["hard"]]
    ts = pd.Timestamp(health.get("run_ts")).strftime("%Y-%m-%d %H:%M UTC")
    by = {c["name"]: c["detail"] for c in checks}
    facts = " &middot; ".join(x for x in (
        f"run {ts}",
        f"slate {health.get('slate_date')}",
        f"games priced {by.get('every upcoming game priced', '?').split(',')[0]}",
        f"prices {by.get('Kalshi prices fresh', '?').split(' priced')[0]} fresh" if "Kalshi prices fresh" in by else "",
        f"weather {by.get('weather at every open-roof game', '?').split(' have')[0]}" if "weather at every open-roof game" in by else "",
        f"starters {by.get('probable pitchers known', '?').split(' games')[0]}" if "probable pitchers known" in by else "",
    ) if x)
    if fails:
        cls, head = "fail", f"{len(fails)} check{'s' if len(fails) > 1 else ''} FAILED: " + ", ".join(c["name"] for c in fails)
    elif warns:
        cls, head = "warn", f"all hard checks passed; {len(warns)} warning{'s' if len(warns) > 1 else ''}: " + ", ".join(c["name"] for c in warns)
    else:
        cls, head = "ok", f"all {len(checks)} checks passed"
    rows = "".join(
        f'<li class="{"ok" if c["ok"] else ("fail" if c["hard"] else "warn")}">'
        f'{c["name"]}: {c["detail"]}</li>' for c in checks)
    return (f'<div class="health {cls}"><b>Run health:</b> {head}<br>'
            f'<span class="facts">{facts}</span>'
            f'<details><summary>every check</summary><ul>{rows}</ul></details></div>')


def card_tier(r):
    return labels.tier_for(r)


def tier_buttons(slate, scope):
    """Filter the slate to one kind of card, with counts, so "what is there
    to act on today" is one tap instead of a scroll."""
    from collections import Counter
    counts = Counter(card_tier(r) for r in slate)
    spec = [("all", "All games", len(slate)), ("bet", "Bets", counts.get("bet", 0)),
            ("small", "Small bets", counts.get("small", 0)), ("pass", "Pass", counts.get("pass", 0)),
            ("stale", "Stale price", counts.get("stale", 0))]
    btns = "".join(
        f'<button id="tier-{scope}-{k}" class="bandbtn{" on" if k == "all" else ""}"'
        f'{"" if n or k == "all" else " disabled"} onclick="mtier(\'{k}\',\'{scope}\')">'
        f'{lab}<span class="cnt">{n}</span></button>' for k, lab, n in spec)
    return f'<div class="bandbar tierbar">{btns}</div>'


def render(slate, today, health=None):
    from src.core.teams import slate_sort_key
    slate = sorted(slate, key=slate_sort_key)
    cards = "".join(game_card(r) for r in slate) or \
        '<p class="sub">No games in the upcoming window.</p>'
    tiers = (start_here(slate) + tier_buttons(slate, "mlb")) if slate else ""
    parlays = build_parlays(slate)
    prows = "".join(
        f'<tr class="prow band-{p["band"]}"><td>{p["legs"]}</td>'
        f'<td>{p["p"]*100:.0f}%</td><td>{100/p["payout_dec"]:.0f}%</td>'
        f'<td>{american(1/p["payout_dec"])}</td>'
        f'<td class="{"ev-hi" if p["ev"] > 0.04 else "ev-md" if p["ev"] > 0 else "ev-lo"}">'
        f'{"+$" + format(p["ev"]*10, ".2f") if p["ev"] >= 0 else "-$" + format(abs(p["ev"])*10, ".2f")}'
        f'</td></tr>' for p in parlays) or \
        '<tr><td colspan="5">Needs upcoming games and prices.</td></tr>'

    return f"""
<nav><button id="b-mweek" class="on" onclick="mtab('mweek')">Today's Slate</button>
<button id="b-mparlays" onclick="mtab('mparlays')">Parlay Lab</button>
<button id="b-mrecord" onclick="mtab('mrecord')">Track Record</button>
<button id="b-mb101" onclick="mtab('mb101')">Bayesian 101</button></nav>
<div class="wrap">
<h1>MLB <span>Model</span> HQ</h1>
<p class="sub">A Bayesian run-differential model &middot; generated {today}</p>
{health_strip(health)}

<div id="mweek" class="panel on">
<h2>Today's Slate</h2>
<p class="sub">Three markets on every card, same order every time: <b>Moneyline</b> (who
wins), <b>Run line</b> (wins by 2 or more, or loses by no more than 1), <b>Total runs</b>
(over or under a number). Each gets one of three words, and each word is an instruction.
<b style="color:var(--green)">BET</b>: one unit. <b style="color:var(--green)">SMALL BET</b>:
half a unit. <b>PASS</b>: do nothing. A market earns its word from its own live record, not
from us: BET needs at least {labels.MIN_BETS} settled bets, profitable overall and over the
most recent half; SMALL BET the same on at least {labels.SMALL_MIN}. When the model likes a
side in a market that has not earned it, the line still shows the side and the edge, marked
PASS, so you can see the disagreement without being told to buy it.
Records so far: picking a team to win {labels.record_phrase("ML")}; the run line
{labels.record_phrase("SPREAD")}; total runs {labels.record_phrase("TOTAL")}. Tap
<b>Details</b> on any card for the numbers. Check the lineup before you buy.</p>
{tiers}
<div class="grid">{cards}</div>
</div>

<div id="mparlays" class="panel">
<h2>Parlay Lab</h2>
<p class="sub"><b>Read this first.</b> Every leg here is a moneyline or run-line pick, the
two markets where the model has lost against the market (picking a team to win
{ml_record()}; run line 40&ndash;44% covers). The "avg profit" column is the model's
own opinion of the combo, not a realized result, and a parlay multiplies each leg's
shortfall along with the payout. This tab is here for transparency about what the model
believes, not as a recommendation.
Combinations of moneylines (at logged Kalshi prices, fees included) and
run lines. Pick your risk appetite: <b>Safe</b> caps the payout near +120 and ranks by
hit chance; these are legs where the model and the market mostly agree, so expect them to
land often but carry little or no edge. <b>Balanced</b> and <b>Longshot</b> rank by the
model's expected profit, which is where real disagreements live. <b>Moonshot</b> is the
lottery tier, including legs where the model disagrees with the market by an amount too
large to fully trust, since a gap that big usually means the model is missing news rather
than the market giving money away. Read each row as: the model says this combo hits X%,
the market's prices imply Y%, and the last column is the average result of a $10 bet
across many tries. Legs are independent games only, and parlays multiply the house's cut
along with the thrill.</p>
<div class="bandbar">
<button id="band-mlb-safe" class="bandbtn" onclick="band('safe','mlb')">Safe &le;+120</button>
<button id="band-mlb-balanced" class="bandbtn" onclick="band('balanced','mlb')">Balanced</button>
<button id="band-mlb-long" class="bandbtn" onclick="band('long','mlb')">Longshot</button>
<button id="band-mlb-moon" class="bandbtn" onclick="band('moon','mlb')">Moonshot +1000</button>
<button id="band-mlb-all" class="bandbtn on" onclick="band('all','mlb')">All</button>
</div>
<table><tr><th>Legs</th><th>Model chance</th><th>Market implied chance</th>
<th>Payout</th><th>Avg profit per $10 bet, win or lose</th></tr>{prows}</table>
</div>

<div id="mrecord" class="panel">{live_performance_html()}<h2>Track Record</h2>{track_record_html()}</div>
<div id="mb101" class="panel"><h2>Bayesian 101</h2>{B101.replace("{ML_RECORD}", ml_record())}</div>

<footer>Every number here states its own uncertainty. Informational only.</footer>
</div>"""
