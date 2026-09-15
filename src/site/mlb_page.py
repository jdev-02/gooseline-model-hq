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


def game_card(r):
    mu, sg, pm = r["mu"], r["sigma"], r["p_home"]
    fav0 = r["home"] if pm >= 0.5 else r["away"]
    fav_p = pm if pm >= 0.5 else 1 - pm
    v0 = str(r.get("verdict", ""))
    vside = ""
    if "&mdash;" in v0 and (v0.startswith("HIGH VALUE") or v0.startswith("CAUTIOUS")):
        vside = v0.split("&mdash;")[-1].strip().replace("small edge on ", "")
    mk = r.get("mkt_home")
    has_price = mk is not None and not pd.isna(mk)
    edge = r.get("edge")
    # Three labelled rows in one block, taken from David's NFL card: the
    # model's line, who it thinks wins, and what (if anything) is priced
    # wrong. The outright call and the price call are different questions;
    # answering the first only inside a sentence about the second is what
    # got "HIGH VALUE -- SF" read as "SF wins" on two real cards.
    lines = (f'<div class="lrow"><span class="llab">Model:</span> '
             f'<span class="lval">{fav_line(mu, r["home"], r["away"])}</span> '
             f'<span class="lnote">&plusmn;{sg:.1f} runs &middot; fair ML {american(fav_p)}</span></div>')
    if abs(fav_p - 0.5) < 0.005:
        ml_row = ('<div class="lrow"><span class="llab">Model picks:</span> '
                  '<span class="lval">too close to call</span></div>')
    else:
        ml_row = (f'<div class="lrow"><span class="llab">Model picks:</span> '
                  f'<span class="lval">{fav0}</span> '
                  f'<span class="lpct">to win <b>{fav_p*100:.0f}%</b> of the time</span></div>')
    if vside:
        e_txt = "" if edge is None or pd.isna(edge) else f" ({float(edge)*100:+.1f}%)"
        small = "" if v0.startswith("HIGH VALUE") else ' <span class="psmall">(small)</span>'
        still = (f' <span class="psmall">&mdash; model still expects {fav0} to win</span>'
                 if vside != fav0 else "")
        dis_html = f'{vside} ML{e_txt}{small}{still}'
    elif has_price:
        dis_html = '<span class="pnone">None at current prices</span>'
    else:
        dis_html = '<span class="pnone">No market price yet</span>'
    dis_row = (f'<div class="lrow"><span class="llab">Price disagreement:</span> '
               f'<span class="lval plist">{dis_html}</span></div>')

    sp = (f'<div class="gap">{r.get("away_sp") or "TBD"} vs {r.get("home_sp") or "TBD"}'
          + (' &middot; <span style="color:var(--yellow)">starter unlisted, range widened</span>'
             if r.get("sp_unknown") else '') + '</div>')

    # Both bars follow the model's favourite, the team the card is about.
    # Fixed to the home side, a card arguing for the away team showed two
    # home-team numbers and the reader had to invert them.
    focus = fav0
    pf = pm if focus == r["home"] else 1 - pm
    model_bar = (f'<div class="brow"><span class="blab">Bayesian model</span>'
                 f'<div class="btrack"><div class="bfill model" '
                 f'style="width:{pf*100:.1f}%"></div></div>'
                 f'<span class="bval">{pf*100:.0f}%</span></div>')
    narr_bar = ""
    pn = r.get("p_home_narrative")
    if r.get("narrative_shift"):
        pnf = pn if focus == r["home"] else 1 - pn
        narr_bar = (f'<div class="brow"><span class="blab">+ narrative</span>'
                    f'<div class="btrack"><div class="bfill model" '
                    f'style="width:{pnf*100:.1f}%;opacity:.5"></div></div>'
                    f'<span class="bval">{pnf*100:.0f}%</span></div>')
    sq3_row = ""
    if has_price:
        mk = float(mk)
        mk_focus = mk if focus == r["home"] else (1 - mk)
        ma = r.get("mkt_away")
        if focus == r["away"] and ma is not None and not pd.isna(ma):
            mk_focus = float(ma)
        gap = (pf - mk_focus) * 100
        mkt_bar = (f'<div class="brow"><span class="blab">Market price</span>'
                   f'<div class="btrack"><div class="bfill mkt" '
                   f'style="width:{mk_focus*100:.1f}%"></div></div>'
                   f'<span class="bval">{mk_focus*100:.0f}&cent;</span></div>')
        age = r.get("price_age_min")
        if age is None:
            agetxt = ""
        elif age <= 2:
            agetxt = ' &middot; price fetched live'
        elif age <= 15:
            agetxt = f' &middot; price {age:.0f} min old'
        else:
            agetxt = (f' &middot; <span style="color:var(--yellow)">price {age:.0f} min '
                      f'old, the market has probably moved</span>')
        gaptxt = (f'<div class="gap">Both bars: chance <b>{focus}</b> wins. '
                  f'Disagreement: <b>{gap:+.0f}</b> points of probability '
                  f'{"toward" if gap > 0 else "against"} {focus}{agetxt}</div>')
        # A gap of more than 0.4 predictive sigmas is more often news the
        # model has not seen than money the market is giving away.
        if vside:
            v_model = pm if vside == r["home"] else 1 - pm
            v_mkt = mk if vside == r["home"] else (float(ma) if ma is not None and not pd.isna(ma) else 1 - mk)
            z = square3_gap(v_model, v_mkt)
            if z > SQUARE3_SIGMAS:
                sq3_row = (f'<div class="gap sq3">Model and market are about <b>{z*sg:.1f}</b> runs '
                           f'apart here, against a typical miss of &plusmn;{sg:.1f}. A gap that '
                           f'size usually means the model has not seen some news; check lineups '
                           f'and the starter before acting.</div>')
    else:
        mkt_bar = ('<div class="brow"><span class="blab">Market price</span>'
                   '<div class="btrack"></div><span class="bval">&mdash;</span></div>')
        gaptxt = '<div class="gap">Market has not opened this game yet</div>'

    # Run line. Backtested at a 40-44% cover rate against Kalshi across the
    # season -- dead, per docs/baselines.md -- so this never gets the "hit"/
    # green styling the moneyline and totals verdicts use: that styling
    # means "this beat the market in testing," and the run line hasn't.
    # Shown for reference only, folded into the collapsed section below.
    rl_row = ""
    if r.get("p_home_cover") is not None and not pd.isna(r.get("p_home_cover")):
        ph_c, pa_c = float(r["p_home_cover"]), float(r["p_away_cover"])
        side, p = ((r["home"], ph_c) if ph_c >= pa_c else (r["away"], pa_c))
        line = "-1.5"
        rl_row = (f'<div class="gap">Run line: model has <b>{side} {line}</b> covering '
                  f'<b>{p*100:.0f}%</b> of the time &middot; fair price {american(p)} '
                  f'&middot; <span style="opacity:.7">not backtested as a bet '
                  f'(season record: 40-44% covers vs. the market) -- reference only</span></div>')

    # Total runs, priced by the negative-binomial model. This used to print
    # "O7.5 51% · O8.5 41% · O9.5 33%" with the actual call tacked on at the
    # end as an aside — a reader had no way to tell over from under without
    # decoding the abbreviation and finding the right clause. Now it leads
    # with one plain-English sentence naming the side, sized and colored the
    # same way the moneyline verdict badge is, so the two calls on a card
    # read the same way.
    tot_row = ""
    ladder_line = ""
    mt = r.get("mu_total")
    if mt is not None and not pd.isna(mt):
        # `tcall`, not `call`: `call` is the "Bayesian Model: X by Y runs"
        # line set at the top of this function and printed in the card's
        # <div class="call">. Reusing the name here silently overwrote it on
        # every card with totals data -- the model's actual moneyline call
        # was replaced by the totals call text ("no edge (best OVER 7.5)")
        # for the rest of the function. Caught while verifying an unrelated
        # layout change; this was already live.
        tcall = str(r.get("total_call") or "")
        te = r.get("total_edge")
        has_edge = bool(tcall) and not tcall.startswith("no edge") and te is not None and not pd.isna(te)
        src = re.sub(r"^no edge \(best (.+)\)$", r"\1", tcall)
        m = re.match(r"(OVER|UNDER)\s+([\d.]+)", src)
        side, line = (m.group(1).title(), float(m.group(2))) if m else (None, None)

        ladder = []
        for ln in (7.5, 8.5, 9.5):
            po = r.get(f"p_over_{ln:g}")
            if po is None or pd.isna(po):
                continue
            po = float(po)
            lean, p = ("Over", po) if po >= 0.5 else ("Under", 1 - po)
            bold = ' style="font-weight:700"' if line == ln else ""
            # `mk_ln`, not `mk`: `mk` is the moneyline home price read above
            # and used again below to name the market's favourite. Reusing
            # it here left `mkt_fav` comparing the over-9.5 price (or None,
            # which raises) -- the same variable-reuse trap as `call`.
            mk_ln = r.get(f"mkt_over_{ln:g}")
            mktxt = ""
            if mk_ln is not None and not pd.isna(mk_ln):
                mask = float(mk_ln) if lean == "Over" else 1 - float(mk_ln)
                mktxt = f" (market {mask*100:.0f}%)"
            ladder.append(f'<span{bold}>{lean} {ln:g}: {p*100:.0f}%{mktxt}</span>')

        # Weather is the input the market prices by hand, so it is shown.
        wx = ""
        if r.get("roof_closed") in (True, 1, 1.0, "True"):
            wx = " &middot; roof closed"
        else:
            tf = r.get("temp_f")
            if tf is not None and not pd.isna(tf):
                wx = f" &middot; {int(float(tf))}&deg;F at first pitch"

        # Same trap as "HIGH VALUE -- SF": the flagged totals side is
        # whichever of OVER/UNDER has the bigger price edge, which is not
        # always the side the model's own number actually leans toward. If
        # the model's raw runs estimate disagrees with the call, say so
        # right here -- not buried in the ladder -- the same fix just made
        # for the moneyline badge, for the same reason.
        tot_disambig = ""
        ladder_line = ""
        if side and line is not None:
            cls = "v-high" if (has_edge and te > 0.04) else "v-caut" if has_edge else "v-avoid"
            edge_txt = (f'edge {float(te)*100:+.1f}% after fees' if has_edge
                       else 'no edge at the current price')
            model_side = "Over" if float(mt) > line else "Under"
            if has_edge and model_side != side:
                tot_disambig = (f'<div class="disambig">Model\'s own number ({float(mt):.1f}) '
                                f'actually leans <b>{model_side}</b> &mdash; this bets the price '
                                f'on <b>{side}</b>, not the model\'s runs call.</div>')
            tot_row = (f'<div class="gap">Total runs call: '
                      f'<span class="verdict {cls}" style="padding:2px 10px;font-size:.78rem">'
                      f'{side} {line:g}</span> &middot; {edge_txt} &middot; '
                      f'model expects <b>{float(mt):.1f}</b> runs{wx}</div>{tot_disambig}')
            ladder_line = (f'<div class="gap" style="font-size:.82rem;opacity:.75">'
                          f'{" &middot; ".join(ladder)}</div>')
        elif ladder:
            tot_row = (f'<div class="gap">Total runs: model expects <b>{float(mt):.1f}</b>{wx} '
                      f'&middot; no price to compare yet</div>')
            ladder_line = (f'<div class="gap" style="font-size:.82rem;opacity:.75">'
                          f'{" &middot; ".join(ladder)}</div>')

    note = ""
    v = str(r["verdict"])
    # "HIGH VALUE -- SF" reads as "the model likes SF to win," and often it
    # doesn't: a HIGH VALUE badge can name the side the model expects to
    # LOSE, when the market's price on it is wrong enough to be worth buying
    # anyway. Demonstrated twice on real cards (CWS@CLE, then SF@STL, where
    # the model favored STL at 52% and the badge said HIGH VALUE -- SF). The
    # collapsed "Why" section already explained this in prose, but hiding
    # the one sentence that prevents the misread is backwards -- it has to
    # be visible without opening anything. The badge itself now says
    # The "Model picks" and "Price disagreement" rows above now answer the two
    # questions on adjacent labelled lines, so no separate disambiguator.
    if has_price:
        mkt_fav = r["home"] if mk >= 0.5 else r["away"]
        if "&mdash;" in v and (v.startswith("HIGH VALUE") or v.startswith("CAUTIOUS")):
            side = v.split("&mdash;")[-1].strip().replace("small edge on ", "").replace(" (underdog)", "")
            soft = "" if v.startswith("HIGH VALUE") else " The edge is small, so treat this one lightly."
            if side != fav0:
                pass  # already said, plainly and visibly, in `disambig` above
            elif side == mkt_fav:
                note = (f'For value: the model and the market agree <b>{side}</b> is '
                        f'the likely winner, but the model is more confident than the '
                        f'price implies. The value play is <b>{side}</b>.' + soft)
            else:
                note = (f'For value: the model calls an upset. It makes <b>{side}</b> '
                        f'the favorite while the market does not, so {side} comes '
                        f'cheap if the model is right.' + soft)
        elif v.startswith("NO VALUE"):
            if fav0 == mkt_fav:
                note = (f'The model and the market see this game the same way: '
                        f'<b>{fav0}</b> likely wins, and the price already says so. '
                        f'Fair price, no bet.')
            else:
                note = (f'The model leans <b>{fav0}</b> while the market leans '
                        f'{mkt_fav}, but not by enough to beat the price after fees. '
                        f'No bet.')
        if note:
            note = f'<div class="gap">{note}</div>'
    if r.get("note"):
        note += (f'<div class="gap"><i>Narrative: {r["note"]}</i> &rarr; '
                 f'model+narrative {pn*100:.0f}%, '
                 f'{str(r.get("verdict_narrative", "")).replace("&mdash;", "-").lower()}</div>')

    # The badge answers "what do I do" -- it used to sit last, after 8+ lines
    # of prose. It's first now, right under the matchup. The reasoning behind
    # it (why this side, not the model's favorite), the full O/U ladder, and
    # the run line (not a backtested bet) are real information, not clutter,
    # so they stay on the card -- just collapsed, the same pattern the
    # run-health strip uses.
    extras = [x for x in (("run line", rl_row), ("O/U ladder", ladder_line)) if x[1]]
    why = "".join(x for x in (note, rl_row, ladder_line) if x)
    label = " &middot; " + " &middot; ".join(t for t, _ in extras) if extras else ""
    why_block = f'<details class="why"><summary>Why{label}</summary>{why}</details>' if why else ""

    te = r.get("total_edge")
    tot_flag = (te is not None and not pd.isna(te) and float(te) > 0.04
                and not str(r.get("total_call") or "").startswith("no edge"))
    classes = f'card tier-{verdict_tier(v)}' + (" tot-flag" if tot_flag else "")
    kick = str(r.get("kick_iso") or "")
    return (f'<div class="{classes}"><div class="match"><span class="teams">{r["away"]} @ '
            f'{r["home"]}</span><span class="date" data-kick="{kick}">{r["date"]}</span></div>'
            f'<div class="leadv">{verdict_badge(v, r.get("edge"))}</div>'
            f'{lines}{ml_row}{dis_row}{sp}'
            f'<div class="bars">{model_bar}{narr_bar}{mkt_bar}</div>'
            f'{gaptxt}{sq3_row}{tot_row}{why_block}</div>')


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
    """The live moneyline record, from the same file Live Bet Performance
    renders, so prose and table can never disagree. Returns a phrase."""
    p = DATA / "paper_trades.csv"
    if not p.exists():
        return "has no settled bets yet"
    t = pd.read_csv(p)
    t = t[t["market"].astype(str).str.upper() == "ML"]
    if not len(t) or not t["stake"].sum():
        return "has no settled bets yet"
    roi = 100 * t["pnl"].sum() / t["stake"].sum()
    w = int(t["won"].sum())
    return (f"has returned {roi:+.1f}% over {len(t)} settled bets ({w}-{len(t) - w}) "
            f"since 2026-08-27")


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
    t["kind"] = np.where(t["market"].astype(str).str.upper() == "ML", "Moneyline", "Totals")
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
not the walk-forward backtest below &mdash; it started 2026-08-27 and is still a small
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
Against the market it has not won: every disagreement it has with Kalshi's moneyline is
"this game is closer than you think", and paper-traded, acting on every such
disagreement {ML_RECORD}. The market is calibrated too, and it has information the model does not
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


def tier_buttons(slate, scope):
    """Filter the slate to one kind of card, with counts, so "what is there to
    act on today" is one tap instead of a scroll. From David's NFL page."""
    from collections import Counter
    counts = Counter(verdict_tier(str(r.get("verdict", ""))) for r in slate)
    tot = sum(1 for r in slate
              if r.get("total_edge") is not None and not pd.isna(r.get("total_edge"))
              and float(r["total_edge"]) > 0.04
              and not str(r.get("total_call") or "").startswith("no edge"))
    spec = [("all", "All", len(slate)), ("tot", "Totals flags", tot),
            ("high", "Disagreements", counts.get("high", 0)),
            ("small", "Small", counts.get("small", 0)),
            ("none", "Fair price", counts.get("none", 0))]
    btns = "".join(
        f'<button id="tier-{scope}-{k}" class="bandbtn{" on" if k == "all" else ""}"'
        f'{"" if n or k == "all" else " disabled"} onclick="mtier(\'{k}\',\'{scope}\')">'
        f'{lab}<span class="cnt">{n}</span></button>' for k, lab, n in spec)
    return f'<div class="bandbar tierbar">{btns}</div>'


def render(slate, today, health=None):
    cards = "".join(game_card(r) for r in slate) or \
        '<p class="sub">No games in the upcoming window.</p>'
    tiers = tier_buttons(slate, "mlb") if slate else ""
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
<p class="sub">Each card shows where the model and the market disagree, and by how much.
<b>Model disagrees</b> means the model prices that side higher than Kalshi does, after
fees. It is not a prediction that the side wins: when the two differ, the card says which
team the model actually expects to win. <b>The record so far:</b> acting on every moneyline
disagreement {ml_record()} (Live Bet Performance, Track Record
tab). The <b>total runs</b> call is the one market Kalshi has not been shown to beat, and
the only one that gets a green badge. Green bar: the model's chance the home team wins.
White bar: the market's price for that. Every flag still gets a human news check.</p>
{tiers}
<div class="grid">{cards}</div>
</div>

<div id="mparlays" class="panel">
<h2>Parlay Lab</h2>
<p class="sub"><b>Read this first.</b> Every leg here is a moneyline or run-line pick, the
two markets where the model has lost against the market (acting on moneyline
disagreements {ml_record()}; run line 40&ndash;44% covers). The "avg profit" column is the model's
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
