"""The MLB half of Model HQ: the same four tabs, card grammar, badges and
copy register as David's NFL page (src/site/nfl_site.py), in run units.

This Week / Parlay Lab / Track Record / Bayesian 101.
"""
from __future__ import annotations

import base64
import itertools
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
    """MLB has one spread that matters, so state the side at +/-1.5."""
    if abs(mu) < 0.05:
        return "pick'em"
    return f"{home if mu > 0 else away} -1.5"


def img_tag(path):
    p = Path(path)
    if not p.exists():
        return ""
    b = base64.b64encode(p.read_bytes()).decode()
    return f'<img class="fig" src="data:image/png;base64,{b}" alt="{p.stem}">'


def verdict_badge(v):
    v = str(v)
    if v.startswith("HIGH VALUE"):
        return f'<span class="verdict v-high">{v}</span>'
    if v.startswith("CAUTIOUS"):
        return f'<span class="verdict v-caut">{v}</span>'
    if v.startswith("NO VALUE"):
        return f'<span class="verdict v-avoid">{v}</span>'
    return '<span class="verdict v-none">No price yet</span>'


def game_card(r):
    mu, sg, pm = r["mu"], r["sigma"], r["p_home"]
    call = (f"Bayesian Model: <b>{r['home']} by {abs(mu):.1f} runs</b>" if mu >= 0
            else f"Bayesian Model: <b>{r['away']} by {abs(mu):.1f} runs</b>")
    call += f" &plusmn;{sg:.1f}"
    fav0 = r["home"] if pm >= 0.5 else r["away"]
    fav_p = pm if pm >= 0.5 else 1 - pm
    gline = (f'Gambler terms: <b>{fav_line(mu, r["home"], r["away"])}</b> &middot; '
             f'{fav0} ML <b>{american(fav_p)}</b>')
    v0 = str(r.get("verdict", ""))
    if "&mdash;" in v0 and (v0.startswith("HIGH VALUE") or v0.startswith("CAUTIOUS")):
        vside = v0.split("&mdash;")[-1].strip().replace("small edge on ", "")
        if vside != fav0:
            gline += (f' &middot; or for value: <b>{vside} ML '
                      f'{american(1 - fav_p)}</b> (why below)')

    sp = (f'<div class="gap">{r.get("away_sp") or "TBD"} vs {r.get("home_sp") or "TBD"}'
          + (' &middot; <span style="color:var(--yellow)">starter unlisted, range widened</span>'
             if r.get("sp_unknown") else '') + '</div>')

    model_bar = (f'<div class="brow"><span class="blab">Bayesian model</span>'
                 f'<div class="btrack"><div class="bfill model" '
                 f'style="width:{pm*100:.1f}%"></div></div>'
                 f'<span class="bval">{pm*100:.0f}%</span></div>')
    narr_bar = ""
    pn = r.get("p_home_narrative")
    if r.get("narrative_shift"):
        narr_bar = (f'<div class="brow"><span class="blab">+ narrative</span>'
                    f'<div class="btrack"><div class="bfill model" '
                    f'style="width:{pn*100:.1f}%;opacity:.5"></div></div>'
                    f'<span class="bval">{pn*100:.0f}%</span></div>')
    mk = r.get("mkt_home")
    has_price = mk is not None and not pd.isna(mk)
    if has_price:
        mk = float(mk)
        gap = (pm - mk) * 100
        mkt_bar = (f'<div class="brow"><span class="blab">Market price</span>'
                   f'<div class="btrack"><div class="bfill mkt" '
                   f'style="width:{mk*100:.1f}%"></div></div>'
                   f'<span class="bval">{mk*100:.0f}&cent;</span></div>')
        gaptxt = (f'<div class="gap">Both bars: chance the home team wins. '
                  f'Disagreement: <b>{gap:+.0f}</b> points of probability '
                  f'{"toward" if gap > 0 else "against"} {r["home"]}</div>')
    else:
        mkt_bar = ('<div class="brow"><span class="blab">Market price</span>'
                   '<div class="btrack"></div><span class="bval">&mdash;</span></div>')
        gaptxt = '<div class="gap">Market has not opened this game yet</div>'

    # Run line, graded the way the NFL page grades the spread
    rl_row = ""
    if r.get("p_home_cover") is not None and not pd.isna(r.get("p_home_cover")):
        ph_c, pa_c = float(r["p_home_cover"]), float(r["p_away_cover"])
        side, p = ((r["home"], ph_c) if ph_c >= pa_c else (r["away"], pa_c))
        line = "-1.5" if side == (r["home"] if ph_c >= pa_c else r["away"]) and p == max(ph_c, pa_c) else "-1.5"
        breakeven = 1 / RUNLINE_JUICE
        if p >= breakeven + 0.05:
            tag = '<span class="hit">value at a book\'s run line</span>'
        elif p >= breakeven:
            tag = '<span style="color:var(--yellow)">slight lean</span>'
        else:
            tag = 'no edge on the run line'
        rl_row = (f'<div class="gap">Run line: model has <b>{side} {line}</b> covering '
                  f'<b>{p*100:.0f}%</b> of the time &middot; fair price {american(p)} '
                  f'&middot; {tag}</div>')

    note = ""
    v = str(r["verdict"])
    if has_price:
        mkt_fav = r["home"] if mk >= 0.5 else r["away"]
        if "&mdash;" in v and (v.startswith("HIGH VALUE") or v.startswith("CAUTIOUS")):
            side = v.split("&mdash;")[-1].strip().replace("small edge on ", "")
            soft = "" if v.startswith("HIGH VALUE") else " The edge is small, so treat this one lightly."
            if side != fav0:
                note = (f'For value: the model still expects <b>{fav0}</b> to win, '
                        f'but the market charges too much for {fav0}. The value '
                        f'play is <b>{side}</b>: buying the underpriced side, not '
                        f'picking the winner.' + soft)
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

    return (f'<div class="card"><div class="match"><span class="teams">{r["away"]} @ '
            f'{r["home"]}</span><span class="date">{r["date"]}</span></div>'
            f'<div class="call">{call}</div><div class="gline">{gline}</div>{sp}'
            f'<div class="bars">{model_bar}{narr_bar}{mkt_bar}</div>{gaptxt}{rl_row}{note}'
            f'{verdict_badge(v)}</div>')


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
    for season, g in hist.groupby("season"):
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
        blocks.append(
            f'<details><summary>{season} season &mdash; every game ({len(g)})</summary>'
            f'<table><tr><th>Date</th><th>Game</th><th>Bayesian Model line</th>'
            f'<th>Model: home team wins</th><th>Final margin (home first)</th>'
            f'<th>Winner pick</th><th>Run-line pick</th></tr>{rows}</table></details>')

    return f"""
<p class="sub">Everything on this tab is stated from the home team's perspective.
Every prediction below was made by the Bayesian Model before it had seen the game:
it refits weekly on games already played, exactly as it runs live. Two report cards:
"winner pick" is the model picking the game outright, and "run-line pick" is its side
at the standard &plusmn;1.5. Baseball is close to a coin flip most nights, so a winner
percentage in the mid-fifties is a real edge, not a weak one.</p>
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
it is knowing your 4.4 honestly while the market prices as if it were smaller.</p>
<p><b>Why baseball is harder than football.</b> An NFL model can separate two teams by
two touchdowns. The best pre-game baseball model in the world separates almost every
matchup by less than two runs, on a spread of four and a half. That is why nearly every
card on This Week reads NO VALUE: the honest range dwarfs the edge, and the market
already knows what the model knows. Finding three real disagreements a week is the
job working, not failing.</p>
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
a day game after a night game, wind blowing out. Every green badge gets a human news
check before anything happens. When the market disagrees with the model, the market is
usually right; this system exists to find the exceptions and to know the difference.</p>
</div>"""


def render(slate, today):
    cards = "".join(game_card(r) for r in slate) or \
        '<p class="sub">No games in the upcoming window.</p>'
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

<div id="mweek" class="panel on">
<h2>Today's Slate</h2>
<p class="sub">Green bar: the Bayesian Model's chance the home team wins. Faded green:
the same after a human narrative tilt, when one was entered. White bar: what the market
charges for that outcome. Badges: green means real value after fees, yellow means an
edge too small to trust, red means the price is fair or worse. Each card also grades the
run line. Every green light still gets a human news check first.</p>
<div class="grid">{cards}</div>
</div>

<div id="mparlays" class="panel">
<h2>Parlay Lab</h2>
<p class="sub">Combinations of moneylines (at logged Kalshi prices, fees included) and
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
<button id="band-safe" class="bandbtn" onclick="band('safe')">Safe &le;+120</button>
<button id="band-balanced" class="bandbtn" onclick="band('balanced')">Balanced</button>
<button id="band-long" class="bandbtn" onclick="band('long')">Longshot</button>
<button id="band-moon" class="bandbtn" onclick="band('moon')">Moonshot +1000</button>
<button id="band-all" class="bandbtn on" onclick="band('all')">All</button>
</div>
<table><tr><th>Legs</th><th>Model chance</th><th>Market implied chance</th>
<th>Payout</th><th>Avg profit per $10 bet, win or lose</th></tr>{prows}</table>
</div>

<div id="mrecord" class="panel"><h2>Track Record</h2>{track_record_html()}</div>
<div id="mb101" class="panel"><h2>Bayesian 101</h2>{B101}</div>

<footer>One model, honestly uncertain. Nothing here is financial advice.</footer>
</div>"""
