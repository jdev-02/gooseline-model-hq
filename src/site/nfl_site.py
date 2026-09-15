import argparse
import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm

import src.nfl.rundown as rd
from src.nfl.features import FEATURE_COLS
from src.core.walkforward import walk_forward
from src.site.names import nfl as nick

V3 = rd.V3_COLS
SPREAD_JUICE = 1.909


def american(p):
    if p <= 0 or p >= 1:
        return "n/a"
    return f"{-round(100 * p / (1 - p))}" if p >= 0.5 else f"+{round(100 * (1 - p) / p)}"


def dec_from_american(a):
    a = float(a)
    return 1 + (a / 100 if a > 0 else 100 / -a)


def fav_line(mu, home, away):
    s = round(mu * 2) / 2
    if s == 0:
        return "pick'em"
    team = home if s > 0 else away
    return f"{team} -{abs(s):g}"


def history_tables(df, seasons=(2021, 2022, 2023, 2024, 2025)):
    frames = []
    for season in seasons:
        p = walk_forward(df, V3, season, lam=rd.LIN_LAM,
                         half_life_seasons=rd.DECAY_HL)
        p["season"] = season
        frames.append(p)
    hist = pd.concat(frames, ignore_index=True)
    hist = hist.merge(df[["game_id", "spread_line"]], on="game_id", how="left")
    hist["p_home"] = norm.cdf(hist["mu"] / hist["sigma"])
    hist["su_pick"] = np.where(hist["mu"] > 0, hist["home_team"], hist["away_team"])
    hist["su_win"] = np.where(hist["y"] > 0, hist["home_team"],
                              np.where(hist["y"] < 0, hist["away_team"], "tie"))
    hist["su_correct"] = hist["su_pick"] == hist["su_win"]
    has_line = hist["spread_line"].notna()
    hist["ats_pick"] = np.where(hist["mu"] > hist["spread_line"],
                                hist["home_team"], hist["away_team"])
    home_cover = hist["y"] > hist["spread_line"]
    push = hist["y"] == hist["spread_line"]
    hist["ats_correct"] = np.where(push, np.nan,
        np.where(hist["ats_pick"] == hist["home_team"], home_cover, ~home_cover))
    hist.loc[~has_line, "ats_correct"] = np.nan

    hist = hist.merge(df[["game_id", "home_score", "away_score"]],
                      on="game_id", how="left")

    by_season = hist.groupby("season").apply(lambda g: pd.Series({
        "games": len(g),
        "winner_pct": 100 * g["su_correct"].mean(),
        "ats_pct": 100 * g["ats_correct"].dropna().astype(float).mean(),
        "avg_miss": (g["y"] - g["mu"]).abs().mean(),
    }), include_groups=False).round(1)

    edges = [0, .35, .45, .55, .65, 1.0]
    labels = ["0-35%", "35-45%", "45-55%", "55-65%", "65-100%"]
    bins = pd.cut(hist["p_home"], edges, labels=labels)
    calib = hist.groupby(bins, observed=True).apply(lambda g: pd.Series({
        "games": len(g),
        "model_said": 100 * g["p_home"].mean(),
        "home_actually_won": 100 * (g["y"] > 0).mean(),
    }), include_groups=False).round(1)
    return hist, by_season, calib


def build_parlays(upcoming_rows, top_n=10):
    legs = []
    for r in upcoming_rows:
        mu, sigma = r["mu"], r["sigma"]
        p_home = norm.cdf(mu / sigma)
        if r.get("mkt_home") is not None and not pd.isna(r.get("mkt_home")):
            side, p = (r["home"], p_home) if p_home >= 0.5 else (r["away"], 1 - p_home)
            price = r["mkt_home"] if side == r["home"] else 1 - r["mkt_home"]
            if 0.02 < price < 0.98:
                legs.append({"game": f"{r['away']}@{r['home']}", "desc": f"{side} ML",
                             "p": p, "dec": 1 / (price + rd.kalshi_fee(price)),
                             "wild": (p - price) > 0.15})
        sl = r.get("spread_line")
        if sl is not None and not pd.isna(sl):
            p_cover_home = norm.cdf((mu - sl) / sigma)
            side, p = ((r["home"], p_cover_home) if p_cover_home >= 0.5
                       else (r["away"], 1 - p_cover_home))
            line = f"-{abs(sl):g}" if (side == r["home"]) == (sl > 0) else f"+{abs(sl):g}"
            legs.append({"game": f"{r['away']}@{r['home']}", "desc": f"{side} {line}",
                         "p": p, "dec": SPREAD_JUICE, "wild": False})
    parlays = []
    for k in (2, 3):
        for combo in itertools.combinations(legs, k):
            if len({c["game"] for c in combo}) < k:
                continue
            p = float(np.prod([c["p"] for c in combo]))
            dec = float(np.prod([c["dec"] for c in combo]))
            wild = any(c["wild"] for c in combo)
            if wild or dec > 11:
                band = "moon"
            elif dec <= 2.2:
                band = "safe"
            elif dec <= 4.0:
                band = "balanced"
            else:
                band = "long"
            parlays.append({"legs": ", ".join(c["desc"] for c in combo),
                            "n": k, "p": p, "payout_dec": dec,
                            "ev": p * dec - 1, "band": band})
    out = []
    for band, key in (("safe", lambda x: x["p"]),
                      ("balanced", lambda x: x["ev"]),
                      ("long", lambda x: x["ev"]),
                      ("moon", lambda x: x["ev"])):
        rows = sorted([x for x in parlays if x["band"] == band],
                      key=key, reverse=True)[:top_n]
        out.extend(rows)
    return out


CSS = """
:root{--bg:#0a0e0b;--panel2:#131a15;--white:#f2f5f2;
  --green:#2ee06f;--yellow:#f5c542;--red:#e5533d;
  --dim:#f2f5f2;--line:#20291f}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--white);
  font:15px/1.5 "Segoe UI",system-ui,sans-serif;margin:0}
.wrap{max-width:920px;margin:0 auto;padding:18px 14px 40px}
h1{font-family:"Arial Narrow","Segoe UI",sans-serif;font-size:1.7rem;
  text-transform:uppercase;letter-spacing:.05em}
h1 span{color:var(--green)}
.sub{color:var(--dim);font-size:.85rem;margin:2px 0 10px}
nav{position:sticky;top:0;z-index:10;background:var(--bg);
  border-bottom:1px solid var(--line);display:flex;gap:6px;padding:8px 14px;
  overflow-x:auto;justify-content:center}
nav button{background:none;border:1px solid var(--white);color:var(--white);
  padding:7px 16px;border-radius:99px;font:600 .85rem "Segoe UI",sans-serif;
  cursor:pointer;white-space:nowrap;transition:background-color .15s,border-color .15s,
  color .15s,transform .1s}
nav button:hover{border-color:var(--green);color:var(--green)}
nav button:active{transform:scale(.96)}
nav button:focus-visible{outline:2px solid var(--green);outline-offset:2px}
nav button.on{background:var(--green);border-color:var(--green);color:#08120b}
nav button.on:hover{color:#08120b}
.panel{display:none}
.panel.on{display:block;animation:fadein .18s ease-out}
@keyframes fadein{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
@media(prefers-reduced-motion:reduce){.panel.on{animation:none}}
.grid{display:grid;gap:10px}
@media(min-width:700px){.grid{grid-template-columns:1fr 1fr}}
.card{background:var(--panel2);border:1px solid var(--line);border-radius:10px;
  padding:11px 13px;transition:border-color .15s}
.card:hover{border-color:#3a4a37}
.match{display:flex;justify-content:space-between;align-items:baseline}
.teams{font-family:"Arial Narrow",sans-serif;font-size:1.05rem;font-weight:700;
  text-transform:uppercase;letter-spacing:.02em}
.date{color:var(--dim);font-size:.75rem}
.call{font-size:.83rem;margin:5px 0 2px}
.gline{color:var(--green);font-size:.8rem;margin-bottom:8px}
.bars{display:grid;gap:4px;margin:4px 0 2px}
.brow{display:grid;grid-template-columns:96px 1fr 44px;align-items:center;
  gap:8px;font-size:.7rem}
.blab{color:var(--dim);text-transform:uppercase;letter-spacing:.04em}
.btrack{height:10px;background:#1a231c;border-radius:5px;overflow:hidden}
.bfill{height:100%;border-radius:5px}
.bfill.model{background:var(--green)}
.bfill.mkt{background:var(--white)}
.bval{text-align:right;font-variant-numeric:tabular-nums}
.gap{font-size:.72rem;margin-top:5px;color:var(--dim)}
.gap b{font-variant-numeric:tabular-nums}
.verdict{display:inline-block;margin-top:7px;padding:2px 10px;border-radius:99px;
  font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em}
/* The lead badge sits right under the matchup header now, not last after a
   wall of prose -- it's the one thing every card must answer at a glance,
   so it gets its own size step instead of sharing the small inline badges'. */
/* Labelled line rows (David's card grammar): model line / model picks /
   price disagreement, same bold block, so the two questions are adjacent. */
.lrow{padding:2px 0;line-height:1.4}
.llab{color:var(--dim);font-size:.72rem;text-transform:uppercase;letter-spacing:.04em}
.lval{font-size:1.05rem;font-weight:700;color:var(--green);margin-left:3px}
.lval.vval{color:var(--white)}
.lnote{color:var(--dim);font-size:.72rem;margin-left:3px}
.lpct{color:var(--white);font-size:.88rem;font-weight:600;margin-left:3px}
.pnone{font-size:.85rem;font-weight:600;color:var(--dim)}
.psmall{font-size:.7rem;font-weight:600;color:var(--dim)}
.sq3{border-left:3px solid var(--yellow);padding-left:8px;color:var(--white)}
.how{font-size:.9rem;margin:6px 0 4px;line-height:1.45}
.how .warn{display:block;margin-top:4px;color:var(--yellow);font-size:.8rem}
.who{font-size:.85rem;color:var(--dim);margin:2px 0 4px}
.also{font-size:.8rem;color:var(--yellow);margin:2px 0}
.tierbar{flex-wrap:wrap}
.cnt{display:inline-block;margin-left:6px;padding:0 6px;border-radius:99px;
  background:rgba(255,255,255,.12);font-size:.7rem}
.bandbtn.on .cnt{background:rgba(0,0,0,.18)}
.bandbtn:disabled{opacity:.35;cursor:default}
.leadv{margin:6px 0 2px}
.leadv .verdict{font-size:.8rem;padding:4px 13px}
/* The one sentence that stops "HIGH VALUE -- SF" reading as "SF wins": when
   the flagged side isn't who the model expects to win, this says so right
   next to the badge -- never inside a collapsed section, since hiding it
   is exactly what let the misread happen twice on real cards. */
.disambig{font-size:.75rem;color:var(--white);background:rgba(240,180,40,.1);
  border-left:3px solid var(--yellow);border-radius:0 6px 6px 0;
  padding:5px 9px;margin:6px 0}
.why{margin-top:6px}
.why summary{cursor:pointer;color:var(--green);font-size:.72rem;
  text-transform:uppercase;letter-spacing:.04em}
.why[open] summary{margin-bottom:2px}
.v-high{background:var(--green);color:#08120b}
/* A disagreement with the price, stated as such: neutral, not green. Green
   would claim value the moneyline record contradicts (docs/baselines.md). */
.v-dis{background:none;border:1px solid var(--white);color:var(--white)}
.v-caut{background:var(--yellow);color:#141005}
.v-avoid{background:none;border:1px solid var(--red);color:var(--red)}
.v-none{background:none;border:1px solid var(--dim);color:var(--dim)}
.bandbar{display:flex;gap:6px;margin:10px 0}
.bandbtn{background:none;border:1px solid var(--white);color:var(--white);
  padding:5px 14px;border-radius:99px;font:600 .8rem "Segoe UI",sans-serif;cursor:pointer;
  transition:background-color .15s,border-color .15s,color .15s,transform .1s}
.bandbtn:hover{border-color:var(--green);color:var(--green)}
.bandbtn:active{transform:scale(.96)}
.bandbtn:focus-visible{outline:2px solid var(--green);outline-offset:2px}
.bandbtn.on{background:var(--green);border-color:var(--green);color:#08120b}
.bandbtn.on:hover{color:#08120b}
h2{font-family:"Arial Narrow",sans-serif;font-size:1.15rem;text-transform:uppercase;
  color:var(--green);margin:18px 0 8px}
table{width:100%;border-collapse:collapse;font-size:.82rem;margin:8px 0}
th{color:var(--green);text-align:left;font-weight:600;padding:6px 7px;
  border-bottom:1px solid var(--line);text-transform:uppercase;font-size:.72rem;
  letter-spacing:.04em}
td{padding:4px 7px;border-bottom:1px solid #17201850;
  font-variant-numeric:tabular-nums}
.hit{color:var(--green)}.miss{color:var(--red)}
.metrics-row{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0 16px}
.mrow{background:var(--panel2);border:1px solid var(--line);border-radius:10px;
  padding:10px 8px;text-align:center}
.mrow .mn{display:block;font-family:"Arial Narrow",sans-serif;font-size:1.3rem;
  font-weight:700;color:var(--green);font-variant-numeric:tabular-nums}
.mrow .ml{display:block;font-size:.68rem;color:var(--dim);text-transform:uppercase;
  letter-spacing:.04em;margin-top:2px}
@media(max-width:760px){.metrics-row{grid-template-columns:repeat(2,1fr)}}
.ev-hi{color:var(--green);font-weight:700}.ev-md{color:var(--yellow)}
.ev-lo{color:var(--red)}
details{background:var(--panel2);border:1px solid var(--line);border-radius:10px;
  padding:10px 14px;margin:8px 0}
summary{cursor:pointer;font-weight:600;color:var(--green);font-size:.9rem;
  transition:color .15s;border-radius:4px}
summary:hover{color:#5cf097}
summary:focus-visible{outline:2px solid var(--green);outline-offset:2px}
a{transition:color .15s}
a:focus-visible{outline:2px solid var(--green);outline-offset:2px}
.b101 p{margin:10px 0;font-size:.92rem}
.b101 b{color:var(--green)}
.b101 table{max-width:640px}
footer{color:var(--dim);font-size:.78rem;margin-top:26px}
"""

TABS_JS = """
// scope defaults to 'nfl' so the original bare band('safe')/tab('week') calls
// on this page keep working unchanged; the MLB page passes 'mlb' explicitly.
// Sharing one band-safe id between two Parlay Labs on the combined page meant
// getElementById always grabbed the first (NFL's), so the MLB filter buttons
// silently updated the wrong element and never visibly selected.
function band(b, scope){
  const root = scope ? (document.getElementById('page-' + scope) || document) : document;
  root.querySelectorAll('.prow').forEach(r=>{
    r.style.display=(b==='all'||r.classList.contains('band-'+b))?'':'none';});
  root.querySelectorAll('.bandbtn').forEach(x=>x.classList.remove('on'));
  const btnId = scope ? ('band-' + scope + '-' + b) : ('band-' + b);
  const btn = document.getElementById(btnId);
  if (btn) btn.classList.add('on');
}
function tab(id, scope){
  scope = scope || 'nfl';
  const root = document.getElementById('page-' + scope) || document;
  root.querySelectorAll('.panel').forEach(p=>p.classList.remove('on'));
  root.querySelectorAll('nav button').forEach(b=>b.classList.remove('on'));
  document.getElementById(id).classList.add('on');
  const btn = document.getElementById('b-'+id);
  if (btn) btn.classList.add('on');
  window.scrollTo(0,0);
}
"""

B101 = """
<div class="b101">
<p><b>The one idea behind everything here.</b> A prediction is not a number, it is a
range of belief. This model never says "the Chiefs will win by 2." It says "our best
guess is Chiefs by 2, and here is exactly how sure we are." Bayesian modeling is the
math of keeping honest track of that sureness: start with a reasonable belief, let
each game's evidence pull it, and never claim more certainty than the evidence paid
for.</p>
<p><b>Why every prediction says plus-or-minus 13.</b> That 13 is measured, not
assumed. Take every NFL game since 2010, compare the final margin to the best
pre-game prediction anyone can make, and the typical miss is about 13 points. Vegas,
with all its money and information, also misses by about 13. Football is decided by
fumbles, tipped balls, and kickers, and no film study predicts those. The skill is
not shrinking the 13; it is knowing your 13 honestly while the market prices as if
it were smaller or larger.</p>
<p><b>Even the model's weights are probabilities.</b> A normal model learns one
number for how much each factor matters, say "quarterback continuity is worth 2.4
points," and treats it as gospel. This model refuses to commit: every learned weight
is a bell curve, a most likely value plus honest error bars, because 16 seasons of
noisy football cannot pin any of these numbers down exactly. To make a prediction it
samples a full set of plausible weights, forming one plausible analyst, and repeats
until it has a room of them. Where the room agrees, the data genuinely supports the
call. Where the room scatters, the model is telling you it does not understand this
game, and no bet should survive that.</p>
<p><b>The Kalman filter: the model's memory.</b> Every team carries a rating, points
better or worse than average on a neutral field, updated after every game by a
Kalman filter, the same math that navigated Apollo to the moon. The filter's genius
is knowing how far to move: a 20-point blowout by a team it already trusts barely
moves the rating; the same blowout by a mystery team moves it a lot. Between
seasons every rating shrinks 30% toward average and its uncertainty balloons,
because offseasons erase certainty. Home field advantage is learned, not assumed,
and converged to almost exactly 2 points.</p>
<p><b>How it all intertwines.</b> The pipeline is a relay. The Kalman filter watches
scores and maintains ratings with error bars. Those ratings, their uncertainty, and
the stat sheet below become the inputs to the room of five neural networks, which
learn the patterns and produce a margin and a per-game uncertainty. A calibration
step then nudges that uncertainty against held-out seasons so the stated confidence
matches reality. Finally the finished probability is compared to the market's price,
minus fees, and only survivors of that comparison and a human news check appear as
value on the This Week tab.</p>
<p><b>The stat sheet.</b> What the model is actually fed, for every game, always
computed only from games played before it:</p>
<table>
<tr><th>Input</th><th>What it is</th></tr>
<tr><td>Kalman rating gap</td><td>Team strength difference, points on a neutral field</td></tr>
<tr><td>Rating uncertainty</td><td>How well the filter currently knows both teams</td></tr>
<tr><td>Scoring form</td><td>Recent point differential, recent games weighted more</td></tr>
<tr><td>EPA, passing and rushing, offense and defense</td><td>Expected Points Added per play: how much each play helped, given down, distance, and field position. Credits efficiency, not raw yards</td></tr>
<tr><td>CPOE</td><td>Completion Percentage Over Expected: does the QB complete throws harder than they look</td></tr>
<tr><td>QB continuity</td><td>Share of the last 16 games started by this week's listed starter. A backup shows up as a 0</td></tr>
<tr><td>Rest difference</td><td>Days off, home minus away</td></tr>
<tr><td>Division game</td><td>Rivals play closer games than ratings suggest</td></tr>
<tr><td>Roof</td><td>Indoor or outdoor stadium</td></tr>
</table>
<p>Notably absent: yards per carry, yards after catch, and other yardage stats.
EPA already contains what they measure, with the context they lack, so adding them
would add noise, not knowledge. Also deliberately absent: betting lines. A model
fed the market's answer can only agree with the market; this one has to form its
own opinion so the two can genuinely disagree.</p>
<p><b>From margin to money.</b> A predicted margin plus its uncertainty gives the
chance of any outcome: the chance the margin beats zero is the moneyline, the
chance it beats the spread is the cover probability. A market price is also a
probability, since 65 cents means 65%. A team looks cheap when the model's number
and the price differ by more than the fees. Whether cheap means <i>worth buying</i>
is an empirical question, and the answer so far is no: over five walk-forward
seasons the market was right more often than the model whenever they differed
(Track Record). Those games are shown as RISKY because that is what the model
produces; the record is shown next to them because that is what happened.</p>
<p><b>What this model honestly cannot see.</b> Injuries announced this week,
coaches resting starters, weather. Every flag gets a human news check before
anything happens. When the market disagrees with the model, the market is usually
right; this system exists to find the exceptions and to know the difference.</p>
</div>"""



def ats_label(x):
    if pd.isna(x.spread_line):
        return "&mdash;"
    if x.ats_pick == x.home_team:
        line = f"-{abs(x.spread_line):g}" if x.spread_line > 0 else f"+{abs(x.spread_line):g}"
    else:
        line = f"-{abs(x.spread_line):g}" if x.spread_line < 0 else f"+{abs(x.spread_line):g}"
    res = ("HIT" if x.ats_correct == 1 else "miss" if x.ats_correct == 0 else "push")
    return f"{x.ats_pick} {line} &middot; {res}"


def verdict_badge(v, edge=None):
    """Display label only; the stored verdict strings are unchanged (the log
    and the paper trade parse them). "HIGH VALUE" in green claimed something
    the record contradicts: moneyline flags lost 12% over five walk-forward
    seasons and the ensemble 17% (docs/baselines.md). What the model
    actually produces is a disagreement with the price, so that is the word,
    in a neutral colour."""
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


SQUARE3_SIGMAS = 0.40


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


def game_card(r):
    """One card, for someone who has never placed a bet. RISKY or NO BET
    only: nothing on the NFL side has a market it has beaten (moneyline
    -12% over five walk-forward seasons, spread 49.2% vs 52.4% breakeven,
    docs/baselines.md), so there is no BET tier here. Full team names;
    every number under a Details tap."""
    mu, sigma, pm = r["mu"], r["sigma"], r["p_home"]
    home, away = r["home"], r["away"]
    H, A = nick(home), nick(away)
    fav0 = home if pm >= 0.5 else away
    fav_p = pm if pm >= 0.5 else 1 - pm
    v = str(r.get("verdict", ""))
    vside = v.split("&mdash;")[-1].strip() if ("&mdash;" in v and v.startswith("HIGH VALUE")) else ""
    edge = r.get("edge")
    mk_h, mk_a = r.get("mkt_home"), r.get("mkt_away")
    has_price = _num(mk_h)

    if vside and has_price:
        price = mk_h if vside == home else mk_a
        ptxt = f" at <b>{_cents(price)}</b>" if _num(price) else ""
        etxt = f" Edge {float(edge)*100:+.1f}% after fees." if _num(edge) else ""
        still = (f" We still expect the <b>{nick(fav0)}</b> to win this game ({fav_p*100:.0f}%); "
                 f"the {nick(vside)} are just cheaper than they should be." if vside != fav0 else "")
        tier = "risky"
        badge = f'<span class="verdict v-caut">RISKY &middot; {nick(vside)} to win</span>'
        body = (f'<div class="how">Buy <b>{nick(vside)} to win</b>{ptxt} on Kalshi. It wins if the '
                f'{nick(vside)} win.{etxt}{still} <span class="warn">Picking a team to win has lost '
                f'12% over five seasons in testing (871 bets) &mdash; that is why this is marked '
                f'risky, not a bet.</span></div>')
    elif v.startswith("STALE"):
        tier, badge = "stale", '<span class="verdict v-caut">PRICE STALE &middot; re-check</span>'
        body = ('<div class="how">The price we saw is too old to act on. Open Kalshi and look '
                'before you buy anything here.</div>')
    else:
        tier, badge = "none", '<span class="verdict v-none">NO BET</span>'
        body = ('<div class="how">The price is fair. Nothing here is worth buying this week.</div>'
                if has_price else '<div class="how">No price on Kalshi yet.</div>')
    if abs(fav_p - 0.5) < 0.005:
        who = '<div class="who">Who we think wins: <b>too close to call</b></div>'
    else:
        who = f'<div class="who">Who we think wins: <b>{nick(fav0)}</b> ({fav_p*100:.0f}% chance)</div>'

    d = []
    sl = r.get("spread_line")
    d.append(f'<div class="gap">Model line: <b>{fav_line(mu, home, away)}</b> &plusmn;{sigma:.0f} '
             f'&middot; fair moneyline {american(fav_p)}'
             + (f' &middot; Vegas line: {fav_line(sl, home, away)}' if _num(sl) else "") + '</div>')
    focus = fav0
    pf = pm if focus == home else 1 - pm
    bars = (f'<div class="brow"><span class="blab">Our model</span><div class="btrack">'
            f'<div class="bfill model" style="width:{pf*100:.1f}%"></div></div>'
            f'<span class="bval">{pf*100:.0f}%</span></div>')
    if has_price:
        mk = float(mk_h)
        mk_focus = mk if focus == home else (float(mk_a) if _num(mk_a) else 1 - mk)
        bars += (f'<div class="brow"><span class="blab">Kalshi price</span><div class="btrack">'
                 f'<div class="bfill mkt" style="width:{mk_focus*100:.1f}%"></div></div>'
                 f'<span class="bval">{_cents(mk_focus)}</span></div>')
        d.append(f'<div class="bars">{bars}</div><div class="gap">Both bars: chance the <b>{nick(focus)}</b> '
                 f'win. Gap: <b>{(pf-mk_focus)*100:+.0f}</b> points</div>')
        if vside:
            v_model = pm if vside == home else 1 - pm
            v_mkt = mk if vside == home else (float(mk_a) if _num(mk_a) else 1 - mk)
            z = square3_gap(v_model, v_mkt)
            if z > SQUARE3_SIGMAS:
                d.append(f'<div class="gap sq3">Model and market are about <b>{z*sigma:.0f}</b> points apart '
                         f'here, against a typical miss of &plusmn;{sigma:.0f}. A gap that size usually '
                         f'means the model has not seen some news; check injuries before acting.</div>')
    else:
        d.append(f'<div class="bars">{bars}</div><div class="gap">Kalshi has not opened this game yet.</div>')
    if _num(sl):
        p_ch = norm.cdf((mu - sl) / sigma)
        side, p = ((home, p_ch) if p_ch >= 0.5 else (away, 1 - p_ch))
        line = f"-{abs(sl):g}" if (side == home) == (sl > 0) else f"+{abs(sl):g}"
        d.append(f'<div class="gap">Spread: model covers {nick(side)} {line} {p*100:.0f}% of the time '
                 f'&middot; <span style="opacity:.7">not a bet: 49.2% covers over five seasons against a '
                 f'52.4% breakeven</span></div>')
    details = f'<details class="why"><summary>Details</summary>{"".join(d)}</details>'
    kick = str(r.get("kick_iso") or "")
    return (f'<div class="card tier-{tier}" data-teams="{away}@{home}"><div class="match"><span class="teams">{A} at {H}</span>'
            f'<span class="date" data-kick="{kick}">{r["date"]}</span></div>'
            f'<div class="leadv">{badge}</div>{body}{who}{details}</div>')


def card_tier(r):
    v = str(r.get("verdict", ""))
    if v.startswith("STALE"):
        return "stale"
    if v.startswith("HIGH VALUE") and r.get("mkt_home") is not None and not pd.isna(r.get("mkt_home")):
        return "risky"
    return "none"


def tier_buttons(rows, scope):
    from collections import Counter
    counts = Counter(card_tier(r) for r in rows)
    spec = [("all", "All games", len(rows)), ("risky", "Risky", counts.get("risky", 0)),
            ("stale", "Stale price", counts.get("stale", 0)), ("none", "No bet", counts.get("none", 0))]
    btns = "".join(
        f'<button id="tier-{scope}-{k}" class="bandbtn{" on" if k == "all" else ""}"'
        f'{"" if n or k == "all" else " disabled"} onclick="mtier(\'{k}\',\'{scope}\')">'
        f'{lab}<span class="cnt">{n}</span></button>' for k, lab, n in spec)
    return f'<div class="bandbar tierbar">{btns}</div>'


def health_strip(today, health_path="data/nfl/health.json"):
    """Same purpose as the MLB page's strip (src/site/mlb_page.py): the run's
    own audit, above the first card, so a reader can trust a number before
    reading it. Built from data/nfl/health.json (ops/healthcheck_nfl.py)."""
    import json
    from pathlib import Path
    p = Path(health_path)
    health = json.loads(p.read_text()) if p.exists() else None
    if health and health.get("run_date") != str(today.date()):
        health = None  # a stale record is worse than none
    if not health:
        return ('<div class="health warn">No health record for this build. '
                'Treat every number below as unverified.</div>')
    checks = [c for c in health.get("checks", []) if c.get("stage") == "data"]
    fails = [c for c in checks if not c["ok"] and c["hard"]]
    warns = [c for c in checks if not c["ok"] and not c["hard"]]
    ts = pd.Timestamp(health.get("run_ts")).strftime("%Y-%m-%d %H:%M UTC")
    by = {c["ok"] for c in checks if c["name"] == "no game priced past its own kickoff"}
    fresh = next((c["detail"] for c in checks if c["name"] == "Kalshi prices fresh"), "")
    facts = " &middot; ".join(x for x in (
        f"run {ts}",
        "kickoff check clean" if by == {True} else ("kickoff check FAILED -- see below" if by else ""),
        fresh,
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


def build_site(out_path="site.html", games_path="data/nfl/games.csv",
               stats_path="data/nfl/team_game_stats.csv", db_path="data/kalshi_prices.db",
               horizon_days=8, edge_threshold=0.04):
    df = rd.build_frame(games_path, stats_path)
    hist, by_season, calib = history_tables(df)

    from src.core.clock import slate_today, now_et as _now_et
    today = slate_today()
    future = df[df["result"].isna() & (df["gameday"] >= today)]
    # Same bug as src/nfl/rundown.py, independently: this path builds the
    # site's own cards from result.isna() and a date window, with no check
    # for whether a game has actually kicked off. On 2026-09-13 that priced
    # eight games against a live, in-progress Kalshi market at rebuild time.
    # This is the path a viewer actually reads on the site, so it gets the
    # identical guard -- applied to `future`, not just the horizon window,
    # so the "show next week" fallback below cannot re-admit a game that
    # has already started (it did: it rebuilt `upcoming` from the unfiltered
    # frame).
    now_et = _now_et()
    kickoff = pd.to_datetime(
        future["gameday"].dt.strftime("%Y-%m-%d") + " " + future["gametime"].fillna("13:00"),
        errors="coerce",
    ).dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    started = kickoff.notna() & (kickoff <= now_et)
    excluded_live = future.loc[started, ["gameday", "away_team", "home_team", "gametime"]]
    if len(excluded_live):
        import sys
        print(f"build_site: excluding {len(excluded_live)} game(s) already past kickoff: "
              + "; ".join(f"{r.away_team}@{r.home_team} {r.gametime}ET"
                          for r in excluded_live.itertuples()), file=sys.stderr)
    future = future[~started]
    upcoming = future[future["gameday"] <= today + pd.Timedelta(days=horizon_days)]
    week_note = ""
    if len(upcoming) == 0 and len(future):
        first = future["gameday"].min()
        upcoming = future[future["gameday"] <= first + pd.Timedelta(days=6)]
        week_note = (f'<p class="sub">No games in the next {horizon_days} days; '
                     f'showing the next scheduled week '
                     f'({first.date()} onward).</p>')
    week_rows, parlays = [], []
    price_age = ""
    # One implementation of pricing: the cards are the rundown's own rows
    # (same fit, same live fetch, same STALE downgrade, same verdict the
    # log and the paper trade see). build_site used to re-fit and re-price
    # on its own and had already drifted (code review, 2026-09-15).
    table = None
    if len(upcoming):
        table = rd.rundown(df=df, db_path=db_path, horizon_days=horizon_days,
                           edge_threshold=edge_threshold, html_out=None,
                           use_live_prices=True, log_path=None)
        if table is None and len(future):
            first = future["gameday"].min()
            span = int((first - today).days) + 7
            table = rd.rundown(df=df, db_path=db_path, horizon_days=span,
                               edge_threshold=edge_threshold, html_out=None,
                               use_live_prices=True, log_path=None)
    if table is not None and len(table):
        started_keys = {(r.gameday.date(), r.away_team, r.home_team)
                        for r in excluded_live.itertuples()}
        live = table["price_age_min"].notna().any() and (table["price_age_min"].fillna(1e9) <= 2).any()
        price_age = ('<p class="sub">Market prices fetched live at build time.</p>' if live else "")
        for r in table.to_dict("records"):
            if (r["date"], r["away"], r["home"]) in started_keys:
                continue
            v = str(r.get("verdict", ""))
            # The rundown's vocabulary (CANDIDATE X / pass / STALE ...) mapped
            # to the card's stored strings; the card decides the word.
            if v.startswith("CANDIDATE "):
                r["verdict"] = f"HIGH VALUE &mdash; {v.split(' ', 1)[1].strip()}"
            elif v == "pass":
                r["verdict"] = "NO VALUE at current price"
            elif v == "no price":
                r["verdict"] = "no price"
            week_rows.append(r)
        parlays = build_parlays(week_rows)

    cards = "".join(game_card(r) for r in week_rows) or \
        '<p class="sub">No games in the upcoming window.</p>'


    srows = "".join(
        f'<tr><td>{s}</td><td>{int(r.games)}</td><td>{r.winner_pct:.1f}%</td>'
        f'<td>{r.ats_pct:.1f}%</td><td>{r.avg_miss:.1f}</td></tr>'
        for s, r in by_season.iterrows())
    crows = "".join(
        f'<tr><td>{idx}</td><td>{int(r.games)}</td><td>{r.model_said:.0f}%</td>'
        f'<td>{r.home_actually_won:.0f}%</td></tr>'
        for idx, r in calib.iterrows())

    prow_html = []
    for p in parlays:
        cls = "ev-hi" if p["ev"] > 0.04 else "ev-md" if p["ev"] > 0 else "ev-lo"
        prow_html.append(
            f'<tr class="prow band-{p["band"]}"><td>{p["legs"]}</td>'
            f'<td>{p["p"]*100:.0f}%</td>'
            f'<td>{100/p["payout_dec"]:.0f}%</td>'
            f'<td>{american(1/p["payout_dec"])}</td>'
            f'<td class="{cls}">{"+$" + format(p["ev"]*10, ".2f") if p["ev"] >= 0 else "-$" + format(abs(p["ev"])*10, ".2f")}</td></tr>')
    prows = "".join(prow_html) or \
        '<tr><td colspan="5">Needs upcoming games and prices.</td></tr>'

    season_blocks = []
    for season, g in hist.groupby("season"):
        rows = "".join(
            f'<tr><td>W{int(x.week)}</td><td>{x.away_team} @ {x.home_team}</td>'
            f'<td>{fav_line(x.mu, x.home_team, x.away_team)}</td>'
            f'<td>{fav_line(x.spread_line, x.home_team, x.away_team) if not pd.isna(x.spread_line) else "&mdash;"}</td>'
            f'<td>{x.p_home*100:.0f}%</td>'
            f'<td>{"+" if x.y > 0 else ""}{x.y:.0f} ({x.home_team} {x.home_score:.0f} - {x.away_team} {x.away_score:.0f})</td>'
            f'<td class="{"hit" if x.su_correct else "miss"}">'
            f'{x.su_pick} &middot; {"HIT" if x.su_correct else "miss"}</td>'
            f'<td class="{"hit" if x.ats_correct == 1 else "miss" if x.ats_correct == 0 else ""}">'
            f'{ats_label(x)}</td></tr>'
            for x in g.itertuples(index=False))
        season_blocks.append(
            f'<details><summary>{season} season &mdash; every game '
            f'({len(g)})</summary><table><tr><th>Wk</th><th>Game</th>'
            f'<th>Bayesian Model line</th><th>Vegas line</th>'
            f'<th>Model: home team wins</th><th>Final margin (home score first)</th>'
            f'<th>Bayesian Model winner pick</th><th>Bayesian Model spread pick</th></tr>{rows}</table></details>')

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NFL Model HQ</title><style>{CSS}</style>
<script>{TABS_JS}</script></head><body>
<nav><button id="b-week" class="on" onclick="tab('week')">This Week</button>
<button id="b-parlays" onclick="tab('parlays')">Parlay Lab</button>
<button id="b-record" onclick="tab('record')">Track Record</button>
<button id="b-b101" onclick="tab('b101')">Bayesian 101</button></nav>
<div class="wrap">
<h1>NFL <span>Model</span> HQ</h1>
<p class="sub">A Bayesian margin model &middot; generated {today.date()}</p>
{health_strip(today)}

<div id="week" class="panel on">
<h2>This Week</h2>
<p class="sub">Two words. <b style="color:var(--yellow)">RISKY</b> means a team looks
cheaper than it should be, but picking a team to win lost 12% over five seasons in
testing (871 bets), and the spread pick covered 49.2% against a 52.4% breakeven
&mdash; so it is your call, not ours. There is no BET tier for football: nothing here
has beaten the market yet. <b>NO BET</b> means the price is fair. Tap <b>Details</b>
on any card for the numbers. Check injuries and inactives before you buy: the model
cannot see them.</p>
{week_note}{price_age}{tier_buttons(week_rows, "nfl") if week_rows else ""}<div class="grid">{cards}</div>
</div>

<div id="parlays" class="panel">
<h2>Parlay Lab</h2>
<p class="sub">Combinations of moneylines (at logged Kalshi prices, fees
included) and spreads (at the standard -110). Pick your risk appetite:
<b>Safe</b> caps the payout near +120 and ranks by hit chance; these are legs
where the model and the market mostly agree, so expect them to land often but
carry little or no edge; you are paying the fees for the fun, not beating anyone.
<b>Balanced</b> (+120 to +300) and <b>Longshot</b> (+300 to +1000) rank by the
model's expected profit, which is where real disagreements live. <b>Moonshot</b>
is the lottery-ticket tier: payouts above +1000, including legs where the model
disagrees with the market by an amount too large to fully trust, since a gap
that big often means the model is missing news rather than the market giving
money away. Moonshot numbers are the model at its least reliable; bet them for
the sweat, not the math. Read each row as: the model says this combo hits X%,
the market's prices imply Y%, and the last column is the average result of a $10 bet, blending
wins and losses at the model's probability: it is not what a winning ticket pays
(the Payout column is), it is what the bet earns or costs on average if you made
it many times. Positive means the model thinks you are being paid to take the
bet; negative means you are paying for the entertainment. Legs
are independent games only, and parlays multiply the house's cut as well as the
thrill.</p>
<div class="bandbar">
<button id="band-safe" class="bandbtn" onclick="band('safe')">Safe &le;+120</button>
<button id="band-balanced" class="bandbtn" onclick="band('balanced')">Balanced</button>
<button id="band-long" class="bandbtn" onclick="band('long')">Longshot</button>
<button id="band-moon" class="bandbtn" onclick="band('moon')">Moonshot +1000</button>
<button id="band-all" class="bandbtn on" onclick="band('all')">All</button>
</div>
<table><tr><th>Legs</th><th>Model chance</th><th>Market implied chance</th>
<th>Payout</th><th>Avg profit per $10 bet, win or lose</th></tr>
{prows}</table>
</div>

<div id="record" class="panel">
<h2>Track Record</h2>
<p class="sub">Everything on this tab is stated from the home team's perspective:
a positive margin means the home team won by that much, and every probability is
the home team's chance of winning. Every prediction below was made by the Bayesian
Model before it had seen the game: each week it trains only on games already
played, exactly as it runs live. Seasons before 2021 are excluded because the model's settings were chosen
using that era. Two separate report cards: Both scorecards below belong to the Bayesian
Model, never to Vegas: "winner pick" is the model picking the game outright, and
"spread pick" is the model's chosen side against the Vegas closing line (the pick
is spelled out in each row, for example "LAC +3"). 52.4% against the spread is
break-even at standard juice. <b>Read both columns against the market, not against
a coin flip:</b> picking 64% of winners is real skill and is not a betting edge,
because the market's price already contains it. Paper-traded against the closing
moneyline the disagreements lost 12%; against the spread the model covered 49.2%.
These are report cards on the model's knowledge, not bets.</p>
<table><tr><th>Season</th><th>Games</th><th>Bayesian Model picks winner</th>
<th>Model vs the spread</th><th>Avg miss (pts)</th></tr>{srows}</table>
<p class="sub">Honesty check. Everything in this table is from the home team's
point of view. Take all the games where the Bayesian Model gave the home team a
certain range of winning chances, then check how often the home team really won.
A single game cannot test a probability, since it either happens or it
does not; only a pile of games can. So games are grouped by what the model
predicted. Reading the first row: take every game where the model gave the home
team somewhere in the 0-35% range; those predictions averaged 28%, and home teams
in that pile actually won 34% of the time. The test is always the last two columns
against each other: when they roughly match in every row, the model's stated
confidence is honest.</p>
<table><tr><th>Games grouped by prediction</th><th>Games</th>
<th>The group's predictions averaged</th><th>Home teams in the group actually won</th></tr>
{crows}</table>
{"".join(season_blocks)}
</div>

<div id="b101" class="panel">
<h2>Bayesian 101</h2>{B101}
</div>

<footer>Every number here states its own uncertainty. Informational only.</footer>
</div></body></html>"""
    import os
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"site written to {out_path} ({len(html)//1024} KB)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site.html")
    ap.add_argument("--games", default="data/nfl/games.csv")
    ap.add_argument("--stats", default="data/nfl/team_game_stats.csv")
    ap.add_argument("--db", default="kalshi_prices.db")
    ap.add_argument("--days", type=int, default=8)
    args = ap.parse_args()
    build_site(args.out, args.games, args.stats, args.db, args.days)
