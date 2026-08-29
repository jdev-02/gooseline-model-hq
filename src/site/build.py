"""Combined NFL + MLB Model HQ -> docs/index.html (GitHub Pages).

  uv run python -m src.site.build --mlb-narrative data/mlb/narrative/2026-08-27.yaml

The NFL page is David's website.py output, untouched, mounted under the NFL
switch. The MLB page reuses the same CSS and card grammar with run units.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import src.site.nfl_site as nfl
from src.mlb.compile import DATA

FIG = Path("figures/mlb/phase0")
SWITCH_CSS = """
.sport{display:flex;gap:8px;justify-content:center;padding:10px 0 4px}
.sport button{background:none;border:2px solid var(--green);color:var(--green);
  padding:8px 22px;border-radius:8px;font:700 .95rem "Segoe UI",sans-serif;cursor:pointer}
.sport button.on{background:var(--green);color:#08120b}
.page{display:none}.page.on{display:block}
img.fig{max-width:100%;border-radius:8px;border:1px solid var(--line)}
.two{display:grid;gap:12px}@media(min-width:700px){.two{grid-template-columns:1fr 1fr}}
"""
SWITCH_JS = """
function sport(s){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('on'));
  document.querySelectorAll('.sport button').forEach(b=>b.classList.remove('on'));
  document.getElementById('page-'+s).classList.add('on');
  document.getElementById('sw-'+s).classList.add('on');
  window.scrollTo(0,0);
}
function mtab(id){
  document.querySelectorAll('#page-mlb .panel').forEach(p=>p.classList.remove('on'));
  document.querySelectorAll('#page-mlb nav button').forEach(b=>b.classList.remove('on'));
  document.getElementById(id).classList.add('on');
  document.getElementById('b-'+id).classList.add('on');
  window.scrollTo(0,0);
}
"""

MLB_METHOD = """
<div class="b101">
<p><b>Same machine, different units.</b> The MLB model is the NFL model's
skeleton pointed at run differential: a Kalman filter keeps a rating for every
club that drifts a little each day and reverts hard between seasons, a
Bayesian ridge turns ratings and the stat sheet into a predicted margin with an
honest give-or-take, and the finished probability is compared with the market
price after fees.</p>
<p><b>The honest give-or-take is 4.4 runs.</b> That is measured from every game
since 2015, and it is larger than any edge the market will ever hand you. A
typical night is a 55/45 game priced at 55/45. Most cards below are passes;
that is the design working.</p>
<p><b>What is fed in, always computed only from games already played:</b></p>
<table>
<tr><th>Input</th><th>What it is</th></tr>
<tr><td>Kalman rating gap, uncertainty</td><td>Club strength difference in runs, and how well the filter knows both clubs</td></tr>
<tr><td>Run-differential form and slope</td><td>Recent margin, recent games weighted more, plus its trajectory over the last ten</td></tr>
<tr><td>Baserunner pressure</td><td>Runners put on per inning, for and against: a club getting runners on every few innings tends to break through</td></tr>
<tr><td>Stranding rate</td><td>Share of baserunners left on base</td></tr>
<tr><td>Late-inning margin</td><td>Innings seven and later: the free bullpen-and-clutch proxy</td></tr>
<tr><td>Starting pitcher FIP, command, consistency, rest</td><td>Strikeouts, walks, homers per inning for the listed starter; strike rate and how steady it is; days since last start</td></tr>
<tr><td>Bullpen FIP and workload</td><td>Relief quality, and relief innings thrown in the last three days</td></tr>
<tr><td>Park, day/night, rest, division</td><td>Expanding-window park factor, so the model never peeks at future seasons</td></tr>
</table>
<p><b>The narrative edge.</b> A human can enter a story about a game: a playoff
push, a club drifting, a clubhouse that has come together. It moves the
predicted margin by at most one run, and it always widens the uncertainty, so an
opinion can shift the number but can never buy confidence. Both the model-only
and the model-plus-narrative calls are logged and scored separately, so whether
the human helps is a number, not a feeling.</p>
<p><b>What this model honestly cannot see.</b> A late scratch, a lineup with
three regulars resting, wind blowing out. Every candidate gets a human news
check before anything happens.</p>
</div>"""


def american(p):
    return nfl.american(p)


def mlb_card(r):
    mu, sg, pm = r["mu"], r["sigma"], r["p_home"]
    call = (f"Bayesian Model: <b>{r['home']} by {abs(mu):.1f} runs</b>" if mu >= 0
            else f"Bayesian Model: <b>{r['away']} by {abs(mu):.1f} runs</b>") + f" &plusmn;{sg:.1f}"
    fav = r["home"] if pm >= 0.5 else r["away"]
    fav_p = pm if pm >= 0.5 else 1 - pm
    gline = f'Gambler terms: <b>{fav} ML {american(fav_p)}</b>'
    bars = (f'<div class="brow"><span class="blab">Bayesian model</span><div class="btrack">'
            f'<div class="bfill model" style="width:{pm*100:.1f}%"></div></div>'
            f'<span class="bval">{pm*100:.0f}%</span></div>')
    pn = r.get("p_home_narrative")
    if r.get("narrative_shift"):
        bars += (f'<div class="brow"><span class="blab">+ narrative</span><div class="btrack">'
                 f'<div class="bfill model" style="width:{pn*100:.1f}%;opacity:.55"></div></div>'
                 f'<span class="bval">{pn*100:.0f}%</span></div>')
    mk = r.get("mkt_home")
    if mk is not None and not pd.isna(mk):
        bars += (f'<div class="brow"><span class="blab">Market price</span><div class="btrack">'
                 f'<div class="bfill mkt" style="width:{mk*100:.1f}%"></div></div>'
                 f'<span class="bval">{mk*100:.0f}&cent;</span></div>')
        gap = f'<div class="gap">Disagreement: <b>{(pm-mk)*100:+.0f}</b> points of probability on {r["home"]}; edge after fees <b>{r["edge"]*100:+.1f}%</b></div>'
    else:
        gap = '<div class="gap">Market has not opened this game yet</div>'
    v = str(r["verdict"])
    badge = (f'<span class="verdict v-high">{v}</span>' if v.startswith("CANDIDATE")
             else '<span class="verdict v-avoid">Pass &mdash; price is fair</span>' if v == "pass"
             else '<span class="verdict v-none">No price yet</span>')
    note = f'<div class="gap"><i>{r["note"]}</i> &rarr; model+narrative {pn*100:.0f}%, {str(r["verdict_narrative"]).lower()}</div>' if r.get("note") else ""
    sp = f'<div class="gap">{r["away_sp"] or "TBD"} vs {r["home_sp"] or "TBD"}</div>'
    return (f'<div class="card"><div class="match"><span class="teams">{r["away"]} @ {r["home"]}</span>'
            f'<span class="date">{r["date"]}</span></div><div class="call">{call}</div>'
            f'<div class="gline">{gline}</div>{sp}<div class="bars">{bars}</div>{gap}{note}{badge}</div>')


def img_tag(path):
    p = Path(path)
    if not p.exists():
        return ""
    b = base64.b64encode(p.read_bytes()).decode()
    return f'<img class="fig" src="data:image/png;base64,{b}" alt="{p.stem}">'


def mlb_page(slate, today):
    cards = "".join(mlb_card(r) for r in slate) if slate else '<p class="sub">No games in the window.</p>'
    res = FIG / "phase0_results.csv"
    rows = ""
    if res.exists():
        t = pd.read_csv(res)
        rows = "".join(f'<tr><td>{r.model}</td><td>{int(r.n)}</td><td>{r.nll:.4f}</td><td>{r.rmse:.3f}</td>'
                       f'<td>{r.brier:.4f}</td><td>{"" if pd.isna(r.max_calib_dev) else f"{r.max_calib_dev:.3f}"}</td></tr>'
                       for r in t.itertuples())
    log = DATA / "narrative" / "log.csv"
    lrows = ""
    if log.exists():
        L = pd.read_csv(log)
        L = L[L["note"].fillna("") != ""].tail(50)
        lrows = "".join(f'<tr><td>{x.date}</td><td>{x.away} @ {x.home}</td><td>{x.narrative_shift:+.2f}</td>'
                        f'<td>{x.p_home*100:.0f}% &rarr; {x.p_home_narrative*100:.0f}%</td>'
                        f'<td>{"" if pd.isna(x.result) else f"{x.result:+.0f}"}</td><td>{x.note}</td></tr>'
                        for x in L.itertuples())
    return f"""
<nav><button id="b-slate" class="on" onclick="mtab('slate')">Today's Slate</button>
<button id="b-mrecord" onclick="mtab('mrecord')">Track Record</button>
<button id="b-nlog" onclick="mtab('nlog')">Narrative Log</button>
<button id="b-method" onclick="mtab('method')">Method</button></nav>
<div class="wrap">
<h1>MLB <span>Model</span> HQ</h1>
<p class="sub">A Bayesian run-differential model &middot; generated {today}</p>
<div id="slate" class="panel on"><h2>Today's Slate</h2>
<p class="sub">Green bar: the model's chance the home team wins. Faded green: the same
after the human narrative tilt. White: what Kalshi charges. Green badge means an edge
above 4% after the 7% fee. Every candidate still gets a news check first.</p>
<div class="grid">{cards}</div></div>
<div id="mrecord" class="panel"><h2>Track Record</h2>
<p class="sub">Walk-forward over 2023&ndash;2025 (7,289 games), refit weekly on games already
played, tuned on 2022 only. Baselines first: no model advances unless it beats both on
negative log-likelihood and Brier with every populated calibration decile within 0.10.
A good MLB margin model explains about 3&ndash;8% of variance; that is the sport, not a bug.</p>
<table><tr><th>Model</th><th>Games</th><th>NLL</th><th>RMSE (runs)</th><th>Brier</th><th>Max calib. dev.</th></tr>{rows}</table>
<div class="two">{img_tag(FIG / "reliability_linear.png")}{img_tag(FIG / "kalman_ratings_2026.png")}</div></div>
<div id="nlog" class="panel"><h2>Narrative Log</h2>
<p class="sub">Every human tilt ever entered, with what it did to the number and how the game
ended. Positive shift favors the home team.</p>
<table><tr><th>Date</th><th>Game</th><th>Shift (runs)</th><th>Home win chance</th><th>Result</th><th>Story</th></tr>
{lrows or '<tr><td colspan="6">No logged tilts yet.</td></tr>'}</table></div>
<div id="method" class="panel"><h2>Method</h2>{MLB_METHOD}</div>
<footer>One model, honestly uncertain. Nothing here is financial advice.</footer>
</div>"""


def build(out="docs/index.html", narrative=None, days=1, db="data/kalshi_prices.db"):
    today = pd.Timestamp.today().date()
    # NFL page: David's site, verbatim, body extracted
    tmp = Path("site_nfl_tmp.html")
    nfl.build_site(str(tmp), db_path=db)
    html = tmp.read_text(encoding="utf-8")
    tmp.unlink()
    nfl_body = re.search(r"<body>(.*)</body>", html, re.S).group(1)
    # MLB slate
    from src.mlb.rundown import rundown
    table = rundown(days=days, db_path=db, narrative_path=narrative, log_path=None)
    slate = table.to_dict("records") if table is not None else []
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model HQ</title><style>{nfl.CSS}{SWITCH_CSS}</style>
<script>{nfl.TABS_JS}{SWITCH_JS}</script></head><body>
<div class="sport"><button id="sw-nfl" onclick="sport('nfl')">NFL</button>
<button id="sw-mlb" class="on" onclick="sport('mlb')">MLB</button></div>
<div id="page-nfl" class="page">{nfl_body}</div>
<div id="page-mlb" class="page on">{mlb_page(slate, today)}</div>
</body></html>"""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(page, encoding="utf-8")
    print(f"site written to {out} ({len(page)//1024} KB)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/index.html")
    ap.add_argument("--mlb-narrative", default=None)
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--db", default="data/kalshi_prices.db")
    a = ap.parse_args()
    build(a.out, a.mlb_narrative, a.days, a.db)
