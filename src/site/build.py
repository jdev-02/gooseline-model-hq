"""Combined NFL + MLB Model HQ -> docs/index.html (GitHub Pages).

  uv run python -m src.site.build --mlb-narrative data/mlb/narrative/2026-08-29.yaml

The NFL page is David's website.py output, mounted verbatim under the NFL
switch. The MLB page (src/site/mlb_page.py) mirrors its four tabs, card
grammar, badges and copy register in run units.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

import src.site.nfl_site as nfl
import src.site.mlb_page as mlb

SWITCH_CSS = """
.masthead{max-width:920px;margin:0 auto;padding:18px 14px 4px}
.masthead .kicker{color:var(--green);font:700 .72rem "Segoe UI",sans-serif;
  letter-spacing:.14em;text-transform:uppercase}
.masthead h1.brand{font-family:"Arial Narrow","Segoe UI",sans-serif;font-size:1.5rem;
  text-transform:uppercase;letter-spacing:.05em;margin:2px 0 6px}
.masthead p{color:var(--white);font-size:.9rem;max-width:70ch;margin-bottom:8px}
.masthead .what{background:var(--panel2);border:1px solid var(--line);
  border-radius:10px;padding:12px 14px;margin-top:10px;font-size:.86rem}
.masthead .what b{color:var(--green)}
.disc{max-width:920px;margin:26px auto 0;padding:16px 14px 40px;
  border-top:1px solid var(--line);font-size:.8rem;color:var(--white)}
.disc h3{color:var(--green);font:700 .78rem "Segoe UI",sans-serif;
  letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px}
.disc ul{margin:0 0 10px 18px}
.disc li{margin-bottom:5px}
.disc .gam{border:1px solid var(--yellow);border-radius:8px;padding:10px 12px;
  color:var(--yellow);margin:10px 0}
.sport{display:flex;gap:8px;justify-content:center;padding:10px 0 4px}
.sport button{background:none;border:2px solid var(--green);color:var(--green);
  padding:8px 22px;border-radius:8px;font:700 .95rem "Segoe UI",sans-serif;cursor:pointer}
.sport button.on{background:var(--green);color:#08120b}
.page{display:none}.page.on{display:block}
img.fig{max-width:100%;border-radius:8px;border:1px solid var(--line)}
.two{display:grid;gap:12px}@media(min-width:700px){.two{grid-template-columns:1fr 1fr}}

/* Mobile. `justify-content:center` on a horizontally scrolling flex row
   clips the first item past the left edge, which is what cut off the
   "Today's Slate" tab. Wrap the tabs instead of scrolling them. */
@media(max-width:760px){
  nav{flex-wrap:wrap;justify-content:flex-start;overflow-x:visible;
      gap:6px;padding:8px 12px}
  nav button{font-size:.78rem;padding:6px 12px}
  .masthead{padding:14px 12px 4px}
  .masthead h1.brand{font-size:1.3rem}
  .masthead p,.masthead .what{font-size:.85rem}
  .wrap{padding:14px 12px 32px}
  h1{font-size:1.4rem}
  .bandbar{flex-wrap:wrap}
  table{font-size:.75rem}
  th,td{padding:4px 5px}
  /* Wide tables scroll inside their own box rather than the page body. */
  details table,.panel>table{display:block;overflow-x:auto;white-space:nowrap}
}
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


MASTHEAD = """
<div class="masthead">
<div class="kicker">Gooseline Solutions &middot; Applied Forecasting</div>
<h1 class="brand">Model HQ</h1>
<p>A working demonstration of the AI and Data practice: Bayesian forecasting,
Kalman state estimation, walk-forward validation, and probability calibration,
applied end to end on a live public dataset that settles itself every night.</p>
<div class="what">
<b>What this is.</b> A portfolio project. Sports are the test bed because the
data is public, the predictions are falsifiable, and the answer arrives in three
hours instead of three quarters. Every number below was produced by a model that
had not seen the game, and the track record shows the misses next to the hits.
<br><br>
<b>What this is not.</b> It is not a betting service, a tipsheet, or a product.
There is nothing to buy, no picks for sale, and no sportsbook links anywhere on
this page. When the model has no edge it says so, which is most nights.
</div>
</div>
"""

DISCLAIMER = """
<div class="disc">
<h3>Important notice</h3>
<ul>
<li><b>For informational and educational purposes only.</b> Nothing here is
betting advice, wagering advice, financial advice, or a recommendation to place
any wager or transaction.</li>
<li><b>No guarantee of accuracy.</b> These are statistical estimates from a model
that is wrong regularly and by design states how wrong it expects to be. Past
performance does not predict future results.</li>
<li><b>Use entirely at your own risk.</b> Any decision you make after reading this
page is yours alone.</li>
<li><b>Where gambling is involved, you must be of legal age</b> in your
jurisdiction, generally 21 or older. Laws differ by state and country; complying
with them is your responsibility.</li>
<li><b>No commercial betting relationship.</b> This page carries no sportsbook
affiliate links, no paid picks, and no referral arrangements of any kind.</li>
</ul>
<div class="gam">If you or someone you know has a gambling problem, help is
available. Call <b>1-800-GAMBLER</b> (1-800-426-2537) or visit
ncpgambling.org. Please gamble responsibly.</div>
<p>Model HQ is a portfolio project of Gooseline Solutions LLC (Derry, New
Hampshire). The NFL model, its Kalman and ensemble design, and the original site
are the work of David (<a href="https://github.com/HowlsCastle97"
style="color:var(--green)">HowlsCastle97</a>), a collaborator on this repository;
the evaluation methodology and the MLB extension are Gooseline's.</p>
</div>
"""


def build(out="docs/index.html", narrative=None, days=1,
          db="data/kalshi_prices.db", skip_nfl=False):
    today = pd.Timestamp.today().date()

    nfl_body = '<div class="wrap"><p class="sub">NFL page not built this run.</p></div>'
    if not skip_nfl:
        tmp = Path("site_nfl_tmp.html")
        nfl.build_site(str(tmp), db_path=db)
        html = tmp.read_text(encoding="utf-8")
        tmp.unlink()
        m = re.search(r"<body>(.*)</body>", html, re.S)
        if m:
            nfl_body = m.group(1)

    from src.mlb.rundown import rundown
    table = rundown(days=days, db_path=db, narrative_path=narrative, log_path=None)
    slate = table.to_dict("records") if table is not None else []

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model HQ | Gooseline Solutions</title>
<meta name="description" content="A portfolio project in applied Bayesian forecasting: Kalman team ratings, walk-forward validation and probability calibration on live public sports data. Informational only; not betting advice.">
<meta name="robots" content="index,follow">
<style>{nfl.CSS}{SWITCH_CSS}</style>
<script>{nfl.TABS_JS}{SWITCH_JS}</script></head><body>
{MASTHEAD}
<div class="sport"><button id="sw-nfl" onclick="sport('nfl')">NFL</button>
<button id="sw-mlb" class="on" onclick="sport('mlb')">MLB</button></div>
<div id="page-nfl" class="page">{nfl_body}</div>
<div id="page-mlb" class="page on">{mlb.render(slate, today)}</div>
{DISCLAIMER}
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
    ap.add_argument("--skip-nfl", action="store_true")
    a = ap.parse_args()
    build(a.out, a.mlb_narrative, a.days, a.db, a.skip_nfl)
