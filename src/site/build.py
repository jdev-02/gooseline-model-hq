"""Combined NFL + MLB Model HQ -> docs/index.html (GitHub Pages).

  uv run python -m src.site.build --mlb-narrative data/mlb/narrative/2026-08-29.yaml

The NFL page is David's website.py output, mounted verbatim under the NFL
switch. The MLB page (src/site/mlb_page.py) mirrors its four tabs, card
grammar, badges and copy register in run units.
"""
from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path as _Path
from pathlib import Path

import pandas as pd

import src.site.nfl_site as nfl
import src.site.mlb_page as mlb

# Gooseline's own favicon, so a browser tab reads as the same company whether
# the visitor is on gooselinesolutions.com or this page's own GitHub Pages URL.
_FAVICON_PATH = _Path(__file__).parent / "assets" / "favicon.png"
_FAVICON_B64 = base64.b64encode(_FAVICON_PATH.read_bytes()).decode() if _FAVICON_PATH.exists() else ""

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
  padding:8px 22px;border-radius:8px;font:700 .95rem "Segoe UI",sans-serif;cursor:pointer;
  transition:background-color .15s,color .15s,transform .1s}
.sport button:hover{background:rgba(46,224,111,.12)}
.sport button:active{transform:scale(.96)}
.sport button:focus-visible{outline:2px solid var(--green);outline-offset:2px}
.sport button.on{background:var(--green);color:#08120b}
.sport button.on:hover{background:var(--green)}
.page{display:none}
.page.on{display:block;animation:pagefade .2s ease-out}
@keyframes pagefade{from{opacity:0}to{opacity:1}}
@media(prefers-reduced-motion:reduce){.page.on{animation:none}}
img.fig{max-width:100%;border-radius:8px;border:1px solid var(--line)}
/* Freshness strip: the run's own audit, above the first card. */
.health{border:1px solid var(--line);border-left:4px solid var(--green);border-radius:8px;
  padding:8px 12px;margin:6px 0 14px;font-size:.82rem;background:var(--panel2)}
.health.warn{border-left-color:var(--yellow)}
.health.fail{border-left-color:#e5484d}
.health .facts{opacity:.8}
.health details{margin-top:4px}
.health summary{cursor:pointer;color:var(--green);font-size:.78rem}
.health ul{margin:6px 0 0 18px;padding:0}
.health li{margin:2px 0}
.health li.ok{opacity:.75}
.health li.warn{color:var(--yellow)}
.health li.fail{color:#e5484d;font-weight:700}
.two{display:grid;gap:12px}@media(min-width:700px){.two{grid-template-columns:1fr 1fr}}
/* Live pill in the sticky nav: how many Start-here bets are still open, when
   the next one closes, how old the prices are. Computed on the reader's clock
   from data-kick / data-run; the page itself does not change between runs. */
.livepill{margin-left:auto;align-self:center;font:600 .74rem "Segoe UI",sans-serif;
  padding:5px 11px;border-radius:99px;border:1px solid var(--green);color:var(--green);
  background:none;cursor:pointer;white-space:nowrap;letter-spacing:.01em}
.livepill:hover{background:rgba(46,224,111,.12)}
.livepill:active{transform:scale(.97)}
.livepill.none{border-color:var(--dim);color:var(--dim)}
.livepill.old{border-color:var(--yellow);color:var(--yellow)}
.start li.gone{opacity:.45;text-decoration:line-through}
.start .closes{color:var(--dim);font-size:.8rem}
@media(max-width:760px){.livepill{margin-left:0;width:100%;text-align:center}}

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
function mtier(t, scope){
  var page = document.getElementById('page-' + scope);
  if (!page) return;
  page.querySelectorAll('.tierbar .bandbtn').forEach(function(b){ b.classList.remove('on'); });
  var on = document.getElementById('tier-' + scope + '-' + t);
  if (on) on.classList.add('on');
  page.querySelectorAll('.grid .card').forEach(function(c){
    var show = t === 'all' || (t === 'tot' ? c.classList.contains('tot-flag')
                                           : c.classList.contains('tier-' + t));
    c.style.display = show ? '' : 'none';
  });
}
/* Kickoffs are stored as UTC instants; show them on the reader's own clock. */
function localiseKickoffs(){
  document.querySelectorAll('.card .date[data-kick]').forEach(function(el){
    var iso = el.dataset.kick; if (!iso) return;
    var d = new Date(iso); if (isNaN(d.getTime())) return;
    el.textContent = d.toLocaleDateString([], {weekday:'short', month:'short', day:'numeric'})
      + ', ' + d.toLocaleTimeString([], {hour:'numeric', minute:'2-digit', timeZoneName:'short'});
  });
}
/* The live pill. Each sport page gets one in its sticky nav, so it follows
   the reader down the slate. It re-reads the Start-here list every 30 s:
   bets whose first pitch has passed are struck through, the count drops,
   the next close is shown on the reader's clock, and the price age turns
   yellow past 90 min. Nothing here fetches anything; between runs the
   numbers on the page are the numbers from the last run. */
function fmtTime(d){ return d.toLocaleTimeString([], {hour:'numeric', minute:'2-digit'}); }
function livePill(){
  var now = Date.now();
  document.querySelectorAll('.page').forEach(function(page){
    var nav = page.querySelector('nav'); var box = page.querySelector('.start');
    if (!nav || !box) return;
    var pill = nav.querySelector('.livepill');
    if (!pill){
      pill = document.createElement('button'); pill.className = 'livepill'; pill.type = 'button';
      pill.title = 'Jump to Start here';
      pill.onclick = function(){
        var slateTab = page.querySelector('nav button'); if (slateTab) slateTab.click();
        box.scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start'});
      };
      nav.appendChild(pill);
    }
    var open = 0, next = null;
    box.querySelectorAll('li[data-kick]').forEach(function(li){
      var t = Date.parse(li.dataset.kick); var c = li.querySelector('.closes');
      if (isNaN(t)){ return; }
      if (t <= now){ li.classList.add('gone'); if (c) c.textContent = 'Started; too late.'; return; }
      open++; if (next === null || t < next) next = t;
      if (c){ var m = Math.round((t - now) / 60000);
        c.textContent = 'Buy before ' + fmtTime(new Date(t)) + (m < 120 ? ' (' + m + ' min)' : '') + '.'; }
    });
    var run = Date.parse(box.dataset.run || ''); var age = isNaN(run) ? null : Math.round((now - run) / 60000);
    var agetxt = age === null ? '' : ' · prices ' + (age < 1 ? 'just now' : age < 90 ? age + ' min old' : Math.round(age/60) + ' h old');
    var total = box.querySelectorAll('li[data-kick]').length;
    pill.textContent = total ? (open + ' of ' + total + ' bet' + (total === 1 ? '' : 's') + ' open' + (next ? ' · buy before ' + fmtTime(new Date(next)) : '') + agetxt)
                             : 'Nothing to bet' + agetxt;
    pill.className = 'livepill' + (total && open ? '' : ' none') + (age !== null && age >= 90 ? ' old' : '');
  });
}
function siteReady(){ localiseKickoffs(); livePill(); setInterval(livePill, 30000); }
if (document.readyState === 'loading'){ document.addEventListener('DOMContentLoaded', siteReady); }
else { siteReady(); }
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
this page. Where the model and the market disagree, the page says so, and the Track
Record tab shows what acting on that disagreement has actually returned, market by
market, recomputed every morning.
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
    from src.core.clock import slate_today
    today = slate_today().date()

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
    health_p = Path("data/mlb/health.json")
    health = None
    if health_p.exists():
        import json
        health = json.loads(health_p.read_text())
        # A stale record is worse than none: only show it for this slate.
        if health.get("slate_date") != str(today):
            health = None

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model HQ | Gooseline Solutions</title>
<link rel="icon" type="image/png" href="data:image/png;base64,{_FAVICON_B64}">
<meta name="description" content="A portfolio project in applied Bayesian forecasting: Kalman team ratings, walk-forward validation and probability calibration on live public sports data. Informational only; not betting advice.">
<meta name="robots" content="index,follow">
<style>{nfl.CSS}{SWITCH_CSS}</style>
<script>{nfl.TABS_JS}{SWITCH_JS}</script></head><body>
{MASTHEAD}
<div class="sport"><button id="sw-nfl" onclick="sport('nfl')">NFL</button>
<button id="sw-mlb" class="on" onclick="sport('mlb')">MLB</button></div>
<div id="page-nfl" class="page">{nfl_body}</div>
<div id="page-mlb" class="page on">{mlb.render(slate, today, health)}</div>
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
