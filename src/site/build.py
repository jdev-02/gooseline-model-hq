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
<title>Model HQ</title><style>{nfl.CSS}{SWITCH_CSS}</style>
<script>{nfl.TABS_JS}{SWITCH_JS}</script></head><body>
<div class="sport"><button id="sw-nfl" onclick="sport('nfl')">NFL</button>
<button id="sw-mlb" class="on" onclick="sport('mlb')">MLB</button></div>
<div id="page-nfl" class="page">{nfl_body}</div>
<div id="page-mlb" class="page on">{mlb.render(slate, today)}</div>
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
