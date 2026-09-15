"""Health gate for the NFL daily run. Same purpose as ops/healthcheck.py on
the MLB side, and one check in particular exists because of a specific
incident: on 2026-09-13 a delayed cron ran mid-Sunday, after eight games had
already kicked off, and both NFL pricing paths (src/nfl/rundown.py and
src/site/nfl_site.py's build_site) compared a static pregame model number to
Kalshi's live, in-progress price for all eight -- three read as 30-44%
"edges" that were never real. Both paths now filter by kickoff time; this
gate independently re-derives kickoff for every card the site actually
shows and fails loudly if one slips through anyway.

  uv run python ops/healthcheck_nfl.py --stage data    # after rundown, before site
  uv run python ops/healthcheck_nfl.py --stage site    # after the site is built

Writes data/nfl/health.json, rendered by src/site/nfl_site.py as the run
health strip.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "nfl"
HEALTH = DATA / "health.json"

RUN_MAX_AGE = 240          # minutes; three cron slots span this window
PRICE_MAX_AGE_HOURS = 30   # a Kalshi price this old is not live, but the
                            # market may simply not be open yet -- soft only


class Report:
    def __init__(self):
        self.checks = []

    def add(self, name, ok, detail, hard=True):
        self.checks.append({"name": name, "ok": bool(ok), "hard": bool(hard), "detail": str(detail)})
        print(f"[{'ok  ' if ok else ('FAIL' if hard else 'warn')}] {name}: {detail}")

    @property
    def hard_failures(self):
        return [c for c in self.checks if not c["ok"] and c["hard"]]


def _file_age_min(p: Path):
    return ((datetime.now(timezone.utc) - datetime.fromtimestamp(p.stat().st_mtime, timezone.utc))
            .total_seconds() / 60.0) if p.exists() else None


def _kickoff_utc(gameday, gametime):
    """gameday: date-like; gametime: 'HH:MM' Eastern (nflverse convention)."""
    t = pd.to_datetime(pd.Series([str(gameday)]) + " " + (gametime or "13:00"), errors="coerce")
    if t.isna().all():
        return None
    t = t.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    return t.dt.tz_convert("UTC").iloc[0]


def stage_data(rep, today):
    gp = DATA / "games.csv"
    age = _file_age_min(gp)
    rep.add("games.csv refreshed", age is not None and age <= RUN_MAX_AGE,
            "missing" if age is None else f"{age:.0f} min old")
    if age is None:
        return
    games = pd.read_csv(gp, low_memory=False)
    games["gameday"] = pd.to_datetime(games["gameday"])

    horizon = games[games["result"].isna() & (games["gameday"] >= today)
                    & (games["gameday"] <= today + timedelta(days=8))]
    if not len(horizon):
        rep.add("rundown ran this run", True, "no games in the 8-day horizon (bye week/offseason); nothing to price")
        return

    log_p = DATA / "rundown_log.csv"
    log = pd.read_csv(log_p, low_memory=False) if log_p.exists() else pd.DataFrame()
    if not len(log):
        rep.add("rundown ran this run", False, "data/nfl/rundown_log.csv missing or empty")
        return
    # Dedup by (away, home, date), not (away, home) alone: divisional
    # opponents meet twice a season, and away/home alone would collapse both
    # meetings into whichever was priced most recently.
    log = log.sort_values("run_ts").drop_duplicates(["away", "home", "date"], keep="last")
    run_age = _file_age_min(log_p)
    rep.add("rundown ran this run", run_age is not None and run_age <= RUN_MAX_AGE,
            "no rundown_log.csv" if run_age is None else f"log last written {run_age:.0f} min old")

    # The check that exists because of the incident: for every row the
    # rundown priced, re-derive that game's kickoff and confirm the row's own
    # run_ts is strictly before it. A row here at all means the pricing code
    # judged the game still upcoming; this independently checks that
    # judgment against the clock, not against the code that made it.
    # game_id formats differ between the two files (nflverse's season_week_
    # away_home vs. the rundown's own date_away_home); (away, home, date) is
    # the safe composite key -- a team cannot play the same opponent twice
    # on the same date, unlike across a season, so away+home alone joins
    # every past meeting of that pair and explodes into nonsense.
    # Scoped to rows from *this* run, not the log's whole history: nflverse's
    # public games.csv can take a day or more to post a final result, so
    # `result.isna()` alone still reads Sunday's already-played games as
    # unsettled Monday morning, and a fixed-forward bug would otherwise stay
    # a permanent hard failure for every game it ever touched. This is meant
    # to catch a fresh recurrence, not re-litigate history.
    g = games.rename(columns={"away_team": "away", "home_team": "home"})
    g["gameday_str"] = g["gameday"].dt.strftime("%Y-%m-%d")
    g = g.merge(log[["away", "home", "date", "run_ts"]].rename(columns={"date": "gameday_str"}),
                on=["away", "home", "gameday_str"], how="inner")
    now_utc = pd.Timestamp.now(tz="UTC")
    bad = []
    for r in g.itertuples():
        run_ts = pd.Timestamp(r.run_ts)
        run_ts = run_ts.tz_localize("UTC") if run_ts.tzinfo is None else run_ts.tz_convert("UTC")
        if (now_utc - run_ts).total_seconds() / 60.0 > RUN_MAX_AGE:
            continue
        ko = _kickoff_utc(r.gameday.date(), getattr(r, "gametime", None))
        if ko is None:
            continue
        if run_ts >= ko:
            bad.append(f"{r.away}@{r.home} (kickoff {ko:%H:%MZ}, priced {run_ts:%H:%MZ})")
    detail = "clean" if not bad else ("; ".join(bad[:5]) + (f"; +{len(bad)-5} more" if len(bad) > 5 else ""))
    rep.add("no game priced past its own kickoff", not bad, detail)

    # Everything the row needs to be a real signal -- scoped to this run's
    # rows, like the kickoff check above. Over the whole log these checks
    # passed on the strength of past weeks when today's pricing had failed
    # outright, and one NaN in September would have failed every day after.
    rts = pd.to_datetime(log["run_ts"], utc=True)
    log = log[(now_utc - rts).dt.total_seconds() / 60.0 <= RUN_MAX_AGE]
    n = len(log)
    rep.add("rows priced this run", n > 0, f"{n} games priced in the last {RUN_MAX_AGE} min")
    if n == 0:
        return
    bad_p = log["p_home"].isna().sum() if "p_home" in log else n
    rep.add("moneyline probabilities finite", bad_p == 0, f"{n - bad_p}/{n}")
    priced = log["mkt_home"].notna().sum() if "mkt_home" in log else 0
    rep.add("Kalshi prices present", priced >= max(1, int(0.5 * n)),
            f"{priced}/{n} games have a price", hard=priced == 0)
    if "price_age_min" in log and priced:
        fresh = (log["price_age_min"].fillna(1e9) <= PRICE_MAX_AGE_HOURS * 60).sum()
        rep.add("Kalshi prices fresh", fresh >= max(1, int(0.6 * priced)),
                f"{fresh}/{priced} priced games under {PRICE_MAX_AGE_HOURS}h", hard=False)

    db = ROOT / "data" / "kalshi_prices.db"
    if db.exists():
        ts = sqlite3.connect(db).execute(
            "SELECT MAX(ts_utc) FROM snapshots WHERE series_ticker LIKE 'KXNFL%'").fetchone()[0]
        if ts:
            t = pd.Timestamp(ts)
            t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
            a = (pd.Timestamp.now(tz="UTC") - t).total_seconds() / 60.0
            rep.add("kalshi-snapshot alive (NFL series)", a <= 90, f"last snapshot {a:.0f} min ago", hard=False)
        else:
            rep.add("kalshi-snapshot alive (NFL series)", False, "no KXNFL snapshots logged", hard=False)


def stage_site(rep, today):
    idx = ROOT / "docs" / "index.html"
    age = _file_age_min(idx)
    rep.add("site rebuilt this run", age is not None and age <= RUN_MAX_AGE,
            "missing" if age is None else f"{age:.0f} min old, {idx.stat().st_size // 1024} KB")
    if age is None:
        return
    html = idx.read_text(encoding="utf-8", errors="replace")

    gp = DATA / "games.csv"
    if not gp.exists():
        return
    games = pd.read_csv(gp, low_memory=False)
    games["gameday"] = pd.to_datetime(games["gameday"])
    upcoming = games[games["result"].isna() & (games["gameday"] >= today)
                     & (games["gameday"] <= today + timedelta(days=8))]
    now = pd.Timestamp.now(tz="UTC")
    live_but_shown, missing = [], []
    for r in upcoming.itertuples():
        label_variants = (f"{r.away_team} @ {r.home_team}", f"{r.away_team}@{r.home_team}")
        shown = any(v in html for v in label_variants)
        ko = _kickoff_utc(r.gameday.date(), getattr(r, "gametime", None))
        started = ko is not None and now >= ko
        if started and shown:
            live_but_shown.append(f"{r.away_team}@{r.home_team}")
        elif not started and not shown and r.gameday <= today + timedelta(days=2):
            missing.append(f"{r.away_team}@{r.home_team}")
    rep.add("no already-started game shown as upcoming on the site", not live_but_shown,
            "clean" if not live_but_shown else "; ".join(live_but_shown))
    rep.add("near-term upcoming games appear on the site", not missing,
            "clean" if not missing else f"missing: {'; '.join(missing)}", hard=False)
    rep.add("site has a freshness strip", 'class="health ' in html, "health strip rendered", hard=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["data", "site"], required=True)
    ap.add_argument("--today", default=None)
    a = ap.parse_args()
    # tz-naive, to compare against games.csv's tz-naive gameday column
    # (rundown.py and nfl_site.py both key off Eastern-local dates this way).
    from src.core.clock import slate_today
    today = pd.Timestamp(a.today) if a.today else slate_today()
    rep = Report()
    (stage_data if a.stage == "data" else stage_site)(rep, today)

    prev = json.loads(HEALTH.read_text()) if HEALTH.exists() else {}
    checks = [c for c in prev.get("checks", []) if c.get("stage") != a.stage] if prev else []
    for c in rep.checks:
        c["stage"] = a.stage
    checks += rep.checks
    out = {"run_ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "run_date": str(today.date()),
           "ok": all(c["ok"] or not c["hard"] for c in checks),
           "hard_failures": [c["name"] for c in checks if not c["ok"] and c["hard"]],
           "warnings": [c["name"] for c in checks if not c["ok"] and not c["hard"]],
           "checks": checks}
    HEALTH.parent.mkdir(parents=True, exist_ok=True)
    HEALTH.write_text(json.dumps(out, indent=2))
    n_fail = len(rep.hard_failures)
    print(f"\n{a.stage}: {len(rep.checks)} checks, {n_fail} hard failures -> {HEALTH}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
