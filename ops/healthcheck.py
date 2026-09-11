"""Health gate for the MLB daily run. Every facet the model depends on is
checked against the clock, not assumed.

  uv run python ops/healthcheck.py --stage data    # after rundown, before site
  uv run python ops/healthcheck.py --stage site    # after the site is built

Each check is HARD (the run is unhealthy; exit 1 so the workflow fails and
the watchdog re-dispatches) or SOFT (recorded and shown, does not fail).
Results accumulate in data/mlb/health.json, which the site renders as the
freshness strip, so a reader can see when the numbers were made and from
what. A run that looks complete and silently is not is the failure mode
this file exists to catch: the day the cron did not fire, the day a game
was missing from the slate, the day every game was priced at 70F.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mlb.totals import TOTAL_FEATURE_COLS  # noqa: E402
from src.mlb.features import MLB_FEATURE_COLS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mlb"
HEALTH = DATA / "health.json"
UPCOMING_STATES = {"Scheduled", "Pre-Game", "Warmup", "Delayed Start"}

# Freshness windows, in minutes.
SCHEDULE_MAX_AGE = 180      # the season pull at the top of the run
RUN_MAX_AGE = 180           # rundown rows must come from this run
PRICE_MAX_AGE = 20          # a Kalshi price older than this is not a price
SNAPSHOT_MAX_AGE = 90       # kalshi-snapshot is its own cron; tolerate one miss


class Report:
    def __init__(self, stage):
        self.stage = stage
        self.checks = []

    def add(self, name, ok, detail, hard=True):
        self.checks.append({"name": name, "ok": bool(ok), "hard": bool(hard), "detail": str(detail)})
        flag = "ok  " if ok else ("FAIL" if hard else "warn")
        print(f"[{flag}] {name}: {detail}")

    @property
    def hard_failures(self):
        return [c for c in self.checks if not c["ok"] and c["hard"]]


def _age_min(ts):
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return (pd.Timestamp.now(tz="UTC") - t).total_seconds() / 60.0


def _file_age_min(p: Path):
    return (datetime.now(timezone.utc) - datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)).total_seconds() / 60.0 if p.exists() else None


def _todays_schedule(today):
    """Today's regular-season games straight from the season pull, so the
    check does not trust games.csv to tell it what games.csv should hold."""
    p = DATA / "raw" / "schedule" / f"{today.year}.json.gz"
    if not p.exists():
        return None, p
    with gzip.open(p, "rt", encoding="utf-8") as f:
        raw = json.load(f)
    # The season pull does not hydrate abbreviations; games.csv learned them
    # from the teams table, so borrow its id -> abbreviation map.
    gp = DATA / "games.csv"
    abbr = {}
    if gp.exists():
        g = pd.read_csv(gp, usecols=["home_team_id", "home_team"], low_memory=False).dropna()
        abbr = dict(zip(g["home_team_id"].astype(int), g["home_team"]))
    rows = []
    for d in raw.get("dates", []):
        if d["date"] != str(today):
            continue
        for g in d["games"]:
            hid, aid = g["teams"]["home"]["team"]["id"], g["teams"]["away"]["team"]["id"]
            rows.append({"game_pk": g["gamePk"], "status": g["status"].get("detailedState"),
                         "home": g["teams"]["home"]["team"].get("abbreviation") or abbr.get(hid, hid),
                         "away": g["teams"]["away"]["team"].get("abbreviation") or abbr.get(aid, aid)})
    return pd.DataFrame(rows), p


def stage_data(rep, today):
    # 1. The schedule was pulled this run.
    sched, sp = _todays_schedule(today)
    age = _file_age_min(sp)
    rep.add("schedule refreshed", age is not None and age <= SCHEDULE_MAX_AGE,
            f"{sp.name} {'missing' if age is None else f'{age:.0f} min old'}")
    if sched is None:
        return
    upcoming = sched[sched["status"].isin(UPCOMING_STATES)]
    rep.add("schedule has today", len(sched) > 0 or today.month in (11, 12, 1, 2, 3),
            f"{len(sched)} games today, {len(upcoming)} still to play")

    # 2. games.csv holds every one of them.
    games = pd.read_csv(DATA / "games.csv", low_memory=False)
    have = set(games.loc[games["gameday"] == str(today), "game_pk"].astype(int))
    missing = sorted(set(sched["game_pk"].astype(int)) - have)
    rep.add("games.csv complete for today", not missing,
            f"{len(have)}/{len(sched)} present" + (f", missing gamePk {missing}" if missing else ""))

    # 3. Yesterday settled: every Final game has scores and a filled log result.
    y = games[(games["gameday"] == str(today - timedelta(days=1)))]
    finals = y[y["status"].fillna("").str.startswith("Final")]
    unsettled = finals[~finals["played"].astype(bool)]
    rep.add("yesterday settled in games.csv", len(unsettled) == 0,
            f"{len(finals) - len(unsettled)}/{len(finals)} finals carry scores")
    log_p = DATA / "narrative" / "log.csv"
    log = pd.read_csv(log_p, low_memory=False) if log_p.exists() else pd.DataFrame()
    if len(log):
        ylog = log[log["date"] == str(today - timedelta(days=1))]
        ylog = ylog.sort_values("run_ts").drop_duplicates("game_pk", keep="last")
        played_pks = set(finals.loc[finals["played"].astype(bool), "game_pk"].astype(int))
        open_rows = ylog[ylog["game_pk"].astype(int).isin(played_pks) & ylog["result"].isna()]
        rep.add("yesterday settled in log", len(open_rows) == 0,
                f"{len(open_rows)} logged games still without a result")

    # 4. Today's rundown: one fresh row per upcoming game, fully populated.
    tlog = log[log["date"] == str(today)] if len(log) else pd.DataFrame()
    if len(tlog):
        tlog = tlog.sort_values("run_ts").drop_duplicates("game_pk", keep="last")
        run_age = _age_min(tlog["run_ts"].max())
    else:
        run_age = None
    rep.add("rundown ran this run", run_age is not None and run_age <= RUN_MAX_AGE,
            "no rows for today" if run_age is None else f"latest row {run_age:.0f} min old")
    if len(tlog) and len(upcoming):
        logged = set(tlog["game_pk"].astype(int))
        miss = sorted(set(upcoming["game_pk"].astype(int)) - logged)
        rep.add("every upcoming game priced", not miss,
                f"{len(logged & set(upcoming['game_pk'].astype(int)))}/{len(upcoming)}"
                + (f", missing {miss}" if miss else ""))
        n = len(tlog)
        bad_p = tlog["p_home"].isna().sum()
        rep.add("moneyline probabilities finite", bad_p == 0, f"{n - bad_p}/{n}")
        bad_t = tlog["mu_total"].isna().sum() if "mu_total" in tlog else n
        rep.add("totals priced", bad_t == 0, f"{n - bad_t}/{n}")
        if "temp_f" in tlog and "roof_closed" in tlog:
            roof = tlog["roof_closed"].astype(str).isin(["True", "1", "1.0"])
            no_wx = (tlog["temp_f"].isna() & ~roof).sum()
            rep.add("weather at every open-roof game", no_wx == 0,
                    f"{n - no_wx}/{n} have a temperature or a closed roof")
        else:
            rep.add("weather columns present", False, "temp_f/roof_closed not in log", hard=True)
        priced = tlog["mkt_home"].notna().sum()
        rep.add("Kalshi prices present", priced >= max(1, int(0.6 * n)),
                f"{priced}/{n} games have a moneyline price", hard=priced == 0)
        fresh = (tlog["price_age_min"].fillna(1e9) <= PRICE_MAX_AGE).sum()
        rep.add("Kalshi prices fresh", fresh >= max(1, int(0.8 * priced)) if priced else True,
                f"{fresh}/{priced} priced games under {PRICE_MAX_AGE} min", hard=False)
        sp_both = (tlog["home_sp"].notna() & tlog["away_sp"].notna()).sum()
        rep.add("probable pitchers known", sp_both >= int(0.5 * n),
                f"{sp_both}/{n} games have both starters", hard=False)

    # 5. The snapshot cron is alive.
    db = ROOT / "data" / "kalshi_prices.db"
    if db.exists():
        ts = sqlite3.connect(db).execute("SELECT MAX(ts_utc) FROM snapshots").fetchone()[0]
        a = _age_min(ts)
        rep.add("kalshi-snapshot alive", a is not None and a <= SNAPSHOT_MAX_AGE,
                f"last snapshot {'never' if a is None else f'{a:.0f} min ago'}", hard=False)
    else:
        rep.add("kalshi-snapshot alive", False, "data/kalshi_prices.db missing", hard=False)

    # 6. Frozen configs match the code that will read them.
    tc = DATA / "totals_config.json"
    if tc.exists():
        cols = json.loads(tc.read_text())["feature_cols"]
        rep.add("totals config matches code", cols == TOTAL_FEATURE_COLS,
                "feature_cols in sync" if cols == TOTAL_FEATURE_COLS else f"config {cols} != code {TOTAL_FEATURE_COLS}")
    else:
        rep.add("totals config present", False, "data/mlb/totals_config.json missing")
    mc = DATA / "model_config.json"
    if mc.exists():
        cols = json.loads(mc.read_text()).get("feature_cols", [])
        extra = sorted(set(cols) - set(MLB_FEATURE_COLS))
        rep.add("moneyline config matches code", not extra,
                "feature_cols in sync" if not extra else f"config names unknown features {extra}")


def stage_site(rep, today):
    idx = ROOT / "docs" / "index.html"
    age = _file_age_min(idx)
    rep.add("site rebuilt this run", age is not None and age <= RUN_MAX_AGE,
            "missing" if age is None else f"{age:.0f} min old, {idx.stat().st_size // 1024} KB")
    if age is None:
        return
    html = idx.read_text(encoding="utf-8", errors="replace")
    rep.add("site dated today", f"generated {today}" in html, f"looked for 'generated {today}'")
    sched, _ = _todays_schedule(today)
    if sched is not None and len(sched):
        up = sched[sched["status"].isin(UPCOMING_STATES)]
        shown = [f"{r.away} @ {r.home}" for r in up.itertuples() if f"{r.away} @ {r.home}" in html]
        rep.add("site shows every upcoming game", len(shown) == len(up),
                f"{len(shown)}/{len(up)} matchups found in docs/index.html")
    rep.add("site has a freshness strip", 'class="health ' in html, "health strip rendered", hard=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["data", "site"], required=True)
    ap.add_argument("--today", default=None, help="YYYY-MM-DD, default: today UTC-4")
    a = ap.parse_args()
    # The slate is an Eastern-time notion; the runners are UTC.
    today = date.fromisoformat(a.today) if a.today else (datetime.now(timezone.utc) - timedelta(hours=4)).date()
    rep = Report(a.stage)
    (stage_data if a.stage == "data" else stage_site)(rep, today)

    prev = json.loads(HEALTH.read_text()) if HEALTH.exists() and a.stage == "site" else {}
    checks = [c for c in prev.get("checks", []) if c.get("stage") != a.stage] if prev else []
    for c in rep.checks:
        c["stage"] = a.stage
    checks += rep.checks
    out = {"run_ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "slate_date": str(today),
           "ok": all(c["ok"] or not c["hard"] for c in checks),
           "hard_failures": [c["name"] for c in checks if not c["ok"] and c["hard"]],
           "warnings": [c["name"] for c in checks if not c["ok"] and not c["hard"]],
           "checks": checks}
    HEALTH.write_text(json.dumps(out, indent=2))
    n_fail = len(rep.hard_failures)
    print(f"\n{a.stage}: {len(rep.checks)} checks, {n_fail} hard failures, "
          f"{sum(1 for c in rep.checks if not c['ok'] and not c['hard'])} warnings -> {HEALTH}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
