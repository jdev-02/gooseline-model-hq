"""MLB StatsAPI client with an on-disk cache. Keyless, public.

Tier A: one schedule call per season (scores, probable pitchers, venue,
day/night, per-inning linescore). Tier B: one boxscore call per game
(per-pitcher lines). The schedule cache is committed; boxscores are not.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE = "https://statsapi.mlb.com/api/v1"
SCHEDULE_HYDRATE = "probablePitcher,decisions,linescore,venue"
RAW = Path("data/mlb/raw")


def get_json(session, path, params=None, retries=4, timeout=60):
    for attempt in range(retries):
        try:
            r = session.get(BASE + path, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} after error: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"gave up on {path}")


def fetch_teams(season, session=None):
    session = session or requests.Session()
    data = get_json(session, "/teams", {"sportId": 1, "season": season})
    return [{"team_id": t["id"], "abbrev": t["abbreviation"], "name": t["name"],
             "division_id": t.get("division", {}).get("id"),
             "league_id": t.get("league", {}).get("id"),
             "venue_id": t.get("venue", {}).get("id")} for t in data["teams"]]


def schedule_path(season, cache_dir=RAW):
    return Path(cache_dir) / "schedule" / f"{season}.json.gz"


def fetch_season_schedule(season, cache_dir=RAW, force=False, session=None):
    fp = schedule_path(season, cache_dir)
    if fp.exists() and not force:
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            return json.load(f)
    session = session or requests.Session()
    data = get_json(session, "/schedule", {
        "sportId": 1, "season": season, "gameType": "R", "hydrate": SCHEDULE_HYDRATE})
    fp.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(fp, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def fetch_live_schedule(start_date, end_date, session=None):
    """Never cached: used at rundown time so probable pitchers are current."""
    session = session or requests.Session()
    return get_json(session, "/schedule", {
        "sportId": 1, "startDate": str(start_date), "endDate": str(end_date),
        "gameType": "R", "hydrate": SCHEDULE_HYDRATE})


def boxscore_path(season, game_pk, cache_dir=RAW):
    return Path(cache_dir) / "boxscore" / str(season) / f"{game_pk}.json"


def fetch_boxscore(season, game_pk, cache_dir=RAW, session=None, force=False):
    fp = boxscore_path(season, game_pk, cache_dir)
    if fp.exists() and fp.stat().st_size > 0 and not force:
        return json.loads(fp.read_text(encoding="utf-8"))
    session = session or requests.Session()
    data = get_json(session, f"/game/{game_pk}/boxscore")
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(data), encoding="utf-8")
    return data


def backfill_schedules(seasons, cache_dir=RAW, force=False):
    session = requests.Session()
    for s in seasons:
        data = fetch_season_schedule(s, cache_dir, force=force, session=session)
        print(f"schedule {s}: {data.get('totalGames')} games")


def backfill_boxscores(pairs, cache_dir=RAW, workers=8, sleep=0.05):
    """pairs: iterable of (season, game_pk). Resumable; skips cached files."""
    todo = [(s, g) for s, g in pairs if not boxscore_path(s, g, cache_dir).exists()]
    print(f"boxscores: {len(todo)} to fetch")
    ok = fail = 0
    session = requests.Session()

    def one(sg):
        time.sleep(sleep)
        return fetch_boxscore(sg[0], sg[1], cache_dir, session=session)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, sg): sg for sg in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
                ok += 1
            except Exception as e:
                fail += 1
                print(f"  {futs[fut]} failed: {e}", file=sys.stderr)
            if i % 500 == 0:
                print(f"  {i}/{len(todo)}")
    return ok, fail
