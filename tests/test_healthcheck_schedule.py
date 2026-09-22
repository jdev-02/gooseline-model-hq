"""A rained-out game keeps its old gamePk under its original date's bucket
in the bulk season schedule pull (alongside a second entry for its makeup
date). The health check's "how many games today" count must drop it the
same way compile_games does, or a normal postponement reads as a broken
pipeline. Found 2026-09-22: KXMLB game 824785 (TOR@BAL) counted as "today"
by the schedule check while correctly absent from games.csv, a false
"games.csv complete for today" hard failure. Synthetic data only."""
import gzip
import json
from datetime import date

import pandas as pd

import ops.healthcheck as hc


def _write_schedule(tmp_path, today, games):
    p = tmp_path / "raw" / "schedule"
    p.mkdir(parents=True)
    payload = {"dates": [{"date": str(today), "games": games}]}
    with gzip.open(p / f"{today.year}.json.gz", "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    return p / f"{today.year}.json.gz"


def _game(pk, status, home_id=100, away_id=200):
    return {"gamePk": pk, "status": {"detailedState": status},
            "teams": {"home": {"team": {"id": home_id, "abbreviation": "BAL"}},
                      "away": {"team": {"id": away_id, "abbreviation": "TOR"}}}}


def test_postponed_game_excluded_from_todays_count(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "DATA", tmp_path)
    today = date(2026, 9, 22)
    _write_schedule(tmp_path, today, [
        _game(824785, "Postponed"),   # moved to its makeup date; not today
        _game(111111, "Scheduled"),
        _game(222222, "Final"),
    ])
    pd.DataFrame({"home_team_id": [], "home_team": []}).to_csv(tmp_path / "games.csv", index=False)
    sched, _ = hc._todays_schedule(today)
    assert set(sched["game_pk"]) == {111111, 222222}


def test_cancelled_and_suspended_also_excluded(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "DATA", tmp_path)
    today = date(2026, 9, 22)
    _write_schedule(tmp_path, today, [
        _game(1, "Cancelled"), _game(2, "Suspended"), _game(3, "Final"),
    ])
    pd.DataFrame({"home_team_id": [], "home_team": []}).to_csv(tmp_path / "games.csv", index=False)
    sched, _ = hc._todays_schedule(today)
    assert set(sched["game_pk"]) == {3}


def test_schedule_count_agrees_with_compile_games_filter():
    """The two functions must never disagree on which statuses are 'dead'."""
    from src.mlb.compile import DEAD_STATUSES
    assert DEAD_STATUSES == ["Postponed", "Cancelled", "Suspended"]
