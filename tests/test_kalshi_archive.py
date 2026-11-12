import gzip
import sqlite3

from src.core.kalshi import SCHEMA
from src.core.kalshi_archive import history, rotate, week_start
from datetime import datetime, timezone


def _live(path, stamps):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    for ts in stamps:
        con.executemany("INSERT INTO snapshots (ts_utc, ticker, series_ticker, yes_bid) VALUES (?,?,?,?)",
                        [(ts, f"KXNFLGAME-T{i}", "KXNFLGAME", 0.5) for i in range(3)])
    con.commit()
    con.close()


def test_week_start_is_monday_utc():
    assert week_start(datetime(2026, 11, 12, 19, 0, tzinfo=timezone.utc)) == "2026-11-09T00:00:00+00:00"
    assert week_start(datetime(2026, 11, 9, 0, 0, tzinfo=timezone.utc)) == "2026-11-09T00:00:00+00:00"


def test_rotate_moves_finished_weeks_only(tmp_path):
    live, arc = tmp_path / "live.db", tmp_path / "arc"
    old = ["2026-10-30T15:07:00+00:00", "2026-11-01T23:37:00+00:00", "2026-11-02T14:07:00+00:00"]
    new = ["2026-11-09T14:07:00+00:00", "2026-11-12T19:07:00+00:00"]
    _live(live, old + new)

    moved = rotate(live, arc, cutoff="2026-11-09T00:00:00+00:00")

    assert moved == 9
    assert sorted(p.name for p in arc.iterdir()) == ["2026-W44.db.gz", "2026-W45.db.gz"]
    con = sqlite3.connect(live)
    assert {r[0] for r in con.execute("SELECT DISTINCT ts_utc FROM snapshots")} == set(new)
    con.close()
    # every row is still readable, archives plus live, nothing duplicated
    h = history(live=live, archive=arc)
    assert len(h) == 15 and set(h.ts_utc) == set(old + new)
    assert gzip.open(arc / "2026-W45.db.gz").read(16).startswith(b"SQLite format 3")


def test_rotate_is_idempotent(tmp_path):
    live, arc = tmp_path / "live.db", tmp_path / "arc"
    _live(live, ["2026-11-02T14:07:00+00:00"])
    rotate(live, arc, cutoff="2026-11-09T00:00:00+00:00")
    # the same snapshot shows up in live again (a run that died between the
    # archive write and the live delete): re-rotating must not duplicate it
    _live(live, ["2026-11-02T14:07:00+00:00"])
    rotate(live, arc, cutoff="2026-11-09T00:00:00+00:00")
    assert len(history(live=live, archive=arc)) == 3
