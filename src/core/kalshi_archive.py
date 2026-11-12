"""Keeps data/kalshi_prices.db under GitHub's 100 MB per-file limit.

kalshi-snapshot appends 700-1,000 rows every 30 minutes and commits the whole
sqlite file back to main. The file crossed 100 MB on 2026-10-19, and GitHub
rejected every push after that (GH001, "exceeds GitHub's file size limit"),
so no snapshot landed for 24 days while the runs kept failing.

Every reader in the daily pipeline asks only for the latest row per ticker
(both rundowns' fallback prices) or for MAX(ts_utc) (both health checks), so
the live file needs nothing older than the current week. Finished ISO weeks
move to data/kalshi_archive/<GGGG>-W<VV>.db.gz. Each archive compresses about
7x, is written when its week ends, and is not rewritten after that. That keeps
the live file, the per-commit churn and the checkout small.

  uv run python -m src.core.kalshi_archive rotate        # after each snapshot (CI)
  uv run python -m src.core.kalshi_archive combine OUT   # one full-history db file

Research code that needs the whole history calls history().
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import sqlite3
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "data" / "kalshi_prices.db"
ARCHIVE = ROOT / "data" / "kalshi_archive"
INDEX = "CREATE INDEX IF NOT EXISTS idx_snap_ticker_ts ON snapshots (ticker, ts_utc)"


def week_start(now: datetime | None = None) -> str:
    """Monday 00:00 UTC of the current ISO week, in the ts_utc string format."""
    now = now or datetime.now(timezone.utc)
    monday = (now - timedelta(days=now.weekday())).date()
    return f"{monday.isoformat()}T00:00:00+00:00"


def _week_key(day: str) -> str:
    y, w, _ = date.fromisoformat(day).isocalendar()
    return f"{y}-W{w:02d}"


def _gunzip(src: Path, dst: Path):
    with gzip.open(src, "rb") as g, open(dst, "wb") as f:
        shutil.copyfileobj(g, f)


def _gzip(src: Path, dst: Path):
    """No filename and no mtime in the gzip header, so the archive's bytes
    depend only on the database inside it."""
    tmp = dst.with_name(dst.name + ".tmp")
    with open(src, "rb") as f, open(tmp, "wb") as raw, \
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as g:
        shutil.copyfileobj(f, g)
    tmp.replace(dst)


def rotate(live: Path = LIVE, archive: Path = ARCHIVE, cutoff: str | None = None) -> int:
    """Move every row older than `cutoff` (default: this week's Monday 00:00
    UTC) into its week's archive, then shrink the live file. Idempotent: rows
    already in an archive under the same snapshot timestamp are replaced, not
    duplicated, so a run that dies between the two writes is safe to repeat."""
    cutoff = cutoff or week_start()
    if not Path(live).exists():
        print(f"{live}: missing, nothing to rotate")
        return 0
    con = sqlite3.connect(live)
    row = con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='snapshots'").fetchone()
    if not row:
        con.close()
        return 0
    create_sql = row[0]
    cols = [r[1] for r in con.execute("PRAGMA table_info(snapshots)")]
    days = [d for (d,) in con.execute(
        "SELECT DISTINCT substr(ts_utc, 1, 10) FROM snapshots WHERE ts_utc < ?", (cutoff,))]
    weeks: dict[str, list[str]] = {}
    for d in days:
        weeks.setdefault(_week_key(d), []).append(d)

    moved = 0
    for wk, wdays in sorted(weeks.items()):
        archive.mkdir(parents=True, exist_ok=True)
        gz = archive / f"{wk}.db.gz"
        where = f"ts_utc < ? AND substr(ts_utc, 1, 10) IN ({','.join('?' * len(wdays))})"
        args = (cutoff, *wdays)
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "week.db"
            if gz.exists():
                _gunzip(gz, tmp)
            a = sqlite3.connect(tmp)
            if not a.execute("SELECT 1 FROM sqlite_master WHERE name='snapshots'").fetchone():
                a.execute(create_sql)
            a.execute(INDEX)
            acols = {r[1] for r in a.execute("PRAGMA table_info(snapshots)")}
            a.commit()
            a.close()

            common = ", ".join(c for c in cols if c in acols)
            con.execute("ATTACH DATABASE ? AS arc", (str(tmp),))
            con.execute(f"DELETE FROM arc.snapshots WHERE ts_utc IN "
                        f"(SELECT DISTINCT ts_utc FROM main.snapshots WHERE {where})", args)
            n = con.execute(f"INSERT INTO arc.snapshots ({common}) "
                            f"SELECT {common} FROM main.snapshots WHERE {where}", args).rowcount
            con.commit()
            con.execute("DETACH DATABASE arc")

            a = sqlite3.connect(tmp)
            a.execute("VACUUM")
            a.close()
            _gzip(tmp, gz)  # archive is safely on disk before the live rows go
        con.execute(f"DELETE FROM snapshots WHERE {where}", args)
        con.commit()
        moved += n
        print(f"{wk}: {n} rows -> {gz.name} ({gz.stat().st_size / 1e6:.1f} MB)")
    if moved:
        con.execute("VACUUM")
    left = con.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
    con.close()
    print(f"rotated {moved} rows; live file keeps {left} rows since {cutoff}, "
          f"{Path(live).stat().st_size / 1e6:.1f} MB")
    return moved


def history(series: str | None = None, columns: str = "*",
            live: Path = LIVE, archive: Path = ARCHIVE) -> pd.DataFrame:
    """Every logged snapshot row, archives first and then the live file."""
    q = f"SELECT {columns} FROM snapshots" + (" WHERE series_ticker = ?" if series else "")
    params = (series,) if series else ()
    frames = []
    for gz in sorted(Path(archive).glob("*.db.gz")):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "week.db"
            _gunzip(gz, tmp)
            c = sqlite3.connect(tmp)
            frames.append(pd.read_sql(q, c, params=params))
            c.close()
    if Path(live).exists():
        c = sqlite3.connect(live)
        if c.execute("SELECT 1 FROM sqlite_master WHERE name='snapshots'").fetchone():
            frames.append(pd.read_sql(q, c, params=params))
        c.close()
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def combine(out: Path):
    """Write the whole history to one sqlite file, for opening by hand."""
    from src.core.kalshi import SCHEMA
    out = Path(out)
    out.unlink(missing_ok=True)
    df = history()
    c = sqlite3.connect(out)
    c.executescript(SCHEMA)
    if len(df):
        df.to_sql("snapshots", c, if_exists="append", index=False)
    c.close()
    print(f"{out}: {len(df)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rotate")
    r.add_argument("--db", default=str(LIVE))
    r.add_argument("--archive", default=str(ARCHIVE))
    r.add_argument("--cutoff", default=None, help="ISO timestamp; default this week's Monday 00:00 UTC")
    c = sub.add_parser("combine")
    c.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "rotate":
        rotate(Path(a.db), Path(a.archive), a.cutoff)
    else:
        combine(Path(a.out))
