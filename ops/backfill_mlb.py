"""Backfill the MLB StatsAPI cache.

  uv run python ops/backfill_mlb.py --schedule --seasons 2015-2026
  uv run python ops/backfill_mlb.py --boxscore --seasons 2026,2025 --workers 8
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mlb.ingest import backfill_schedules, backfill_boxscores, fetch_season_schedule  # noqa: E402


def parse_seasons(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", action="store_true")
    ap.add_argument("--boxscore", action="store_true")
    ap.add_argument("--seasons", default="2015-2026")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    seasons = parse_seasons(args.seasons)
    if args.schedule:
        backfill_schedules(seasons, force=args.force)
    if args.boxscore:
        pairs = []
        for s in seasons:
            sched = fetch_season_schedule(s)
            for d in sched["dates"]:
                for g in d["games"]:
                    if g["status"].get("abstractGameState") == "Final":
                        pairs.append((s, g["gamePk"]))
        ok, fail = backfill_boxscores(pairs, workers=args.workers)
        print(f"boxscores done: {ok} ok, {fail} failed")


if __name__ == "__main__":
    main()
