"""One Kalshi snapshot for NFL + MLB series (Task Scheduler friendly).

  uv run python ops/kalshi_snapshot.py            # one snapshot, exit
  uv run python ops/kalshi_snapshot.py --loop 15  # every 15 minutes
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.kalshi import snapshot, DEFAULT_SERIES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="kalshi_prices.db")
ap.add_argument("--series", nargs="+", default=DEFAULT_SERIES)
ap.add_argument("--loop", type=float, default=None, help="minutes between snapshots")
args = ap.parse_args()
while True:
    n = snapshot(args.db, args.series)
    if args.loop is None:
        sys.exit(0 if n > 0 else 1)
    time.sleep(args.loop * 60)
