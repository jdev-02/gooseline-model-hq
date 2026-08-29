"""Compile the StatsAPI cache into data/mlb/*.csv.

  uv run python ops/compile_mlb.py            # games.csv from schedules, plus
                                              # team/pitcher stats from any cached boxscores
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mlb.compile import compile_games, compile_boxscores, DATA  # noqa: E402
from src.mlb.ingest import fetch_teams  # noqa: E402

teams_path = DATA / "teams.csv"
teams = pd.read_csv(teams_path) if teams_path.exists() else pd.DataFrame(fetch_teams(2026))
teams.to_csv(teams_path, index=False)

games = compile_games(range(2008, 2027), teams=teams)
games.to_csv(DATA / "games.csv", index=False)
print(f"games.csv: {len(games)} rows, {int(games.played.sum())} played")

if "--games-only" in sys.argv:
    sys.exit(0)
team_stats, pitcher_stats = compile_boxscores(games)
if len(team_stats):
    team_stats.to_csv(DATA / "team_game_stats.csv", index=False)
    pitcher_stats.to_csv(DATA / "pitcher_game_stats.csv.gz", index=False, compression="gzip")
    print(f"team_game_stats.csv: {len(team_stats)} rows over "
          f"{team_stats.season.min()}-{team_stats.season.max()}")
    print(f"pitcher_game_stats.csv.gz: {len(pitcher_stats)} rows")
else:
    print("no boxscores cached; Tier A only")
