"""Every team the data names must resolve to one canonical code with a
nickname, from every alias any source uses; and a slate must sort the same
way every time. Uses the committed games.csv files only as a list of codes."""
import random
from pathlib import Path

import pandas as pd
import pytest

from src.core.teams import canon, nickname, aliases, codes, slate_sort_key

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("league,path,cols", [
    ("mlb", "data/mlb/games.csv", ("home_team", "away_team")),
    ("nfl", "data/nfl/games.csv", ("home_team", "away_team")),
])
def test_every_code_in_the_data_resolves(league, path, cols):
    p = ROOT / path
    if not p.exists():
        pytest.skip(f"{path} not present")
    g = pd.read_csv(p, usecols=list(cols) + ["season"], low_memory=False)
    g = g[g["season"] >= 2024]
    seen = set(g[cols[0]].dropna()) | set(g[cols[1]].dropna())
    unresolved = sorted(c for c in seen if canon(league, c) not in codes(league))
    assert not unresolved, f"{league}: no registry entry for {unresolved}"
    for c in seen:
        assert nickname(league, c) != canon(league, c), f"{league} {c} has no nickname"


def test_aliases_round_trip():
    assert canon("mlb", "ARI") == "AZ" and canon("mlb", "CHW") == "CWS"
    assert canon("mlb", "WAS") == "WSH" and canon("mlb", "OAK") == "ATH"
    assert canon("nfl", "JAC") == "JAX" and canon("nfl", "LAR") == "LA"
    assert canon("nfl", "WSH") == "WAS" and canon("nfl", "LVR") == "LV"
    for lg in ("mlb", "nfl"):
        for c in codes(lg):
            for a in aliases(lg, c):
                assert canon(lg, a) == c
    # Same code, different leagues, different teams: never cross the streams.
    assert nickname("mlb", "SF") == "Giants" and nickname("nfl", "SF") == "49ers"
    assert nickname("mlb", "WAS") == "Nationals" and nickname("nfl", "WAS") == "Commanders"


def test_unknown_code_passes_through_without_crashing():
    assert canon("mlb", "XYZ") == "XYZ"
    assert nickname("mlb", "XYZ") == "XYZ"
    assert canon("mlb", None) is None


def test_slate_order_is_deterministic():
    rows = [
        {"kick_iso": "2026-09-15T23:05:00+00:00", "away": "NYM", "home": "NYY"},
        {"kick_iso": "2026-09-15T18:20:00+00:00", "away": "PIT", "home": "CHC"},
        {"kick_iso": "2026-09-15T23:05:00+00:00", "away": "BOS", "home": "TEX"},
        {"kick_iso": "", "away": "SEA", "home": "LAA"},
        {"kick_iso": "2026-09-16T02:15:00+00:00", "away": "SD", "home": "SF"},
    ]
    expected = ["PIT@CHC", "BOS@TEX", "NYM@NYY", "SD@SF", "SEA@LAA"]
    for seed in range(5):
        rng = random.Random(seed)
        shuffled = rows[:]
        rng.shuffle(shuffled)
        got = [f'{r["away"]}@{r["home"]}' for r in sorted(shuffled, key=slate_sort_key)]
        assert got == expected
