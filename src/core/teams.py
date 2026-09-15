"""One registry of teams per league: canonical code, the name a person says,
and every alias any source uses for it (StatsAPI, nflverse, Kalshi, the
common sportsbook codes). Everything that names a team goes through here.

Before this, identity was spread over four places -- a nickname map, a
Kalshi alias dict in the NFL rundown, an assumption that StatsAPI and Kalshi
share MLB codes, and regex ticker matching -- with no test that every team
in the data resolved. `canon()` is the single mapping; tests/test_teams.py
checks every code that appears in games.csv for both leagues resolves and
has a nickname.
"""
from __future__ import annotations

MLB = {
    "AZ":  ("Diamondbacks", ("ARI",)),
    "ATL": ("Braves", ()),
    "BAL": ("Orioles", ()),
    "BOS": ("Red Sox", ()),
    "CHC": ("Cubs", ()),
    "CWS": ("White Sox", ("CHW",)),
    "CIN": ("Reds", ()),
    "CLE": ("Guardians", ()),
    "COL": ("Rockies", ()),
    "DET": ("Tigers", ()),
    "HOU": ("Astros", ()),
    "KC":  ("Royals", ("KCR",)),
    "LAA": ("Angels", ("ANA",)),
    "LAD": ("Dodgers", ()),
    "MIA": ("Marlins", ()),
    "MIL": ("Brewers", ()),
    "MIN": ("Twins", ()),
    "NYM": ("Mets", ()),
    "NYY": ("Yankees", ()),
    "ATH": ("Athletics", ("OAK",)),
    "PHI": ("Phillies", ()),
    "PIT": ("Pirates", ()),
    "SD":  ("Padres", ("SDP",)),
    "SF":  ("Giants", ("SFG",)),
    "SEA": ("Mariners", ()),
    "STL": ("Cardinals", ()),
    "TB":  ("Rays", ("TBR",)),
    "TEX": ("Rangers", ()),
    "TOR": ("Blue Jays", ()),
    "WSH": ("Nationals", ("WAS", "WSN")),
}

NFL = {
    "ARI": ("Cardinals", ()),
    "ATL": ("Falcons", ()),
    "BAL": ("Ravens", ()),
    "BUF": ("Bills", ()),
    "CAR": ("Panthers", ()),
    "CHI": ("Bears", ()),
    "CIN": ("Bengals", ()),
    "CLE": ("Browns", ()),
    "DAL": ("Cowboys", ()),
    "DEN": ("Broncos", ()),
    "DET": ("Lions", ()),
    "GB":  ("Packers", ("GNB",)),
    "HOU": ("Texans", ()),
    "IND": ("Colts", ()),
    "JAX": ("Jaguars", ("JAC",)),
    "KC":  ("Chiefs", ("KAN",)),
    "LV":  ("Raiders", ("LVR", "OAK")),
    "LAC": ("Chargers", ("SD",)),
    "LA":  ("Rams", ("LAR", "STL")),
    "MIA": ("Dolphins", ()),
    "MIN": ("Vikings", ()),
    "NE":  ("Patriots", ("NWE",)),
    "NO":  ("Saints", ("NOR",)),
    "NYG": ("Giants", ()),
    "NYJ": ("Jets", ()),
    "PHI": ("Eagles", ()),
    "PIT": ("Steelers", ()),
    "SF":  ("49ers", ("SFO",)),
    "SEA": ("Seahawks", ()),
    "TB":  ("Buccaneers", ("TAM",)),
    "TEN": ("Titans", ()),
    "WAS": ("Commanders", ("WSH",)),
}

_LEAGUES = {"mlb": MLB, "nfl": NFL}
_ALIAS = {lg: {a: code for code, (_, aliases) in reg.items() for a in (code,) + aliases}
          for lg, reg in _LEAGUES.items()}


def canon(league, code):
    """The canonical code for any alias. Unknown codes pass through unchanged
    so a new expansion team degrades to its raw code, never to a crash."""
    if code is None:
        return code
    return _ALIAS[league].get(str(code).upper(), str(code))


def nickname(league, code):
    c = canon(league, code)
    return _LEAGUES[league].get(c, (str(code), ()))[0]


def aliases(league, code):
    """Every code a source might use for this team, canonical first."""
    c = canon(league, code)
    return (c,) + _LEAGUES[league].get(c, ("", ()))[1]


def codes(league):
    return tuple(_LEAGUES[league].keys())


def slate_sort_key(row):
    """Deterministic slate order: first pitch / kickoff instant, then away
    code, then home code. Rows without a kickoff sort last, still
    deterministically."""
    k = row.get("kick_iso") if hasattr(row, "get") else getattr(row, "kick_iso", "")
    k = str(k or "")
    return (k == "", k, str(row.get("away") if hasattr(row, "get") else getattr(row, "away", "")),
            str(row.get("home") if hasattr(row, "get") else getattr(row, "home", "")))
