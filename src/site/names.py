"""Team codes to the names a person actually says. Thin shim over the one
registry in src/core/teams.py; kept so the site modules read as before."""
from src.core.teams import nickname


def mlb(code):
    return nickname("mlb", code)


def nfl(code):
    return nickname("nfl", code)
