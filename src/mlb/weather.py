"""Game-time temperature for games that have not been played yet.

StatsAPI only posts a game's weather around first pitch, which is hours after
the daily run, so the totals model would otherwise price every upcoming game
at 70F. Open-Meteo (free, no key) gives an hourly forecast for any lat/lon;
`data/mlb/venues.csv` has the coordinates. One request covers every venue on
the horizon.

Wind is deliberately not taken from the forecast: the model's wind term is
"out to" or "in from" the field, which needs each park's orientation, and the
walk-forward found wind worth a tenth of what temperature is. Temperature is
the term that carries the weather block.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
VENUES = Path(__file__).resolve().parents[2] / "data" / "mlb" / "venues.csv"
GAME_HOURS = 3   # average the forecast over the hours a game is actually played


def load_venues(path=VENUES):
    v = pd.read_csv(path)
    return v.dropna(subset=["lat", "lon"]).set_index("venue_id")


def forecast_temps(games, venues=None, session=None, timeout=20):
    """{game_pk: temp_f} for the rows of `games` (needs venue_id and
    game_datetime_utc). Rows whose venue has no coordinates, or that fall
    outside the forecast window, are left out; the caller keeps its fallback.
    Never raises on a network failure: an empty dict is the honest answer."""
    venues = load_venues() if venues is None else venues
    g = games.dropna(subset=["venue_id", "game_datetime_utc"]).copy()
    g["venue_id"] = g["venue_id"].astype(int)
    g = g[g["venue_id"].isin(venues.index)]
    if g.empty:
        return {}
    g["t"] = pd.to_datetime(g["game_datetime_utc"], utc=True).dt.tz_localize(None)
    start, end = g["t"].min().date(), (g["t"].max() + pd.Timedelta(hours=GAME_HOURS)).date()
    vids = sorted(g["venue_id"].unique())
    params = {
        "latitude": ",".join(f"{venues.at[v, 'lat']:.4f}" for v in vids),
        "longitude": ",".join(f"{venues.at[v, 'lon']:.4f}" for v in vids),
        "hourly": "temperature_2m", "temperature_unit": "fahrenheit",
        "timezone": "UTC", "start_date": str(start), "end_date": str(end),
    }
    try:
        r = (session or requests).get(OPEN_METEO, params=params, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
    except Exception:
        return {}
    if isinstance(payload, dict):
        payload = [payload]
    series = {}
    for v, block in zip(vids, payload):
        h = block.get("hourly", {})
        if not h.get("time"):
            continue
        idx = pd.to_datetime(h["time"])
        series[v] = pd.Series(h["temperature_2m"], index=idx, dtype=float)
    out = {}
    for r in g.itertuples(index=False):
        s = series.get(r.venue_id)
        if s is None:
            continue
        t0 = r.t.floor("h")
        window = s.loc[t0:t0 + pd.Timedelta(hours=GAME_HOURS - 1)].dropna()
        if len(window):
            out[int(r.game_pk)] = float(np.round(window.mean()))
    return out
