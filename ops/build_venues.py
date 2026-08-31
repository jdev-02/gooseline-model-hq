"""Cache venue coordinates so travel distance between consecutive games can be
computed. One StatsAPI call; written to data/mlb/venues.csv."""
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mlb.compile import DATA  # noqa: E402

r = requests.get("https://statsapi.mlb.com/api/v1/venues",
                 params={"hydrate": "location", "sportIds": 1}, timeout=60)
r.raise_for_status()
rows = []
for v in r.json().get("venues", []):
    loc = v.get("location", {})
    coord = loc.get("defaultCoordinates", {})
    if coord.get("latitude") is None:
        continue
    rows.append({"venue_id": v["id"], "venue_name": v.get("name"),
                 "lat": coord["latitude"], "lon": coord["longitude"],
                 "tz": (loc.get("azimuthAngle") or ""), "city": loc.get("city")})
df = pd.DataFrame(rows).drop_duplicates("venue_id")
df.to_csv(DATA / "venues.csv", index=False)
print(f"wrote {len(df)} venues to {DATA / 'venues.csv'}")
