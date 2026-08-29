"""Does game 1 of a doubleheader tell you anything about game 2?

Specifically the claim worth testing: a club shut out in the opener bounces
back in the nightcap. This is the shape of read the narrative layer exists to
capture, so it deserves a real answer from the data rather than a prior.

  uv run python ops/study_doubleheader.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mlb.compile import DATA, load_games  # noqa: E402

df = load_games(first_season=2008)
df = df.sort_values(["gameday", "game_number", "game_pk"])

pairs = []
for (day, h, a), grp in df.groupby(["gameday", "home_team", "away_team"]):
    if len(grp) < 2:
        continue
    grp = grp.sort_values("game_number")
    g1, g2 = grp.iloc[0], grp.iloc[1]
    pairs.append({
        "date": day, "home": h, "away": a,
        "g1_home": g1.home_score, "g1_away": g1.away_score,
        "g2_home": g2.home_score, "g2_away": g2.away_score,
        "g1_total": g1.home_score + g1.away_score,
        "g2_total": g2.home_score + g2.away_score,
    })
p = pd.DataFrame(pairs)
print(f"{len(p)} doubleheader pairs, 2008-2026\n")

base_home = (p["g2_home"] > p["g2_away"]).mean()
print(f"Baseline: home team wins game 2 {100*base_home:.1f}% of the time")
print(f"Baseline: mean game-2 total {p['g2_total'].mean():.2f} runs\n")


def bucket(mask, label):
    n = int(mask.sum())
    if n < 25:
        print(f"{label:<44} n={n:<4} (too thin to read)")
        return
    hw = (p.loc[mask, "g2_home"] > p.loc[mask, "g2_away"]).mean()
    tot = p.loc[mask, "g2_total"].mean()
    print(f"{label:<44} n={n:<4} home wins G2 {100*hw:5.1f}%  "
          f"({100*(hw-base_home):+5.1f})   G2 total {tot:5.2f} "
          f"({tot-p['g2_total'].mean():+5.2f})")


print("--- Was a club shut out in game 1? ---")
bucket(p["g1_home"] == 0, "home shut out in G1 -> home in G2")
bucket(p["g1_away"] == 0, "away shut out in G1 -> home in G2")
print("\n--- Did a club get blown out in game 1? ---")
bucket((p["g1_away"] - p["g1_home"]) >= 5, "home lost G1 by 5+")
bucket((p["g1_home"] - p["g1_away"]) >= 5, "away lost G1 by 5+")
print("\n--- Did a club just win game 1? ---")
bucket(p["g1_home"] > p["g1_away"], "home won G1")
bucket(p["g1_away"] > p["g1_home"], "away won G1")
print("\n--- Was game 1 low or high scoring? ---")
bucket(p["g1_total"] <= 5, "G1 total <= 5")
bucket(p["g1_total"] >= 12, "G1 total >= 12")

r = np.corrcoef(p["g1_total"], p["g2_total"])[0, 1]
print(f"\nCorrelation between game-1 and game-2 totals: r = {r:+.4f}")
print("(r near zero means the opener carries no information about the nightcap)")
p.to_csv(DATA / "doubleheader_pairs.csv", index=False)
