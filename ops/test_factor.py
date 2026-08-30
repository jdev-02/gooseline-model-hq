"""Measure a narrative factor against history and update its registry status.

A read is a hypothesis. This turns it into a number: find every historical
game matching the factor's condition, compare what actually happened against
what the model expected, and report whether the read carries information the
model does not already have.

  uv run python ops/test_factor.py --factor starter_returning
  uv run python ops/test_factor.py --factor spot_starter

The comparison is against the model's own residual, not against a raw win
rate. A factor only earns `supported` if it predicts the part of the outcome
the model gets wrong.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mlb.compile import DATA  # noqa: E402

HIST = DATA / "mlb_history.csv"


def load_hist():
    if not HIST.exists():
        print("need data/mlb/mlb_history.csv first: run ops/export_history.py")
        sys.exit(1)
    h = pd.read_csv(HIST)
    g = pd.read_csv(DATA / "games.csv", low_memory=False)
    keep = ["game_id", "gameday", "home_team", "away_team", "home_sp_id",
            "away_sp_id", "game_number", "venue_id", "season"]
    h["game_id"] = h["game_id"].astype(str)
    g["game_id"] = g["game_id"].astype(str)
    return h.merge(g[keep], on="game_id", how="left", suffixes=("", "_g"))


def report(h, mask, label, side_col):
    """side_col: +1 where the factor favors home, -1 where it favors away."""
    sub = h[mask].copy()
    n = len(sub)
    if n < 50:
        print(f"{label}: n={n} — too thin to call. Leave it untested.")
        return None
    # Residual in the direction the factor claims to help.
    resid = (sub["y"] - sub["mu"]) * sub[side_col]
    mean, sd = resid.mean(), resid.std()
    se = sd / np.sqrt(n)
    t = mean / se if se > 0 else 0.0
    print(f"\n{label}")
    print(f"  n = {n}")
    print(f"  mean residual in the claimed direction: {mean:+.3f} runs "
          f"(se {se:.3f}, t = {t:+.2f})")
    print(f"  the model already expected: {sub['mu'].mean() * sub[side_col].mean():+.3f}")
    if abs(t) < 2:
        print("  VERDICT: no measurable effect beyond the model. -> untested/rejected")
    elif mean > 0:
        print(f"  VERDICT: supported, worth about {mean:.2f} runs. "
              f"Consider promoting it to a model feature.")
    else:
        print("  VERDICT: effect runs OPPOSITE to the claim. -> rejected")
    return {"n": n, "mean": float(mean), "t": float(t)}


def factor_starter_returning(h, gap_days=15):
    """Starts preceded by a long gap for that pitcher."""
    g = pd.read_csv(DATA / "games.csv", low_memory=False)
    g["gameday"] = pd.to_datetime(g["gameday"])
    apps = []
    for side in ("home", "away"):
        d = g[[f"{side}_sp_id", "gameday", "game_id"]].dropna()
        d.columns = ["pid", "gameday", "game_id"]
        d["side"] = side
        apps.append(d)
    a = pd.concat(apps).sort_values(["pid", "gameday"])
    a["gap"] = a.groupby("pid")["gameday"].diff().dt.days
    ret = a[a["gap"] >= gap_days]
    h = h.copy()
    h["game_id"] = h["game_id"].astype(str)
    ret["game_id"] = ret["game_id"].astype(str)
    home_ret = set(ret[ret.side == "home"]["game_id"])
    away_ret = set(ret[ret.side == "away"]["game_id"])
    h["side"] = np.where(h["game_id"].isin(home_ret), 1.0,
                         np.where(h["game_id"].isin(away_ret), -1.0, 0.0))
    return h["side"] != 0, "side"


def factor_spot_starter(h):
    """Games where a probable was never listed -> model used league mean."""
    g = pd.read_csv(DATA / "games.csv", low_memory=False)
    g["game_id"] = g["game_id"].astype(str)
    h = h.copy()
    h["game_id"] = h["game_id"].astype(str)
    m = g.set_index("game_id")
    h["home_missing"] = h["game_id"].map(m["home_sp_id"].isna())
    h["away_missing"] = h["game_id"].map(m["away_sp_id"].isna())
    # claim: the club WITHOUT the missing starter is favored
    h["side"] = np.where(h["home_missing"] & ~h["away_missing"], -1.0,
                         np.where(h["away_missing"] & ~h["home_missing"], 1.0, 0.0))
    return h["side"] != 0, "side"


FACTORS = {
    "starter_returning": factor_starter_returning,
    "spot_starter": factor_spot_starter,
}

ap = argparse.ArgumentParser()
ap.add_argument("--factor", required=True, choices=sorted(FACTORS))
args = ap.parse_args()

h = load_hist()
print(f"walk-forward history: {len(h)} games, "
      f"{h['season'].min()}-{h['season'].max()}")
mask, side_col = FACTORS[args.factor](h)
h = h.loc[h.index]
report(h, mask, args.factor, side_col)
print("\nA factor only earns `supported` when it predicts the part of the "
      "outcome the model gets wrong. Update data/mlb/factors.yaml by hand "
      "with the verdict and the evidence.")
