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


def factor_starter_returning(h, gap_days=15, gap_max=400):
    """Starts preceded by a gap of [gap_days, gap_max] for that pitcher."""
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
    ret = a[(a["gap"] >= gap_days) & (a["gap"] <= gap_max)]
    h = h.copy()
    h["game_id"] = h["game_id"].astype(str)
    ret["game_id"] = ret["game_id"].astype(str)
    home_ret = set(ret[ret.side == "home"]["game_id"])
    away_ret = set(ret[ret.side == "away"]["game_id"])
    h["side"] = np.where(h["game_id"].isin(home_ret), 1.0,
                         np.where(h["game_id"].isin(away_ret), -1.0, 0.0))
    return h, h["side"] != 0, "side"


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
    return h, h["side"] != 0, "side"


def factor_playoff_push(h, month_from=9, max_gb=6.0):
    """Late-season games where BOTH clubs are in contention.

    The claim is that a game with seeding or a head-to-head tiebreaker on the
    line is played harder than the season-long profile implies. If that is
    true of both clubs at once it should show up as extra variance or as a
    home-field effect, not as a directional edge — so this tests whether the
    model's margin is systematically off in these spots at all.
    """
    g = pd.read_csv(DATA / "games.csv", low_memory=False)
    g["gameday"] = pd.to_datetime(g["gameday"])
    g = g[g["played"]].sort_values("gameday")

    # Running wins/losses per club, strictly before each game.
    rec = {}
    rows = []
    for r in g.itertuples(index=False):
        hw, hl = rec.get((r.season, r.home_team), (0, 0))
        aw, al = rec.get((r.season, r.away_team), (0, 0))
        rows.append((str(r.game_id), hw, hl, aw, al))
        if r.result > 0:
            rec[(r.season, r.home_team)] = (hw + 1, hl)
            rec[(r.season, r.away_team)] = (aw, al + 1)
        else:
            rec[(r.season, r.home_team)] = (hw, hl + 1)
            rec[(r.season, r.away_team)] = (aw + 1, al)
    pre = pd.DataFrame(rows, columns=["game_id", "hw", "hl", "aw", "al"])

    h = h.copy()
    h["game_id"] = h["game_id"].astype(str)
    h = h.merge(pre, on="game_id", how="left")
    h["gameday"] = pd.to_datetime(h["gameday"])
    h["hpct"] = h["hw"] / (h["hw"] + h["hl"]).clip(lower=1)
    h["apct"] = h["aw"] / (h["aw"] + h["al"]).clip(lower=1)
    # Contention proxy: both clubs above .500 late in the year.
    late = h["gameday"].dt.month >= month_from
    both_live = (h["hpct"] >= 0.500) & (h["apct"] >= 0.500)
    h["side"] = 1.0  # test for a directional bias toward the home club
    return h, late & both_live, "side"


def factor_day_night_split(h, min_prior=5, gap_thresh=1.0):
    """Does a starter's own day/night history predict his next such start?

    The trap: ERA splits over ten starts are mostly balls-in-play luck. This
    builds each pitcher's split from FIP peripherals only (HR, BB, K), uses
    strictly prior starts, and asks whether a large prior split predicts the
    model's residual in the next same-condition start.
    """
    pit = pd.read_csv(DATA / "pitcher_game_stats.csv.gz")
    g = pd.read_csv(DATA / "games.csv", low_memory=False)
    g["game_id"] = g["game_id"].astype(str)
    pit["game_id"] = pit["game_id"].astype(str)
    dn = g.set_index("game_id")["day_night"]
    starts = pit[pit["is_starter"] == 1].copy()
    starts["day_night"] = starts["game_id"].map(dn)
    starts["gameday"] = pd.to_datetime(starts["gameday"])
    starts = starts.sort_values("gameday")
    starts["fipnum"] = (13 * starts["hr"].fillna(0)
                        + 3 * (starts["bb"].fillna(0) + starts["hbp"].fillna(0))
                        - 2 * starts["so"].fillna(0))

    # Expanding, strictly-prior FIP numerator per inning, split by condition.
    acc = {}
    rows = []
    for r in starts.itertuples(index=False):
        key_d = (r.pitcher_id, "day")
        key_n = (r.pitcher_id, "night")
        dn_i, di = acc.get(key_d, (0.0, 0.0))
        nn_i, ni = acc.get(key_n, (0.0, 0.0))
        prior_day = dn_i / di if di > 0 else np.nan
        prior_night = nn_i / ni if ni > 0 else np.nan
        n_day = acc.get((r.pitcher_id, "nday"), 0)
        n_night = acc.get((r.pitcher_id, "nnight"), 0)
        rows.append((r.game_id, r.pitcher_id, r.is_home, r.day_night,
                     prior_day, prior_night, n_day, n_night))
        ip = float(r.ip or 0)
        if ip > 0:
            if r.day_night == "day":
                acc[key_d] = (dn_i + r.fipnum, di + ip)
                acc[(r.pitcher_id, "nday")] = n_day + 1
            else:
                acc[key_n] = (nn_i + r.fipnum, ni + ip)
                acc[(r.pitcher_id, "nnight")] = n_night + 1
    s = pd.DataFrame(rows, columns=["game_id", "pid", "is_home", "day_night",
                                    "prior_day", "prior_night", "n_day", "n_night"])
    # A pitcher is "split-favoured" tonight if his prior FIP rate in tonight's
    # condition is at least gap_thresh better than in the other condition.
    s["this"] = np.where(s["day_night"] == "day", s["prior_day"], s["prior_night"])
    s["other"] = np.where(s["day_night"] == "day", s["prior_night"], s["prior_day"])
    s["favoured"] = ((s["other"] - s["this"]) >= gap_thresh) & \
                    (s[["n_day", "n_night"]].min(axis=1) >= min_prior)
    s["hurt"] = ((s["this"] - s["other"]) >= gap_thresh) & \
                (s[["n_day", "n_night"]].min(axis=1) >= min_prior)

    h = h.copy()
    h["game_id"] = h["game_id"].astype(str)
    fav_home = set(s[(s.favoured) & (s.is_home == 1)]["game_id"])
    fav_away = set(s[(s.favoured) & (s.is_home == 0)]["game_id"])
    # claim: the club whose starter is in his favoured condition is helped
    h["side"] = np.where(h["game_id"].isin(fav_home), 1.0,
                         np.where(h["game_id"].isin(fav_away), -1.0, 0.0))
    return h, h["side"] != 0, "side"


def _haversine(lat1, lon1, lat2, lon2):
    R = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def factor_travel(h, miles=1500, max_rest=1):
    """A club that just crossed the country and plays the next day.

    The model carries rest in days but knows nothing about distance or time
    zones, so a coast-to-coast trip on one day's rest looks identical to a bus
    ride across town.
    """
    ven = pd.read_csv(DATA / "venues.csv")
    vmap = ven.set_index("venue_id")[["lat", "lon"]]
    g = pd.read_csv(DATA / "games.csv", low_memory=False)
    g["gameday"] = pd.to_datetime(g["gameday"])
    g = g[g["played"]].sort_values("gameday")

    last = {}          # team -> (gameday, venue_id)
    trav = {}          # game_id -> (home_miles, away_miles, home_rest, away_rest)
    for r in g.itertuples(index=False):
        rec = {}
        for side in ("home", "away"):
            t = getattr(r, f"{side}_team")
            prev = last.get(t)
            if prev is None or prev[1] not in vmap.index or r.venue_id not in vmap.index:
                rec[side] = (0.0, 99)
            else:
                a, b = vmap.loc[prev[1]], vmap.loc[r.venue_id]
                d = _haversine(a.lat, a.lon, b.lat, b.lon)
                rec[side] = (float(d), int((r.gameday - prev[0]).days))
        trav[str(r.game_id)] = rec
        for side in ("home", "away"):
            last[getattr(r, f"{side}_team")] = (r.gameday, r.venue_id)

    h = h.copy()
    h["game_id"] = h["game_id"].astype(str)
    hm = h["game_id"].map(lambda k: trav.get(k, {}).get("home", (0, 99))[0])
    hr = h["game_id"].map(lambda k: trav.get(k, {}).get("home", (0, 99))[1])
    am = h["game_id"].map(lambda k: trav.get(k, {}).get("away", (0, 99))[0])
    ar = h["game_id"].map(lambda k: trav.get(k, {}).get("away", (0, 99))[1])
    home_tired = (hm >= miles) & (hr <= max_rest)
    away_tired = (am >= miles) & (ar <= max_rest)
    # claim: the club that just travelled far on short rest is hurt
    h["side"] = np.where(away_tired & ~home_tired, 1.0,
                         np.where(home_tired & ~away_tired, -1.0, 0.0))
    return h, h["side"] != 0, "side"


FACTORS = {
    "starter_returning": factor_starter_returning,
    "spot_starter": factor_spot_starter,
    "playoff_push": factor_playoff_push,
    "day_night_split": factor_day_night_split,
    "travel": factor_travel,
}

ap = argparse.ArgumentParser()
ap.add_argument("--factor", required=True, choices=sorted(FACTORS))
ap.add_argument("--sweep", action="store_true",
                help="for starter_returning, sweep the gap threshold")
args = ap.parse_args()

h0 = load_hist()
print(f"walk-forward history: {len(h0)} games, "
      f"{h0['season'].min()}-{h0['season'].max()}")

if args.sweep and args.factor == "starter_returning":
    # A layoff is not one thing. A skipped turn is not a rehab stint, and
    # lumping them together can hide an effect that lives in only one band.
    for lo, hi in ((6, 9), (10, 14), (15, 20), (21, 40), (41, 400)):
        h = h0.copy()
        h, mask, side = factor_starter_returning(h, gap_days=lo, gap_max=hi)
        report(h, mask, f"starter_returning gap {lo}-{hi}d", side)
else:
    h, mask, side_col = FACTORS[args.factor](h0)
    report(h, mask, args.factor, side_col)
print("\nA factor only earns `supported` when it predicts the part of the "
      "outcome the model gets wrong. Update data/mlb/factors.yaml by hand "
      "with the verdict and the evidence.")
