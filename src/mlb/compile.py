"""Compile the raw StatsAPI cache into the flat CSVs the model reads.

games.csv            one row per regular-season game (Tier A: schedule only)
team_game_stats.csv  two rows per game, keyed (game_id, team)   (Tier B)
pitcher_game_stats   one row per pitcher appearance             (Tier B)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.mlb.ingest import RAW, fetch_season_schedule, boxscore_path

DATA = Path("data/mlb")
TRAINABLE = ("Final",)
UPCOMING = ("Scheduled", "Pre-Game", "Warmup", "Delayed Start", "In Progress")


def _ip_to_outs(ip):
    if ip is None or ip == "":
        return 0
    s = str(ip)
    whole, _, frac = s.partition(".")
    return int(whole) * 3 + int(frac or 0)


def _linescore_side(ls, side):
    inns = ls.get("innings", []) if ls else []
    first6 = sum((i.get(side, {}).get("runs") or 0) for i in inns if i.get("num", 0) <= 6)
    late = sum((i.get(side, {}).get("runs") or 0) for i in inns if i.get("num", 0) >= 7)
    tot = ls.get("teams", {}).get(side, {}) if ls else {}
    return {
        "hits": tot.get("hits"), "errors": tot.get("errors"), "lob": tot.get("leftOnBase"),
        "runs_first6": first6, "runs_late": late, "innings": len(inns),
    }


def compile_games(seasons, cache_dir=RAW, teams=None):
    rows = []
    for season in seasons:
        sched = fetch_season_schedule(season, cache_dir)
        for d in sched["dates"]:
            for g in d["games"]:
                st = g["status"]
                h, a = g["teams"]["home"], g["teams"]["away"]
                ls = g.get("linescore", {})
                hs, as_ = _linescore_side(ls, "home"), _linescore_side(ls, "away")
                played = st.get("abstractGameState") == "Final" and \
                    h.get("score") is not None and a.get("score") is not None
                rows.append({
                    "game_id": str(g["gamePk"]), "game_pk": g["gamePk"], "season": season,
                    "gameday": d["date"], "game_datetime_utc": g["gameDate"],
                    "day_night": g.get("dayNight"), "game_number": g.get("gameNumber", 1),
                    "doubleheader": g.get("doubleHeader", "N"),
                    "series_game_number": g.get("seriesGameNumber"),
                    "games_in_series": g.get("gamesInSeries"),
                    "home_team_id": h["team"]["id"], "away_team_id": a["team"]["id"],
                    "home_team": h["team"].get("abbreviation"),
                    "away_team": a["team"].get("abbreviation"),
                    "home_div_id": h["team"].get("division", {}).get("id"),
                    "away_div_id": a["team"].get("division", {}).get("id"),
                    "venue_id": g.get("venue", {}).get("id"),
                    "venue_name": g.get("venue", {}).get("name"),
                    "home_score": h.get("score"), "away_score": a.get("score"),
                    "status": st.get("detailedState"),
                    "abstract_state": st.get("abstractGameState"),
                    "played": bool(played),
                    "scheduled_innings": g.get("scheduledInnings", 9),
                    "final_inning": ls.get("currentInning") if ls else None,
                    "home_sp_id": h.get("probablePitcher", {}).get("id"),
                    "away_sp_id": a.get("probablePitcher", {}).get("id"),
                    "home_sp_name": h.get("probablePitcher", {}).get("fullName"),
                    "away_sp_name": a.get("probablePitcher", {}).get("fullName"),
                    "home_hits": hs["hits"], "away_hits": as_["hits"],
                    "home_errors": hs["errors"], "away_errors": as_["errors"],
                    "home_lob": hs["lob"], "away_lob": as_["lob"],
                    "home_runs_first6": hs["runs_first6"], "away_runs_first6": as_["runs_first6"],
                    "home_runs_late": hs["runs_late"], "away_runs_late": as_["runs_late"],
                    "innings_played": hs["innings"],
                })
    df = pd.DataFrame(rows)
    # Team abbreviations / divisions are only hydrated on some seasons; fill
    # from the teams table when given, else from any row that has them.
    if teams is not None:
        tmap = teams.set_index("team_id")
        for side in ("home", "away"):
            df[f"{side}_team"] = df[f"{side}_team_id"].map(tmap["abbrev"])
            df[f"{side}_div_id"] = df[f"{side}_team_id"].map(tmap["division_id"])
    df = df[df["abstract_state"].isin(["Final", "Preview", "Live"])]
    df = df[~df["status"].isin(["Postponed", "Cancelled", "Suspended"])]
    df = df.dropna(subset=["home_team", "away_team"])
    df["gameday"] = pd.to_datetime(df["gameday"])
    df["result"] = np.where(df["played"], df["home_score"] - df["away_score"], np.nan)
    df["div_game"] = (df["home_div_id"] == df["away_div_id"]).astype(int)
    df = df.sort_values(["gameday", "game_number", "game_pk"]).reset_index(drop=True)
    # A suspended game appears on both its start date and its resumption date;
    # the resumption row carries the final score, so keep the last one.
    df = df.drop_duplicates("game_pk", keep="last").reset_index(drop=True)
    first = df.groupby("season")["gameday"].transform("min")
    df["day_index"] = (df["gameday"] - first).dt.days
    # rest days per team (doubleheader game 2 gets 0)
    last_seen = {}
    hr, ar = np.empty(len(df)), np.empty(len(df))
    for i, r in enumerate(df.itertuples(index=False)):
        for side, arr in (("home", hr), ("away", ar)):
            t = getattr(r, f"{side}_team")
            prev = last_seen.get((r.season, t))
            arr[i] = np.nan if prev is None else (r.gameday - prev).days
            last_seen[(r.season, t)] = r.gameday
    df["home_rest"], df["away_rest"] = hr, ar
    return df


def load_games(path=DATA / "games.csv", first_season=2015, keep_unplayed=False):
    df = pd.read_csv(path, low_memory=False, dtype={"game_id": str})
    df = df[df["season"] >= first_season]
    if not keep_unplayed:
        df = df[df["played"]]
    df = df.copy()
    df["gameday"] = pd.to_datetime(df["gameday"])
    return df.sort_values(["gameday", "game_number", "game_pk"]).reset_index(drop=True)


# ---------------------------------------------------------------- Tier B ---

def _team_lines(box, side, opp, game, is_home):
    t = box["teams"][side]
    bat = t.get("teamStats", {}).get("batting", {})
    pit = t.get("teamStats", {}).get("pitching", {})
    outs = _ip_to_outs(pit.get("inningsPitched"))
    ip = outs / 3.0
    pa = bat.get("plateAppearances") or 0
    br_off = (bat.get("hits", 0) + bat.get("baseOnBalls", 0) + bat.get("hitByPitch", 0))
    br_def = (pit.get("hits", 0) + pit.get("baseOnBalls", 0) + pit.get("hitByPitch", 0))
    inn_bat = max(1.0, (game["innings_played"] or 9))
    late_for = game["home_runs_late"] if is_home else game["away_runs_late"]
    late_against = game["away_runs_late"] if is_home else game["home_runs_late"]
    return {
        "game_id": game["game_id"], "game_pk": game["game_pk"], "season": game["season"],
        "gameday": game["gameday"], "team": game["home_team" if is_home else "away_team"],
        "opponent": opp, "is_home": int(is_home),
        "runs_scored": bat.get("runs"), "runs_allowed": pit.get("runs"),
        "run_diff": (bat.get("runs") or 0) - (pit.get("runs") or 0),
        "hits": bat.get("hits"), "doubles": bat.get("doubles"), "triples": bat.get("triples"),
        "home_runs": bat.get("homeRuns"), "bb": bat.get("baseOnBalls"), "so": bat.get("strikeOuts"),
        "hbp": bat.get("hitByPitch"), "pa": pa, "ab": bat.get("atBats"),
        "tb": bat.get("totalBases"), "lob": bat.get("leftOnBase"), "sb": bat.get("stolenBases"),
        "obp": bat.get("obp"), "slg": bat.get("slg"),
        "p_ip": ip, "p_outs": outs, "p_h": pit.get("hits"), "p_bb": pit.get("baseOnBalls"),
        "p_hbp": pit.get("hitByPitch"), "p_so": pit.get("strikeOuts"), "p_hr": pit.get("homeRuns"),
        "p_er": pit.get("earnedRuns"), "p_r": pit.get("runs"), "p_bf": pit.get("battersFaced"),
        "p_pitches": pit.get("pitchesThrown") or pit.get("numberOfPitches"),
        "p_strikes": pit.get("strikes"),
        "p_strike_pct": (float(pit["strikePercentage"]) if pit.get("strikePercentage") else np.nan),
        "brpi_off": br_off / inn_bat, "brpi_def": br_def / max(ip, 1.0),
        "lob_rate_off": (bat.get("leftOnBase") or 0) / max(1, br_off),
        "runs_late": late_for, "runs_allowed_late": late_against,
    }


def _pitcher_lines(box, side, game, is_home):
    t = box["teams"][side]
    order = t.get("pitchers", [])
    out = []
    for k, pid in enumerate(order):
        p = t["players"].get(f"ID{pid}", {})
        st = p.get("stats", {}).get("pitching", {})
        if not st:
            continue
        outs = _ip_to_outs(st.get("inningsPitched"))
        pitches = st.get("pitchesThrown") or st.get("numberOfPitches")
        strikes = st.get("strikes")
        out.append({
            "game_id": game["game_id"], "game_pk": game["game_pk"], "season": game["season"],
            "gameday": game["gameday"], "team": game["home_team" if is_home else "away_team"],
            "opponent": game["away_team" if is_home else "home_team"], "is_home": int(is_home),
            "pitcher_id": pid, "pitcher_name": p.get("person", {}).get("fullName"),
            "is_starter": int(k == 0), "bullpen_order": k,
            "ip": outs / 3.0, "outs": outs, "bf": st.get("battersFaced"),
            "h": st.get("hits"), "hr": st.get("homeRuns"), "bb": st.get("baseOnBalls"),
            "ibb": st.get("intentionalWalks"), "hbp": st.get("hitByPitch"),
            "so": st.get("strikeOuts"), "er": st.get("earnedRuns"), "r": st.get("runs"),
            "pitches": pitches, "strikes": strikes,
            "strike_pct": (strikes / pitches) if pitches and strikes is not None else np.nan,
        })
    return out


def compile_boxscores(games, cache_dir=RAW):
    """-> (team_game_stats_df, pitcher_game_stats_df) for games with a cached boxscore."""
    team_rows, pit_rows = [], []
    for g in games[games["played"]].itertuples(index=False):
        fp = boxscore_path(g.season, g.game_pk, cache_dir)
        if not fp.exists():
            continue
        box = json.loads(fp.read_text(encoding="utf-8"))
        gd = g._asdict()
        team_rows.append(_team_lines(box, "home", g.away_team, gd, True))
        team_rows.append(_team_lines(box, "away", g.home_team, gd, False))
        pit_rows.extend(_pitcher_lines(box, "home", gd, True))
        pit_rows.extend(_pitcher_lines(box, "away", gd, False))
    return pd.DataFrame(team_rows), pd.DataFrame(pit_rows)


def load_team_game_stats(path=DATA / "team_game_stats.csv"):
    if not Path(path).exists():
        return None
    stats = pd.read_csv(path, dtype={"game_id": str})
    return {(r.game_id, r.team): r for r in stats.itertuples(index=False)}


def load_pitcher_game_stats(path=DATA / "pitcher_game_stats.csv.gz"):
    if not Path(path).exists():
        return None
    return pd.read_csv(path, dtype={"game_id": str})
