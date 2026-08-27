"""Daily MLB rundown: fit on every played game, predict the slate, compare
with the latest logged Kalshi price after fees, apply the human narrative
tilt as a separate stream, and log both.

  uv run python -m src.mlb.rundown --days 1 --narrative data/mlb/narrative/2026-08-27.yaml
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.core.kalman import TeamKalman
from src.core.kalshi import kalshi_fee, latest_prices, mlb_event_key, match_mlb_event
from src.core.models import LinearGaussianModel
from src.core.walkforward import season_decay_weights
from src.mlb.compile import DATA, load_games, load_team_game_stats, load_pitcher_game_stats
from src.mlb.features import build_features, MLB_FEATURE_COLS
from src.mlb.ingest import fetch_live_schedule
from src.mlb.narrative import load_narrative, apply_narrative
from src.mlb.park import build_park_factors, park_lookup

UPCOMING_STATES = {"Scheduled", "Pre-Game", "Warmup", "Delayed Start"}


def load_config():
    p = DATA / "model_config.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"kalman": {"obs_var": 20.0, "step_q": 0.005, "season_inflate": 1.0,
                       "season_revert": 0.75, "step_col": "day_index", "init_hfa": 0.2,
                       "init_hfa_var": 0.25, "init_var": 1.0, "hfa_q": 1e-5},
            "lam": 10.0, "half_life_seasons": None, "recal_scale": 1.0,
            "feature_cols": MLB_FEATURE_COLS}


def refresh_probables(games, start, end):
    """Overwrite probable pitchers / status for the horizon from the live API
    (never cached) so a late scratch is seen before the run."""
    live = fetch_live_schedule(start, end)
    upd = {}
    for d in live.get("dates", []):
        for g in d["games"]:
            upd[g["gamePk"]] = (g["teams"]["home"].get("probablePitcher", {}).get("id"),
                                g["teams"]["away"].get("probablePitcher", {}).get("id"),
                                g["status"].get("detailedState"))
    for pk, (hsp, asp, st) in upd.items():
        m = games["game_pk"] == pk
        if m.any():
            games.loc[m, ["home_sp_id", "away_sp_id", "status"]] = [hsp, asp, st]
    return games


def build_frame(cfg, refresh_window=None):
    games = load_games(keep_unplayed=True)
    if refresh_window:
        games = refresh_probables(games, *refresh_window)
    park = build_park_factors(games)
    df = build_features(games, load_team_game_stats(), load_pitcher_game_stats(), park_lookup(park))
    kal = dict(cfg["kalman"])
    return TeamKalman(**kal).run(df)


def fit_model(df, cfg, asof_season, cols):
    train = df[df["y"].notna()]
    assert train["home_score"].notna().all(), "unplayed rows leaked into training"
    hl = cfg.get("half_life_seasons") or np.inf
    sw = season_decay_weights(train["season"].values, asof_season, hl)
    return LinearGaussianModel(lam=cfg["lam"]).fit(train[cols].values, train["y"].values, sample_weight=sw)


def try_ensemble(df, cfg, asof_season, cols):
    try:
        from src.core.ensemble import DeepEnsemble
    except Exception:
        return None
    train = df[df["y"].notna()]
    hl = cfg.get("half_life_seasons") or np.inf
    sw = season_decay_weights(train["season"].values, asof_season, hl)
    return DeepEnsemble(n_members=5, hidden=16, weight_decay=1e-2, epochs=200, seed=0).fit(
        train[cols].values, train["y"].values, sample_weight=sw)


def rundown(days=1, db_path="kalshi_prices.db", edge_threshold=0.04, narrative_path=None,
            use_ensemble=False, log_path=DATA / "narrative" / "log.csv", asof=None):
    cfg = load_config()
    cols = cfg["feature_cols"]
    today = pd.Timestamp(asof or date.today()).normalize()
    end = today + pd.Timedelta(days=days - 1)
    df = build_frame(cfg, refresh_window=(today.date(), end.date()))
    up = df[df["y"].isna() & (df["gameday"] >= today) & (df["gameday"] <= end)
            & df["status"].isin(UPCOMING_STATES)]
    if len(up) == 0:
        print("No upcoming games in the window.")
        return None
    season = int(up["season"].max())
    lin = fit_model(df, cfg, season, cols)
    X = up[cols].values
    mu, sigma = lin.predict_dist(X)
    ens = try_ensemble(df, cfg, season, cols) if use_ensemble else None
    if ens is not None:
        emu, ale, epi = ens.predict_split(X)
        mu, sigma = emu, np.sqrt(ale + epi)
    recal = float(cfg.get("recal_scale", 1.0))
    p_home = norm.cdf(mu / (recal * sigma))
    prices = latest_prices(db_path, "KXMLBGAME", mlb_event_key)
    narr = load_narrative(narrative_path)

    rows = []
    for j, r in enumerate(up.itertuples(index=False)):
        ent = narr.get((r.away_team, r.home_team))
        mu_n, sg_n, shift = apply_narrative(mu[j], sigma[j], r.home_team, r.away_team, ent)
        p_n = float(norm.cdf(mu_n / (recal * sg_n)))
        rec = {"date": r.gameday.date(), "game_pk": r.game_pk, "away": r.away_team, "home": r.home_team,
               "home_sp": r.home_sp_name, "away_sp": r.away_sp_name,
               "mu": round(float(mu[j]), 2), "sigma": round(float(sigma[j]), 3), "p_home": round(float(p_home[j]), 3),
               "mkt_home": None, "mkt_away": None, "edge": None, "verdict": "no price",
               "narrative_shift": round(shift, 2), "mu_narrative": round(mu_n, 2),
               "sigma_narrative": round(sg_n, 3), "p_home_narrative": round(p_n, 3),
               "edge_narrative": None, "verdict_narrative": "no price",
               "note": ent.note if ent else ""}
        ev = match_mlb_event(prices, r.gameday.date(), r.away_team, r.home_team, int(r.game_number))
        if ev:
            hp, ap_ = ev.get(r.home_team, {}).get("ask"), ev.get(r.away_team, {}).get("ask")
            rec["mkt_home"], rec["mkt_away"] = hp, ap_
            for p, ek, vk in ((p_home[j], "edge", "verdict"), (p_n, "edge_narrative", "verdict_narrative")):
                e_h = (p - hp - kalshi_fee(hp)) if hp is not None else -1
                e_a = ((1 - p) - ap_ - kalshi_fee(ap_)) if ap_ is not None else -1
                best, side = max((e_h, r.home_team), (e_a, r.away_team))
                rec[ek] = round(float(best), 3)
                rec[vk] = f"CANDIDATE {side}" if best > edge_threshold else "pass"
        rows.append(rec)
    table = pd.DataFrame(rows)
    trained = df[df["y"].notna()]
    print(f"\n=== MLB rundown {today.date()} (+{days - 1}d), trained through "
          f"{trained['gameday'].max().date()} on {len(trained)} games, "
          f"{'ensemble' if ens is not None else 'linear'} ===")
    print(table.drop(columns=["note"]).to_string(index=False))
    print("\nCANDIDATE = model edge over Kalshi ask after 7% fee. Apply the news check "
          "(scratches, lineups, weather) before acting; the model cannot see them.")
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        table.assign(run_ts=pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                     result=np.nan).to_csv(log_path, mode="a", index=False,
                                          header=not Path(log_path).exists())
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--db", default="kalshi_prices.db")
    ap.add_argument("--edge", type=float, default=0.04)
    ap.add_argument("--narrative", default=None)
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD (default today)")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    rundown(a.days, a.db, a.edge, a.narrative, a.ensemble,
            None if a.no_log else DATA / "narrative" / "log.csv", a.asof)
