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
import sys

from src.core.kalshi import (kalshi_fee, latest_prices, latest_snapshot_ts,
                             live_prices, mlb_event_key, mlb_total_key,
                             match_mlb_event)
from src.core.models import LinearGaussianModel, prob_margin_over
from src.core.walkforward import season_decay_weights
from src.mlb.compile import DATA, load_games, load_team_game_stats, load_pitcher_game_stats
from src.mlb.features import build_features, MLB_FEATURE_COLS
from src.mlb.ingest import fetch_live_schedule
from src.mlb.narrative import load_narrative, apply_narrative
from src.mlb.park import build_park_factors, park_lookup

UPCOMING_STATES = {"Scheduled", "Pre-Game", "Warmup", "Delayed Start"}
RUN_LINE = 1.5          # the standard MLB spread
HIGH_VALUE_EDGE = 0.04  # same threshold David uses for NFL
STALE_MINUTES = 15      # past this, a price is not actionable


def verdict_for(edge, side, price_age_min=None):
    """Three-tier verdict, worded exactly as the NFL page.

    A stale price is the most dangerous output this system can produce: a
    HIGH VALUE badge computed against a quote the market has already moved
    past looks identical to a live one. So staleness downgrades the badge
    rather than being reported alongside it.
    """
    if edge is None:
        return "no price"
    if price_age_min is not None and price_age_min > STALE_MINUTES:
        return f"STALE PRICE &mdash; re-check before acting ({price_age_min:.0f} min old)"
    if edge > HIGH_VALUE_EDGE:
        return f"HIGH VALUE &mdash; {side}"
    if edge > 0:
        return f"CAUTIOUS &mdash; small edge on {side}"
    return "NO VALUE at current price"


TOTAL_LINES = (7.5, 8.5, 9.5)


def load_totals_config():
    p = DATA / "totals_config.json"
    return json.loads(p.read_text()) if p.exists() else None


def price_totals(df, upcoming, cfg_t):
    """Fit the negative-binomial totals model and price the Kalshi ladder.

    Returns {game_pk: {"mu_total": float, "lines": {strike: p_over}}} or {}
    when the totals config has not been frozen by ops/run_totals.py yet.
    """
    if cfg_t is None:
        print("totals: no data/mlb/totals_config.json, run ops/run_totals.py",
              file=sys.stderr)
        return {}
    from src.mlb.totals import build_total_features, NegBinomTotal
    from src.mlb.park import build_park_factors, park_lookup
    games = load_games(keep_unplayed=True, first_season=cfg_t.get("first_season", 2008))
    park = build_park_factors(games)
    tdf = build_total_features(games, load_team_game_stats(),
                               load_pitcher_game_stats(), park_lookup(park))
    cols = cfg_t["feature_cols"]
    train = tdf[tdf["y"].notna()]
    hl = cfg_t.get("half_life_seasons") or np.inf
    sw = season_decay_weights(train["season"].values,
                              int(upcoming["season"].max()), hl)
    nb = NegBinomTotal().fit(train[cols].values, train["y"].values, sample_weight=sw)
    want = set(int(x) for x in upcoming["game_pk"].tolist())
    up_t = tdf[tdf["game_pk"].astype(int).isin(want)]
    print(f"totals: {len(train)} train rows, k={nb.k_:.1f}, "
          f"{len(up_t)}/{len(want)} upcoming matched", file=sys.stderr)
    if not len(up_t):
        return {}
    X = up_t[cols].values
    mu, _ = nb.predict_dist(X)
    out = {}
    for j, pk in enumerate(up_t["game_pk"].tolist()):
        out[pk] = {"mu_total": round(float(mu[j]), 2),
                   "lines": {ln: round(float(nb.prob_over(X[j:j + 1], ln)[0]), 3)
                             for ln in TOTAL_LINES}}
    return out


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
            hp = g["teams"]["home"].get("probablePitcher", {})
            ap_ = g["teams"]["away"].get("probablePitcher", {})
            upd[g["gamePk"]] = (hp.get("id"), ap_.get("id"),
                                hp.get("fullName"), ap_.get("fullName"),
                                g["status"].get("detailedState"))
    for pk, (hsp, asp, hnm, anm, st) in upd.items():
        m = games["game_pk"] == pk
        if m.any():
            # Names must refresh with the ids: updating only the ids left a
            # freshly announced starter rendering as TBD on the card.
            games.loc[m, ["home_sp_id", "away_sp_id", "status"]] = [hsp, asp, st]
            games.loc[m, ["home_sp_name", "away_sp_name"]] = [hnm, anm]
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


def rundown(days=1, db_path="data/kalshi_prices.db", edge_threshold=0.04, narrative_path=None,
            use_ensemble=False, log_path=DATA / "narrative" / "log.csv", asof=None,
            use_live_prices=True):
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
    # Unlisted probable starter: the pitcher block fell back to league mean,
    # so widen the honest range by 10% and flag the row.
    sp_unknown = (up["home_sp_id"].isna() | up["away_sp_id"].isna()).values
    sigma = np.where(sp_unknown, sigma * 1.10, sigma)
    recal = float(cfg.get("recal_scale", 1.0))
    p_home = norm.cdf(mu / (recal * sigma))
    prices = live_prices("KXMLBGAME", mlb_event_key) if use_live_prices else {}
    price_source = "live"
    if not prices:
        prices = latest_prices(db_path, "KXMLBGAME", mlb_event_key)
        price_source = "snapshot log"
    narr = load_narrative(narrative_path)
    now = pd.Timestamp.now(tz="UTC")
    try:
        totals = price_totals(df, up, load_totals_config())
    except Exception as e:
        print(f"totals pricing skipped ({e})", file=sys.stderr)
        totals = {}
    tot_prices = (live_prices("KXMLBTOTAL", mlb_total_key)
                  if (totals and use_live_prices) else {})

    rows = []
    for j, r in enumerate(up.itertuples(index=False)):
        ent = narr.get((r.away_team, r.home_team))
        mu_n, sg_n, shift = apply_narrative(mu[j], sigma[j], r.home_team, r.away_team, ent)
        p_n = float(norm.cdf(mu_n / (recal * sg_n)))
        rec = {"date": r.gameday.date(), "game_pk": r.game_pk, "away": r.away_team, "home": r.home_team,
               "home_sp": r.home_sp_name, "away_sp": r.away_sp_name,
               "sp_unknown": bool(sp_unknown[j]),
               "mu": round(float(mu[j]), 2), "sigma": round(float(sigma[j]), 3), "p_home": round(float(p_home[j]), 3),
               "mkt_home": None, "mkt_away": None, "edge": None, "verdict": "no price",
               "narrative_shift": round(shift, 2), "mu_narrative": round(mu_n, 2),
               "sigma_narrative": round(sg_n, 3), "p_home_narrative": round(p_n, 3),
               "edge_narrative": None, "verdict_narrative": "no price",
               "run_line": RUN_LINE,
               "p_home_cover": round(float(prob_margin_over(mu[j], sigma[j], RUN_LINE)), 3),
               "p_away_cover": round(float(1 - prob_margin_over(mu[j], sigma[j], -RUN_LINE)), 3),
               "note": ent.note if ent else ""}
        ev = match_mlb_event(prices, r.gameday.date(), r.away_team, r.home_team, int(r.game_number))
        if ev:
            hp, ap_ = ev.get(r.home_team, {}).get("ask"), ev.get(r.away_team, {}).get("ask")
            rec["mkt_home"], rec["mkt_away"] = hp, ap_
            asof = ev.get(r.home_team, {}).get("asof") or ev.get(r.away_team, {}).get("asof")
            age = ((now - pd.Timestamp(asof)).total_seconds() / 60.0) if asof is not None else None
            if age is None and price_source != "live":
                ts = latest_snapshot_ts(db_path)
                age = ((now - pd.Timestamp(ts).tz_localize("UTC")).total_seconds() / 60.0
                       if ts else None)
            rec["price_age_min"] = None if age is None else round(age, 1)
            for p, ek, vk in ((p_home[j], "edge", "verdict"), (p_n, "edge_narrative", "verdict_narrative")):
                e_h = (p - hp - kalshi_fee(hp)) if hp is not None else -1
                e_a = ((1 - p) - ap_ - kalshi_fee(ap_)) if ap_ is not None else -1
                best, side = max((e_h, r.home_team), (e_a, r.away_team))
                rec[ek] = round(float(best), 3)
                rec[vk] = verdict_for(best, side, age)
        t = totals.get(r.game_pk)
        if t:
            rec["mu_total"] = t["mu_total"]
            tp = tot_prices.get((str(r.gameday.date()), r.away_team,
                                 r.home_team, int(r.game_number)), {})
            best_t, best_desc = None, ""
            for ln, p_over in t["lines"].items():
                rec[f"p_over_{ln:g}"] = p_over
                q = tp.get(f"{ln:g}")
                ask = q.get("ask") if q else None
                if ask is None:
                    continue
                e_o = p_over - ask - kalshi_fee(ask)
                e_u = (1 - p_over) - (1 - ask) - kalshi_fee(1 - ask)
                for e, side in ((e_o, f"OVER {ln:g}"), (e_u, f"UNDER {ln:g}")):
                    if best_t is None or e > best_t:
                        best_t, best_desc = e, side
            if best_t is not None:
                rec["total_edge"] = round(float(best_t), 3)
                rec["total_call"] = (best_desc if best_t > edge_threshold
                                     else f"no edge (best {best_desc})")
        rows.append(rec)
    table = pd.DataFrame(rows)
    trained = df[df["y"].notna()]
    print(f"\n=== MLB rundown {today.date()} (+{days - 1}d), trained through "
          f"{trained['gameday'].max().date()} on {len(trained)} games, "
          f"{'ensemble' if ens is not None else 'linear'}, prices: {price_source} ===")
    print(table.drop(columns=["note"]).to_string(index=False))
    print("\nHIGH VALUE = model edge over the Kalshi ask after the 7% fee. Apply the news check "
          "(scratches, lineups, weather) before acting; the model cannot see them.")
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        new = table.assign(run_ts=pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                           result=np.nan)
        if Path(log_path).exists():
            # schema-tolerant append: new columns become NaN on old rows
            new = pd.concat([pd.read_csv(log_path), new], ignore_index=True)
        new.to_csv(log_path, index=False)
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--db", default="data/kalshi_prices.db")
    ap.add_argument("--edge", type=float, default=0.04)
    ap.add_argument("--narrative", default=None)
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD (default today)")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--snapshot-prices", action="store_true",
                    help="read the sqlite log instead of fetching live prices")
    a = ap.parse_args()
    rundown(a.days, a.db, a.edge, a.narrative, a.ensemble,
            None if a.no_log else DATA / "narrative" / "log.csv", a.asof,
            use_live_prices=not a.snapshot_prices)
