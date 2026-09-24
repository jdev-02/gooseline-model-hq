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

# FAVNO: a base-rate bet, not a model call. Across 44,456 played games
# (2008-2026), a winning team wins by 2+ runs 71.5% of the time and by
# exactly 1 run the other 28.5%. So for a team with true win probability p,
# P(that team fails to cover -1.5) = 1 - WIN_BY_2_GIVEN_WIN * p, which is
# above 50% for any p below ~70% -- meaning "No, the favorite doesn't cover"
# is the statistically true favorite for every game short of a real
# blowout price. Found 2026-09-15 from screenshots of a Kalshi account
# buying No on ordinary favorites' run lines across a whole slate; verified
# against src/mlb/compile games.csv rather than taken on faith. Tracked
# here exactly like totals and spread were before they earned a card: the
# rundown logs it and paper_trade.py settles it, and it appears on the
# site only if its live record earns a word.
WIN_BY_2_GIVEN_WIN = 0.715
FAVNO_MIN_FAV = 0.50    # below this there is no favorite to fade
FAVNO_MAX_FAV = 0.70    # at/above this the favorite covers more than it doesn't


def _num(x):
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def favno_pick(mkt_home, mkt_away, home_team, away_team, sp_home_bid, sp_away_bid):
    """The base-rate fade of an ordinary favorite's run line (see
    WIN_BY_2_GIVEN_WIN above): identify the favorite from the moneyline
    ask, skip it if it is a toss-up or a real blowout price, and price the
    No side off the run-line bid. Returns (team, no_price, p_no) or None.
    A pure function so the base-rate arithmetic is unit-tested on its own,
    without running the whole rundown pipeline."""
    if not (_num(mkt_home) and _num(mkt_away)):
        return None
    if mkt_home >= mkt_away:
        fav_p, fav_team, fav_bid = mkt_home, home_team, sp_home_bid
    else:
        fav_p, fav_team, fav_bid = mkt_away, away_team, sp_away_bid
    if not (FAVNO_MIN_FAV <= fav_p < FAVNO_MAX_FAV) or not _num(fav_bid):
        return None
    no_price = round(1 - float(fav_bid), 3)
    p_no = round(1 - WIN_BY_2_GIVEN_WIN * float(fav_p), 3)
    return fav_team, no_price, p_no


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


# Every rung Kalshi lists on a normal night. Until 2026-09-15 this was
# (7.5, 8.5, 9.5), so at Coors Field, where the market's main line sits at
# 10.5 or 11.5, the model priced only the low tail and flagged a long-shot
# Under 8.5 while never looking at the rung everyone was trading. Rungs
# below 6.5 and above 12.5 are listed on a minority of games and sit in the
# tails where the count model is least trusted; they stay out.
TOTAL_LINES = (6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5)


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
    # `df` carries the live refresh (probables, status, forecast weather) for
    # the horizon; games.csv on disk does not. Without this overlay the
    # totals model priced every game at 70F and never saw a scratch.
    live_cols = [c for c in ("home_sp_id", "away_sp_id", "status", "temp_f",
                             "wind_mph", "wind_dir", "condition", "hp_umpire_id")
                 if c in df.columns and c in games.columns]
    fresh = df[df["game_pk"].isin(upcoming["game_pk"])].drop_duplicates("game_pk").set_index("game_pk")
    m = games["game_pk"].isin(fresh.index)
    for c in live_cols:
        games.loc[m, c] = games.loc[m, "game_pk"].map(fresh[c]).values
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
    temps = up_t["temp_f"].tolist() if "temp_f" in up_t else [None] * len(up_t)
    roofs = up_t["roof_closed"].tolist()
    for j, pk in enumerate(up_t["game_pk"].tolist()):
        t = temps[j]
        out[pk] = {"mu_total": round(float(mu[j]), 2),
                   "temp_f": None if t is None or pd.isna(t) else int(t),
                   "roof_closed": bool(roofs[j]),
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
    upd, wx_upd = {}, {}
    for d in live.get("dates", []):
        for g in d["games"]:
            hp = g["teams"]["home"].get("probablePitcher", {})
            ap_ = g["teams"]["away"].get("probablePitcher", {})
            upd[g["gamePk"]] = (hp.get("id"), ap_.get("id"),
                                hp.get("fullName"), ap_.get("fullName"),
                                g["status"].get("detailedState"))
            # Weather is a forecast until first pitch and the plate umpire
            # posts with the lineups, so both are re-read every run. Only
            # fields the API actually carries are written; a blank never
            # overwrites what the season pull already had.
            wx = g.get("weather", {}) or {}
            off = next((o for o in g.get("officials", []) or []
                        if o.get("officialType") == "Home Plate"), {})
            row = {}
            if wx.get("temp") not in (None, ""):
                try:
                    row["temp_f"] = int(wx["temp"])
                except (TypeError, ValueError):
                    pass
            wind = wx.get("wind") or ""
            if " mph" in wind:
                try:
                    row["wind_mph"] = int(wind.split(" mph")[0].strip())
                except ValueError:
                    pass
                if "," in wind:
                    row["wind_dir"] = wind.split(",", 1)[1].strip()
            if wx.get("condition"):
                row["condition"] = wx["condition"]
            if off.get("official", {}).get("id"):
                row["hp_umpire_id"] = off["official"]["id"]
                row["hp_umpire"] = off["official"].get("fullName")
            if row:
                wx_upd[g["gamePk"]] = row
    for pk, (hsp, asp, hnm, anm, st) in upd.items():
        m = games["game_pk"] == pk
        if m.any():
            # Names must refresh with the ids: updating only the ids left a
            # freshly announced starter rendering as TBD on the card.
            games.loc[m, ["home_sp_id", "away_sp_id", "status"]] = [hsp, asp, st]
            games.loc[m, ["home_sp_name", "away_sp_name"]] = [hnm, anm]
            for k, v in wx_upd.get(pk, {}).items():
                if k in games.columns:
                    games.loc[m, k] = v
    # StatsAPI leaves the weather blank until about first pitch, so for the
    # games still ahead take the game-hour temperature from a forecast.
    if "temp_f" in games.columns:
        from src.mlb.weather import forecast_temps
        horizon = games["game_pk"].isin(list(upd)) & games["temp_f"].isna()
        if horizon.any():
            fc = forecast_temps(games[horizon])
            for pk, t in fc.items():
                games.loc[games["game_pk"] == pk, "temp_f"] = t
            print(f"weather: forecast temperature for {len(fc)}/{int(horizon.sum())} "
                  f"upcoming games", file=sys.stderr)
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
    from src.core.clock import slate_today
    today = pd.Timestamp(asof).normalize() if asof else slate_today()
    end = today + pd.Timedelta(days=days - 1)
    df = build_frame(cfg, refresh_window=(today.date(), end.date()))
    up = df[df["y"].isna() & (df["gameday"] >= today) & (df["gameday"] <= end)
            & df["status"].isin(UPCOMING_STATES)]
    if len(up) == 0:
        print("No upcoming games in the window.")
        return None
    season = int(up["season"].max())
    # Last scheduled day of this season's own slate, so labels.py can flag
    # a game inside the last two weeks -- see docs/baselines.md's
    # "late-season totals" section for why: 2023-2025 held-out data shows
    # the totals model gets systematically MORE bullish on overs in that
    # window while real scoring goes down, a real, measured miscalibration
    # (found 2026-09-24, ops/experiment_late_season.py).
    season_end = df.loc[df["season"] == season, "gameday"].max()
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
    # Same live-then-snapshot fallback the moneyline gets. Without it one
    # Kalshi outage silently emptied the totals market for the day -- no
    # total_call logged, no paper-trade row, every card "no price to
    # compare yet" -- and the health gate did not notice because it only
    # tests mu_total.
    tot_prices = {}
    if totals:
        tot_prices = live_prices("KXMLBTOTAL", mlb_total_key) if use_live_prices else {}
        if not tot_prices:
            tot_prices = latest_prices(db_path, "KXMLBTOTAL", mlb_total_key)
    # The run line is the third market, priced against Kalshi's actual
    # "wins by over 1.5 runs" contracts, logged and paper-traded like the
    # other two. It used to be a probability on the card with no price and
    # no record -- which is how "the model went 5-1 on the spread" could be
    # said and never checked.
    from src.core.kalshi import mlb_spread_key, spread_sides
    sp_prices = live_prices("KXMLBSPREAD", mlb_spread_key) if use_live_prices else {}
    if not sp_prices:
        sp_prices = latest_prices(db_path, "KXMLBSPREAD", mlb_spread_key)

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
               # First pitch as a real instant, so the page can render it on
               # the reader's own clock instead of the build machine's.
               "kick_iso": str(getattr(r, "game_datetime_utc", "") or ""),
               "days_before_season_end": (int((season_end - r.gameday).days)
                                          if pd.notna(season_end) else None),
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
                if ts:
                    # Snapshot timestamps may carry an offset or not; this
                    # path only runs when the live API is down, and it used
                    # to crash the whole run on a tz-aware value.
                    t = pd.Timestamp(ts)
                    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
                    age = (now - t).total_seconds() / 60.0
            rec["price_age_min"] = None if age is None else round(age, 1)
            for p, ek, vk in ((p_home[j], "edge", "verdict"), (p_n, "edge_narrative", "verdict_narrative")):
                e_h = (p - hp - kalshi_fee(hp)) if hp is not None else -1
                e_a = ((1 - p) - ap_ - kalshi_fee(ap_)) if ap_ is not None else -1
                best, side = max((e_h, r.home_team), (e_a, r.away_team))
                rec[ek] = round(float(best), 3)
                rec[vk] = verdict_for(best, side, age)
        sq = sp_prices.get((str(r.gameday.date()), r.away_team, r.home_team, int(r.game_number)), {})
        rec["mkt_sp_home"] = (sq.get(f"{r.home_team}:{RUN_LINE:g}") or {}).get("ask")
        rec["mkt_sp_home_bid"] = (sq.get(f"{r.home_team}:{RUN_LINE:g}") or {}).get("bid")
        rec["mkt_sp_away"] = (sq.get(f"{r.away_team}:{RUN_LINE:g}") or {}).get("ask")
        rec["mkt_sp_away_bid"] = (sq.get(f"{r.away_team}:{RUN_LINE:g}") or {}).get("bid")
        sides = spread_sides(sq, r.home_team, r.away_team, RUN_LINE,
                             rec["p_home_cover"], rec["p_away_cover"])
        best_s = None
        for call, p_s, cost in sides:
            e_s = p_s - cost - kalshi_fee(cost)
            if best_s is None or e_s > best_s[0]:
                best_s = (e_s, call, cost, p_s)
        if best_s is not None:
            rec["spread_edge"] = round(float(best_s[0]), 3)
            rec["spread_price"] = round(float(best_s[2]), 3)
            rec["spread_p"] = round(float(best_s[3]), 3)
            rec["spread_call"] = (best_s[1] if best_s[0] > edge_threshold
                                  else f"no edge (best {best_s[1]})")
        # FAVNO: a base-rate bet, not a model call. Silent (favno_call left
        # unset) when there is no favorite in the 50-70% window or no
        # run-line quote -- most games, by design, since a heavy favorite
        # or a true toss-up gives this strategy nothing to say.
        pick = favno_pick(rec["mkt_home"], rec["mkt_away"], r.home_team, r.away_team,
                          rec.get("mkt_sp_home_bid"), rec.get("mkt_sp_away_bid"))
        if pick:
            fav_team, no_price, p_no = pick
            e_no = p_no - no_price - kalshi_fee(no_price)
            rec["favno_team"] = fav_team
            rec["favno_price"] = no_price
            rec["favno_p"] = p_no
            rec["favno_edge"] = round(float(e_no), 3)
            rec["favno_call"] = (f"{fav_team} NO -1.5" if e_no > edge_threshold
                                 else f"no edge (best {fav_team} NO -1.5)")
        t = totals.get(r.game_pk)
        if t:
            rec["mu_total"] = t["mu_total"]
            rec["temp_f"] = t.get("temp_f")
            rec["roof_closed"] = t.get("roof_closed")
            tp = tot_prices.get((str(r.gameday.date()), r.away_team,
                                 r.home_team, int(r.game_number)), {})
            best_t, best_desc = None, ""
            for ln, p_over in t["lines"].items():
                rec[f"p_over_{ln:g}"] = p_over
                q = tp.get(f"{ln:g}")
                ask = q.get("ask") if q else None
                bid = q.get("bid") if q else None
                rec[f"mkt_over_{ln:g}"] = ask
                if ask is None:
                    continue
                # A NO contract costs 1 minus the YES *bid*, not 1 minus the
                # ask. Pricing the under from the ask overstated every UNDER
                # edge by the whole bid-ask spread (a 6-cent spread became a
                # phantom +6% edge and a green badge). Fall back to the ask
                # only when the quote carries no bid.
                under_cost = (1 - bid) if bid is not None else (1 - ask)
                rec[f"mkt_under_{ln:g}"] = round(float(under_cost), 3)
                e_o = p_over - ask - kalshi_fee(ask)
                e_u = (1 - p_over) - under_cost - kalshi_fee(under_cost)
                for e, side in ((e_o, f"OVER {ln:g}"), (e_u, f"UNDER {ln:g}")):
                    if best_t is None or e > best_t:
                        best_t, best_desc = e, side
            if best_t is not None:
                rec["total_edge"] = round(float(best_t), 3)
                rec["total_call"] = (best_desc if best_t > edge_threshold
                                     else f"no edge (best {best_desc})")
        # The word a reader sees, decided by each market's live record
        # (src/mlb/labels.py). Logged here so the public feed and anyone
        # rendering it (David's MLB tab) carry the same label as our page.
        from src.mlb.labels import tier_for, label_for
        from src.site.names import mlb as _nick
        rec["tier"] = tier_for(rec)
        rec["label"] = label_for(rec, _nick)
        rows.append(rec)
    # Deterministic slate order: first pitch, then away, then home.
    from src.core.teams import slate_sort_key
    rows.sort(key=slate_sort_key)
    table = pd.DataFrame(rows)
    trained = df[df["y"].notna()]
    print(f"\n=== MLB rundown {today.date()} (+{days - 1}d), trained through "
          f"{trained['gameday'].max().date()} on {len(trained)} games, "
          f"{'ensemble' if ens is not None else 'linear'}, prices: {price_source} ===")
    print(table.drop(columns=["note"]).to_string(index=False))
    print("\nHIGH VALUE = model edge over the Kalshi ask after the 7% fee. Apply the news check "
          "(scratches, lineups, a roof or forecast change since the run) before acting; "
          "the model cannot see them.")
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
