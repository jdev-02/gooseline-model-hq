"""Paper-trade the logged verdicts and report realized P&L.

The walk-forward backtest already measured whether the model knows anything.
This measures something the backtest structurally cannot see: execution. How
often was the flagged price already gone, what did the fee actually cost, and
how much of the modeled edge survived to settlement.

  uv run python ops/paper_trade.py --stake 15 --min-edge 0.04

Writes data/mlb/paper_trades.csv and prints the summary.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.kalshi import kalshi_fee  # noqa: E402
from src.mlb.compile import DATA  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stake", type=float, default=15.0, help="flat stake per bet")
ap.add_argument("--min-edge", type=float, default=0.04)
ap.add_argument("--kelly", type=float, default=0.0,
                help="if >0, fraction of Kelly instead of a flat stake")
ap.add_argument("--bankroll", type=float, default=1000.0)
ap.add_argument("--stream", default="model", choices=["model", "narrative"])
ap.add_argument("--sport", default="mlb", choices=["mlb", "nfl"])
args = ap.parse_args()

if args.sport == "nfl":
    log_path = Path("data/nfl/rundown_log.csv")
    games_path = Path("data/nfl/games.csv")
else:
    log_path = DATA / "narrative" / "log.csv"
    games_path = DATA / "games.csv"

if not log_path.exists():
    print(f"no {args.sport} rundown log yet")
    sys.exit(0)
log = pd.read_csv(log_path)

if args.sport == "nfl":
    gf = pd.read_csv(games_path, low_memory=False)
    gf = gf[gf["result"].notna()].copy()
    gf["key"] = (gf["gameday"].astype(str) + "_" + gf["away_team"]
                 + "_" + gf["home_team"])
    gf["played"] = True
    gf["home_score"] = gf["home_score"]
    gf["away_score"] = gf["away_score"]
    g = gf.set_index("key")
    log["game_pk"] = log.get("game_id", log.get("date"))
else:
    games = pd.read_csv(games_path,
                        usecols=["game_pk", "played", "result", "home_score", "away_score"])
    g = games.set_index("game_pk")

edge_col = "edge" if args.stream == "model" else "edge_narrative"
verd_col = "verdict" if args.stream == "model" else "verdict_narrative"

# One row per game per run; keep the LAST run before the game, which is the
# price a human acting on the page would most plausibly have seen.
log = log.sort_values("run_ts").drop_duplicates(["game_pk", "date"], keep="last")

rows = []
for r in log.itertuples(index=False):
    pk = r.game_pk if args.sport == "nfl" else int(r.game_pk)
    if pk not in g.index or not bool(g.loc[pk, "played"]):
        continue
    result = float(g.loc[pk, "result"])
    total = float(g.loc[pk, "home_score"] + g.loc[pk, "away_score"])

    # ---- moneyline ----
    edge = getattr(r, edge_col, np.nan)
    verdict = str(getattr(r, verd_col, ""))
    if pd.notna(edge) and edge >= args.min_edge and "STALE" not in verdict:
        side = (verdict.split("&mdash;")[-1].strip()
                .replace("small edge on ", "")
                .replace("CANDIDATE ", "").strip())
        if side in (r.home, r.away):
            mh = r.mkt_home
            ma = getattr(r, "mkt_away", None)
            if side == r.home:
                ask = float(mh) if pd.notna(mh) else np.nan
            elif ma is not None and pd.notna(ma):
                ask = float(ma)
            elif pd.notna(mh):
                ask = 1.0 - float(mh)   # NFL logs only the home ask
            else:
                ask = np.nan
            if not np.isfinite(ask):
                continue
            p_model = float(r.p_home if side == r.home else 1 - r.p_home)
            won = (result > 0) if side == r.home else (result < 0)
            rows.append(dict(date=r.date, game_pk=pk, market="ML", pick=side,
                             ask=ask, p_model=p_model, edge=float(edge), won=bool(won),
                             price_age_min=getattr(r, "price_age_min", np.nan)))

    # ---- total ----
    te = getattr(r, "total_edge", np.nan)
    call = str(getattr(r, "total_call", ""))
    if pd.notna(te) and te >= args.min_edge and call and not call.startswith("no edge"):
        parts = call.split()
        if len(parts) == 2:
            direction, line = parts[0], float(parts[1])
            p_over = getattr(r, f"p_over_{line:g}", np.nan)
            if pd.notna(p_over):
                p_model = float(p_over) if direction == "OVER" else 1 - float(p_over)
                # the logged ask is for the over rung; under is its complement
                ask = np.nan
                won = (total > line) if direction == "OVER" else (total < line)
                # reconstruct the traded price from model prob minus edge and fee
                ask = float(p_model - te - kalshi_fee(max(min(p_model - te, .99), .01)))
                rows.append(dict(date=r.date, game_pk=pk, market=f"{direction} {line:g}",
                                 pick=call, ask=ask, p_model=p_model, edge=float(te),
                                 won=bool(won),
                                 price_age_min=getattr(r, "price_age_min", np.nan)))

if not rows:
    print("no settled bets at this threshold yet")
    sys.exit(0)

t = pd.DataFrame(rows)
t["fee"] = t["ask"].clip(0.01, 0.99).map(kalshi_fee)
t["cost_per_contract"] = t["ask"] + t["fee"]
if args.kelly > 0:
    b = (1 - t["cost_per_contract"]) / t["cost_per_contract"]
    kf = ((b * t["p_model"] - (1 - t["p_model"])) / b).clip(lower=0)
    t["stake"] = (args.kelly * kf * args.bankroll).round(2)
else:
    t["stake"] = args.stake
t["contracts"] = t["stake"] / t["cost_per_contract"]
t["pnl"] = np.where(t["won"], t["contracts"] * 1.0 - t["stake"], -t["stake"]).round(2)
t["cum_pnl"] = t["pnl"].cumsum().round(2)
t.to_csv(DATA / "paper_trades.csv", index=False)

n, w = len(t), int(t["won"].sum())
staked, pnl = t["stake"].sum(), t["pnl"].sum()
print(f"\n=== Paper trades ({args.stream} stream, min edge {args.min_edge:.0%}) ===")
print(t[["date", "market", "pick", "ask", "p_model", "edge", "won", "pnl"]].to_string(index=False))
print(f"\nbets {n}  wins {w} ({100*w/n:.1f}%)  staked ${staked:.2f}  "
      f"P&L ${pnl:+.2f}  ROI {100*pnl/staked:+.1f}%")
print(f"model expected win rate {100*t['p_model'].mean():.1f}%  "
      f"modeled edge {100*t['edge'].mean():+.1f}%")
print(f"realized minus expected: {100*(w/n - t['p_model'].mean()):+.1f} points")
stale = t["price_age_min"].dropna()
if len(stale):
    print(f"price age at flag: median {stale.median():.0f} min, "
          f"max {stale.max():.0f} min")
print("\nA few dozen bets measures execution, not skill. Treat the ROI as a "
      "slippage estimate until the sample is in the hundreds.")
