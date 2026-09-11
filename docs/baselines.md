# Baselines

The named floors every model must beat on the walk-forward harness. No model
advances unless it beats both on NLL **and** Brier, with every populated
calibration decile within 0.10. If it does not, investigate features, not
hyperparameters, and do not add another block as a rescue.

## MLB

Defined in `src/mlb/baselines.py`. Walk-forward 2023–2025 (7,289 games),
each season's baseline fit on strictly prior seasons.

| Name | Definition |
|---|---|
| `home_always` | `mu = mean(result)` on train; `p_home = P(home win)` on train |
| `elo_lite` | Elo on wins, K=4, home edge 24 Elo points; one OLS scale from Elo gap to run margin |
| `market_implied` | de-vigged Kalshi mid, 2026-forward only where a price was logged (the ceiling) |

### Results

Produced by `uv run python -m src.mlb.run_phase0`, written to
`figures/mlb/phase0/phase0_results.csv`.

| Model | Games | NLL | RMSE (runs) | Brier | Max calib. dev. | Beats home_always | Beats elo_lite |
|---|---|---|---|---|---|---|---|
| home_always | 7289 | 2.9255 | 4.511 | 0.2496 | 0.019 | — | — |
| elo_lite | 7289 | 2.9129 | 4.454 | 0.2445 | — | yes | — |
| linear, Tier A (schedule-only features, quick Kalman grid) | 7289 | 2.9090 | 4.437 | 0.2449 | 0.062 | yes | NLL yes, Brier **no** (0.2449 vs 0.2445) |
| linear, Tier A + pitcher block (boxscores 2021–2026, train from 2015) | 7289 | 2.9076 | 4.431 | 0.2438 | 0.071 | yes | yes |
| **linear, Tier A + pitcher block, train from 2008** (boxscores 2008–2026) | 7289 | **2.9075** | **4.429** | **0.2436** | **0.064** | **yes** | **yes** |

Embargo check: NLL 2.9075 with `embargo_steps=1` vs 2.9076 without. No
within-series leakage signal. `assert_no_leakage(r2_ceiling=0.60)` passed on
every fold (R² = 0.035).

Frozen config (`data/mlb/model_config.json`, 2026-08-27, train from 2008):
Kalman `obs_var=19, step_q=0.001, season_inflate=0.1, season_revert=0.75`
(HFA learned at 0.04 runs); ridge `lam=10`, season half-life 4.0;
`RECAL_SCALE=0.90`.

Reading: the gate passes, narrowly. The pitcher block is worth about 0.0011
Brier over the schedule-only model; extending training history from 2015 back
to 2008 (with boxscores throughout) adds another 0.0002 and tightens
calibration from 0.071 to 0.064. On a 4.4-run sigma that is a real but small
edge, and it is the reason most cards on the site are passes. StatsAPI
boxscores are reliable back to 2008; 2008 is now the floor of the training
set.

## NFL

David's model. Walk-forward 2021–2025 season tables live on the site's Track
Record tab (`src/site/nfl_site.py:history_tables`).

### Against the market (`ops/backtest_nfl_ml.py`, run 2026-09-10)

The Track Record tab reports the model picks winners 63.9% of the time. That
is not the bettor's question. The bettor's question is whether the model's
disagreements with the market's *price* make money. Paper-traded against the
de-vigged Vegas closing moneyline, $15 flat, betting every game where the
model's win probability beat the market by the threshold:

| Slice | Bets | Win % | ROI |
|---|---|---|---|
| 2021 | 182 | 35.2% | −5.2% |
| 2022 | 169 | 32.5% | −19.5% |
| 2023 | 165 | 33.3% | −11.3% |
| 2024 | 179 | 36.3% | −19.2% |
| 2025 | 176 | 40.3% | −6.4% |
| **All, edge ≥ 4%** | **871** | **35.6%** | **−12.3%** |
| edge ≥ 8% | 474 | 36.3% | −6.6% |
| edge ≥ 12% | 240 | 35.8% | −4.4% |
| edge ≥ 20% | 58 | 37.9% | +1.1% |
| model backs the favorite | 170 | 61.2% | −4.5% |
| model backs the underdog | 701 | 29.4% | −14.1% |
| week 1 | 55 | 41.8% | −8.3% |
| weeks 5+ | 650 | 34.0% | −13.3% |

Every season negative. Every threshold below 20% negative. Every week
negative. Against the spread the same walk-forward shows 49.2% (breakeven
52.4%). The favorite row is the cleanest reading: −4.5% is the vig, meaning
when the model backs a favorite it carries no information the price does
not already hold. The underdog row is where the money goes: the model
systematically overrates underdogs, and 701 of its 871 flags are underdogs.

**Verdict: the NFL model does not beat the market on any tested market.**
By this repo's own rule it has not passed its gate, and its HIGH VALUE
badges should not be read as bets until something changes that.

What this does *not* say: it does not say the model is badly built. Picking
64% of winners with public data is competent. It says the market is better,
which is the normal finding for public-data NFL models and the reason the
methodology insists on this test before any probability is bet.

### Ensemble check (`ops/backtest_nfl_ml_ensemble.py --season 2025`, run 2026-09-10)

The table above walks forward the linear model, because `history_tables`
uses it as a fast proxy. The live badges come from the deep ensemble. Walked
forward over 2025 alone (272 games, refit weekly):

| Slice | Bets | Win % | ROI |
|---|---|---|---|
| edge ≥ 4% | 189 | 38.6% | **−17.5%** |
| edge ≥ 8% | 110 | 37.3% | −20.0% |
| edge ≥ 12% | 61 | 42.6% | −5.1% |
| backs the favorite | 65 | 60.0% | −10.4% |
| backs the underdog | 124 | 27.4% | −21.2% |

The ensemble is worse than the linear proxy on every slice (linear 2025:
−6.4%). The nonlinearity buys confidence, not information. Caveat closed:
the model that produces the site's verdicts does not beat the market.

### Does the model add anything to the market? (`ops/experiment_nfl_market_blend.py`)

Blend `a*model + (1-a)*market` and find the `a` that minimises Brier. If it is
zero, the model carries no information the price does not already hold.

| | Brier |
|---|---|
| Vegas alone (a=0) | **0.2118** |
| model alone (a=1) | 0.2215 |
| best blend, pooled | a = **0.0** |
| best blend, walk-forward (choose a on prior seasons) | a* -> 0.00 by 2024; market wins every season |

Calibration by market bucket shows the mechanism. Where Vegas says 23%, home
teams win 23% and the model says 31%. Where Vegas says 78%, they win 80% and
the model says 73%. **The model is shrunk toward 50/50**; every disagreement
it has with the market is "this game is closer than you think", and it is
wrong. That is not a tuning problem. The feature set does not contain
information Vegas lacks, and no reweighting of it will. An NFL edge would
need new inputs (injury reports, weather, line movement), not a better fit.

Kalshi tracks Vegas within about 1.2 cents on NFL moneylines (max 3.3c over
week 1, 2026), so there is no Kalshi-specific mispricing to exploit either.

Caveat on comparability: the MLB gate above compared against Elo and
home-always baselines, not against the market, because `games.csv` for MLB
carries no historical odds. MLB's only market evidence is the live paper
trade since 2026-08-27, which is positive but small. MLB is unproven against
the market, not proven; running this same backtest for MLB needs a
historical-odds source and is the next thing to do.
