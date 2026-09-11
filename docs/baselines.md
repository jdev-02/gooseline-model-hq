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

### MLB moneyline against Kalshi (`ops/backtest_mlb_market.py`, run 2026-09-11)

Every game the daily run priced since 2026-08-27, not just the flagged ones,
against the live Kalshi ask it was priced at, settled by `settle_log.py`.
166 games. This is the market the model actually bets, so it is the test
that matters; the Elo gate above only showed the model knows *something*.

| | Brier |
|---|---|
| Kalshi alone (a=0) | **0.2397** |
| model alone (a=1) | 0.2464 |
| best blend weight on the model | **0.00** (pooled and on an honest first-half/second-half split) |

| Paper trade, $15 flat, after fee | Bets | Win % | ROI |
|---|---|---|---|
| edge ≥ 4% (the HIGH VALUE rule) | 46 | 34.8% | **−20.3%** |
| edge ≥ 8% | 11 | 18.2% | −47.2% |
| backs the underdog (≥ 4%) | 40 | 32.5% | −22.2% |

The ROI gets *worse* as the threshold rises: the more the model disagrees
with Kalshi, the more wrong it is. Same shape as the NFL result, same
mechanism — where Kalshi says 69% the model says 61%; where Kalshi says 60%
the model says 55%. Shrunk toward 50/50, over-rating underdogs.

`ops/paper_trade.py` (the site's Live Bet Performance section) agrees to the
bet: 46 bets, 34.8%, −20.3%. The +24.6% it showed on 2026-09-03 was 17 bets.

**Verdict: the MLB moneyline model does not beat Kalshi either.** Same
disease as NFL, same conclusion: the moneyline feature set is exhausted.

 the MLB gate above compared against Elo and
home-always baselines, not against the market, because `games.csv` for MLB
carries no historical odds. MLB's only market evidence is the live paper
trade since 2026-08-27, which is positive but small. MLB is unproven against
the market, not proven; running this same backtest for MLB needs a
historical-odds source and is the next thing to do.

### MLB totals against Kalshi (`ops/backtest_mlb_totals_market.py`, run 2026-09-11)

The only market in either sport where the model has not been shown to be
worse than the price. 85 games with both a model and a Kalshi quote.

| Line | Brier, pooled | Brier, 2nd half only | ≥4% paper, pooled | ≥4% paper, 2nd half |
|---|---|---|---|---|
| O/U 7.5 | model 0.2357 < Kalshi 0.2375 | **Kalshi** 0.2323 < model 0.2360 | 26 bets, +10.3% | 12 bets, −23.3% |
| O/U 8.5 | model 0.2398 < Kalshi 0.2439 | **Kalshi** 0.2449 < model 0.2455 | 29 bets, +2.9% | 11 bets, +3.7% |
| O/U 9.5 | Kalshi 0.2273 < model 0.2285 | Kalshi 0.2301 < model 0.2322 | 22 bets, +25.1% | 8 bets, +30.9% |

Read carefully: the pooled Brier advantage at 7.5 and 8.5 is carried
entirely by the first 37 games and **does not replicate on the second 48**,
where Kalshi wins at every line. The positive paper ROIs sit on 8–29 bets
and swing from −100% to +54% as the threshold moves, which is what noise
looks like. One mildly consistent lean: when the model says UNDER, the
over hits 2–4 points less often than Kalshi implies, at all three lines.

**Verdict: unknown, not alive.** Totals is the only market not proven
worse than Kalshi, and the only one worth continuing to track. It is not
bettable on 85 games. It needs several hundred, and it needs the inputs
that actually move totals and that the market may under-price: home-plate
umpire, weather and wind at outdoor parks, posted lineups. Those are the
research direction; nothing else in this file is.

### Totals: umpire and weather (`ops/run_totals.py`, run 2026-09-11)

StatsAPI's schedule hydrates the home-plate umpire and game-time weather
for every game back to 2008 (44,631 games; umpire and temperature 100%,
wind ≥ 93%). Walk-forward 2023–2025, negative binomial, against the
park-adjusted floor:

| Feature set | Brier @7.5 | Brier @8.5 | Brier @9.5 | RMSE | count NLL |
|---|---|---|---|---|---|
| park_adjusted_mean (floor) | — | 0.2499 | — | 4.482 | — |
| previous set (pitchers, park, form) | 0.2424 | 0.2485 | 0.2377 | 4.465 | 2.8633 |
| + umpire + temperature + wind + roof | 0.2416 | 0.2475 | 0.2366 | 4.452 | 2.8605 |
| − umpire | 0.2416 | 0.2476 | 0.2367 | 4.453 | 2.8607 |
| − temperature | 0.2424 | 0.2483 | 0.2376 | 4.463 | 2.8628 |
| − wind | 0.2417 | 0.2476 | 0.2367 | 4.453 | 2.8608 |

The margin over the floor went from 0.0014 to 0.0024 Brier, the first
feature addition to move this model since the pitcher block, and
**temperature is the whole of it**. The raw relationship is a monotone
two-run climb: open-roof games average 8.43 runs below 60°F and 10.56
above 90°F. Its standardised coefficient (+0.035) is second only to park
factor (+0.054). Wind and the umpire are each worth 0.0001, which is noise;
wind stays for its physical sign, the umpire is computed for study but not
shipped (`STUDY_COLS`), since the assignment is also not public until
lineups post, hours after the daily run.

Shipped set (`data/mlb/totals_config.json`): previous columns plus
`temp_c70`, `wind_out`, `roof_closed`. Gate PASS; every populated decile
within 0.049 at all three traded lines.

The inference problem: StatsAPI leaves the weather blank until about first
pitch, so the daily run would have priced every game at 70°F. The rundown
now takes game-hour temperature from the Open-Meteo forecast
(`src/mlb/weather.py`), falling back to the venue's month-of-year history
(R² 0.54 against the actual, residual 7.6°F vs 11.1°F raw) and treating a
venue that closes its roof in half or more of that month's games as closed.
The temperature the model used is written to the narrative log so the
forecast can be checked against the recorded value once the game settles.

What this does not say: it does not say the market misses temperature.
Every sportsbook prices weather. It says the model now has the input, and
the paper trade against Kalshi (`ops/backtest_mlb_totals_market.py`,
re-run nightly) is the only test of whether the price already holds it.

### Where this leaves the mandate

| Market | Result | Status |
|---|---|---|
| NFL moneyline | −12% (linear) / −17% (ensemble), blend weight 0 | dead on this feature set |
| NFL spread | 49.2% vs 52.4% breakeven | dead |
| MLB moneyline | −20% at the HIGH VALUE rule, blend weight 0 | dead on this feature set |
| MLB run line | 40–44% cover rate | dead |
| MLB totals | indistinguishable from Kalshi on 85 games; temperature added 2026-09-11 (Brier 0.2485 → 0.2476) | **track; lineups next** |
| NFL totals | no model | not built |

Both moneyline models share one mechanism: tuned to minimise error against
*outcomes*, they converge on the base rate and sit shrunk toward 50/50,
under-rating favorites by 5–7 points. A market is already calibrated to
outcomes; being calibrated too is not an edge. An edge needs information
the market misprices, and this feature set does not contain any.
