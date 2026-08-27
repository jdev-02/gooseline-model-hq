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
| **linear, Tier A + pitcher block** (boxscores 2021–2026, widened Kalman grid) | 7289 | **2.9076** | **4.431** | **0.2438** | 0.071 | **yes** | **yes** |

Embargo check: NLL 2.9075 with `embargo_steps=1` vs 2.9076 without. No
within-series leakage signal. `assert_no_leakage(r2_ceiling=0.60)` passed on
every fold (R² = 0.035).

Frozen config (`data/mlb/model_config.json`, 2026-08-27): Kalman
`obs_var=19, step_q=0.001, season_inflate=0.1, season_revert=0.75`
(HFA learned at 0.04 runs); ridge `lam=1000`, season half-life 2.0;
`RECAL_SCALE=0.975`.

Reading: the gate passes, narrowly. The pitcher block is worth about 0.0011
Brier over the schedule-only model and 0.0007 over Elo-lite; on a 4.4-run
sigma that is a real but small edge, and it is the reason most cards on the
site are passes. Boxscores before 2021 are not yet cached, so pitcher
features for 2015–2020 training rows sit at the league mean; backfilling them
is the next cheap improvement to try.

## NFL

David's model. Walk-forward 2021–2025 season tables live on the site's Track
Record tab (`src/site/nfl_site.py:history_tables`). A `home_always` / Elo
row for NFL in this format is still to be added (`src/nfl/baselines.py`).
