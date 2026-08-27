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
| linear, Tier A + pitcher block | pending | | | | | | |

Embargo check (Tier A): NLL 2.909 with `embargo_steps=1` vs 2.909 without.
No within-series leakage signal.

Reading: the Tier-A linear model is a hair short of the Brier gate against
Elo-lite. The Kalman grid pinned every parameter at its lower edge (slow
drift, strong between-season reversion), so the grid was widened downward
and the run is being repeated with the pitcher block once the boxscore cache
is complete. Until a row in this table shows both "yes", the MLB verdicts on
the site are for paper tracking only.

## NFL

David's model. Walk-forward 2021–2025 season tables live on the site's Track
Record tab (`src/site/nfl_site.py:history_tables`). A `home_always` / Elo
row for NFL in this format is still to be added (`src/nfl/baselines.py`).
