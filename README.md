# Model HQ — NFL + MLB

Bayesian margin models with Kalman team ratings, evaluated walk-forward, priced
against Kalshi after fees, published as a static site.

The NFL model is David's (`HowlsCastle97/nfl-model-hq`, kept as the `upstream`
remote with history intact): a closed-form Bayesian ridge over the scoring
margin, a joint Kalman filter over team ratings and home-field advantage, a
heteroscedastic deep ensemble, and a fee-aware rundown. This repo keeps that
core verbatim under `src/core/`, mounts the NFL pipeline under `src/nfl/`, and
adds an MLB pipeline under `src/mlb/` built on the same skeleton with
run-differential units and a baseball feature set.

Nothing here is financial advice. One model, honestly uncertain.

## Layout

```
src/core/     models, ensemble, kalman, walkforward, kalshi, eval, splits, plotting
src/nfl/      features, prep_pbp, rundown, runners            (David's, imports repointed)
src/mlb/      ingest, compile, park, features, baselines, narrative, rundown, run_phase0
src/site/     nfl_site (David's website.py), build (combined NFL | MLB page)
ops/          backfill_mlb, kalshi_snapshot
tests/        synthetic-only pytest
data/nfl/     games.csv, team_game_stats.csv
data/mlb/     schedule cache (committed), compiled CSVs, park factors, narrative/
docs/         index.html (GitHub Pages), methodology, baselines, mlb_data
figures/      evaluation figures by phase
```

## Quickstart

```
uv sync --extra dev            # add --extra torch for the deep ensemble (CPU wheel)
uv run pytest

# MLB data (Tier A is enough to run; Tier B adds the pitcher block)
uv run python ops/backfill_mlb.py --schedule --seasons 2008-2026
uv run python ops/backfill_mlb.py --boxscore --seasons 2026,2025,2024,2023,2022,2021

# Evaluate (baselines, walk-forward 2023-2025, calibration) and freeze config
uv run python -m src.mlb.run_phase0

# Log market prices, then price today's slate
uv run python ops/kalshi_snapshot.py
uv run python -m src.mlb.rundown --days 1 --narrative data/mlb/narrative/2026-08-27.yaml

# Publish
uv run python -m src.site.build --mlb-narrative data/mlb/narrative/2026-08-27.yaml
```

## How the MLB model works

1. **Ingest** the MLB StatsAPI schedule (scores, probable pitchers, linescore)
   and, optionally, per-game boxscores. Keyless.
2. **Features**, computed in one causal pass: Kalman rating gap and
   uncertainty; run-differential form and its slope; baserunner pressure,
   stranding rate, late-inning margin; starter FIP, command, command
   consistency, rest; bullpen FIP and workload; park factor, day/night, rest,
   division.
3. **Model**: the same `LinearGaussianModel` as the NFL, giving a margin and a
   sigma (about 4.4 runs). `p_home = Φ(mu / (recal · sigma))`.
4. **Gate**: walk-forward over 2023–2025 must beat `home_always` and `elo_lite`
   on NLL and Brier with calibration within 0.10 (`docs/baselines.md`).
5. **Narrative edge**: a human YAML entry per game shifts the margin by at
   most one run and always widens sigma; both streams are logged and scored.
6. **Rundown**: compare with the latest Kalshi ask minus the 7% fee.
   `HIGH VALUE` means edge above 4% on a price fetched live; most games are passes, by design.

## Process

- `docs/methodology.md` is the governing document: uv only, baselines first,
  walk-forward with embargo, Brier and reliability before any probability is
  bet, `regression_report` after every fit, synthetic data only in tests.
- Conventional Commits, `feat/` `fix/` `chore/` `docs/` branches, PR to `main`.
- `uv run pytest` must be green before pushing.

## Credits

NFL model, Kalman/ensemble design, and the site: David (HowlsCastle97).
Evaluation method and MLB extension: GooseLine Solutions.
