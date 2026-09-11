# Syncing with David's NFL repo

`HowlsCastle97/nfl-model-hq` is the source of truth for the NFL model. This
repo carries a restructured copy under `src/nfl/` and `src/site/nfl_site.py`.
The `upstream-watch` workflow checks weekly and keeps one issue open listing
commits landed upstream since the SHA in `.upstream-reviewed`. Porting is a
human step, because his model-logic changes must pass `ops/backtest_nfl_ml.py`
before they ship here.

## Reviewed through `a9e6296` (2026-09-10)

**Ported**
- `features.py`: `epa_games`, `qb_fam_home/away`, opt-in `epa_season_revert`
  / `form_season_revert` knobs (default 1.0, no behaviour change)
- `rundown.py`: `neutral_row`, `feature_contributions` (ablation-based
  per-game explainer against the deployed ensemble)
- `prep_pbp.py`: `--refresh-latest` incremental EPA mode, replacing
  build-once-and-cache-forever
- His `weekly-site.yml` games.csv fetch-and-validate guard, adapted into
  `nfl-daily.yml`

**Deliberately not ported, and why**
- `website.py` redesign (~1,100 lines): our site diverged into a combined
  NFL+MLB dark theme; his card copy and kickoff ordering are worth mining
  for MLB parity, not wholesale replacement
- `healthcheck.py`, `publish_prices.py`, `prices.yml`, `install_*.ps1`,
  `run_*.cmd`: his local-machine price logger and watchdog. We run live
  prices at render time from GitHub Actions instead
- `weekly-site.yml` cadence: ours runs daily
- `CLAUDE.md`, `requirements.txt`: his tooling; we use uv/pyproject and
  keep no agent-process files in this repo
- `test_healthcheck.py`, `test_card_render.py`, `test_summary_fuzz.py`:
  test code we did not port
- `test_prep_pbp.py`, `test_select_week.py`: worth porting once paths are
  adapted to `src.nfl.*`; not done yet
- `experiment_epa.py`: his experiment; the knobs it exercises are ported,
  the script is not

**Finding that post-dates his commits:** `ops/backtest_nfl_ml.py` shows the
NFL model's moneyline edges lose 12–17% against Vegas over 2021–2025 and
the model adds no information to the market (blend weight 0). See
`docs/baselines.md`. This is not in his repo yet.
