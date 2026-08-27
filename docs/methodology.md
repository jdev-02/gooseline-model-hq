# Methodology

Rules every model in this repo follows. The betting-system extensions were
first written for `crude-edge` and carried over unchanged; the NFL model is
David's coursework (Bayesian ridge, Kalman ratings, deep ensemble) and sets
the pattern the MLB model copies.

## Source-of-truth principles

| Principle | Where it shows up |
|---|---|
| uv only, never pip/conda/venv directly | `pyproject.toml`, `uv sync`, `uv run` |
| `regression_report()` after every regression fit | `src/core/eval.py` |
| Scaler fit on train only | `LinearGaussianModel.fit` standardizes on train; `DeepEnsemble` likewise |
| RMSE=0 or R² near 1 = leakage red flag | `src/core/eval.py:assert_no_leakage` (MLB ceiling 0.60, see below) |
| Baseline first, then complexity | `src/{nfl,mlb}/baselines.py` + `docs/baselines.md` |
| Synthetic data only for plumbing tests | `tests/` never trains a model on real data |

## Extensions for a betting system

### 1. Walk-forward with embargo

`src/core/walkforward.py` refits on every game strictly before the test step
(`week` for NFL, `day_index` for MLB) and predicts that step. `refit_every`
reuses a fit for N steps to keep daily sports tractable; it is still causal.
`embargo_steps` drops the last k steps before the test step from training,
mirroring `src/core/splits.py`. Report the primary run and the embargo run;
if NLL degrades more than 2% under embargo, something inside a series is
leaking and the model does not ship until it is found.

### 2. Calibration before any probability is bet

- **Brier score** against the constant base-rate floor `p(1-p)`.
- **Reliability diagram**: no populated decile (n ≥ 50) may deviate more than
  0.10 from the diagonal. If it does, isotonic/Platt on the *validation*
  season only, never on test.
- `RECAL_SCALE` (a multiplier on sigma before Φ) is fit on the validation
  season by binary NLL and frozen.

If calibration cannot be made acceptable, the model is not safe to bet,
however well it ranks games.

## MLB-specific rules

- **Tune on 2022 only. Headline test 2023–2025. 2026 is partial and labeled so.**
- **Trainable game** = `abstractGameState == "Final"` and both scores present.
  Postponed/cancelled rows are dropped; suspended games keep only the
  resumption row (it carries the final score).
- **Leakage traps named:** `decisions` (W/L/SV) in the schedule payload is
  post-game and is never a feature. `leagueRecord` is confusable with the
  end-of-season record; form is derived from `result` in the causal loop
  instead. Park factors and the league FIP constant are expanding-window.
- **`assert_no_leakage(r2_ceiling=0.60)`**: a legitimate MLB run-differential
  model explains roughly 3–8% of variance. The default 0.999 ceiling would
  never fire on a broken model.
- **Narrative edge is not a feature.** It is applied to the predictive
  distribution at rundown time, bounded to ±1 run, and always widens sigma.
  Both streams are logged (`data/mlb/narrative/log.csv`) and scored
  separately.

## Anti-patterns this repo will not repeat

- Synthetic data for training (the deleted `add_consensus()` in the old
  crude-edge made a backtest meaningless).
- Reporting a metric that was not produced by the harness in this repo.
- A full-sample park factor or season-average stat used as a pre-game feature.
