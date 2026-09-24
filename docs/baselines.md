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

**Caveat found 2026-09-15 (code review):** every UNDER edge in this table
was priced as `1 - over_ask`. A NO contract costs `1 - over_bid`, so the
UNDER side was overstated by the full bid-ask spread (typically 2–6¢), and
the "model says UNDER" rows above are inflated by that amount. The rundown
now prices the under off the bid and logs `mkt_under_X`; this backtest
must be re-run against bid-priced unders (the snapshot db carries bids)
before the totals result is quoted again. The OVER rows are unaffected.

### Totals against Kalshi, under priced off the bid (re-run 2026-09-15, 145 games)

| Line | Brier, pooled | ≥4% paper, under off bid | 1st half | 2nd half |
|---|---|---|---|---|
| O/U 7.5 | Kalshi 0.2326 = model 0.2326 | 37 bets, **−6.2%** | +3.6% | **−17.8%** |
| O/U 8.5 | **Kalshi** 0.2436 < model 0.2448 | 32 bets, +6.8% | +27.6% | **−20.0%** |
| O/U 9.5 | **Kalshi** 0.2449 < model 0.2468 | 35 bets, +7.6% | +31.2% | **−14.7%** |

At 145 games Kalshi's Brier is better at every line, and every line's
second half is negative. The pooled positive ROI is the first half, the
sample the idea was born on. The ask-priced version (the "old" line in the
script) flattered 7.5 by 10 points and the under side generally.

**Verdict: not alive.** No market in either sport has beaten Kalshi.
The site now says so by rule rather than by prose: a market shows BET only
when its live paper trade (`data/mlb/paper_trades.csv`) has ≥100 settled
bets, positive overall and positive over the most recent half
(`src/mlb/labels.py`); SMALL BET (half a unit) is the same test on ≥20 bets;
everything else the model flags is PASS, shown with the market's record on
the card so the disagreement is visible but not acted on. Totals at
79 settled bets (+1.9%, recent +4.0%) is SMALL BET. It is not
bettable on 85 games. It needs several hundred, and it needs the inputs
that actually move totals and that the market may under-price: home-plate
umpire, weather and wind at outdoor parks, posted lineups. Those are the
research direction; nothing else in this file is.

### Late-season totals: Over calls get overconfident (`ops/experiment_late_season.py`, run 2026-09-24)

User's instinct, checked rather than taken on faith: late September has
teams locked into (or out of) playoff seeding, and games stop being played
at full effort — division winners rest starters, eliminated teams run out
call-ups, bullpens get emptied for evaluation rather than the win. None of
that is a feature this model sees. Split 2023-2025 held-out predictions
from the exact frozen configs the site runs (moneyline and the negative-
binomial totals model) into the last 14 days of each season's own schedule
vs. everything earlier:

| Market | Window | n | Brier | Actual rate | Model rate |
|---|---|---|---|---|---|
| Moneyline | early | 6754 | 0.2439 | 53.0% home | 50.7% home |
| Moneyline | last 14 days | 535 | 0.2392 | 50.5% home | 51.2% home |
| Total O/U 7.5 | early | 6754 | 0.2469 | 57.6% over | 59.2% over |
| Total O/U 7.5 | last 14 days | 535 | **0.2617** | **52.2% over** | **61.5% over** |
| Total O/U 8.5 | early | 6754 | 0.2530 | 50.5% over | 49.9% over |
| Total O/U 8.5 | last 14 days | 535 | **0.2579** | **43.2% over** | **52.3% over** |
| Total O/U 9.5 | early | 6754 | 0.2437 | 40.2% over | 41.1% over |
| Total O/U 9.5 | last 14 days | 535 | 0.2395 | 35.9% over | 43.5% over |

Moneyline shows no late-season effect (if anything, marginally better —
535 games is a small, noisy sample either way). **Totals shows a real,
directional failure**: at every line, the real over-rate drops 5–9 points
in the last two weeks of the season while the model's own stated P(over)
goes *up* or holds flat — the model does not adjust for lower-scoring
late-season baseball, and at 7.5/8.5 the Brier cost is well above sampling
noise. The miscalibration has a direction: it is specifically Over calls
that get overconfident, not Under calls.

**Fix shipped same day**: `rundown.py` logs `days_before_season_end` on
every row (last day of that season's own schedule minus the game's date,
so it needs no external clinch/elimination data). `labels.py` caps any
Over total call inside that window one tier down (BET → SMALL BET → PASS)
regardless of the market's aggregate record, and the card's "why" sentence
says so specifically rather than quoting the generic market record. Under
calls in the same window are untouched — nothing here shows they are
harmed, and by the same asymmetry they may be mildly *underconfident*, an
open question for after the season. `LATE_SEASON_DAYS = 14` is a proxy for
"seeding is mostly decided," not a reconstructed clinch date; a sharper
version (per-team, from actual standings) is future work.

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

### Spread / run line, tracked live from 2026-09-15

Until 2026-09-15 the run line (MLB ±1.5) and the NFL spread were shown as a
probability with no Kalshi price, no logged bet and no live record, so a
claim like "the model went 5-1 on the spread" could not be checked. Both
rundowns now price the spread against Kalshi's actual "wins by over X.5"
contracts (`KXMLBSPREAD`, `KXNFLSPREAD`; NFL at the rung nearest the Vegas
line), on all four sides (a +line side is the NO of the opponent's −line
contract, costed at 1 − bid), log `spread_call` / `spread_edge` /
`spread_price`, and paper-trade it as its own market (`SPREAD`). The
record accrues under Live Bet Performance; the card word for the run line
is decided by that record like the other two markets.

### Totals ladder widened to 6.5–12.5 (2026-09-15)

Until this date the rundown priced three rungs, 7.5 / 8.5 / 9.5, which is
where Kalshi's main line sits on an ordinary night. Kalshi lists eleven
rungs (5.5–15.5) and at Coors Field the main line is 10.5 or 11.5, so the
model was pricing only the low tail there and flagging a long-shot Under
8.5 while never looking at the rung being traded. `TOTAL_LINES` is now
6.5–12.5 (the rungs listed on nearly every game; the outer tails stay out
because the count model is least trusted there and they are listed on a
minority of games). The pick is still the best fee-adjusted edge across
the ladder, and the card now says how often the model expects the pick to
win, what the price implies, and where the main line is when the pick sits
elsewhere. **Record continuity:** the 79-bet live totals record was earned
on the three-rung ladder; bets from here on can land on 6.5, 10.5, 11.5
or 12.5. The backtest (`ops/backtest_mlb_totals_market.py`) still evaluates
7.5–9.5 only; re-run it on the full ladder once the log has a month of
wide-ladder rows.

### Team skill block: offensive K%, ISO, staff K% (`ops/experiment_skill_feats.py`, run 2026-09-15)

A public analysis of World Series winners 2010–2025 (season z-scores, lasso
logistic, 15 positives in 450 team seasons) kept wRC+/xwOBA, starter
ERA/xERA, offensive strikeout rate, DRS/OAA, ISO and bullpen swinging-strike
rate. Three of those are free from the boxscores we already hold, so they
were built as per-team EWMAs in the causal feature loop (`SKILL_COLS` in
`src/mlb/features.py`: `off_k_diff` = opponent K/PA − own, `iso_diff`,
`pit_k_diff` = staff K/BF, all oriented positive-favors-home) and put
through the same gate as everything else: tune on 2022, walk forward
2023–2025 weekly refits, same frozen Kalman.

| Arm | NLL | RMSE | Brier | max calib dev |
|---|---|---|---|---|
| A. shipped set, ridge | 2.9071 | 4.428 | 0.2433 | 0.066 |
| B. A + skill, ridge | 2.9072 | 4.429 | 0.2434 | 0.073 |
| C. A + skill, lasso (α=0.002) | 2.9072 | 4.429 | 0.2435 | 0.077 |

The raw signal is there (correlation with home margin over 2015+:
`pit_k_diff` 0.14, `iso_diff` 0.11, `off_k_diff` 0.06) but it is already
carried by the Kalman rating, the FIP block and the form EWMAs; adding it
changes nothing at the fourth decimal and loosens calibration slightly.
The lasso arm, which is the method the source used, prunes to the same
answer. **Not shipped**: the columns are computed and tested
(`tests/test_mlb_skill_feats.py`) but stay out of `MLB_FEATURE_COLS`.

What that analysis found is a *season-level* ranking of who is built for
October, which is not the same question as who wins tonight, and the
market already knows both. The remaining candidates from it need new
ingest: xwOBA / xERA / OAA from Baseball Savant (a per-team-season CSV
export, cheap; per-game splits need the statcast search endpoint), and
bullpen swinging-strike rate needs pitch-level data. Neither is worth
building for the moneyline, which is dead on this feature set regardless;
the only place they could matter is totals, where xwOBA-against by the
probable starter is the one untested input. Logged as the next totals
experiment after lineups.

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
