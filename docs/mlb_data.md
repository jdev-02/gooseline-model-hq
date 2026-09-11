# MLB data

All sources are keyless and public.

## Endpoints

| Purpose | Endpoint | Tier |
|---|---|---|
| Teams | `statsapi.mlb.com/api/v1/teams?sportId=1&season=Y` | A |
| Schedule, scores, probable pitchers, venue, day/night, per-inning linescore, home-plate umpire, game-time weather | `/api/v1/schedule?sportId=1&season=Y&gameType=R&hydrate=probablePitcher,decisions,linescore,venue,weather,officials` | A (1 call/season) |
| Per-game team and per-pitcher lines | `/api/v1/game/{gamePk}/boxscore` | B (1 call/game, ~170 KB) |
| Live probables at rundown time | schedule endpoint with `startDate`/`endDate`, never cached | — |
| Game-hour temperature forecast for upcoming games | `api.open-meteo.com/v1/forecast` (hourly `temperature_2m`, batched by venue lat/lon) | — |
| Market | Kalshi `KXMLBGAME`, `KXMLBSPREAD`, `KXMLBTOTAL` | — |

Tier A alone yields a valid, degraded model (pitcher features fall back to
league mean). Tier B is ~29k requests / ~5 GB and is gitignored;
`ops/backfill_mlb.py --boxscore` regenerates it, resumably.

```
uv run python ops/backfill_mlb.py --schedule --seasons 2008-2026
uv run python ops/backfill_mlb.py --boxscore --seasons 2026,2025,2024,2023,2022,2021 --workers 8
uv run python -c "from src.mlb.compile import *; ..."   # see run_phase0 / README
```

## Files

- `data/mlb/raw/schedule/{season}.json.gz` — committed (~1.3 MB each).
- `data/mlb/raw/boxscore/{season}/{gamePk}.json` — gitignored.
- `data/mlb/games.csv` — one row per game. Key `game_id` = gamePk (unique
  including doubleheaders). `result = home_score - away_score` is the target.
  `played` = Final with both scores. `day_index` = days since that season's
  first game (the Kalman step). `home_rest`/`away_rest` in days (doubleheader
  game 2 = 0). Linescore-derived: `*_hits`, `*_errors`, `*_lob`,
  `*_runs_first6`, `*_runs_late` (innings 7+). Totals inputs from the
  schedule hydrate: `hp_umpire_id`/`hp_umpire`, `temp_f`, `wind_mph`,
  `wind_dir` (`Out To CF`, `In From LF`, `L To R`, `Calm`, `Varies`, ...),
  `condition` (`Roof Closed`/`Dome` mark the weather terms moot). Coverage
  2008–2026: umpire and temperature 100%, wind ≥ 93%. Before first pitch the
  weather and umpire are blank until about first pitch, so
  `rundown.refresh_probables` re-reads both every run and, for the games
  still ahead, takes game-hour temperature from the Open-Meteo forecast
  (`src/mlb/weather.py`, free, no key, one batched call using
  `data/mlb/venues.csv` coordinates). If that fails too, the feature builder
  falls back to the venue's own month-of-year history, and treats a venue
  that has closed its roof in at least half of that month's games as closed.
- `data/mlb/team_game_stats.csv` — two rows per game keyed `(game_id, team)`:
  batting and pitching lines, `p_strike_pct`, `brpi_off/def` (baserunners per
  inning), `lob_rate_off`, late-inning runs.
- `data/mlb/pitcher_game_stats.csv.gz` — one row per pitcher appearance keyed
  `(game_id, pitcher_id)`: `is_starter`, IP/outs, BF, H, HR, BB, HBP, K, ER,
  pitches, strikes, `strike_pct`.
- `data/mlb/park_factors.csv` — `(venue_id, season)` expanding-window factor
  from the prior three completed seasons, shrunk with K=150 games.
- `data/mlb/model_config.json` — Kalman params, `lam`, `recal_scale`, feature
  list from the last `run_phase0`.
- `data/mlb/narrative/YYYY-MM-DD.yaml` + `log.csv` — human tilts and every
  rundown row (model-only and model+narrative), with `result` filled in after
  the game.

## Kalshi tickers

Event `KXMLBGAME-26AUG271905HOUNYY[G2]`: date, ET start HHMM, `AWAYHOME`,
optional doubleheader suffix. Team codes are 2–3 chars so `AZSF`, `CWSMIN`,
`KCTOR`, `SDTB` are ambiguous; the market ticker's trailing team
(`...-NYY`) disambiguates (`src/core/kalshi.py:split_mlb_pair`). Kalshi and
StatsAPI use the same abbreviations today (`AZ CWS KC SD SF TB WSH ATH`).
Join key: `(officialDate, away, home, game_number)`.

Snapshot cadence: `ops/kalshi_snapshot.py` every 15 minutes, 10:00–23:45 ET,
via Task Scheduler. Kalshi keeps postponed markets open ~2 days, so the
rundown only prices games whose status is Scheduled / Pre-Game / Warmup.

## Known gaps

- First-pitch-strike rate needs `/playByPlay`; not ingested (Phase 2).
- Statcast (Baseball Savant CSV export) works but is not on the critical path.
- Retrosheet game logs (`retrosheet.org/gamelogs/glYYYY.zip`) are available for
  pre-2015 history if the Kalman burn-in ever needs it.
