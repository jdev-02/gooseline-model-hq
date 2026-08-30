# Deriving a bet from the model

The site shows numbers. This is the procedure for turning them into a
decision, in the order the checks actually matter.

## 1. Is the price live?

Look at the card's price age. Anything over 15 minutes shows **STALE PRICE**
and is not a decision — the market moves hardest in the hour before first
pitch. On 2026-08-29 a badge computed against a 46¢ snapshot was still
showing HIGH VALUE when the live ask had moved to 51¢, turning a +5.5% edge
into +0.55%. Refresh before acting, every time.

## 2. Does anything clear 4% after fees?

The edge column already subtracts the Kalshi fee (`0.07 · p · (1−p)`).

| Badge | Edge | What it means |
|---|---|---|
| HIGH VALUE | > 4% | the only tier worth a full unit |
| CAUTIOUS | 0 to 4% | real but inside the noise; not a bet |
| NO VALUE | < 0 | the price is fair or worse |

A CAUTIOUS badge is a pass. It is tempting because it names a side, but the
model's measured edge over a plain Elo baseline is 0.0007 Brier — far too
thin to justify acting on a 2% signal that the next price refresh could
erase.

**Most nights, everything is a pass. That is the design, not a failure.**
Over 2023–2025 the model picked winners 56–57% of the time while the market
priced those same games at roughly the same number.

## 3. Check the market you are actually in

- **Moneyline** is the only market the margin model is validated on.
- **Run line (±1.5)** is graded on every card but the track record is poor:
  40–44% across three seasons. The model predicts *who*, not *by how much*.
  Skip it.
- **Totals** are priced by a separate negative-binomial model that passed its
  own gate (Brier 0.2485 at the 8.5 line, better than league-mean and
  park-adjusted baselines). Its margin over those baselines is 0.0014, so
  the same 4% rule applies and will rarely be met.
- **Player props** have no model at all. Kalshi lists them
  (`KXMLBHIT`, `KXMLBHR`, `KXMLBRBI`, `KXMLBTB`, `KXMLBOUTS`) but nothing
  here prices them. Check the bid–ask spread before assuming the ask is
  meaningful; thin props quote 60/97 and the ask is fiction.

## 4. Run the news check the model cannot

The model is blind to: a starter scratched after the probables posted, three
regulars resting in a day game after a night game, wind, and whether an
unlisted starter is a real arm or a bullpen game. When `sp_unknown` is true
the model has substituted a league-average starter, which usually flatters
the club that is actually piecing the game together.

## 5. Check your read against the factor registry

`data/mlb/factors.yaml` records every read type that has been measured.
Before sizing on an intuition, look it up:

| Read | Status |
|---|---|
| Starter back from a 21–40 day absence | **supported** — fade that club, −0.68 runs |
| Starter back from any other layoff | rejected, no effect at any other band |
| Club shut out in game 1 rebounds in game 2 | rejected |
| Last game was low-scoring so expect the reverse | rejected |
| Both clubs chasing a playoff spot | rejected |

A rejected factor is loaded with its shift zeroed, so a disproven read cannot
move a number again under a new note.

## 6. Size it

Flat 1 unit on HIGH VALUE. Nothing on CAUTIOUS. Half a unit when
`sp_unknown` is true, because the model is guessing at one of the two
starters.

Quarter-Kelly on a 5% edge at even money is roughly 2.5% of bankroll, which
is more than this model has earned: it has no live track record, only
walk-forward backtest. `ops/paper_trade.py` runs daily and will have an
answer by the end of the season.

## Worked example — 2026-08-30, BOS @ NYY

| | Model | Market | Edge |
|---|---|---|---|
| Moneyline | NYY 47.2% | NYY 43¢ | +2.4% |
| Total | 8.14 runs | U8.5 at 55¢ | +2.8% |

Both real, both under 4%, neither is a bet. The model agrees with the market
that Suárez is the better starter; it just thinks 59¢ overcharges for that.
No factor in the registry applies. **Verdict: pass.**
