"""The word is an instruction, decided by the record. Synthetic ledgers only."""
import pandas as pd
import pytest

from src.mlb import labels


def _ledger(tmp_path, monkeypatch, rows):
    p = tmp_path / "paper_trades.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    monkeypatch.setattr(labels, "DATA", tmp_path)
    return p


def _rows(market, n, pnl_each, start="2026-08-01"):
    dates = pd.date_range(start, periods=n, freq="D").strftime("%Y-%m-%d")
    return [dict(date=d, market=market, stake=1.0, pnl=pnl_each, won=int(pnl_each > 0))
            for d in dates]


def _row(total_call="OVER 9.5", total_edge=0.06, verdict="HIGH VALUE &mdash; SF", edge=0.05,
         spread_call="SF -1.5", spread_edge=0.05):
    return dict(home="STL", away="SF", verdict=verdict, edge=edge, mkt_home=0.62, mkt_away=0.39,
                spread_call=spread_call, spread_edge=spread_edge, total_call=total_call,
                total_edge=total_edge)


def test_no_ledger_means_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(labels, "DATA", tmp_path)
    assert labels.action("TOTAL") == "pass"
    assert labels.tier_for(_row()) == "pass"
    assert labels.label_for(_row()) == "PASS"
    assert labels.ranked([_row()]) == []


def test_small_bet_needs_twenty_profitable(tmp_path, monkeypatch):
    _ledger(tmp_path, monkeypatch, _rows("TOTAL", 19, 0.1))
    assert labels.action("TOTAL") == "pass"
    _ledger(tmp_path, monkeypatch, _rows("TOTAL", 20, 0.1))
    assert labels.action("TOTAL") == "small"
    assert labels.label_for(_row()) == "SMALL BET · Over 9.5 runs"


def test_bet_needs_hundred_and_recent_half_positive(tmp_path, monkeypatch):
    _ledger(tmp_path, monkeypatch, _rows("TOTAL", 100, 0.1))
    assert labels.action("TOTAL") == "bet"
    # profitable overall, losing lately: back to pass, not small
    rows = _rows("TOTAL", 60, 0.5) + _rows("TOTAL", 40, -0.2, start="2026-10-01")
    _ledger(tmp_path, monkeypatch, rows)
    assert labels.action("TOTAL") == "pass"


def test_headline_prefers_the_higher_word_then_fixed_order(tmp_path, monkeypatch):
    _ledger(tmp_path, monkeypatch, _rows("TOTAL", 30, 0.1) + _rows("ML", 30, -0.1))
    h = labels.headline(_row())
    assert h[0] == "TOTAL" and h[1] == "small"
    # with totals also pass, the moneyline (first in fixed order) leads
    _ledger(tmp_path, monkeypatch, _rows("TOTAL", 30, -0.1) + _rows("ML", 30, -0.1))
    assert labels.headline(_row())[0] == "ML"
    assert labels.tier_for(_row()) == "pass"


def test_late_season_over_is_capped_but_under_is_not(tmp_path, monkeypatch):
    # TOTAL earns a full BET on its own record...
    _ledger(tmp_path, monkeypatch, _rows("TOTAL", 120, 0.05))
    assert labels.action("TOTAL") == "bet"
    over_late = _row(total_call="OVER 9.5", total_edge=0.06)
    over_late["days_before_season_end"] = 5
    under_late = _row(total_call="UNDER 8.5", total_edge=0.06)
    under_late["days_before_season_end"] = 5
    over_early = _row(total_call="OVER 9.5", total_edge=0.06)
    over_early["days_before_season_end"] = 30
    # ...but an Over inside the last 14 days is capped one tier down,
    # an Under in the same window is not, and the same Over earlier in
    # the season is not either.
    assert labels.tier_for(over_late) == "small"
    assert labels.tier_for(under_late) == "bet"
    assert labels.tier_for(over_early) == "bet"
    assert "last 14 days" in labels.totals_why(over_late, "Over 9.5 runs")
    assert labels.totals_why(under_late, "Under 8.5 runs") == labels.why_not_bet("TOTAL")

    # A SMALL BET record caps to pass, not to "small" twice over.
    _ledger(tmp_path, monkeypatch, _rows("TOTAL", 25, 0.05))
    assert labels.action("TOTAL") == "small"
    assert labels.tier_for(over_late) == "pass"


def test_ranked_orders_by_word_then_market_return_then_edge(tmp_path, monkeypatch):
    _ledger(tmp_path, monkeypatch,
            _rows("TOTAL", 120, 0.05) + _rows("SPREAD", 30, 0.2) + _rows("ML", 30, -0.1))
    a = _row(total_edge=0.05, spread_edge=0.09)
    b = _row(total_edge=0.08, spread_edge=0.05)
    out = labels.ranked([a, b])
    kinds = [(f[0], f[1], f[3]) for _, f in out]
    # BET (totals) first regardless of the spread's larger return, then by edge
    assert kinds[:2] == [("TOTAL", "bet", 0.08), ("TOTAL", "bet", 0.05)]
    assert kinds[2:] == [("SPREAD", "small", 0.09), ("SPREAD", "small", 0.05)]
    assert all(k[0] != "ML" for k in kinds)
