"""The four ways to bet one spread line, priced the way Kalshi sells them."""
import pytest
from scipy.stats import norm

from src.core.kalshi import spread_sides, mlb_spread_key, nfl_spread_key, kalshi_fee


def test_four_sides_and_their_costs():
    quotes = {"TEX:1.5": {"bid": 0.35, "ask": 0.37}, "BOS:1.5": {"bid": 0.33, "ask": 0.34}}
    sides = dict((c, (p, cost)) for c, p, cost in
                 spread_sides(quotes, "TEX", "BOS", 1.5, 0.40, 0.30))
    assert sides["TEX -1.5"] == (0.40, 0.37)            # own YES ask
    assert sides["BOS -1.5"] == (0.30, 0.34)
    assert sides["TEX +1.5"] == (pytest.approx(0.70), pytest.approx(0.67))  # NO of BOS -1.5: 1 - bid
    assert sides["BOS +1.5"] == (pytest.approx(0.60), pytest.approx(0.65))  # NO of TEX -1.5: 1 - bid


def test_plus_side_probability_is_complement_of_opponent_minus_side():
    mu, sigma, line = 0.6, 4.4, 1.5
    p_home_cover = 1 - norm.cdf((line - mu) / sigma)      # home wins by > 1.5
    p_away_cover = norm.cdf((-line - mu) / sigma)         # away wins by > 1.5
    sides = dict((c, p) for c, p, _ in spread_sides(
        {"H:1.5": {"bid": .3, "ask": .32}, "A:1.5": {"bid": .3, "ask": .32}},
        "H", "A", line, p_home_cover, p_away_cover))
    assert sides["H +1.5"] == pytest.approx(1 - p_away_cover)
    assert sides["A +1.5"] == pytest.approx(1 - p_home_cover)
    # Probability mass is consistent: P(H covers -1.5) + P(A covers +1.5) == 1.
    assert sides["H -1.5"] + sides["A +1.5"] == pytest.approx(1.0)


def test_missing_quotes_yield_no_sides_not_errors():
    assert spread_sides({}, "H", "A", 1.5, 0.4, 0.3) == []
    assert spread_sides({"H:1.5": {"bid": None, "ask": None}}, "H", "A", 1.5, .4, .3) == []


def test_ticker_parsers():
    key, sub = mlb_spread_key("KXMLBSPREAD-26SEP152005BOSTEX", "KXMLBSPREAD-26SEP152005BOSTEX-TEX2")
    assert key == ("2026-09-15", "BOS", "TEX", 1) and sub == "TEX:1.5"
    key, sub = nfl_spread_key("KXNFLSPREAD-26SEP21NYGLAR", "KXNFLSPREAD-26SEP21NYGLAR-NYG8")
    assert key == "NYGLAR" and sub == "NYG:7.5"
    assert mlb_spread_key("KXMLBGAME-26SEP152005BOSTEX", "x-TEX") is None


def test_spread_edge_after_fee_is_the_same_formula_as_every_other_market():
    p, cost = 0.55, 0.47
    assert p - cost - kalshi_fee(cost) == pytest.approx(0.55 - 0.47 - 0.07 * 0.47 * 0.53)
