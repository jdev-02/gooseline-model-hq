"""The arithmetic between a model probability and a dollar. Every formula a
card or a paper trade depends on, pinned as a number, so a change that gets
one wrong fails here instead of on the site. Synthetic inputs only."""
import numpy as np
import pytest
from scipy.stats import norm

from src.core.kalshi import kalshi_fee
from src.mlb.totals import prob_over, NegBinomTotal
from src.site.mlb_page import american, square3_gap


# ---- fee ---------------------------------------------------------------

def test_fee_is_seven_percent_of_p_times_one_minus_p():
    assert kalshi_fee(0.5) == pytest.approx(0.0175)
    assert kalshi_fee(0.25) == pytest.approx(kalshi_fee(0.75))   # symmetric
    assert kalshi_fee(0.0) == 0.0 and kalshi_fee(1.0) == 0.0
    assert max(kalshi_fee(p) for p in np.linspace(0, 1, 101)) == pytest.approx(0.0175)


# ---- moneyline edge, both sides ---------------------------------------

def ml_edge(p_side, ask):
    """The formula src/mlb/rundown.py and src/nfl/rundown.py both use."""
    return p_side - ask - kalshi_fee(ask)


def test_moneyline_edge_uses_each_side_own_ask():
    p_home = 0.60
    home_ask, away_ask = 0.55, 0.47          # sums to 1.02: the vig
    e_home = ml_edge(p_home, home_ask)
    e_away = ml_edge(1 - p_home, away_ask)
    assert e_home == pytest.approx(0.60 - 0.55 - kalshi_fee(0.55))
    assert e_away == pytest.approx(0.40 - 0.47 - kalshi_fee(0.47))
    # Using 1 - home_ask for the away side would overstate it by the vig.
    wrong = ml_edge(1 - p_home, 1 - home_ask)
    assert wrong > e_away


def test_a_fair_price_has_negative_edge_after_fees():
    # Model and market agree exactly: the fee is the whole result.
    assert ml_edge(0.60, 0.60) == pytest.approx(-kalshi_fee(0.60))


# ---- totals: the under is priced off the bid --------------------------

def test_under_cost_is_one_minus_bid_not_one_minus_ask():
    p_over, bid, ask = 0.42, 0.44, 0.50       # six-cent spread
    under_from_bid = (1 - p_over) - (1 - bid) - kalshi_fee(1 - bid)
    under_from_ask = (1 - p_over) - (1 - ask) - kalshi_fee(1 - ask)
    # Pricing off the ask overstates the edge by the spread (plus a fee sliver).
    assert under_from_ask - under_from_bid == pytest.approx(
        (ask - bid) + kalshi_fee(1 - bid) - kalshi_fee(1 - ask))
    assert under_from_ask > 0.04 > under_from_bid    # the exact phantom flag


# ---- paper-trade payout ------------------------------------------------

def paper_pnl(stake, ask, won):
    """ops/paper_trade.py: contracts bought at ask plus fee, each pays $1."""
    contracts = stake / (ask + kalshi_fee(ask))
    return contracts * 1.0 - stake if won else -stake


def test_paper_pnl_arithmetic():
    assert paper_pnl(15.0, 0.50, True) == pytest.approx(15 / (0.5 + 0.0175) - 15)
    assert paper_pnl(15.0, 0.50, False) == -15.0
    # A win at any price below $1 minus fee is a profit; at $1 it is not.
    assert paper_pnl(15.0, 0.10, True) > 0
    assert paper_pnl(15.0, 0.99, True) < 0


def test_break_even_win_rate_equals_price_plus_fee():
    ask = 0.42
    win_needed = ask + kalshi_fee(ask)
    # Expected P&L at exactly that win rate is zero.
    ev = win_needed * paper_pnl(1.0, ask, True) + (1 - win_needed) * paper_pnl(1.0, ask, False)
    assert ev == pytest.approx(0.0, abs=1e-12)


# ---- probabilities from the margin model -------------------------------

def test_home_win_probability_is_symmetric_in_mu():
    sigma, recal = 4.4, 0.9
    p = norm.cdf(1.2 / (recal * sigma))
    q = norm.cdf(-1.2 / (recal * sigma))
    assert p + q == pytest.approx(1.0)
    assert norm.cdf(0.0) == 0.5


def test_gaussian_prob_over_matches_norm_and_is_monotone():
    mu, sigma = 8.8, 4.4
    assert prob_over(mu, sigma, 8.5) == pytest.approx(1 - norm.cdf((8.5 - mu) / sigma))
    ps = [prob_over(mu, sigma, s) for s in (6.5, 7.5, 8.5, 9.5, 10.5)]
    assert all(a > b for a, b in zip(ps, ps[1:]))


def test_negbinom_prob_over_is_a_valid_ladder():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = rng.poisson(8.8 + X[:, 0], size=400).astype(float)
    nb = NegBinomTotal().fit(X, y)
    x = X[:5]
    p75, p85, p95 = (nb.prob_over(x, s) for s in (7.5, 8.5, 9.5))
    assert np.all((0 <= p95) & (p95 <= p85) & (p85 <= p75) & (p75 <= 1))
    mu, sd = nb.predict_dist(x)
    assert np.all(sd ** 2 >= mu - 1e-9)           # overdispersed, never under


# ---- display arithmetic ------------------------------------------------

def test_american_odds():
    assert american(0.5) == "-100"
    assert american(0.6) == "-150"
    assert american(0.4) == "+150"
    assert american(0.75) == "-300"
    assert american(0.0) == "n/a" and american(1.0) == "n/a"


def test_square3_gap_is_antisymmetric_and_zero_at_agreement():
    assert square3_gap(0.6, 0.6) == 0.0
    assert square3_gap(0.6, 0.5) == pytest.approx(-square3_gap(0.5, 0.6))
    assert square3_gap(0.7, 0.5) > square3_gap(0.6, 0.5) > 0
