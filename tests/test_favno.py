"""The favorite-fade base rate: buy No on an ordinary favorite's run line.
Pure arithmetic, checked against the 44,456-game constant it is built on."""
import pytest

from src.mlb.rundown import favno_pick, WIN_BY_2_GIVEN_WIN, FAVNO_MIN_FAV, FAVNO_MAX_FAV


def test_picks_the_higher_priced_side_as_favorite():
    pick = favno_pick(0.60, 0.42, "NYY", "BOS", sp_home_bid=0.50, sp_away_bid=0.30)
    assert pick is not None
    team, no_price, p_no = pick
    assert team == "NYY"
    assert no_price == pytest.approx(0.50)                       # 1 - home bid
    assert p_no == pytest.approx(round(1 - WIN_BY_2_GIVEN_WIN * 0.60, 3))


def test_away_favorite_uses_away_bid():
    pick = favno_pick(0.35, 0.65, "COL", "LAD", sp_home_bid=0.20, sp_away_bid=0.55)
    team, no_price, p_no = pick
    assert team == "LAD"
    assert no_price == pytest.approx(0.45)                        # 1 - away bid


def test_near_toss_up_still_gets_a_pick():
    # The favorite is always the higher-priced side, so it is always >= 50%
    # by construction; FAVNO_MIN_FAV documents that floor rather than
    # filtering anything most games would ever hit.
    pick = favno_pick(FAVNO_MIN_FAV, 1 - FAVNO_MIN_FAV, "H", "A", 0.5, 0.5)
    assert pick is not None


def test_heavy_favorite_is_skipped():
    assert favno_pick(FAVNO_MAX_FAV, 1 - FAVNO_MAX_FAV, "H", "A", 0.7, 0.3) is None
    assert favno_pick(FAVNO_MAX_FAV - 0.001, 1 - FAVNO_MAX_FAV + 0.001,
                      "H", "A", 0.69, 0.31) is not None


def test_missing_price_yields_none_not_a_crash():
    assert favno_pick(None, None, "H", "A", 0.5, 0.5) is None
    assert favno_pick(0.6, 0.4, "H", "A", None, None) is None      # no run-line quote


def test_probability_and_edge_are_monotone_in_favorite_strength():
    # A bigger favorite is more likely to cover, so p_no falls as fav_p rises.
    _, _, p_lo = favno_pick(0.55, 0.45, "H", "A", 0.5, 0.5)
    _, _, p_hi = favno_pick(0.65, 0.35, "H", "A", 0.5, 0.5)
    assert p_hi < p_lo
    assert p_lo == pytest.approx(round(1 - WIN_BY_2_GIVEN_WIN * 0.55, 3))


def test_break_even_matches_the_documented_threshold():
    # At fav_p just under FAVNO_MAX_FAV (0.70), p_no should sit just above 0.50 --
    # this is the reasoning in rundown.py's own comment, pinned so it can't drift.
    _, _, p_no = favno_pick(FAVNO_MAX_FAV - 0.001, 1 - FAVNO_MAX_FAV + 0.001,
                            "H", "A", 0.6, 0.4)
    assert p_no == pytest.approx(0.5, abs=0.005)
