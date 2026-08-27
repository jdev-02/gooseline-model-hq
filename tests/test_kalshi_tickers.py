"""Kalshi MLB ticker parsing, using strings captured live on 2026-08-27."""
import pytest
from src.core.kalshi import parse_mlb_event, split_mlb_pair, match_mlb_event, mlb_event_key

CASES = [
    ("KXMLBGAME-26AUG271905HOUNYY", "KXMLBGAME-26AUG271905HOUNYY-NYY", "HOU", "NYY", 1),
    ("KXMLBGAME-26AUG291305BOSNYYG1", "KXMLBGAME-26AUG291305BOSNYYG1-BOS", "BOS", "NYY", 1),
    ("KXMLBGAME-26AUG291915BOSNYYG2", "KXMLBGAME-26AUG291915BOSNYYG2-NYY", "BOS", "NYY", 2),
    ("KXMLBGAME-26AUG292205AZSFG2", "KXMLBGAME-26AUG292205AZSFG2-SF", "AZ", "SF", 2),
    ("KXMLBGAME-26AUG282010CWSMIN", "KXMLBGAME-26AUG282010CWSMIN-CWS", "CWS", "MIN", 1),
    ("KXMLBGAME-26AUG271907KCTOR", "KXMLBGAME-26AUG271907KCTOR-TOR", "KC", "TOR", 1),
    ("KXMLBGAME-26AUG281910SDTB", "KXMLBGAME-26AUG281910SDTB-TB", "SD", "TB", 1),
]


@pytest.mark.parametrize("event,market,away,home,gnum", CASES)
def test_parse_live_tickers(event, market, away, home, gnum):
    ev = parse_mlb_event(event, market)
    assert ev["away"] == away and ev["home"] == home and ev["game_number"] == gnum
    assert ev["date"].startswith("2026-08-")


def test_date_and_time():
    ev = parse_mlb_event("KXMLBGAME-26AUG271905HOUNYY", "KXMLBGAME-26AUG271905HOUNYY-HOU")
    assert ev["date"] == "2026-08-27" and ev["time_et"] == "1905"


def test_split_without_suffix():
    assert split_mlb_pair("HOUNYY") == ("HOU", "NYY")
    assert split_mlb_pair("AZSF") == ("AZ", "SF")
    assert split_mlb_pair("SDTB", "TB") == ("SD", "TB")


def test_match_event_by_key():
    key, team = mlb_event_key("KXMLBGAME-26AUG271905HOUNYY", "KXMLBGAME-26AUG271905HOUNYY-NYY")
    prices = {key: {team: {"ask": 0.60}}}
    ev = match_mlb_event(prices, "2026-08-27", "HOU", "NYY")
    assert ev["NYY"]["ask"] == 0.60
    assert match_mlb_event(prices, "2026-08-27", "HOU", "NYY", game_number=2) is None
