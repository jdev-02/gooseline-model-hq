import math
import pytest

from src.mlb.narrative import NarrativeEntry, apply_narrative, load_narrative


def _e(team, delta, conf):
    return NarrativeEntry(game="HOU@NYY", team=team, delta_runs=delta, confidence=conf)


def test_home_tilt_moves_mean_toward_home_and_widens_sigma():
    mu, sg, shift = apply_narrative(0.3, 4.4, "NYY", "HOU", _e("NYY", 0.35, 0.6))
    assert shift == pytest.approx(0.35)
    assert mu == pytest.approx(0.65)
    assert sg > 4.4


def test_away_tilt_is_negative():
    mu, sg, shift = apply_narrative(0.3, 4.4, "NYY", "HOU", _e("HOU", 0.5, 0.9))
    assert shift == pytest.approx(-0.5) and mu == pytest.approx(-0.2)


def test_shift_is_bounded():
    _, _, shift = apply_narrative(0.0, 4.4, "NYY", "HOU", _e("NYY", 5.0, 1.0))
    assert abs(shift) <= 1.0


def test_lower_confidence_pays_more_variance():
    _, hi, _ = apply_narrative(0.0, 4.4, "NYY", "HOU", _e("NYY", 0.5, 0.9))
    _, lo, _ = apply_narrative(0.0, 4.4, "NYY", "HOU", _e("NYY", 0.5, 0.1))
    assert lo > hi > 4.4


def test_sigma_never_shrinks_even_at_full_confidence():
    _, sg, _ = apply_narrative(0.0, 4.4, "NYY", "HOU", _e("NYY", 0.0, 1.0))
    assert sg > 4.4  # sigma floor always applies


def test_unrelated_team_is_ignored():
    mu, sg, shift = apply_narrative(0.3, 4.4, "NYY", "HOU", _e("BOS", 0.5, 0.9))
    assert (mu, sg, shift) == (0.3, 4.4, 0.0)


def test_load_yaml(tmp_path):
    p = tmp_path / "n.yaml"
    p.write_text("date: 2026-08-27\nauthor: jon\nentries:\n"
                 "  - game: HOU@NYY\n    team: nyy\n    delta_runs: 0.35\n"
                 "    confidence: 0.6\n    note: push\n", encoding="utf-8")
    d = load_narrative(p)
    e = d[("HOU", "NYY")]
    assert e.team == "NYY" and e.home == "NYY" and e.author == "jon"
    assert load_narrative(None) == {}
