"""The human 'narrative edge' input.

Never a model feature. A per-game qualitative prior (playoff push, a team
drifting, a clubhouse story) is applied to the model's predictive
distribution at rundown time as a bounded shift in the mean that is always
paid for with extra variance: an opinion can move the number but can never
buy confidence. Both the model-only and model+narrative streams are logged
so the track record can say whether the human helped.

data/mlb/narrative/YYYY-MM-DD.yaml:
    date: 2026-08-27
    author: jon
    entries:
      - game: HOU@NYY          # AWAY@HOME
        team: NYY              # who the story favors
        delta_runs: 0.35       # clipped to +/- max_delta
        confidence: 0.6        # 0..1; low confidence widens sigma more
        note: "Yankees in a playoff push; Astros 3-7 L10 and drifting"
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

MAX_DELTA = 1.0
SIGMA_FLOOR = 0.15


FACTORS_PATH = Path("data/mlb/factors.yaml")


def load_factors(path=FACTORS_PATH):
    """-> {factor_id: dict}. The registry of read types and their status."""
    if not Path(path).exists():
        return {}
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {f["id"]: f for f in doc.get("factors", [])}


@dataclass
class NarrativeEntry:
    game: str
    team: str
    delta_runs: float
    confidence: float
    note: str = ""
    author: str = ""
    factor: str = ""
    factor_status: str = "unregistered"

    @property
    def away(self):
        return self.game.split("@")[0].strip().upper()

    @property
    def home(self):
        return self.game.split("@")[1].strip().upper()


def load_narrative(path, factors_path=FACTORS_PATH):
    """-> {(away, home): NarrativeEntry}. Missing/None path -> {}.

    An entry naming a factor whose registry status is `rejected` is loaded
    with its shift zeroed: the read has already been measured against history
    and did not hold, so it must not move a number again under a new note.
    """
    if not path or not Path(path).exists():
        return {}
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    reg = load_factors(factors_path)
    out = {}
    for e in doc.get("entries", []):
        fid = str(e.get("factor", "") or "")
        status = reg.get(fid, {}).get("status", "unregistered" if fid else "")
        delta = float(e.get("delta_runs", 0.0))
        if status == "rejected":
            print(f"narrative: factor '{fid}' is rejected in the registry; "
                  f"zeroing its shift for {e.get('game')}")
            delta = 0.0
        ent = NarrativeEntry(game=e["game"], team=str(e["team"]).upper(),
                             delta_runs=delta,
                             confidence=float(e.get("confidence", 0.5)),
                             note=e.get("note", ""), author=doc.get("author", ""),
                             factor=fid, factor_status=status)
        out[(ent.away, ent.home)] = ent
    return out


def apply_narrative(mu, sigma, home, away, entry, max_delta=MAX_DELTA,
                    sigma_floor=SIGMA_FLOOR):
    """-> (mu_h, sigma_h, signed_shift). sigma_h >= sigma always."""
    if entry is None:
        return float(mu), float(sigma), 0.0
    delta = float(np.clip(entry.delta_runs, -max_delta, max_delta))
    if entry.team == home:
        signed = delta
    elif entry.team == away:
        signed = -delta
    else:
        return float(mu), float(sigma), 0.0
    conf = float(np.clip(entry.confidence, 0.0, 1.0))
    sig_add = abs(signed) * (1.0 - conf) + sigma_floor
    return float(mu + signed), float(np.sqrt(sigma ** 2 + sig_add ** 2)), signed
