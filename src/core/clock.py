"""One definition of "today" for the whole pipeline.

A slate is an Eastern-time notion (a 7:05 PM ET game on the 14th is the
14th's game even though it is the 15th in UTC by the 8th inning). The runners
are UTC. Before this helper, "today" was defined five different ways across
the producers and the checkers -- runner-local UTC date, fixed UTC-4, true
America/New_York -- and every cross-file comparison of those dates
(rundown log date vs health gate, "generated" stamp vs site check) broke
between 00:00 and 04:00 UTC, which is exactly when the watchdog's 22:00 ET
repair run fires. Every producer and checker now calls this.
"""
from __future__ import annotations

import pandas as pd

ZONE = "America/New_York"


def now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=ZONE)


def slate_today() -> pd.Timestamp:
    """Midnight of the current Eastern calendar date, tz-naive, so it compares
    directly against the tz-naive `gameday` columns in games.csv."""
    return now_et().normalize().tz_localize(None)
