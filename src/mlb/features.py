"""MLB feature construction. Same causal single-pass loop as the NFL
features: for each game, read state -> write features -> only then update
state from that game's outcome. Nothing a feature sees was observed after
the first pitch of that game.

Target y = home run differential. Every feature is oriented so that a
positive value favors the home team.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

MOMENTUM_COLS = ["rd_ewma_diff", "rd_slope_diff", "pressure_diff",
                 "lob_rate_diff", "late_inning_diff"]
PITCHER_COLS = ["sp_fip_diff", "sp_command_diff", "sp_command_consistency_diff",
                "sp_rest_diff", "sp_short_il_diff", "bp_fip_diff",
                "bp_workload_diff"]

# A starter back from a 21-40 day absence underperforms the model by about
# 0.68 runs (n=229, t=-2.41; ops/test_factor.py --factor starter_returning
# --sweep). Long enough to be a real injury, short enough that he is back
# before he is right. sp_rest_diff clips at 7 days and cannot see this.
SHORT_IL_BAND = (21, 40)
CONTEXT_COLS = ["log_park_factor", "day_night", "rest_diff", "div_game"]
KALMAN_COLS = ["kalman_diff", "kalman_var"]
TIER_A_COLS = KALMAN_COLS + MOMENTUM_COLS + CONTEXT_COLS
MLB_FEATURE_COLS = TIER_A_COLS + PITCHER_COLS

FIP_LEAGUE_DEFAULT = 4.20
STRIKE_PCT_DEFAULT = 0.64


def _decay(h):
    return 0.5 ** (1.0 / h)


def _slope(hist):
    n = len(hist)
    if n < 3:
        return 0.0
    x = np.arange(n) - (n - 1) / 2.0
    y = np.asarray(hist, dtype=float)
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def _fip_num(hr, bb, hbp, k):
    return 13.0 * hr + 3.0 * (bb + hbp) - 2.0 * k


class _Ewma:
    __slots__ = ("d", "v", "n")

    def __init__(self, d, v0=0.0):
        self.d, self.v, self.n = d, v0, 0

    def push(self, x):
        self.v = self.d * self.v + (1 - self.d) * x
        self.n += 1


def build_features(df, team_lookup=None, pitcher_stats=None, park_fn=None,
                   form_half_life_games=20, pressure_half_life=15,
                   pitch_half_life_starts=6, bp_half_life=30, slope_window=10,
                   rest_clip=(0, 4), sp_rest_clip=(3, 7), fip_shrink_k=5.0,
                   cmd_shrink_k=3.0):
    df = df.copy()
    n = len(df)
    d_form = _decay(form_half_life_games)
    d_press = _decay(pressure_half_life)
    d_sp = _decay(pitch_half_life_starts)
    d_bp = _decay(bp_half_life)

    # --- team state (Tier A; derivable from the schedule/linescore alone) ---
    rd = {}                      # team -> EWMA run diff
    rd_hist = {}                 # team -> deque of last run diffs
    late = {}                    # team -> EWMA late-inning run diff
    press = {}                   # team -> (EWMA brpi_off, EWMA brpi_def)
    lobr = {}                    # team -> EWMA lob rate
    # --- pitcher state (Tier B) ---
    sp_fip = {}                  # pitcher_id -> _Ewma of FIP numerator/IP
    sp_cmd = {}                  # pitcher_id -> _Ewma strike pct
    sp_cmd_var = {}              # pitcher_id -> _Ewma |strike - ewma|
    sp_last = {}                 # pitcher_id -> last gameday
    bp_fip = {}                  # team -> _Ewma bullpen FIP
    bp_ip_log = {}               # team -> deque[(gameday, relief_ip)]
    # expanding league FIP constant
    lg_num = lg_ip = lg_er = 0.0

    # pre-index pitcher appearances per game
    pit_by_game = {}
    if pitcher_stats is not None and len(pitcher_stats):
        for key, grp in pitcher_stats.groupby("game_id"):
            pit_by_game[key] = grp

    cols = {c: np.zeros(n) for c in MOMENTUM_COLS + PITCHER_COLS + ["log_park_factor"]}

    def g(dct, k, default=0.0):
        return dct.get(k, default)

    for i, r in enumerate(df.itertuples(index=False)):
        h, a = r.home_team, r.away_team
        # ---------- read state ----------
        cols["rd_ewma_diff"][i] = g(rd, h) - g(rd, a)
        cols["rd_slope_diff"][i] = _slope(rd_hist.get(h, [])) - _slope(rd_hist.get(a, []))
        ph, pa = press.get(h, (0.0, 0.0)), press.get(a, (0.0, 0.0))
        cols["pressure_diff"][i] = (ph[0] - ph[1]) - (pa[0] - pa[1])
        cols["lob_rate_diff"][i] = g(lobr, a) - g(lobr, h)
        cols["late_inning_diff"][i] = g(late, h) - g(late, a)
        cols["log_park_factor"][i] = park_fn(r.venue_id, r.season) if park_fn else 0.0

        lg_c = (lg_er / lg_ip * 9.0 - lg_num / lg_ip) if lg_ip > 100 else FIP_LEAGUE_DEFAULT - 0.0
        lg_fip = FIP_LEAGUE_DEFAULT if lg_ip <= 100 else (lg_num / lg_ip + lg_c)

        def sp_feats(pid):
            e = sp_fip.get(pid)
            fip = lg_fip if e is None else (e.n * (e.v + lg_c) + fip_shrink_k * lg_fip) / (e.n + fip_shrink_k)
            c = sp_cmd.get(pid)
            cmd = STRIKE_PCT_DEFAULT if c is None else (c.n * c.v + cmd_shrink_k * STRIKE_PCT_DEFAULT) / (c.n + cmd_shrink_k)
            cv = sp_cmd_var.get(pid)
            cvar = 0.03 if cv is None else cv.v
            last = sp_last.get(pid)
            gap = None if last is None else (r.gameday - last).days
            rest = 5.0 if gap is None else float(np.clip(gap, *sp_rest_clip))
            short_il = float(gap is not None
                             and SHORT_IL_BAND[0] <= gap <= SHORT_IL_BAND[1])
            return fip, cmd, cvar, rest, short_il

        hf, hc, hv, hr_, hil = sp_feats(r.home_sp_id)
        af, ac, av, ar_, ail = sp_feats(r.away_sp_id)
        cols["sp_fip_diff"][i] = af - hf
        cols["sp_command_diff"][i] = hc - ac
        cols["sp_command_consistency_diff"][i] = av - hv
        cols["sp_rest_diff"][i] = hr_ - ar_
        # positive favors home: the away starter is the compromised one
        cols["sp_short_il_diff"][i] = ail - hil
        bh, ba = bp_fip.get(h), bp_fip.get(a)
        cols["bp_fip_diff"][i] = (lg_fip if ba is None else ba.v + lg_c) - (lg_fip if bh is None else bh.v + lg_c)

        def bp_load(team):
            q = bp_ip_log.get(team, ())
            return sum(ip for gd, ip in q if 0 < (r.gameday - gd).days <= 3)
        cols["bp_workload_diff"][i] = bp_load(a) - bp_load(h)

        if pd.isna(r.result):
            continue
        # ---------- update state from this game's outcome ----------
        margin = float(r.result)
        for team, m in ((h, margin), (a, -margin)):
            rd[team] = d_form * g(rd, team) + (1 - d_form) * m
            rd_hist.setdefault(team, deque(maxlen=slope_window)).append(m)
        hl = (r.home_runs_late or 0) - (r.away_runs_late or 0)
        late[h] = d_form * g(late, h) + (1 - d_form) * hl
        late[a] = d_form * g(late, a) + (1 - d_form) * (-hl)

        if team_lookup is not None:
            for team in (h, a):
                obs = team_lookup.get((r.game_id, team))
                if obs is None:
                    continue
                po, pdf_ = press.get(team, (0.0, 0.0))
                press[team] = (d_press * po + (1 - d_press) * float(obs.brpi_off),
                               d_press * pdf_ + (1 - d_press) * float(obs.brpi_def))
                lobr[team] = d_form * g(lobr, team) + (1 - d_form) * float(obs.lob_rate_off)
        else:
            # Tier-A proxy: hits + LOB from the linescore
            for team, hits, lob in ((h, r.home_hits, r.home_lob), (a, r.away_hits, r.away_lob)):
                if hits is None or pd.isna(hits):
                    continue
                inn = max(1.0, float(r.innings_played or 9))
                po, pdf_ = press.get(team, (0.0, 0.0))
                press[team] = (d_press * po + (1 - d_press) * float(hits) / inn, pdf_)
                if lob is not None and not pd.isna(lob):
                    lobr[team] = d_form * g(lobr, team) + (1 - d_form) * float(lob) / max(1.0, float(hits))

        grp = pit_by_game.get(r.game_id)
        if grp is not None:
            for p in grp.itertuples(index=False):
                ip = float(p.ip or 0)
                num = _fip_num(p.hr or 0, p.bb or 0, p.hbp or 0, p.so or 0)
                lg_num += num
                lg_ip += ip
                lg_er += float(p.er or 0)
                if p.is_starter:
                    if ip > 0:
                        sp_fip.setdefault(p.pitcher_id, _Ewma(d_sp, 0.0)).push(num / ip)
                    if not pd.isna(p.strike_pct):
                        c = sp_cmd.setdefault(p.pitcher_id, _Ewma(d_sp, STRIKE_PCT_DEFAULT))
                        dev = abs(float(p.strike_pct) - c.v)
                        c.push(float(p.strike_pct))
                        sp_cmd_var.setdefault(p.pitcher_id, _Ewma(d_sp, 0.03)).push(dev)
                    sp_last[p.pitcher_id] = r.gameday
                else:
                    if ip > 0:
                        bp_fip.setdefault(p.team, _Ewma(d_bp, 0.0)).push(num / ip)
                    bp_ip_log.setdefault(p.team, deque(maxlen=40)).append((r.gameday, ip))

    for c, v in cols.items():
        df[c] = v
    df["day_night"] = (df["day_night"] == "night").astype(int)
    hr = df["home_rest"].clip(*rest_clip).fillna(1)
    ar = df["away_rest"].clip(*rest_clip).fillna(1)
    df["rest_diff"] = hr - ar
    df["div_game"] = df["div_game"].fillna(0).astype(int)
    df["y"] = df["result"].astype(float)
    return df
