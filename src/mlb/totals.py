"""Total-runs model: same data and same machinery as the margin model, with
every feature re-oriented from a difference to a sum.

The margin model asks "who is better and by how much." Totals ask "how much
scoring will this environment produce," which is a different question with a
different answer set: park and the two starters dominate, and team strength
barely matters at all. So the features here are sums (both offenses, both
staffs, the yard) rather than home-minus-away differences.

Target: y_total = home_score + away_score.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from src.mlb.features import _Ewma, _decay, _fip_num, FIP_LEAGUE_DEFAULT, STRIKE_PCT_DEFAULT

TOTAL_FEATURE_COLS = [
    "off_sum",          # both offenses: EWMA runs scored per game, summed
    "def_sum",          # both run-prevention units: EWMA runs allowed, summed
    "pressure_sum",     # baserunners per inning created by both clubs
    "sp_fip_sum",       # both listed starters, shrunk to the league mean
    "sp_k9_sum",        # strikeouts suppress balls in play, so they suppress runs
    "bp_fip_sum",       # both bullpens
    "bp_workload_sum",  # tired pens give up runs
    "log_park_factor",  # the single largest environmental term for totals
    "day_night",
    "roof_or_dh",       # DH-game-2 fatigue flag
    # Weather. Temperature is the one addition since the pitcher block that
    # moved the walk-forward: 2023-2025 Brier at 8.5 from 0.2485 to 0.2475,
    # and the raw relationship is a monotone two-run climb from cold to hot.
    # Wind is kept for its physical sign; it was worth a tenth of temperature.
    "temp_c70",         # (game-time temp - 70F) / 10; zero under a roof
    "wind_out",         # signed mph/10: out to the field +, in from it -; zero under a roof
    "roof_closed",      # dome or roof closed: weather terms are moot
]
# Computed alongside but not shipped: the home-plate umpire's running effect
# on total runs. Ablation on 2023-2025 put it at 0.0001 Brier, and the
# assignment is not public until lineups post, hours after the daily run.
STUDY_COLS = ["ump_runs"]

LEAGUE_TOTAL_DEFAULT = 8.8
K9_DEFAULT = 8.5
UMP_SHRINK_K = 30.0        # games behind an umpire before his number is half his own
UMP_HALF_LIFE = 120.0      # games; zones drift with the rulebook, so not a plain mean
ROOF_CLOSED = {"Roof Closed", "Dome"}
CLIM_MIN_GAMES = 10        # venue-month history before it stands in for a missing temp
ROOF_ASSUME_RATE = 0.5     # unplayed game at a venue that closes this often: assume closed


def _wind_signed(mph, direction):
    """Out To * is positive, In From * negative, cross/varies/calm zero."""
    if mph is None or direction is None or (isinstance(mph, float) and np.isnan(mph)):
        return 0.0
    d = str(direction)
    if d.startswith("Out To"):
        return float(mph)
    if d.startswith("In From"):
        return -float(mph)
    return 0.0


def _nan_to_none(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x


def build_total_features(df, team_lookup=None, pitcher_stats=None, park_fn=None,
                         form_half_life_games=25, pressure_half_life=15,
                         pitch_half_life_starts=6, bp_half_life=30,
                         fip_shrink_k=5.0):
    """Causal single pass, identical discipline to the margin features:
    read state, write the row, then update state from that row's outcome."""
    df = df.copy()
    n = len(df)
    d_form = _decay(form_half_life_games)
    d_press = _decay(pressure_half_life)
    d_sp = _decay(pitch_half_life_starts)
    d_bp = _decay(bp_half_life)

    rs, ra = {}, {}          # team -> EWMA runs scored / allowed
    press_off = {}           # team -> EWMA baserunners per inning created
    sp_fip, sp_k9, sp_last = {}, {}, {}
    bp_fip, bp_ip_log = {}, {}
    lg_num = lg_ip = lg_er = lg_k = 0.0
    # Umpire effect is measured as a residual against the league's running
    # total, so a hitter-friendly era does not read as a hitter-friendly crew.
    d_ump = _decay(UMP_HALF_LIFE)
    ump = {}
    lg_total = _Ewma(_decay(400.0), LEAGUE_TOTAL_DEFAULT)
    has_wx = {"hp_umpire_id", "temp_f", "wind_mph", "wind_dir", "condition"} <= set(df.columns)
    # Fallbacks for games not yet played, whose weather is a blank until first
    # pitch: the venue's own history for that month. Both are expanding over
    # prior games only, like everything else in this loop.
    clim = {}        # (venue_id, month) -> [n, sum of temp_f], open-roof games
    roof_hist = {}   # (venue_id, month) -> [n, n closed]

    pit_by_game = {}
    if pitcher_stats is not None and len(pitcher_stats):
        for key, grp in pitcher_stats.groupby("game_id"):
            pit_by_game[key] = grp

    cols = {c: np.zeros(n) for c in TOTAL_FEATURE_COLS + STUDY_COLS}

    for i, r in enumerate(df.itertuples(index=False)):
        h, a = r.home_team, r.away_team
        lg_c = (lg_er / lg_ip * 9.0 - lg_num / lg_ip) if lg_ip > 100 else 3.10
        lg_fip = FIP_LEAGUE_DEFAULT if lg_ip <= 100 else (lg_num / lg_ip + lg_c)
        lg_k9 = (lg_k / lg_ip * 9.0) if lg_ip > 100 else K9_DEFAULT

        half = LEAGUE_TOTAL_DEFAULT / 2.0
        cols["off_sum"][i] = rs.get(h, half) + rs.get(a, half)
        cols["def_sum"][i] = ra.get(h, half) + ra.get(a, half)
        cols["pressure_sum"][i] = press_off.get(h, 1.4) + press_off.get(a, 1.4)

        def sp(pid):
            e = sp_fip.get(pid)
            fip = (lg_fip if e is None else
                   (e.n * (e.v + lg_c) + fip_shrink_k * lg_fip) / (e.n + fip_shrink_k))
            k = sp_k9.get(pid)
            k9 = (lg_k9 if k is None else
                  (k.n * k.v + fip_shrink_k * lg_k9) / (k.n + fip_shrink_k))
            return fip, k9

        hf, hk = sp(r.home_sp_id)
        af, ak = sp(r.away_sp_id)
        cols["sp_fip_sum"][i] = hf + af
        cols["sp_k9_sum"][i] = hk + ak
        bh, ba = bp_fip.get(h), bp_fip.get(a)
        cols["bp_fip_sum"][i] = ((lg_fip if bh is None else bh.v + lg_c) +
                                 (lg_fip if ba is None else ba.v + lg_c))

        def bp_load(team):
            q = bp_ip_log.get(team, ())
            return sum(ip for gd, ip in q if 0 < (r.gameday - gd).days <= 3)
        cols["bp_workload_sum"][i] = bp_load(h) + bp_load(a)

        cols["log_park_factor"][i] = park_fn(r.venue_id, r.season) if park_fn else 0.0
        cols["day_night"][i] = 1.0 if r.day_night == "night" else 0.0
        cols["roof_or_dh"][i] = 1.0 if int(getattr(r, "game_number", 1)) > 1 else 0.0

        ump_id = _nan_to_none(getattr(r, "hp_umpire_id", None)) if has_wx else None
        if ump_id is not None and ump_id in ump:
            e = ump[ump_id]
            cols["ump_runs"][i] = e.v * e.n / (e.n + UMP_SHRINK_K)
        vm = (r.venue_id, r.gameday.month)
        cond = _nan_to_none(getattr(r, "condition", None)) if has_wx else None
        if cond is not None:
            roof = str(cond) in ROOF_CLOSED
        else:
            rh = roof_hist.get(vm)
            roof = bool(rh and rh[0] >= CLIM_MIN_GAMES and rh[1] / rh[0] >= ROOF_ASSUME_RATE)
        cols["roof_closed"][i] = 1.0 if roof else 0.0
        if has_wx and not roof:
            t = _nan_to_none(getattr(r, "temp_f", None))
            if t is None:
                c = clim.get(vm)
                t = c[1] / c[0] if c and c[0] >= CLIM_MIN_GAMES else None
            cols["temp_c70"][i] = (float(t) - 70.0) / 10.0 if t is not None else 0.0
            cols["wind_out"][i] = _wind_signed(_nan_to_none(getattr(r, "wind_mph", None)),
                                               _nan_to_none(getattr(r, "wind_dir", None))) / 10.0

        if pd.isna(r.result):
            continue
        # ---- update state from this game ----
        hs, as_ = float(r.home_score), float(r.away_score)
        if ump_id is not None:
            ump.setdefault(ump_id, _Ewma(d_ump, 0.0)).push((hs + as_) - lg_total.v)
        lg_total.push(hs + as_)
        if cond is not None:
            rh = roof_hist.setdefault(vm, [0, 0])
            rh[0] += 1
            rh[1] += int(roof)
            t_obs = _nan_to_none(getattr(r, "temp_f", None))
            if not roof and t_obs is not None:
                c = clim.setdefault(vm, [0, 0.0])
                c[0] += 1
                c[1] += float(t_obs)
        rs[h] = d_form * rs.get(h, half) + (1 - d_form) * hs
        rs[a] = d_form * rs.get(a, half) + (1 - d_form) * as_
        ra[h] = d_form * ra.get(h, half) + (1 - d_form) * as_
        ra[a] = d_form * ra.get(a, half) + (1 - d_form) * hs

        if team_lookup is not None:
            for team in (h, a):
                obs = team_lookup.get((r.game_id, team))
                if obs is not None:
                    press_off[team] = (d_press * press_off.get(team, 1.4)
                                       + (1 - d_press) * float(obs.brpi_off))

        grp = pit_by_game.get(r.game_id)
        if grp is not None:
            for p in grp.itertuples(index=False):
                ip = float(p.ip or 0)
                if ip <= 0:
                    continue
                num = _fip_num(p.hr or 0, p.bb or 0, p.hbp or 0, p.so or 0)
                lg_num += num
                lg_ip += ip
                lg_er += float(p.er or 0)
                lg_k += float(p.so or 0)
                if p.is_starter:
                    sp_fip.setdefault(p.pitcher_id, _Ewma(d_sp, 0.0)).push(num / ip)
                    sp_k9.setdefault(p.pitcher_id, _Ewma(d_sp, K9_DEFAULT)).push(
                        float(p.so or 0) / ip * 9.0)
                    sp_last[p.pitcher_id] = r.gameday
                else:
                    bp_fip.setdefault(p.team, _Ewma(d_bp, 0.0)).push(num / ip)
                    bp_ip_log.setdefault(p.team, deque(maxlen=60)).append((r.gameday, ip))

    for c, v in cols.items():
        df[c] = v
    df["y"] = (df["home_score"] + df["away_score"]).astype(float)
    return df


# --------------------------------------------------------------------------
# Baselines. A totals model has to beat the yard and the league average before
# anyone should believe it knows anything about pitching.
# --------------------------------------------------------------------------

class LeagueMeanTotal:
    name = "league_mean_total"

    def fit(self, df):
        y = df["y"].dropna()
        self.mu_, self.sigma_ = float(y.mean()), float(y.std())
        return self

    def predict_dist(self, df):
        n = len(df)
        return np.full(n, self.mu_), np.full(n, self.sigma_)


class ParkAdjustedMeanTotal:
    """League mean scaled by the venue's expanding park factor. This is the
    honest floor: it uses no pitching or batting information at all."""
    name = "park_adjusted_mean"

    def fit(self, df):
        y = df["y"].dropna()
        self.mu_ = float(y.mean())
        resid = y - self.mu_ * np.exp(df.loc[y.index, "log_park_factor"])
        self.sigma_ = float(resid.std())
        return self

    def predict_dist(self, df):
        mu = self.mu_ * np.exp(df["log_park_factor"].values)
        return mu, np.full(len(df), self.sigma_)


def prob_over(mu, sigma, strike):
    """P(total > strike). Kalshi's floor_strike is already the .5 line, so no
    continuity correction is needed: floor_strike 8.5 means 9 runs or more."""
    from scipy.stats import norm
    return 1.0 - norm.cdf((strike - np.asarray(mu)) / np.asarray(sigma))


class NegBinomTotal:
    """Negative-binomial regression on total runs, log link.

    The Gaussian version got the mean right and the shape wrong: runs are
    non-negative integer counts with a fat right tail (big innings cluster),
    so a symmetric bell misprices exactly the outcomes the O/U ladder is
    made of. Poisson alone is too tight because baseball is overdispersed;
    the negative binomial adds the extra variance as a fitted parameter.

    Mean comes from sklearn's PoissonRegressor (a log-link GLM, consistent
    under overdispersion); the dispersion is then fit by method of moments,
    Var = mu + mu^2 / k.
    """

    name = "negbinom_totals"

    def __init__(self, alpha=1e-4, max_iter=300):
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, y, sample_weight=None):
        from sklearn.linear_model import PoissonRegressor
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.x_mean_, self.x_std_ = X.mean(axis=0), X.std(axis=0)
        self.x_std_[self.x_std_ == 0] = 1.0
        Z = (X - self.x_mean_) / self.x_std_
        self.glm_ = PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter)
        self.glm_.fit(Z, y, sample_weight=sample_weight)
        mu = self.glm_.predict(Z)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
        var = np.average((y - mu) ** 2, weights=w)
        mbar = np.average(mu, weights=w)
        excess = var - mbar
        self.k_ = float(mbar ** 2 / excess) if excess > 1e-6 else 1e6
        return self

    def _mu(self, X):
        Z = (np.asarray(X, dtype=float) - self.x_mean_) / self.x_std_
        return self.glm_.predict(Z)

    def predict_dist(self, X):
        """Mean and sd, so the walk-forward harness can score it unchanged."""
        mu = self._mu(X)
        return mu, np.sqrt(mu + mu ** 2 / self.k_)

    def _nb_params(self, X):
        mu = self._mu(X)
        n = self.k_
        return mu, n, n / (n + mu)

    def prob_over(self, X, strike):
        """P(total > strike) under the fitted negative binomial."""
        from scipy.stats import nbinom
        mu, n, p = self._nb_params(X)
        return 1.0 - nbinom.cdf(np.floor(strike), n, p)

    def nll(self, X, y):
        """Per-game negative log-likelihood, comparable across models only
        against another *count* likelihood, never against the Gaussian one."""
        from scipy.stats import nbinom
        mu, n, p = self._nb_params(X)
        return -nbinom.logpmf(np.asarray(y, dtype=int), n, p)
