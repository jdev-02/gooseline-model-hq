import numpy as np
import pandas as pd
from src.core.models import LinearGaussianModel, gaussian_nll


def season_decay_weights(train_seasons, asof_season, half_life_seasons):
    age = asof_season - np.asarray(train_seasons, dtype=float)
    if np.isinf(half_life_seasons):
        return np.ones_like(age)
    return 0.5 ** (age / half_life_seasons)


def walk_forward(df, feature_cols, test_season, lam=0.0, half_life_seasons=np.inf,
                 model_factory=None, step_col="week", min_train=100,
                 refit_every=1, embargo_steps=0):
    """Refit on all games strictly before each test step; predict that step.

    model_factory() must return an object with fit(X, y, sample_weight) and
    predict_dist(X). Defaults to LinearGaussianModel(lam).

    step_col     : the time index within a season ("week" for NFL,
                   "day_index" for MLB).
    refit_every  : refit only every Nth test step and reuse the previous fit
                   in between. Still strictly causal; keeps daily sports
                   tractable.
    embargo_steps: exclude games within this many steps before the test step
                   from training (mirrors src.core.splits embargo semantics).
    """
    if model_factory is None:
        model_factory = lambda: LinearGaussianModel(lam=lam)

    out = []
    test_steps = sorted(df.loc[df["season"] == test_season, step_col].unique())
    model = None
    for k, st in enumerate(test_steps):
        test = df[(df["season"] == test_season) & (df[step_col] == st)]
        if len(test) == 0:
            continue
        if model is None or k % refit_every == 0:
            train = df[(df["season"] < test_season) |
                       ((df["season"] == test_season) &
                        (df[step_col] < st - embargo_steps))]
            train = train[train["y"].notna()]
            if len(train) < min_train:
                continue
            sw = season_decay_weights(train["season"].values, test_season,
                                      half_life_seasons)
            model = model_factory()
            model.fit(train[feature_cols].values, train["y"].values, sample_weight=sw)
        mu, sigma = model.predict_dist(test[feature_cols].values)
        chunk = test[["game_id", "season", step_col, "home_team", "away_team", "y"]].copy()
        chunk["mu"] = mu
        chunk["sigma"] = sigma
        out.append(chunk)
    return pd.concat(out, ignore_index=True)


def evaluate(preds):
    r = preds["y"] - preds["mu"]
    return {
        "n": len(preds),
        "nll": float(gaussian_nll(preds["y"].values, preds["mu"].values,
                                  preds["sigma"].values).mean()),
        "rmse": float(np.sqrt((r**2).mean())),
        "mae": float(r.abs().mean()),
    }


def tune(df, feature_cols, val_season, lam_grid, half_life_grid, **kw):
    rows = []
    for hl in half_life_grid:
        for lam in lam_grid:
            preds = walk_forward(df, feature_cols, val_season,
                                 lam=lam, half_life_seasons=hl, **kw)
            m = evaluate(preds)
            rows.append({"half_life_seasons": hl, "lam": lam, **m})
    table = pd.DataFrame(rows).sort_values("nll").reset_index(drop=True)
    best = table.iloc[0]
    return {"half_life_seasons": float(best["half_life_seasons"]),
            "lam": float(best["lam"])}, table
