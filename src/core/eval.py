"""Evaluation primitives. Honors the NPS CS3315 hard requirements:

- regression_report() is the standard output for any regression model
  (verbatim from gitlab-nps/.../CLAUDE.md:85-137, lightly adapted to
  return the figure for save_figure() instead of plt.show()).
- assert_no_leakage() trips a loud failure on RMSE=0 or R^2~=1.0.
- For probabilistic outputs we add Brier score + reliability diagram,
  which the NPS coursework does not cover but which a betting system requires.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


def regression_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "Model",
    units: str = "",
    show: bool = True,
):
    """Report MSE/RMSE/MAE/MAPE/R^2 + residual plots.

    Returns a dict of metrics and the figure (caller can save_figure() it).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("nan")

    unit_str = f" {units}" if units else ""
    print(f"\n=== Regression Report: {label} ===")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}{unit_str}  <- interpret in original units")
    print(f"  MAE:  {mae:.4f}{unit_str}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R^2:  {r2:.4f}  <- fraction of variance explained")

    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(0, color="red", linewidth=1)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"{label} - Residuals vs Fitted")
    axes[1].hist(residuals, bins=30)
    axes[1].set_xlabel("Residual")
    axes[1].set_title(f"{label} - Residual Distribution")
    fig.tight_layout()
    if show:
        plt.show()

    return {
        "label": label,
        "mse": float(mse),
        "rmse": rmse,
        "mae": float(mae),
        "mape_pct": mape,
        "r2": float(r2),
        "n": int(len(y_true)),
    }, fig


def assert_no_leakage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rmse_floor: float = 1e-6,
    r2_ceiling: float = 0.999,
) -> None:
    """Raise loudly if metrics look impossibly good. Per NPS CLAUDE.md:134."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    if rmse < rmse_floor:
        raise AssertionError(
            f"RMSE={rmse:.2e} is suspiciously close to zero. "
            "Likely data leakage or a degenerate split."
        )
    if r2 > r2_ceiling:
        raise AssertionError(
            f"R^2={r2:.6f} > {r2_ceiling}. Likely data leakage or train/test contamination."
        )


def classification_report_test_only(y_test, y_pred_test, class_names=None) -> str:
    """Print confusion matrix + report for the *test set* only.

    Per NPS CLAUDE.md:140 confusion matrices on training data are forbidden.
    Function name encodes the rule.
    """
    cm = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix - Test Set")
    fig.tight_layout()
    plt.show()
    report = classification_report(y_test, y_pred_test, target_names=class_names)
    print(report)
    return report


# ---------- Probabilistic-output metrics (extension; not in NPS coursework) ----------


def brier_score(y_true_binary: np.ndarray, p_pred: np.ndarray) -> float:
    """Mean squared error between predicted probability and {0,1} outcome.

    Lower is better. A perfectly calibrated, perfectly informative model = 0.
    A constant prediction of base-rate p on data with prevalence p has Brier = p(1-p).
    Compare candidate models against that floor.
    """
    y = np.asarray(y_true_binary).ravel().astype(float)
    p = np.asarray(p_pred).ravel().astype(float)
    return float(np.mean((p - y) ** 2))


def reliability_diagram(
    y_true_binary: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
    label: str = "Model",
):
    """Bin predictions by predicted probability, plot empirical vs stated.

    A perfectly calibrated model lies on y=x. Returns (fig, calibration_table).
    Use to decide whether isotonic / sigmoid calibration is needed before betting.
    """
    y = np.asarray(y_true_binary).ravel().astype(float)
    p = np.asarray(p_pred).ravel().astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append((b, (bins[b] + bins[b + 1]) / 2, float("nan"), float("nan"), 0))
            continue
        stated = float(p[mask].mean())
        empirical = float(y[mask].mean())
        rows.append((b, (bins[b] + bins[b + 1]) / 2, stated, empirical, n))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    xs = [r[2] for r in rows if r[4] > 0]
    ys = [r[3] for r in rows if r[4] > 0]
    ns = [r[4] for r in rows if r[4] > 0]
    ax.scatter(xs, ys, s=[max(20, n * 4) for n in ns], alpha=0.7, label=label)
    ax.set_xlabel("Stated probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Reliability - {label} (Brier={brier_score(y, p):.4f})")
    ax.legend()
    fig.tight_layout()
    plt.show()

    return fig, rows
