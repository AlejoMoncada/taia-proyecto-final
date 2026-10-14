"""Metricas para clasificacion multietiqueta (una probabilidad por clase)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    multilabel_confusion_matrix,
    roc_auc_score,
)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE de una clase con bins de igual ancho."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def tune_thresholds(y: np.ndarray, p: np.ndarray, grid: np.ndarray | None = None) -> np.ndarray:
    """Umbral por clase que maximiza F1. Se usa SOLO con validacion."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    best = []
    for k in range(y.shape[1]):
        scores = [f1_score(y[:, k], p[:, k] >= t, zero_division=0) for t in grid]
        best.append(grid[int(np.argmax(scores))])
    return np.array(best)


def evaluate(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray, classes: list[str]) -> dict:
    pred = (p >= thresholds).astype(int)
    cms = multilabel_confusion_matrix(y, pred)
    per_class = {}
    for k, c in enumerate(classes):
        per_class[c] = {
            "f1": float(f1_score(y[:, k], pred[:, k], zero_division=0)),
            "roc_auc": float(roc_auc_score(y[:, k], p[:, k])),
            "brier": float(brier_score_loss(y[:, k], p[:, k])),
            "ece": expected_calibration_error(y[:, k], p[:, k]),
            "threshold": float(thresholds[k]),
            # [[TN, FP], [FN, TP]]
            "confusion_matrix": cms[k].tolist(),
            "support": int(y[:, k].sum()),
        }
    return {
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc_macro": float(roc_auc_score(y, p, average="macro")),
        "brier_mean": float(np.mean([v["brier"] for v in per_class.values()])),
        "ece_mean": float(np.mean([v["ece"] for v in per_class.values()])),
        "per_class": per_class,
    }


def plot_reliability(y: np.ndarray, p: np.ndarray, classes: list[str], path, n_bins: int = 10) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="calibracion perfecta")
    for k, c in enumerate(classes):
        frac, mean = calibration_curve(y[:, k], p[:, k], n_bins=n_bins)
        ax.plot(mean, frac, marker="o", label=c)
    ax.set_xlabel("probabilidad predicha")
    ax.set_ylabel("frecuencia observada")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cms: dict[str, list], path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(cms), figsize=(3 * len(cms), 3))
    for ax, (c, cm) in zip(axes, cms.items()):
        cm = np.asarray(cm)
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        ax.set_title(c)
        ax.set_xticks([0, 1], ["no", "si"])
        ax.set_yticks([0, 1], ["no", "si"])
        ax.set_xlabel("predicho")
        ax.set_ylabel("real")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
