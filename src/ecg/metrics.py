"""Metricas para clasificacion multietiqueta (una probabilidad por clase)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    multilabel_confusion_matrix,
    roc_auc_score,
)
#EJMO-Ini
import torch
from torch import nn, optim
#EJMO-Fin

#EJMO-Ini
def find_best_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """
    Encuentra la temperatura escalar óptima para recalibrar el modelo.
    Se debe usar SOLO con los logits del conjunto de validación.
    """
    # Convertimos los arreglos de numpy a tensores de PyTorch
    logits_tensor = torch.tensor(val_logits, dtype=torch.float32)
    labels_tensor = torch.tensor(val_labels, dtype=torch.float32)
    
    # Parámetro de temperatura inicializado en 1.0
    temperature = nn.Parameter(torch.ones(1))
    
    # Usamos L-BFGS, estándar para Temperature Scaling por su convergencia rápida
    nll_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    
    def eval_loss():
        optimizer.zero_grad()
        # Escala los logits por la temperatura y calcula el BCE
        loss = nll_criterion(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss
        
    optimizer.step(eval_loss)
    
    # Retornamos el valor numérico (float) de la mejor temperatura
    return float(temperature.item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Aplica la temperatura calculada a los logits y los transforma 
    en probabilidades finales [0, 1].
    """
    scaled_logits = torch.tensor(logits, dtype=torch.float32) / temperature
    # Convertimos a probabilidades usando sigmoide y devolvemos a numpy
    probs = torch.sigmoid(scaled_logits).numpy()
    return probs
#EJMO-Fin

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

#EJMO-Ini
def bootstrap_patient_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, patient_ids: np.ndarray, 
                              thresholds: np.ndarray, n_iterations: int = 1000) -> dict:
    """
    Realiza un bootstrap estratificado por paciente sobre predicciones fijas 
    para obtener intervalos de confianza del 95%.
    """
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    
    f1_scores = []
    auc_scores = []
    
    # Pre-calcular los índices de cada paciente para acelerar el ciclo
    patient_to_idx = {p: np.where(patient_ids == p)[0] for p in unique_patients}
    
    for _ in range(n_iterations):
        # 1. Remuestreo aleatorio CON reemplazo de pacientes (no de registros)
        boot_patients = np.random.choice(unique_patients, size=n_patients, replace=True)
        
        # 2. Reconstruir los índices basados en los pacientes elegidos
        boot_indices = np.concatenate([patient_to_idx[p] for p in boot_patients])
        
        # 3. Filtrar las etiquetas y probabilidades usando el remuestreo
        y_true_boot = y_true[boot_indices]
        y_probs_boot = y_pred_probs[boot_indices]
        y_pred_boot = (y_probs_boot >= thresholds).astype(int)
        
        # 4. Calcular métricas (ignorando si en una muestra muy rara falta una clase)
        try:
            f1 = f1_score(y_true_boot, y_pred_boot, average="macro", zero_division=0)
            auc = roc_auc_score(y_true_boot, y_probs_boot, average="macro")
            f1_scores.append(f1)
            auc_scores.append(auc)
        except ValueError:
            continue
            
    # Extraer media e intervalos de confianza (percentiles 2.5 y 97.5)
    return {
        "f1_mean": float(np.mean(f1_scores)),
        "f1_ci": [float(x) for x in np.percentile(f1_scores, [2.5, 97.5])],
        "auc_mean": float(np.mean(auc_scores)),
        "auc_ci": [float(x) for x in np.percentile(auc_scores, [2.5, 97.5])]
    }
#EJMO-Fin