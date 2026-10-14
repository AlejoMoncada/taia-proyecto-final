"""Comparacion final de modelos a partir de resultados ya generados (no entrena nada)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecg.data import DATASET_DIRNAME, load_metadata, split_by_folds
from ecg.utils import load_config, resolve

MODELS = ["random_forest", "xgboost", "xgboost_tuned", "cnn1d", "cnn_lstm"]
DEEP = ["cnn1d", "cnn_lstm"]
SEEDS = [42, 43, 44]

# Rutas relativas a outputs/
SUMMARY = {
    "random_forest": "baseline/seeds/summary_seeds.csv",
    "xgboost": "baseline/seeds/summary_seeds.csv",
    "xgboost_tuned": "baseline/search/xgboost/final/summary_seeds.csv",
    "cnn1d": "deep/final/cnn1d/summary_seeds.csv",
    "cnn_lstm": "deep/final/cnn_lstm/summary_seeds.csv",
}
# Familia de busqueda -> (trials.csv, columna con segundos por trial)
TRIALS = {
    "xgboost": ("baseline/search/xgboost/trials.csv", "fit_seconds"),
    "cnn1d": ("deep/search/cnn1d/trials.csv", "train_seconds"),
    "cnn_lstm": ("deep/search/cnn_lstm/trials.csv", "train_seconds"),
}
SEARCH_OF = {"xgboost_tuned": "xgboost", "cnn1d": "cnn1d", "cnn_lstm": "cnn_lstm"}
# Modelo -> carpeta con test_proba.npy (seed 42), alineado con el split test
PROBA = {
    "random_forest": "baseline/seeds/seed_42/random_forest",
    "xgboost": "baseline/seeds/seed_42/xgboost",
    "xgboost_tuned": "baseline/search/xgboost/final/seed_42",
    "cnn1d": "deep/final/cnn1d/seed_42",
    "cnn_lstm": "deep/final/cnn_lstm/seed_42",
}


def format_pm(mean: float, std: float | None, decimals: int) -> str:
    """Texto 'media ± desviacion'; sin desviacion muestra solo la media."""
    if std is None or pd.isna(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def read_summary(path: Path) -> pd.DataFrame:
    """Lee summary_seeds.csv y devuelve columnas planas <metrica>_mean / <metrica>_std."""
    if "model" in pd.read_csv(path, nrows=0).columns:
        return pd.read_csv(path)
    # Formato de DataFrame.to_csv con columnas MultiIndex (metrica, estadistico)
    raw = pd.read_csv(path, header=[0, 1], index_col=[0, 1])
    raw.columns = [f"{m}_{s}" for m, s in raw.columns]
    return raw.reset_index()


def _pick(df: pd.DataFrame, model: str, split: str) -> pd.Series:
    """Fila del modelo y split. Si el archivo trae un solo modelo, se usa esa fila."""
    sub = df[df["split"] == split]
    if model in set(sub["model"]):
        return sub[sub["model"] == model].iloc[0]
    if sub["model"].nunique() == 1:
        return sub.iloc[0]
    raise ValueError(f"No encuentro '{model}' en el split {split}")


def select_rows(summaries: dict[str, Path], split: str) -> pd.DataFrame:
    """Una fila por modelo (indice = nombre) con todas las columnas de metricas."""
    rows = {}
    for name, path in summaries.items():
        s = _pick(read_summary(path), name, split).drop(["model", "split"])
        rows[name] = s
    return pd.DataFrame(rows).T.apply(pd.to_numeric)


def _classes(rows: pd.DataFrame) -> list[str]:
    return [c[3:-5] for c in rows.columns if c.startswith("f1_") and c.endswith("_mean") and c != "f1_macro_mean"]


def build_table(summaries: dict[str, Path], split: str, search: dict[str, float] | None = None) -> pd.DataFrame:
    """Tabla de comparacion: 'media ± desv' por modelo; search son los segundos totales de busqueda."""
    search = search or {}
    rows = select_rows(summaries, split)
    specs = [("f1_macro", "F1 macro", 3), ("roc_auc_macro", "ROC-AUC macro", 3),
             ("brier_mean", "Brier", 3), ("ece_mean", "ECE", 3)]
    specs += [(f"f1_{c}", f"F1 {c}", 3) for c in _classes(rows)]
    specs += [("fit_seconds", "Entrenamiento (s)", 0), ("search", "Busqueda (s)", 0),
              ("inference_ms_per_ecg", "Inferencia (ms/ECG)", 3)]

    out = []
    for name, r in rows.iterrows():
        line = {"Modelo": name}
        for key, label, dec in specs:
            if key == "search":
                s = search.get(name)
                line[label] = "-" if s is None else f"{s:.0f}"
            else:
                line[label] = format_pm(r[f"{key}_mean"], r.get(f"{key}_std"), dec)
        out.append(line)
    return pd.DataFrame(out)


def to_markdown(table: pd.DataFrame) -> str:
    """Tabla Markdown con coma decimal."""
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(v).replace(".", ",") for v in row) + " |")
    return "\n".join(lines) + "\n"


def search_totals(base: Path) -> dict[str, float]:
    """Segundos totales de busqueda por modelo (RF y XGBoost por defecto no tienen busqueda)."""
    totals = {}
    for model, fam in SEARCH_OF.items():
        if fam == "xgboost":
            info = json.loads((base / "baseline/search/xgboost/search_time.json").read_text(encoding="utf-8"))
            totals[model] = float(info["search_seconds"])
        else:
            rel, col = TRIALS[fam]
            totals[model] = float(pd.read_csv(base / rel)[col].sum())
    return totals


def load_search_trials(base: Path) -> pd.DataFrame:
    frames = []
    for fam, (rel, _) in TRIALS.items():
        df = pd.read_csv(base / rel)
        df.insert(0, "family", fam)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_test_labels(classes: list[str]) -> np.ndarray:
    """Etiquetas del split test en el mismo orden que usan los scripts de entrenamiento."""
    cfg = load_config()
    df = load_metadata(resolve(cfg["data"]["raw_dir"]) / DATASET_DIRNAME, classes)
    parts = split_by_folds(df, cfg["data"]["folds"])
    return parts["test"][classes].to_numpy()


def _calibration(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidad media predicha y frecuencia observada por bin de igual ancho."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    pred, frac = [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            pred.append(p[m].mean())
            frac.append(y[m].mean())
    return np.array(pred), np.array(frac)


def plot_search_trials(trials: pd.DataFrame, path: Path) -> None:
    fams = list(dict.fromkeys(trials["family"]))
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, fam in enumerate(fams):
        sub = trials[trials["family"] == fam]
        x = i + rng.uniform(-0.12, 0.12, len(sub))
        ax.scatter(x, sub["val_roc_auc_macro"], s=18, alpha=0.7)
        best = sub.loc[sub["val_roc_auc_macro"].idxmax()]
        ax.scatter(i, best["val_roc_auc_macro"], marker="*", s=220, color="red",
                   label="seleccionado" if i == 0 else None)
    ax.set_xticks(range(len(fams)), fams)
    ax.set_ylabel("AUC macro en validacion")
    ax.set_title("Busqueda aleatoria: AUC val por trial")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_curves(base: Path, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    colors = plt.get_cmap("tab10").colors
    for r, model in enumerate(DEEP):
        ax_loss, ax_auc = axes[r]
        for i, seed in enumerate(SEEDS):
            h = pd.read_csv(base / "deep/final" / model / f"seed_{seed}/history.csv")
            c = colors[i]
            ax_loss.plot(h["epoch"], h["train_loss"], color=c, label=f"train, seed {seed}")
            ax_loss.plot(h["epoch"], h["val_loss"], color=c, ls=":", label=f"val, seed {seed}")
            ax_auc.plot(h["epoch"], h["val_roc_auc_macro"], color=c, label=f"seed {seed}")
            best = int(h["epoch"].iloc[h["val_roc_auc_macro"].idxmax()])
            ax_loss.axvline(best, color=c, ls="--", lw=1)
            ax_auc.axvline(best, color=c, ls="--", lw=1)
        ax_loss.set_title(f"{model}: perdida")
        ax_auc.set_title(f"{model}: AUC macro en validacion")
        ax_loss.set_xlabel("epoca")
        ax_auc.set_xlabel("epoca")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_f1_per_class(rows: pd.DataFrame, classes: list[str], path: Path) -> None:
    x = np.arange(len(classes))
    n = len(rows)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (name, r) in enumerate(rows.iterrows()):
        mean = [r[f"f1_{c}_mean"] for c in classes]
        std = [r[f"f1_{c}_std"] for c in classes]
        ax.bar(x + (i - (n - 1) / 2) * width, mean, width, yerr=std, capsize=2, label=name)
    ax.set_xticks(x, classes)
    ax.set_ylabel("F1 en test")
    ax.set_ylim(0, 1)
    ax.set_title("F1 por clase (media y desviacion de 3 semillas)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_reliability(labels: np.ndarray, probas: dict[str, np.ndarray], classes: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(classes), figsize=(3.2 * len(classes), 3.6), sharex=True, sharey=True)
    for k, (ax, cls) in enumerate(zip(axes, classes)):
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        for name, p in probas.items():
            pred, frac = _calibration(labels[:, k], p[:, k])
            ax.plot(pred, frac, marker="o", ms=3, label=name)
        ax.set_title(cls)
        ax.set_xlabel("prob. predicha")
    axes[0].set_ylabel("frecuencia observada")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_times(rows: pd.DataFrame, search: dict[str, float], path: Path) -> None:
    names = list(rows.index)
    x = np.arange(len(names))
    fit = rows["fit_seconds_mean"].to_numpy()
    srch = np.array([search.get(n, np.nan) for n in names], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, fit, 0.4, label="entrenamiento final (media)")
    ax.bar(x + 0.2, srch, 0.4, label="busqueda (total)")
    ax.set_yscale("log")
    ax.set_xticks(x, names)
    ax.set_ylabel("segundos (escala log)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compara modelos con resultados ya generados (sin entrenar).")
    ap.add_argument("--outputs", type=Path, default=None, help="Carpeta outputs/ (por defecto la del YAML)")
    args = ap.parse_args()

    cfg = load_config()
    classes = cfg["data"]["classes"]
    base = args.outputs or resolve(cfg["outputs"]["dir"])
    out = base / "comparison"
    out.mkdir(parents=True, exist_ok=True)

    summaries = {m: base / rel for m, rel in SUMMARY.items()}
    required = set(summaries.values())
    required |= {base / rel for rel, _ in TRIALS.values()}
    required |= {base / "baseline/search/xgboost/search_time.json"}
    required |= {base / f"deep/final/{m}/seed_{s}/history.csv" for m in DEEP for s in SEEDS}
    missing = sorted(str(p) for p in required if not p.exists())
    if missing:
        raise SystemExit("Faltan resultados (correr antes los entrenamientos):\n" + "\n".join(missing))

    search = search_totals(base)
    for split in ("test", "val"):
        table = build_table(summaries, split, search)
        table.to_csv(out / f"table_{split}.csv", index=False)
        if split == "test":
            (out / "table_test.md").write_text(to_markdown(table), encoding="utf-8")

    trials = load_search_trials(base)
    trials.to_csv(out / "search_trials.csv", index=False, float_format="%.4f")
    plot_search_trials(trials, out / "search_trials.png")

    plot_curves(base, out / "curves.png")

    rows_test = select_rows(summaries, "test")
    plot_f1_per_class(rows_test, classes, out / "f1_per_class.png")
    plot_times(rows_test, search, out / "times.png")

    labels = load_test_labels(classes)
    probas = {}
    for name, rel in PROBA.items():
        p = base / rel / "test_proba.npy"
        if not p.exists():
            print(f"Sin test_proba para {name}: se omite de la confiabilidad")
            continue
        arr = np.load(p)
        if len(arr) != len(labels):
            raise ValueError(f"{name}: {len(arr)} probabilidades y {len(labels)} etiquetas de test")
        probas[name] = arr
    plot_reliability(labels, probas, classes, out / "reliability_test.png")

    print(f"Comparacion escrita en {out}")


if __name__ == "__main__":
    main()
