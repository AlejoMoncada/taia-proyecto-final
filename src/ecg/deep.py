"""Redes profundas (CNN 1D y CNN 1D + LSTM) sobre ECG de PTB-XL.

Uso:
    uv run python -m ecg.deep search --model cnn1d [--config configs/default.yaml] [--smoke]
    uv run python -m ecg.deep final --model cnn1d [--smoke]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ecg.data import load_or_build_cache, split_by_folds
from ecg.metrics import evaluate, plot_confusion, plot_reliability, tune_thresholds
from ecg.models import build_model, count_parameters
from ecg.search import sample_configs
from ecg.utils import load_config, resolve, save_json, set_seed

SMOKE_ROWS = {"train": 512, "val": 256, "test": 256}
INFER_BATCH = 256
N_LEADS = 12


# ---------------------------------------------------------------- datos

def _chunks(x: Any, rows: np.ndarray, chunk: int) -> Iterator[np.ndarray]:
    """Lee filas de x en bloques (x es (n, muestras, 12); puede ser memmap)."""
    for s in range(0, len(rows), chunk):
        yield np.asarray(x[np.sort(rows[s:s + chunk])], dtype=np.float64)


def lead_stats(x_train: Any, rows: np.ndarray | None = None, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Media y desviacion por derivacion (12,). Solo usa el arreglo recibido (train)."""
    rows = np.arange(len(x_train)) if rows is None else np.asarray(rows)
    total = np.zeros(N_LEADS, dtype=np.float64)
    count = 0
    for block in _chunks(x_train, rows, chunk):
        total += block.sum(axis=(0, 1))
        count += block.shape[0] * block.shape[1]
    mean = total / count
    sq = np.zeros(N_LEADS, dtype=np.float64)
    for block in _chunks(x_train, rows, chunk):
        sq += ((block - mean) ** 2).sum(axis=(0, 1))
    std = np.sqrt(sq / count)
    # Evita division entre cero en derivaciones planas.
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


class EcgSet(Dataset):
    """Registros de un split: senal normalizada con estadisticas de train, etiqueta multietiqueta."""

    def __init__(self, signals: Any, rows: np.ndarray, labels: np.ndarray, mean: np.ndarray, std: np.ndarray):
        self.signals = signals
        self.rows = np.asarray(rows)
        self.labels = np.asarray(labels).astype(np.int64)
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = (np.asarray(self.signals[self.rows[i]], dtype=np.float32) - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(x.T)), torch.from_numpy(self.labels[i].astype(np.float32))


def load_data(cfg: dict, smoke: bool) -> tuple[Any, dict[str, np.ndarray], np.ndarray]:
    """Senales, filas por split y etiquetas. En smoke submuestrea con semilla."""
    df, signals = load_or_build_cache(cfg)
    classes = cfg["data"]["classes"]
    labels = df[classes].to_numpy()
    rng = np.random.default_rng(cfg["seed"])
    rows = {}
    for name, part in split_by_folds(df, cfg["data"]["folds"]).items():
        pos = df.index.get_indexer(part.index)
        if smoke and len(pos) > SMOKE_ROWS[name]:
            pos = rng.choice(pos, size=SMOKE_ROWS[name], replace=False)
        rows[name] = np.sort(pos)
    return signals, rows, labels


def make_sets(signals: Any, rows: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, EcgSet]:
    # Estadisticas solo con train; se aplican a todos los splits.
    mean, std = lead_stats(signals, rows["train"])
    return {name: EcgSet(signals, r, labels[r], mean, std) for name, r in rows.items()}


# ---------------------------------------------------------------- entrenamiento

def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def device_label(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "cpu"


def _forward_all(model: nn.Module, data: Dataset, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Logits y etiquetas de todo un split, sin barajar."""
    model.eval()
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    logits, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits.append(model(x.to(device)).cpu())
            labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def predict_proba(model: nn.Module, data: Dataset, device: torch.device) -> np.ndarray:
    logits, _ = _forward_all(model, data, device, INFER_BATCH)
    return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))


def train_model(model: nn.Module, train_data: Dataset, val_data: Dataset, params: dict,
                deep_cfg: dict, device: torch.device, seed: int) -> tuple[nn.Module, pd.DataFrame, dict]:
    """Entrena con AdamW + BCE, para por paciencia o por tiempo y restaura el mejor AUC en val."""
    set_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    loss_fn = nn.BCEWithLogitsLoss()
    bs = deep_cfg["batch_size"]
    train_dl = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=0,
                          generator=torch.Generator().manual_seed(seed))

    max_epochs = deep_cfg["max_epochs"]
    patience = deep_cfg["early_stopping_patience"]
    max_seconds = deep_cfg["max_minutes"] * 60
    history: list[dict] = []
    best_auc, best_epoch, best_state, stale = -np.inf, 0, None, 0
    stop_reason = "max_epochs"
    t0 = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        lr = opt.param_groups[0]["lr"]
        loss_sum, n_seen = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            n_seen += len(y)
        train_loss = loss_sum / n_seen

        val_logits, val_y = _forward_all(model, val_data, device, INFER_BATCH)
        val_loss = loss_fn(torch.from_numpy(val_logits), torch.from_numpy(val_y.astype(np.float32))).item()
        val_proba = 1.0 / (1.0 + np.exp(-val_logits.astype(np.float64)))
        val_y_int = val_y.astype(int)
        val_auc = roc_auc_score(val_y_int, val_proba, average="macro")
        val_f1 = f1_score(val_y_int, val_proba >= 0.5, average="macro", zero_division=0)
        sched.step(val_auc)

        elapsed = time.perf_counter() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_roc_auc_macro": val_auc, "val_f1_macro": val_f1,
                        "lr": lr, "elapsed_s": elapsed})
        print(f"  ep {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"AUC {val_auc:.4f}  F1 {val_f1:.4f}  lr {lr:.1e}  {elapsed:.0f}s")

        if val_auc > best_auc:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1

        if stale >= patience:
            stop_reason = "early_stopping"
            break
        if elapsed > max_seconds:
            stop_reason = "time_budget"
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    info = {"best_epoch": best_epoch, "epochs_run": len(history),
            "train_seconds": time.perf_counter() - t0, "stop_reason": stop_reason}
    return model, pd.DataFrame(history), info


def fit_once(model_name: str, params: dict, sets: dict[str, EcgSet], deep_cfg: dict,
             device: torch.device, seed: int) -> tuple[nn.Module, pd.DataFrame, dict]:
    set_seed(seed)
    model = build_model(model_name, params)
    return train_model(model, sets["train"], sets["val"], params, deep_cfg, device, seed)


# ---------------------------------------------------------------- graficas

def plot_curves(hist: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(hist["epoch"], hist["train_loss"], marker="o", label="train")
    ax1.plot(hist["epoch"], hist["val_loss"], marker="o", label="val")
    ax1.set_xlabel("epoca")
    ax1.set_ylabel("BCE")
    ax1.legend()
    ax2.plot(hist["epoch"], hist["val_roc_auc_macro"], marker="o", color="C2")
    ax2.set_xlabel("epoca")
    ax2.set_ylabel("ROC-AUC macro (val)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- busqueda y final

def run_search(cfg: dict, model_name: str, root: Path, device: torch.device, sets: dict[str, EcgSet]) -> None:
    dcfg, scfg, classes = cfg["deep"], cfg["search"], cfg["data"]["classes"]
    base = dcfg[model_name]
    sampled = sample_configs(scfg["space"][model_name], scfg["n_trials"], scfg["seed"])
    yv = sets["val"].labels
    metric_key = f"val_{scfg['metric']}"
    model_dir = root / "search" / model_name
    rows, all_params = [], []

    for k, extra in enumerate(sampled, 1):
        params = {**base, **extra}
        all_params.append(params)
        print(f"\n== trial {k}/{len(sampled)}  {json.dumps(extra, sort_keys=True)} ==")
        model, hist, info = fit_once(model_name, params, sets, dcfg, device, cfg["seed"])
        trial_dir = model_dir / f"trial_{k}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        hist.to_csv(trial_dir / "history.csv", index=False)

        p_val = predict_proba(model, sets["val"], device)
        res = evaluate(yv, p_val, tune_thresholds(yv, p_val), classes)
        rows.append({"trial": k, "params": json.dumps(params, sort_keys=True),
                     "n_params": count_parameters(model), **info,
                     "val_roc_auc_macro": res["roc_auc_macro"], "val_f1_macro": res["f1_macro"]})
        print(f"  trial {k}: AUC val {res['roc_auc_macro']:.4f}  F1 val {res['f1_macro']:.4f}")

    trials = pd.DataFrame(rows)
    model_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(model_dir / "trials.csv", index=False, float_format="%.4f")
    best_idx = int(np.argmax([r[metric_key] for r in rows]))
    save_json({"trial": best_idx + 1, "metric": metric_key, "value": rows[best_idx][metric_key],
               "params": all_params[best_idx]}, model_dir / "best.json")
    print(f"\nMejor trial: {best_idx + 1} ({metric_key}={rows[best_idx][metric_key]:.4f}). "
          f"Resumen en {model_dir / 'trials.csv'}")


def run_final(cfg: dict, model_name: str, root: Path, device: torch.device, sets: dict[str, EcgSet]) -> None:
    best_path = root / "search" / model_name / "best.json"
    if not best_path.exists():
        raise SystemExit(f"Falta {best_path}. Ejecuta primero: "
                         f"uv run python -m ecg.deep search --model {model_name}")
    params = json.loads(best_path.read_text(encoding="utf-8"))["params"]
    dcfg, classes = cfg["deep"], cfg["data"]["classes"]
    out_root = root / "final" / model_name
    yv, yt = sets["val"].labels, sets["test"].labels
    rows = []

    for seed in cfg["search"]["final_seeds"]:
        print(f"\n######## semilla {seed} ({model_name}) ########")
        seed_dir = out_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model, hist, info = fit_once(model_name, params, sets, dcfg, device, seed)
        hist.to_csv(seed_dir / "history.csv", index=False)

        p_val = predict_proba(model, sets["val"], device)
        thr = tune_thresholds(yv, p_val)
        res_val = evaluate(yv, p_val, thr, classes)

        t0 = time.perf_counter()
        p_test = predict_proba(model, sets["test"], device)
        infer_ms = 1000 * (time.perf_counter() - t0) / len(sets["test"])
        res_test = evaluate(yt, p_test, thr, classes)

        np.save(seed_dir / "test_proba.npy", p_test)
        torch.save(model.state_dict(), seed_dir / "model.pt")
        plot_curves(hist, seed_dir / "curves.png")
        plot_reliability(yt, p_test, classes, seed_dir / "calibration_test.png")
        plot_confusion({c: v["confusion_matrix"] for c, v in res_test["per_class"].items()},
                       seed_dir / "confusion_test.png")

        res = {"model": model_name, "params": params, "seed": seed,
               "fit_seconds": info["train_seconds"], "train_seconds": info["train_seconds"],
               "inference_ms_per_ecg": infer_ms, "n_params": count_parameters(model),
               "history": info, "device": device_label(device),
               "env": {"platform": platform.platform(), "cpu_count": os.cpu_count(), "torch": torch.__version__},
               "val": res_val, "test": res_test}
        save_json(res, seed_dir / "metrics.json")

        for split, r in (("val", res_val), ("test", res_test)):
            row = {"model": model_name, "split": split, "f1_macro": r["f1_macro"],
                   "roc_auc_macro": r["roc_auc_macro"], "brier_mean": r["brier_mean"], "ece_mean": r["ece_mean"]}
            row |= {f"f1_{c}": r["per_class"][c]["f1"] for c in classes}
            row |= {f"auc_{c}": r["per_class"][c]["roc_auc"] for c in classes}
            row |= {"fit_seconds": info["train_seconds"], "inference_ms_per_ecg": infer_ms, "seed": seed}
            rows.append(row)
        print(f"val  F1 macro={res_val['f1_macro']:.3f} AUC macro={res_val['roc_auc_macro']:.3f}")
        print(f"test F1 macro={res_test['f1_macro']:.3f} AUC macro={res_test['roc_auc_macro']:.3f}"
              f"  (entrenamiento {info['train_seconds']:.1f} s, {infer_ms:.3f} ms/ECG)")

    runs = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out_root / "runs.csv", index=False, float_format="%.4f")
    metrics = [c for c in runs.columns if c not in ("model", "split", "seed")]
    agg = runs.groupby(["model", "split"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg.to_csv(out_root / "summary_seeds.csv", float_format="%.4f")
    print(f"\nResumen en {out_root / 'summary_seeds.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Busqueda de hiperparametros y entrenamiento final de redes profundas.")
    ap.add_argument("mode", choices=["search", "final"])
    ap.add_argument("--model", required=True, choices=["cnn1d", "cnn_lstm"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="Submuestra y reduce epocas/trials; escribe en outputs/deep_smoke")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = copy.deepcopy(cfg)
        cfg["deep"]["max_epochs"] = 2
        cfg["search"]["n_trials"] = 2
        cfg["search"]["final_seeds"] = [42]
        root = resolve(cfg["outputs"]["dir"]) / "deep_smoke"
    else:
        root = resolve(cfg["outputs"]["dir"]) / "deep"

    device = resolve_device(cfg["deep"]["device"])
    print(f"Dispositivo: {device_label(device)} | modo: {args.mode} | modelo: {args.model}"
          f"{' | smoke' if args.smoke else ''}")
    signals, rows, labels = load_data(cfg, args.smoke)
    sets = make_sets(signals, rows, labels)
    print("Filas por split: " + ", ".join(f"{k}={len(v)}" for k, v in sets.items()))

    if args.mode == "search":
        run_search(cfg, args.model, root, device, sets)
    else:
        run_final(cfg, args.model, root, device, sets)


if __name__ == "__main__":
    main()
