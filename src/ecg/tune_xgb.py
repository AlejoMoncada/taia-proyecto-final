"""Busqueda aleatoria de hiperparametros para XGBoost y evaluacion final en test.

La busqueda solo ve train y val. Test se evalua una vez, con los mejores parametros.

Uso:
    uv run python -m ecg.tune_xgb --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ecg.baseline import build_model, get_features, predict_proba
from ecg.data import split_by_folds
from ecg.metrics import evaluate, plot_confusion, plot_reliability, tune_thresholds
from ecg.search import sample_configs
from ecg.utils import load_config, resolve, save_json, set_seed

MODEL = "xgboost"
TUNED = "xgboost_tuned"


def load_splits(cfg: dict) -> tuple[dict, dict, float]:
    """Features y particiones por fold, igual que baseline.main."""
    df, feats, feat_time = get_features(cfg)
    if not cfg["features"]["include_metadata"]:
        feats = feats.drop(columns=["age", "sex"], errors="ignore")
    parts = split_by_folds(df, cfg["data"]["folds"])
    classes = cfg["data"]["classes"]
    X = {k: feats.loc[v.index] for k, v in parts.items()}
    y = {k: v[classes].to_numpy() for k, v in parts.items()}
    return X, y, feat_time


def fit_xgb(params: dict, seed: int, X_train, y_train):
    """Entrena un modelo XGBoost con una semilla; devuelve (modelo, segundos de entrenamiento)."""
    set_seed(seed)
    model = build_model(MODEL, params, seed)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    return model, time.perf_counter() - t0


def search_xgb(space: dict, n_trials: int, search_seed: int, base_params: dict, train_seed: int,
               X_train, y_train, X_val, y_val, classes: list[str]) -> pd.DataFrame:
    """Prueba n_trials configuraciones muestreadas y las evalua en val. No recibe test."""
    rows = []
    for i, sampled in enumerate(sample_configs(space, n_trials, search_seed), start=1):
        model, fit_s = fit_xgb(base_params | sampled, train_seed, X_train, y_train)
        p_val = predict_proba(model, X_val)
        thr = tune_thresholds(y_val, p_val)
        res = evaluate(y_val, p_val, thr, classes)
        rows.append({
            "trial": i,
            "params": json.dumps(sampled, sort_keys=True),
            "fit_seconds": fit_s,
            "val_roc_auc_macro": res["roc_auc_macro"],
            "val_f1_macro": res["f1_macro"],
        })
        print(f"trial {i}: val AUC macro={res['roc_auc_macro']:.4f} F1 macro={res['f1_macro']:.4f}"
              f"  ({fit_s:.1f} s) {sampled}")
    return pd.DataFrame(rows)


def _rows(name: str, seed: int, res: dict, fit_s: float, infer_ms: float, classes: list[str]) -> list[dict]:
    """Filas val y test con el mismo formato que baseline.run_seed."""
    rows = []
    for split in ("val", "test"):
        r = res[split]
        row = {"model": name, "split": split, "f1_macro": r["f1_macro"],
               "roc_auc_macro": r["roc_auc_macro"], "brier_mean": r["brier_mean"], "ece_mean": r["ece_mean"]}
        row |= {f"f1_{c}": r["per_class"][c]["f1"] for c in classes}
        row |= {f"auc_{c}": r["per_class"][c]["roc_auc"] for c in classes}
        row |= {"fit_seconds": fit_s, "inference_ms_per_ecg": infer_ms, "seed": seed}
        rows.append(row)
    return rows


def final_xgb(params: dict, seeds: list[int], X: dict, y: dict, classes: list[str],
              env: dict, out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entrena con cada semilla final y guarda metricas, graficas y probabilidades de test."""
    runs = []
    for seed in seeds:
        print(f"\n######## semilla {seed} ########")
        model, fit_s = fit_xgb(params, seed, X["train"], y["train"])
        p_val = predict_proba(model, X["val"])
        thr = tune_thresholds(y["val"], p_val)

        t0 = time.perf_counter()
        p_test = predict_proba(model, X["test"])
        infer_ms = 1000 * (time.perf_counter() - t0) / len(X["test"])

        res = {
            "model": TUNED,
            "params": params,
            "seed": seed,
            "fit_seconds": fit_s,
            "inference_ms_per_ecg": infer_ms,
            "val": evaluate(y["val"], p_val, thr, classes),
            "test": evaluate(y["test"], p_test, thr, classes),
            "env": env,
        }
        out = out_root / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        save_json(res, out / "metrics.json")
        np.save(out / "test_proba.npy", p_test)
        plot_reliability(y["test"], p_test, classes, out / "calibration_test.png")
        plot_confusion({c: v["confusion_matrix"] for c, v in res["test"]["per_class"].items()},
                       out / "confusion_test.png")
        runs += _rows(TUNED, seed, res, fit_s, infer_ms, classes)
        print(f"val  F1 macro={res['val']['f1_macro']:.3f} AUC macro={res['val']['roc_auc_macro']:.3f}")
        print(f"test F1 macro={res['test']['f1_macro']:.3f} AUC macro={res['test']['roc_auc_macro']:.3f}")

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(out_root / "runs.csv", index=False, float_format="%.4f")
    metrics = [c for c in runs_df.columns if c not in ("model", "split", "seed")]
    agg = runs_df.groupby(["model", "split"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg.to_csv(out_root / "summary_seeds.csv", float_format="%.4f")
    return runs_df, agg


def main() -> None:
    ap = argparse.ArgumentParser(description="Busqueda de hiperparametros para XGBoost y evaluacion final.")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    scfg = cfg["search"]
    classes = cfg["data"]["classes"]

    X, y, feat_time = load_splits(cfg)
    out_root = resolve(cfg["outputs"]["dir"]) / "baseline" / "search" / MODEL
    out_root.mkdir(parents=True, exist_ok=True)
    env = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "feature_extraction_s": feat_time,
        "n_features": X["train"].shape[1],
        "n": {k: len(v) for k, v in X.items()},
    }

    t_start = time.perf_counter()
    trials = search_xgb(scfg["space"][MODEL], scfg["n_trials"], scfg["seed"], cfg["baseline"][MODEL],
                        cfg["seed"], X["train"], y["train"], X["val"], y["val"], classes)
    trials.to_csv(out_root / "trials.csv", index=False, float_format="%.4f")
    t_search = time.perf_counter() - t_start

    metric_col = f"val_{scfg['metric']}"
    best = trials.loc[trials[metric_col].idxmax()]
    best_params = json.loads(best["params"])
    save_json({"trial": int(best["trial"]), "metric": scfg["metric"], "value": float(best[metric_col]),
               "params": best_params}, out_root / "best.json")

    t_final = time.perf_counter()
    full_params = cfg["baseline"][MODEL] | best_params
    _, agg = final_xgb(full_params, scfg["final_seeds"], X, y, classes, env, out_root / "final")
    final_s = time.perf_counter() - t_final

    total_s = time.perf_counter() - t_start
    save_json({"search_seconds": t_search, "final_seconds": final_s, "total_seconds": total_s},
              out_root / "search_time.json")

    print(f"\nMejores parametros (trial {int(best['trial'])}): {best_params}")
    for _, r in trials.iterrows():
        print(f"  trial {int(r['trial'])}: val AUC macro {r['val_roc_auc_macro']:.4f}")
    t = agg.loc[(TUNED, "test")]
    print(f"Test (semillas {scfg['final_seeds']}): F1 macro {t['f1_macro_mean']:.4f} ± {t['f1_macro_std']:.4f}"
          f"  AUC macro {t['roc_auc_macro_mean']:.4f} ± {t['roc_auc_macro_std']:.4f}")
    print(f"Tiempo busqueda {t_search:.0f} s, final {final_s:.0f} s. Resumen en {out_root / 'final'}")


if __name__ == "__main__":
    main()
