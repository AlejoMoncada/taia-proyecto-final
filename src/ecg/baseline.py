"""Linea base clasica: features hechas a mano + Random Forest / XGBoost.

Uso:
    uv run python -m ecg.baseline --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ecg.data import load_or_build_cache, split_by_folds
from ecg.features import extract_features
from ecg.metrics import evaluate, plot_confusion, plot_reliability, tune_thresholds
from ecg.utils import load_config, resolve, save_json, set_seed


def build_model(name: str, params: dict, seed: int):
    # Imputador y escalador viven dentro del pipeline: se ajustan solo con train.
    if name == "random_forest":
        clf = RandomForestClassifier(random_state=seed, **params)
    elif name == "xgboost":
        clf = OneVsRestClassifier(XGBClassifier(random_state=seed, eval_metric="logloss", **params))
    else:
        raise ValueError(f"Modelo desconocido: {name}")
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf)


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    p = model.predict_proba(X)
    # RandomForest multietiqueta devuelve una lista (n, 2) por clase.
    if isinstance(p, list):
        p = np.stack([pk[:, 1] for pk in p], axis=1)
    return p


def get_features(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Extrae (o carga de cache) la tabla de features para todos los registros."""
    df, signals = load_or_build_cache(cfg)
    fcfg = cfg["features"]
    sr = cfg["data"]["sampling_rate"]
    path = resolve(cfg["data"]["processed_dir"]) / f"features_{sr}hz.pkl"
    if path.exists():
        feats = pd.read_pickle(path)
        if np.array_equal(feats.index.values, df.index.values):
            return df, feats, float("nan")
    t0 = time.perf_counter()
    feats = extract_features(signals, df, sr, fcfg["bands_hz"])
    elapsed = time.perf_counter() - t0
    feats.index = df.index
    feats.to_pickle(path)
    return df, feats, elapsed


def main() -> None:
    ap = argparse.ArgumentParser(description="Entrena y evalua la linea base clasica.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seeds", type=int, nargs="+",
                    help="Repite el entrenamiento con varias semillas y reporta media y desviacion")
    args = ap.parse_args()
    cfg = load_config(args.config)
    classes = cfg["data"]["classes"]

    df, feats, feat_time = get_features(cfg)
    if not cfg["features"]["include_metadata"]:
        feats = feats.drop(columns=["age", "sex"], errors="ignore")
    parts = split_by_folds(df, cfg["data"]["folds"])
    X = {k: feats.loc[v.index] for k, v in parts.items()}
    y = {k: v[classes].to_numpy() for k, v in parts.items()}

    out_root = resolve(cfg["outputs"]["dir"]) / "baseline"
    env = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "feature_extraction_s": feat_time,
        "n_features": X["train"].shape[1],
        "n": {k: len(v) for k, v in X.items()},
    }

    if not args.seeds:
        summary = pd.DataFrame(run_seed(cfg, cfg["seed"], X, y, env, out_root))
        summary.to_csv(out_root / "summary.csv", index=False, float_format="%.4f")
        print(f"\nResumen en {out_root / 'summary.csv'}")
        return

    # Mismas particiones y features; solo cambia la semilla de los modelos.
    seeds_root = out_root / "seeds"
    rows = []
    for seed in args.seeds:
        print(f"\n######## semilla {seed} ########")
        rows += [r | {"seed": seed} for r in run_seed(cfg, seed, X, y, env, seeds_root / f"seed_{seed}")]
    runs = pd.DataFrame(rows)
    runs.to_csv(seeds_root / "runs.csv", index=False, float_format="%.4f")
    metrics = [c for c in runs.columns if c not in ("model", "split", "seed")]
    agg = runs.groupby(["model", "split"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg.to_csv(seeds_root / "summary_seeds.csv", float_format="%.4f")
    print(f"\nSemillas {args.seeds}. Resumen en {seeds_root / 'summary_seeds.csv'}")
    for (model, split), r in agg.iterrows():
        if split == "test":
            print(f"  {model:14s} F1 macro {r['f1_macro_mean']:.3f} ± {r['f1_macro_std']:.3f}"
                  f"  AUC macro {r['roc_auc_macro_mean']:.3f} ± {r['roc_auc_macro_std']:.3f}")


def run_seed(cfg: dict, seed: int, X: dict, y: dict, env: dict, out_root: Path) -> list[dict]:
    """Entrena y evalua todos los modelos con una semilla; guarda resultados en out_root/<modelo>."""
    set_seed(seed)
    classes = cfg["data"]["classes"]
    bcfg = cfg["baseline"]
    rows = []
    for name in bcfg["models"]:
        print(f"\n== {name} ==")
        model = build_model(name, bcfg[name], seed)
        t0 = time.perf_counter()
        model.fit(X["train"], y["train"])
        fit_s = time.perf_counter() - t0

        p_val = predict_proba(model, X["val"])
        thr = tune_thresholds(y["val"], p_val) if bcfg["tune_thresholds_on_val"] else np.full(len(classes), 0.5)

        t0 = time.perf_counter()
        p_test = predict_proba(model, X["test"])
        infer_s = time.perf_counter() - t0

        res = {
            "model": name,
            "params": bcfg[name],
            "seed": seed,
            "fit_seconds": fit_s,
            "inference_ms_per_ecg": 1000 * infer_s / len(X["test"]),
            "val": evaluate(y["val"], p_val, thr, classes),
            "test": evaluate(y["test"], p_test, thr, classes),
            "env": env,
        }
        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        save_json(res, out / "metrics.json")
        joblib.dump(model, out / "model.joblib")
        np.save(out / "test_proba.npy", p_test)
        plot_reliability(y["test"], p_test, classes, out / "calibration_test.png")
        plot_confusion({c: v["confusion_matrix"] for c, v in res["test"]["per_class"].items()},
                       out / "confusion_test.png")

        for split in ("val", "test"):
            r = res[split]
            row = {"model": name, "split": split, "f1_macro": r["f1_macro"],
                   "roc_auc_macro": r["roc_auc_macro"], "brier_mean": r["brier_mean"], "ece_mean": r["ece_mean"]}
            row |= {f"f1_{c}": r["per_class"][c]["f1"] for c in classes}
            row |= {f"auc_{c}": r["per_class"][c]["roc_auc"] for c in classes}
            row |= {"fit_seconds": fit_s, "inference_ms_per_ecg": res["inference_ms_per_ecg"]}
            rows.append(row)
        print(f"val  F1 macro={res['val']['f1_macro']:.3f} AUC macro={res['val']['roc_auc_macro']:.3f}")
        print(f"test F1 macro={res['test']['f1_macro']:.3f} AUC macro={res['test']['roc_auc_macro']:.3f}"
              f"  (entrenamiento {fit_s:.1f} s)")
    return rows


if __name__ == "__main__":
    main()
