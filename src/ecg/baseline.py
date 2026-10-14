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
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = cfg["seed"]
    set_seed(seed)
    classes = cfg["data"]["classes"]
    bcfg = cfg["baseline"]

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

    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / "summary.csv", index=False, float_format="%.4f")
    print(f"\nResumen en {Path(out_root) / 'summary.csv'}")


if __name__ == "__main__":
    main()
