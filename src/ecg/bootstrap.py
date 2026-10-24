"""Script para calcular la incertidumbre poblacional con Bootstrap Pareado."""

import json
import numpy as np
from pathlib import Path

from ecg.data import load_or_build_cache, split_by_folds
from ecg.utils import load_config, resolve
from ecg.metrics import bootstrap_patient_metrics

def main():
    # 1. Cargar configuración y clases
    cfg = load_config("configs/default.yaml")
    classes = cfg["data"]["classes"]
    
    # 2. Cargar el DataFrame original para extraer patient_ids del conjunto de Test
    df, _ = load_or_build_cache(cfg)
    folds = split_by_folds(df, cfg["data"]["folds"])
    test_df = folds["test"]
    
    test_y = test_df[classes].to_numpy()
    test_patient_ids = test_df["patient_id"].to_numpy()
    
    # === EVALUACIÓN PARA CNN 1D (Semilla 42 como ejemplo) ===
    model_name = "cnn1d"
    seed = 42
    root = resolve(cfg["outputs"]["dir"]) / "deep" / "final" / model_name / f"seed_{seed}"
    
    prob_path = root / "test_proba.npy"
    metrics_path = root / "metrics.json"
    
    if not prob_path.exists() or not metrics_path.exists():
        print(f"Error: No se encontraron los resultados de {model_name} en {root}")
        return
        
    # Cargar probabilidades
    test_probs = np.load(prob_path)
    
    # Cargar umbrales óptimos calculados durante el entrenamiento
    with open(metrics_path, "r", encoding="utf-8") as f:
        res = json.load(f)
    thresholds = np.array([res["val"]["per_class"][c]["threshold"] for c in classes])
    
    print(f"Iniciando Bootstrap (1000 iteraciones) para {model_name} (semilla {seed})...")
    print("Agrupando por patient_id para respetar la dependencia de registros...")
    
    # 3. Ejecutar el cálculo
    stats = bootstrap_patient_metrics(test_y, test_probs, test_patient_ids, thresholds)
    
    print("\n--- Resultados con Intervalo de Confianza (95%) ---")
    print(f"F1 Macro:  {stats['f1_mean']:.3f}  [95% CI: {stats['f1_ci'][0]:.3f} - {stats['f1_ci'][1]:.3f}]")
    print(f"AUC Macro: {stats['auc_mean']:.3f}  [95% CI: {stats['auc_ci'][0]:.3f} - {stats['auc_ci'][1]:.3f}]")

if __name__ == "__main__":
    main()