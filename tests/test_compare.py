import pandas as pd

from ecg.compare import build_table, format_pm, to_markdown

MODELS_BY_FILE = {
    "baseline.csv": ["random_forest", "xgboost"],
    "tuned.csv": ["xgboost"],  # un solo modelo en el archivo: se usa esa fila
    "cnn1d.csv": ["cnn1d"],
    "cnn_lstm.csv": ["cnn_lstm"],
}


def _write_summary(path, models):
    """Escribe un summary_seeds.csv con el mismo formato que deep/baseline (2 semillas)."""
    rows = []
    for i, model in enumerate(models):
        for split in ("val", "test"):
            for seed, f1 in ((42, 0.740), (43, 0.748)):
                rows.append({
                    "model": model, "split": split, "seed": seed,
                    "f1_macro": f1 + 0.01 * i, "roc_auc_macro": 0.9, "brier_mean": 0.05,
                    "ece_mean": 0.02, "f1_NORM": 0.8, "f1_MI": 0.6,
                    "fit_seconds": 10.0 + seed - 42, "inference_ms_per_ecg": 0.5,
                })
    runs = pd.DataFrame(rows)
    metrics = [c for c in runs.columns if c not in ("model", "split", "seed")]
    agg = runs.groupby(["model", "split"])[metrics].agg(["mean", "std"])
    agg.to_csv(path, float_format="%.4f")


def _summaries(tmp_path):
    paths = {}
    for name, models in MODELS_BY_FILE.items():
        _write_summary(tmp_path / name, models)
        paths[name] = tmp_path / name
    return {
        "random_forest": paths["baseline.csv"],
        "xgboost": paths["baseline.csv"],
        "xgboost_tuned": paths["tuned.csv"],
        "cnn1d": paths["cnn1d.csv"],
        "cnn_lstm": paths["cnn_lstm.csv"],
    }


def test_format_pm():
    assert format_pm(0.7441, 0.0049, 3) == "0.744 ± 0.005"
    assert format_pm(120.4, 3.2, 0) == "120 ± 3"
    assert format_pm(1.0, float("nan"), 3) == "1.000"


def test_build_table_row_order_and_columns(tmp_path):
    table = build_table(_summaries(tmp_path), "test", search={"xgboost_tuned": 900.0, "cnn1d": 120.2})

    assert list(table["Modelo"]) == ["random_forest", "xgboost", "xgboost_tuned", "cnn1d", "cnn_lstm"]
    assert list(table.columns) == [
        "Modelo", "F1 macro", "ROC-AUC macro", "Brier", "ECE", "F1 NORM", "F1 MI",
        "Entrenamiento (s)", "Busqueda (s)", "Inferencia (ms/ECG)",
    ]
    rf = table.iloc[0]
    assert rf["F1 macro"] == "0.744 ± 0.006"
    assert rf["Entrenamiento (s)"] == "10 ± 1"
    assert rf["Busqueda (s)"] == "-"
    assert table.set_index("Modelo").loc["cnn1d", "Busqueda (s)"] == "120"
    assert table.set_index("Modelo").loc["xgboost_tuned", "Busqueda (s)"] == "900"


def test_to_markdown_uses_decimal_comma(tmp_path):
    table = pd.DataFrame({"Modelo": ["random_forest"], "F1 macro": ["0.744 ± 0.005"]})
    md = to_markdown(table)

    assert md.splitlines()[0] == "| Modelo | F1 macro |"
    assert "| random_forest | 0,744 ± 0,005 |" in md
