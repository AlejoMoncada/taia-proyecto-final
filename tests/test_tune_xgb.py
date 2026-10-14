import inspect

import numpy as np
import pandas as pd

from ecg import tune_xgb

CLASSES = ["A", "B", "C"]
SPACE = {"n_estimators": [5, 10, 15], "max_depth": [2, 3]}
BASE = {"n_estimators": 5, "max_depth": 2, "tree_method": "hist", "n_jobs": 1}


def _synthetic(n, rng):
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = (rng.random(size=(n, len(CLASSES))) < 0.3).astype(int)
    return X, y


def _search(n_trials, X_tr, y_tr, X_va, y_va):
    return tune_xgb.search_xgb(SPACE, n_trials, 123, BASE, 42, X_tr, y_tr, X_va, y_va, CLASSES)


def test_search_signature_has_no_test_split():
    names = inspect.signature(tune_xgb.search_xgb).parameters
    assert not any("test" in name for name in names)


def test_search_only_predicts_on_val(monkeypatch):
    rng = np.random.default_rng(0)
    X_tr, y_tr = _synthetic(80, rng)
    X_va, y_va = _synthetic(40, rng)
    seen = []
    real_predict = tune_xgb.predict_proba

    def spy(model, X):
        seen.append(len(X))
        return real_predict(model, X)

    monkeypatch.setattr(tune_xgb, "predict_proba", spy)
    _search(3, X_tr, y_tr, X_va, y_va)
    assert seen == [len(X_va)] * 3


def test_trials_table_has_one_row_per_trial():
    rng = np.random.default_rng(1)
    X_tr, y_tr = _synthetic(80, rng)
    X_va, y_va = _synthetic(40, rng)
    trials = _search(4, X_tr, y_tr, X_va, y_va)
    assert len(trials) == 4
    assert list(trials.columns) == ["trial", "params", "fit_seconds", "val_roc_auc_macro", "val_f1_macro"]
    assert trials["trial"].tolist() == [1, 2, 3, 4]
