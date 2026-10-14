"""Pruebas de las redes profundas con datos sinteticos (CPU, sin dataset)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ecg.deep import EcgSet, lead_stats, predict_proba, train_model
from ecg.models import build_model
from ecg.utils import load_config

CFG = load_config()
CLASSES = 5
EXPECTED_COLUMNS = ["epoch", "train_loss", "val_loss", "val_roc_auc_macro", "val_f1_macro", "lr", "elapsed_s"]


def _synthetic(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Senal con una derivacion por clase: separable de forma sencilla."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=(n, CLASSES))
    x = rng.normal(0.0, 1.0, size=(n, 1000, 12)).astype(np.float32)
    x[:, :, :CLASSES] += 3.0 * y[:, None, :]
    return x, y


def _sets(n_train: int = 96, n_val: int = 64) -> tuple[EcgSet, EcgSet]:
    x_tr, y_tr = _synthetic(n_train, 0)
    x_va, y_va = _synthetic(n_val, 1)
    mean, std = np.zeros(12, np.float32), np.ones(12, np.float32)
    train = EcgSet(x_tr, np.arange(n_train), y_tr, mean, std)
    val = EcgSet(x_va, np.arange(n_val), y_va, mean, std)
    return train, val


@pytest.mark.parametrize("channels", CFG["search"]["space"]["cnn1d"]["channels"])
def test_cnn1d_output_shape_for_every_channel_option(channels):
    model = build_model("cnn1d", {**CFG["deep"]["cnn1d"], "channels": channels})
    assert model(torch.randn(4, 12, 1000)).shape == (4, 5)


@pytest.mark.parametrize("lstm_hidden", CFG["search"]["space"]["cnn_lstm"]["lstm_hidden"])
def test_cnn_lstm_output_shape_for_every_hidden_option(lstm_hidden):
    model = build_model("cnn_lstm", {**CFG["deep"]["cnn_lstm"], "lstm_hidden": lstm_hidden})
    assert model(torch.randn(4, 12, 1000)).shape == (4, 5)


def test_lead_stats_use_only_the_array_passed():
    rng = np.random.default_rng(0)
    train = rng.normal(0.0, 1.0, size=(64, 200, 12)).astype(np.float32)
    mean, std = lead_stats(train)
    assert mean.shape == (12,) and std.shape == (12,)
    assert np.allclose(mean, 0.0, atol=0.1)
    assert np.allclose(std, 1.0, atol=0.1)
    # Un arreglo de validacion con media 100 no debe contaminar las estadisticas de train.
    val = rng.normal(100.0, 1.0, size=(64, 200, 12)).astype(np.float32)
    assert np.allclose(lead_stats(train)[0], mean)
    assert np.all(lead_stats(val)[0] > 99.0)


def test_train_model_returns_history_and_respects_max_epochs():
    train, val = _sets()
    params = {"channels": [16, 32], "kernel_size": 7, "dropout": 0.0, "lr": 3e-3, "weight_decay": 0.0}
    deep_cfg = {"batch_size": 16, "max_epochs": 3, "max_minutes": 5, "early_stopping_patience": 8}
    model = build_model("cnn1d", params)
    model, hist, info = train_model(model, train, val, params, deep_cfg, torch.device("cpu"), seed=42)

    assert list(hist.columns) == EXPECTED_COLUMNS
    assert 1 <= len(hist) <= 3
    assert info["epochs_run"] == len(hist)
    assert info["stop_reason"] in {"max_epochs", "early_stopping", "time_budget"}
    proba = predict_proba(model, val, torch.device("cpu"))
    assert proba.shape == (len(val), CLASSES)
    assert np.all((proba >= 0) & (proba <= 1))


def test_train_model_stops_after_one_epoch_when_time_budget_is_zero():
    train, val = _sets()
    params = {"channels": [16, 32], "kernel_size": 7, "dropout": 0.0, "lr": 3e-3, "weight_decay": 0.0}
    deep_cfg = {"batch_size": 16, "max_epochs": 3, "max_minutes": 0, "early_stopping_patience": 8}
    model = build_model("cnn1d", params)
    _, hist, info = train_model(model, train, val, params, deep_cfg, torch.device("cpu"), seed=42)

    assert len(hist) == 1
    assert info["stop_reason"] == "time_budget"
