"""Arquitecturas para ECG multietiqueta: CNN 1D residual y CNN 1D + LSTM bidireccional.

Entrada: (B, derivaciones, muestras). Salida: logits (B, clases).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

# Claves del optimizador: las usa deep.py, el modelo las ignora.
OPTIMIZER_KEYS = ("lr", "weight_decay")


class ResBlock1d(nn.Module):
    """Bloque residual 1D: conv-BN-ReLU-conv-BN, suma con el atajo y ReLU."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size debe ser impar para padding 'same'")
        pad = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(c_out, c_out, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(c_out),
        )
        # Atajo 1x1 solo cuando cambia el ancho.
        self.shortcut = nn.Identity() if c_in == c_out else nn.Conv1d(c_in, c_out, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.body(x) + self.shortcut(x))


def _trunk(n_leads: int, channels: Sequence[int], kernel_size: int) -> nn.Sequential:
    """Bloques residuales, cada uno seguido de MaxPool1d(2)."""
    layers: list[nn.Module] = []
    c_in = n_leads
    for width in channels:
        layers += [ResBlock1d(c_in, width, kernel_size), nn.MaxPool1d(2)]
        c_in = width
    return nn.Sequential(*layers)


class CNN1D(nn.Module):
    """CNN 1D residual; cabeza con promedio y maximo global en el tiempo."""

    def __init__(self, n_leads: int = 12, n_classes: int = 5,
                 channels: Sequence[int] = (64, 64, 128, 128),
                 kernel_size: int = 7, dropout: float = 0.2):
        super().__init__()
        self.features = _trunk(n_leads, channels, kernel_size)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * channels[-1], n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(pooled)


class CNNLSTM(nn.Module):
    """Extractor convolucional + LSTM bidireccional sobre el tiempo; cabeza con media y maximo."""

    def __init__(self, n_leads: int = 12, n_classes: int = 5,
                 channels: Sequence[int] = (64, 64, 128),
                 kernel_size: int = 7, lstm_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.features = _trunk(n_leads, channels, kernel_size)
        self.lstm = nn.LSTM(channels[-1], lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * lstm_hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).transpose(1, 2)  # (B, L, C)
        out, _ = self.lstm(h)                  # (B, L, 2H)
        pooled = torch.cat([out.mean(dim=1), out.amax(dim=1)], dim=1)
        return self.head(pooled)


def build_model(name: str, params: dict[str, Any]) -> nn.Module:
    """Crea el modelo pedido; ignora lr y weight_decay (son del optimizador)."""
    kwargs = {k: v for k, v in params.items() if k not in OPTIMIZER_KEYS}
    if name == "cnn1d":
        return CNN1D(**kwargs)
    if name == "cnn_lstm":
        return CNNLSTM(**kwargs)
    raise ValueError(f"Modelo desconocido: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
