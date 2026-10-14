"""Utilidades comunes: configuracion, semillas y rutas."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Convierte una ruta del YAML (relativa a la raiz) en absoluta."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
