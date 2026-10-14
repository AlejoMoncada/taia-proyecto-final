"""Busqueda aleatoria de hiperparametros (comun a todas las familias de modelos)."""

from __future__ import annotations

import json
import random
from typing import Any


def sample_configs(space: dict[str, list[Any]], n_trials: int, seed: int) -> list[dict[str, Any]]:
    """Devuelve n_trials configuraciones distintas muestreadas del espacio, de forma reproducible."""
    rng = random.Random(seed)
    keys = sorted(space)
    total = 1
    for k in keys:
        total *= len(space[k])
    n_trials = min(n_trials, total)

    seen, configs = set(), []
    while len(configs) < n_trials:
        cfg = {k: space[k][rng.randrange(len(space[k]))] for k in keys}
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    return configs
