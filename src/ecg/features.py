"""Features hechas a mano para la linea base clasica."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal, stats

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
PERCENTILES = [5, 25, 50, 75, 95]


def lead_features(x: np.ndarray, fs: int, bands: list[list[float]]) -> np.ndarray:
    """Features de una senal (n, muestras) -> (n, k). Vectorizado sobre registros."""
    feats = [
        x.mean(1), x.std(1), x.min(1), x.max(1), np.ptp(x, 1),
        stats.skew(x, axis=1), stats.kurtosis(x, axis=1),
        np.sqrt((x ** 2).mean(1)),
        (np.diff(np.signbit(x - x.mean(1, keepdims=True)), axis=1) != 0).mean(1),
    ]
    feats += list(np.percentile(x, PERCENTILES, axis=1))
    freqs, psd = signal.welch(x, fs=fs, nperseg=min(256, x.shape[1]), axis=1)
    total = psd.sum(1) + 1e-12
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        feats.append(psd[:, mask].sum(1) / total)
    return np.stack(feats, axis=1)


def lead_feature_names(bands: list[list[float]]) -> list[str]:
    names = ["mean", "std", "min", "max", "ptp", "skew", "kurt", "rms", "zcr"]
    names += [f"p{p}" for p in PERCENTILES]
    names += [f"bp_{lo:g}_{hi:g}" for lo, hi in bands]
    return names


def rr_features(lead_ii: np.ndarray, fs: int) -> np.ndarray:
    """Frecuencia cardiaca y variabilidad RR a partir de la derivacion II."""
    b, a = signal.butter(2, [5, 15], btype="band", fs=fs)
    out = np.full((len(lead_ii), 4), np.nan)
    for i, sig in enumerate(lead_ii):
        env = np.abs(signal.filtfilt(b, a, sig))
        peaks, _ = signal.find_peaks(env, distance=int(0.3 * fs), height=0.5 * np.percentile(env, 99))
        if len(peaks) >= 3:
            rr = np.diff(peaks) / fs
            out[i] = [60.0 / rr.mean(), rr.std(), np.sqrt(np.mean(np.diff(rr) ** 2)), len(peaks)]
    return out


def extract_features(
    signals: np.ndarray,
    meta: pd.DataFrame | None,
    fs: int,
    bands: list[list[float]],
) -> pd.DataFrame:
    """signals: (n, muestras, 12). meta: filas alineadas con signals (edad, sexo)."""
    blocks, cols = [], []
    names = lead_feature_names(bands)
    for j, lead in enumerate(LEADS):
        blocks.append(lead_features(np.asarray(signals[:, :, j], dtype=np.float64), fs, bands))
        cols += [f"{lead}_{n}" for n in names]
    blocks.append(rr_features(np.asarray(signals[:, :, 1], dtype=np.float64), fs))
    cols += ["hr_bpm", "rr_std", "rr_rmssd", "n_beats"]

    df = pd.DataFrame(np.concatenate(blocks, axis=1), columns=cols)
    if meta is not None:
        age = meta["age"].to_numpy(dtype=float, copy=True)
        age[age > 120] = np.nan  # PTB-XL codifica >89 anos como 300
        df["age"] = age
        df["sex"] = meta["sex"].astype(float).to_numpy()
    return df
