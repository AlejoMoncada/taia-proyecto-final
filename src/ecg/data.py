"""Descarga de PTB-XL, etiquetas de superclase y particion por paciente.

Uso:
    uv run python -m ecg.data --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from ecg.utils import load_config, resolve

DATASET_DIRNAME = "ptb-xl"


# ---------------------------------------------------------------- descarga

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(session: requests.Session, url: str, dest: Path, sha: str | None) -> None:
    """Descarga un archivo si falta o si su hash no coincide con el oficial."""
    if dest.exists() and (sha is None or _sha256(dest) == sha):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(r.content)
            if sha is not None and _sha256(tmp) != sha:
                raise OSError(f"hash distinto en {dest.name}")
            tmp.replace(dest)
            return
        except (requests.RequestException, OSError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def download_ptbxl(base_url: str, raw_dir: Path, sampling_rate: int = 100, workers: int = 16) -> Path:
    """Descarga los CSV y los registros de la frecuencia pedida desde el mirror de PhysioNet.

    Cada archivo se verifica contra SHA256SUMS.txt. Es idempotente y se puede reanudar:
    los archivos que ya estan bien no se vuelven a bajar.
    """
    base_url = base_url.rstrip("/") + "/"
    target = raw_dir / DATASET_DIRNAME
    target.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    _fetch(session, base_url + "SHA256SUMS.txt", target / "SHA256SUMS.txt", None)
    sums = {}
    for line in (target / "SHA256SUMS.txt").read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        sums[name.strip()] = sha

    for name in ("ptbxl_database.csv", "scp_statements.csv", "LICENSE.txt"):
        _fetch(session, base_url + name, target / name, sums.get(name))

    col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    stems = pd.read_csv(target / "ptbxl_database.csv", usecols=[col])[col]
    files = [f"{s}{ext}" for s in stems for ext in (".hea", ".dat")]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(_fetch, session, base_url + f, target / f, sums.get(f)) for f in files]
        for job in tqdm(as_completed(jobs), total=len(jobs), desc=f"records{sampling_rate}"):
            job.result()
    return target


# ---------------------------------------------------------------- etiquetas

def load_metadata(ptbxl_dir: Path, classes: list[str]) -> pd.DataFrame:
    """Lee ptbxl_database.csv y agrega una columna binaria por superclase."""
    df = pd.read_csv(ptbxl_dir / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    scp = pd.read_csv(ptbxl_dir / "scp_statements.csv", index_col=0)
    scp = scp[scp["diagnostic"] == 1]
    code_to_class = scp["diagnostic_class"].to_dict()

    df["superclasses"] = df["scp_codes"].apply(
        lambda codes: sorted({code_to_class[c] for c in codes if c in code_to_class})
    )
    for c in classes:
        df[c] = df["superclasses"].apply(lambda s, c=c: int(c in s))
    # Se descartan registros sin ninguna superclase diagnostica (practica estandar del benchmark).
    return df[df["superclasses"].map(len) > 0].copy()


# ---------------------------------------------------------------- particion

def split_by_folds(df: pd.DataFrame, folds: dict[str, list[int]]) -> dict[str, pd.DataFrame]:
    """Particion con strat_fold. Verifica que ningun paciente quede en dos particiones."""
    parts = {name: df[df["strat_fold"].isin(fs)] for name, fs in folds.items()}
    check_no_patient_leakage(parts)
    return parts


def check_no_patient_leakage(parts: dict[str, pd.DataFrame]) -> None:
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(parts[a]["patient_id"]) & set(parts[b]["patient_id"])
            if shared:
                raise AssertionError(f"{len(shared)} pacientes compartidos entre {a} y {b}")


# ---------------------------------------------------------------- senales

def load_signals(df: pd.DataFrame, ptbxl_dir: Path, sampling_rate: int = 100) -> np.ndarray:
    """Devuelve un arreglo (n, muestras, 12) en float32."""
    import wfdb

    col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    sigs = [wfdb.rdsamp(str(ptbxl_dir / f))[0] for f in tqdm(df[col], desc="Leyendo ECG")]
    return np.asarray(sigs, dtype=np.float32)


def load_or_build_cache(cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """Carga metadatos + senales; guarda un .npy en processed/ para no releer los 21k archivos."""
    dcfg = cfg["data"]
    ptbxl_dir = resolve(dcfg["raw_dir"]) / DATASET_DIRNAME
    proc = resolve(dcfg["processed_dir"])
    proc.mkdir(parents=True, exist_ok=True)
    sr = dcfg["sampling_rate"]

    df = load_metadata(ptbxl_dir, dcfg["classes"])
    cache = proc / f"signals_{sr}hz.npy"
    ids_cache = proc / f"signals_{sr}hz_ids.npy"
    if cache.exists() and ids_cache.exists() and np.array_equal(np.load(ids_cache), df.index.values):
        X = np.load(cache, mmap_mode="r")
    else:
        X = load_signals(df, ptbxl_dir, sr)
        np.save(cache, X)
        np.save(ids_cache, df.index.values)
    return df, X


def main() -> None:
    ap = argparse.ArgumentParser(description="Descarga PTB-XL y genera la particion por paciente.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--skip-signals", action="store_true", help="Solo descarga y split, sin cache de senales")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg = cfg["data"]

    ptbxl_dir = download_ptbxl(dcfg["url"], resolve(dcfg["raw_dir"]), dcfg["sampling_rate"])
    df = load_metadata(ptbxl_dir, dcfg["classes"])
    parts = split_by_folds(df, dcfg["folds"])

    proc = resolve(dcfg["processed_dir"])
    proc.mkdir(parents=True, exist_ok=True)
    split = pd.concat([p.assign(split=name)[["patient_id", "strat_fold", "split"]] for name, p in parts.items()])
    split.to_csv(proc / "split.csv")

    print(f"Registros con superclase: {len(df)}")
    for name, p in parts.items():
        prev = p[dcfg["classes"]].mean().round(3).to_dict()
        print(f"  {name:5s} n={len(p):5d} pacientes={p['patient_id'].nunique():5d} prevalencia={prev}")
    print("Sin pacientes compartidos entre particiones.")

    if not args.skip_signals:
        load_or_build_cache(cfg)
        print(f"Cache de senales en {proc}")


if __name__ == "__main__":
    main()
