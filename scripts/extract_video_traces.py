"""Extrae trazados reales de PTB-XL para el video introductorio.

Genera tres bloques en assets/video/traces.json:
  - superclasses: un registro de etiqueta unica por superclase (fold de test).
  - leads: las 12 derivaciones de un mismo registro normal.
  - beat: un latido aislado de ese registro, con los instantes de P, QRS y T.

La salida son muestras normalizadas, no la senal cruda, para que el video no
dependa de data/ al renderizar.

Uso:
    uv run python scripts/extract_video_traces.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "ptb-xl"
OUT = ROOT / "assets" / "video" / "traces.json"

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEAD_NAMES = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]
LEAD_II = 1
SAMPLES = 1000  # 10 s a 100 Hz
FS = 100


def superclasses_from(scp_codes: dict[str, float], mapping: pd.DataFrame) -> set[str]:
    """Superclases diagnosticas presentes, segun scp_statements.csv."""
    out = set()
    for code in scp_codes:
        if code not in mapping.index:
            continue
        row = mapping.loc[code]
        if not isinstance(row, pd.Series):
            row = mapping.loc[[code]].iloc[0]
        value = row.get("diagnostic_class")
        if isinstance(value, str) and value:
            out.add(value)
    return out


def normalise(signal: np.ndarray) -> list[float]:
    """Centra en la mediana y escala a [-1, 1] usando el propio registro."""
    signal = np.nan_to_num(signal, nan=0.0).astype(float)
    signal = signal - np.median(signal)
    span = float(np.max(np.abs(signal)))
    if span > 0:
        signal = signal / span
    return [round(float(v), 4) for v in signal]


def landmarks(signal: np.ndarray) -> dict[str, int]:
    """Ubica los picos de P, R y T alrededor del primer R detectado.

    A 100 Hz la resolucion es de 10 ms, suficiente para ubicar etiquetas.
    """
    flat = signal - np.median(signal)
    r_peak = int(np.argmax(np.abs(flat)))

    # P: maximo local en la ventana 120-260 ms antes de R.
    p_lo = max(0, r_peak - int(0.26 * FS))
    p_hi = max(0, r_peak - int(0.10 * FS))
    p_peak = p_lo + int(np.argmax(np.abs(flat[p_lo:p_hi]))) if p_hi > p_lo else r_peak

    # T: maximo local en la ventana 160-420 ms despues de R.
    t_lo = min(len(flat) - 1, r_peak + int(0.16 * FS))
    t_hi = min(len(flat), r_peak + int(0.42 * FS))
    t_peak = t_lo + int(np.argmax(np.abs(flat[t_lo:t_hi]))) if t_hi > t_lo else r_peak

    # QRS: el tramo de pendiente alta alrededor de R.
    deriv = np.abs(np.diff(flat))
    thresh = 0.25 * float(np.max(deriv))
    qrs_on = r_peak
    while qrs_on > 0 and deriv[qrs_on - 1] > thresh:
        qrs_on -= 1
    qrs_off = r_peak
    while qrs_off < len(deriv) - 1 and deriv[qrs_off] > thresh:
        qrs_off += 1

    return {"p": p_peak, "r": r_peak, "qrs_on": qrs_on, "qrs_off": qrs_off, "t": t_peak}


def main() -> None:
    df = pd.read_csv(DATA / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    mapping = pd.read_csv(DATA / "scp_statements.csv", index_col=0)

    df["superclasses"] = df["scp_codes"].apply(
        lambda codes: superclasses_from(codes, mapping)
    )

    payload: dict[str, object] = {"fs": FS, "lead": "II", "superclasses": {}}
    superclass_block: dict[str, dict] = {}

    for name in SUPERCLASSES:
        candidates = df[
            (df["strat_fold"] == 10)
            & (df["superclasses"].map(len) == 1)
            & (df["superclasses"].map(lambda s: name in s))
        ]
        if candidates.empty:
            raise SystemExit(f"Sin registros de {name} en el fold de test")

        ordered = candidates.sort_values("baseline_drift", na_position="last")
        ecg_id = int(ordered.index[0])
        row = ordered.iloc[0]
        record = wfdb.rdrecord(str(DATA / row["filename_lr"]))
        signal = record.p_signal[:SAMPLES, LEAD_II]

        superclass_block[name] = {
            "ecg_id": ecg_id,
            "n_samples": int(len(signal)),
            "samples": normalise(signal),
        }
        print(f"{name:5s} ecg_id={ecg_id} muestras={len(signal)}")

    payload["superclasses"] = superclass_block

    # Un mismo registro normal aporta las 12 derivaciones y el latido ampliado.
    norm_id = superclass_block["NORM"]["ecg_id"]
    norm_row = df.loc[norm_id]
    norm_record = wfdb.rdrecord(str(DATA / norm_row["filename_lr"]))
    payload["reference_ecg_id"] = norm_id

    payload["leads"] = {
        name: normalise(norm_record.p_signal[:SAMPLES, i])
        for i, name in enumerate(LEAD_NAMES)
    }
    print(f"leads      ecg_id={norm_id} derivaciones={len(LEAD_NAMES)}")

    beat_signal = norm_record.p_signal[:, LEAD_II]
    marks = landmarks(beat_signal)
    pad_lo = int(0.30 * FS)
    pad_hi = int(0.42 * FS)
    lo = max(0, marks["p"] - pad_lo)
    hi = min(len(beat_signal), marks["t"] + pad_hi)
    window = beat_signal[lo:hi]

    payload["beat"] = {
        "ecg_id": norm_id,
        "offset": lo,
        "n_samples": int(len(window)),
        "samples": normalise(window),
        "marks": {key: int(value - lo) for key, value in marks.items()},
    }
    print(
        "beat       "
        f"muestras={len(window)} P={marks['p'] - lo} R={marks['r'] - lo} "
        f"T={marks['t'] - lo}"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Escrito {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
