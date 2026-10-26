"""Extract one textbook-clear PTB-XL example per superclass for the defense deck.

Every record is validated by a human, has a single diagnostic superclass and its main
statement at 100 % likelihood. Each one is shown in the lead where the finding is visible.
A 0.5 Hz high-pass removes baseline wander for display, as an electrocardiograph does;
the models never see this filtered version.

Writes assets/slides/ecg_examples.json (normalized samples, no raw data).

Usage:
    uv run python scripts/extract_slide_examples.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "ptb-xl"
OUT = ROOT / "assets" / "slides" / "ecg_examples.json"
FS = 100
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Chosen by reviewing the candidate traces; see the deck's slide on the five superclasses.
EXAMPLES = {
    "NORM": (4805, "II", "Ritmo sinusal normal"),
    "MI": (16052, "V2", "Infarto anteroseptal con elevación del ST"),
    "STTC": (941, "V5", "Isquemia: descenso del ST e inversión de la onda T"),
    "CD": (4387, "II", "Bloqueo AV de primer grado: PR prolongado"),
    "HYP": (1996, "V5", "Hipertrofia ventricular izquierda: criterio de voltaje"),
}


def main() -> None:
    db = pd.read_csv(DATA / "ptbxl_database.csv", index_col="ecg_id")
    b, a = butter(2, 0.5 / (FS / 2), "highpass")
    out = {"fs": FS, "highpass_hz": 0.5, "examples": {}}
    for cls, (ecg_id, lead, finding) in EXAMPLES.items():
        row = db.loc[ecg_id]
        assert row["validated_by_human"], f"{ecg_id} is not human-validated"
        signal = wfdb.rdrecord(str(DATA / row["filename_lr"])).p_signal[:, LEADS.index(lead)]
        signal = filtfilt(b, a, signal)
        signal = signal / float(np.max(np.abs(signal)))
        codes = {k: v for k, v in ast.literal_eval(row["scp_codes"]).items() if k != "SR"}
        out["examples"][cls] = {"ecg_id": ecg_id, "lead": lead, "finding": finding, "scp_codes": codes,
                                "samples": [round(float(v), 4) for v in signal]}
        print(f"{cls}: ECG {ecg_id}, {lead}, {codes}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"Escrito {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
