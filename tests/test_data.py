import pandas as pd
import pytest

from ecg.data import check_no_patient_leakage, load_metadata, split_by_folds

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
FOLDS = {"train": [1, 2, 3, 4, 5, 6, 7, 8], "val": [9], "test": [10]}


def _write_tiny_ptbxl(tmp_path):
    db = pd.DataFrame({
        "ecg_id": [1, 2, 3, 4],
        "patient_id": [10, 10, 20, 30],
        "strat_fold": [1, 1, 9, 10],
        "scp_codes": [
            "{'NORM': 100.0, 'SR': 0.0}",
            "{'IMI': 50.0, 'NDT': 100.0}",
            "{'SR': 0.0}",               # sin superclase diagnostica -> se descarta
            "{'LVH': 100.0, 'CLBBB': 100.0}",
        ],
    })
    db.to_csv(tmp_path / "ptbxl_database.csv", index=False)
    scp = pd.DataFrame({
        "code": ["NORM", "IMI", "NDT", "LVH", "CLBBB", "SR"],
        "diagnostic": [1, 1, 1, 1, 1, None],
        "diagnostic_class": ["NORM", "MI", "STTC", "HYP", "CD", None],
    })
    scp.to_csv(tmp_path / "scp_statements.csv", index=False)


def test_load_metadata_maps_codes_to_superclasses(tmp_path):
    _write_tiny_ptbxl(tmp_path)
    df = load_metadata(tmp_path, CLASSES)

    assert list(df.index) == [1, 2, 4]
    assert df.loc[1, CLASSES].tolist() == [1, 0, 0, 0, 0]
    assert df.loc[2, CLASSES].tolist() == [0, 1, 1, 0, 0]  # multietiqueta
    assert df.loc[4, CLASSES].tolist() == [0, 0, 0, 1, 1]


def test_split_uses_official_folds():
    df = pd.DataFrame({"patient_id": [1, 2, 3, 4], "strat_fold": [1, 8, 9, 10]})
    parts = split_by_folds(df, FOLDS)

    assert len(parts["train"]) == 2
    assert parts["val"]["patient_id"].tolist() == [3]
    assert parts["test"]["patient_id"].tolist() == [4]


def test_split_detects_patient_leakage():
    df = pd.DataFrame({"patient_id": [1, 1, 2], "strat_fold": [1, 10, 9]})
    with pytest.raises(AssertionError, match="pacientes compartidos"):
        split_by_folds(df, FOLDS)


def test_no_leakage_passes_on_disjoint_patients():
    parts = {"a": pd.DataFrame({"patient_id": [1, 2]}), "b": pd.DataFrame({"patient_id": [3]})}
    check_no_patient_leakage(parts)
