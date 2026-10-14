import numpy as np
import pandas as pd

from ecg.features import LEADS, extract_features, lead_feature_names, rr_features
from ecg.metrics import evaluate, expected_calibration_error, tune_thresholds

FS = 100
BANDS = [[0.5, 5], [5, 15], [15, 40]]


def _synthetic_ecg(n=3, bpm=60, seconds=10):
    """Tren de pulsos estrechos a frecuencia fija, igual en las 12 derivaciones."""
    t = np.arange(seconds * FS)
    sig = np.zeros(len(t))
    period = int(FS * 60 / bpm)
    for start in range(period // 2, len(t), period):
        sig[start:start + 3] = [0.5, 1.0, 0.5]
    rng = np.random.default_rng(0)
    return np.stack([np.tile(sig[:, None], (1, 12)) + 0.01 * rng.standard_normal((len(t), 12))
                     for _ in range(n)]).astype(np.float32)


def test_extract_features_shape_and_columns():
    x = _synthetic_ecg()
    meta = pd.DataFrame({"age": [50, 300, 70], "sex": [0, 1, 0]})
    df = extract_features(x, meta, FS, BANDS)

    expected = 12 * len(lead_feature_names(BANDS)) + 4 + 2
    assert df.shape == (3, expected)
    assert f"{LEADS[0]}_mean" in df.columns
    assert np.isnan(df.loc[1, "age"])  # 300 = edad enmascarada en PTB-XL


def test_rr_features_recover_heart_rate():
    x = _synthetic_ecg(n=1, bpm=75)
    hr = rr_features(x[:, :, 1].astype(float), FS)[0, 0]
    assert abs(hr - 75) < 3


def test_ece_is_zero_when_perfectly_calibrated():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.0, 1.0, 0.0, 1.0])
    assert expected_calibration_error(y, p) == 0.0


def test_tune_thresholds_and_evaluate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(200, 2))
    p = np.clip(y * 0.3 + 0.35 + 0.1 * rng.standard_normal(y.shape), 0, 1)
    thr = tune_thresholds(y, p)
    res = evaluate(y, p, thr, ["A", "B"])

    assert thr.shape == (2,)
    assert 0.35 < thr[0] < 0.65
    assert res["f1_macro"] > 0.9
    assert res["per_class"]["A"]["support"] == y[:, 0].sum()
    assert np.array(res["per_class"]["B"]["confusion_matrix"]).sum() == 200
