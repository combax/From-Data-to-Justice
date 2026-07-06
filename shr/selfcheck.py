"""The one runnable check: python -m shr.selfcheck

Assert-based, no framework. Covers the pure logic that would silently corrupt
results (cleaning, encoding, the split contract, metric formulas) and — when
artifacts exist — that the served models load and predict sanely.
"""

import numpy as np
import pandas as pd

from shr.data import CATEGORICAL, FEATURES, NUMERIC, TARGET_SEX, clean_weapon, make_preprocessor, split
from shr.train import classification_metrics, regression_metrics


def check_clean_weapon() -> None:
    got = clean_weapon(pd.Series([
        "Strangulation - hanging",
        "Knife or cutting instrument",
        "Firearm, type not stated",
        "Handgun",
    ]))
    assert list(got) == ["Strangulation", "Knife", "Firearm", "Handgun"], list(got)


def _toy_frame(n: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "VicAge": rng.integers(1, 90, n),
        "Year": rng.integers(1976, 2022, n),
        "Weapon": rng.choice(["Handgun", "Knife", "Shotgun"], n),
        "VicSex": rng.choice(["Male", "Female"], n),
        "VicRace": rng.choice(["White", "Black"], n),
        "State": rng.choice(["Texas", "California"], n),
        "Agentype": ["Municipal police"] * n,
        "Month": rng.choice(["January", "July"], n),
        "Homicide": ["Murder and non-negligent manslaughter"] * n,
        "ActionType": ["Normal update"] * n,
    })


def check_preprocessor() -> None:
    toy = _toy_frame()
    prep = make_preprocessor().fit(toy)
    out = prep.transform(toy)
    assert out.shape == (len(toy), len(FEATURES)), out.shape
    # numeric block is standardised
    assert abs(out[:, 0].mean()) < 1e-9, "VicAge not centred"
    # unseen category must encode to -1 (column order: numerics then categoricals)
    unseen = toy.head(1).copy()
    unseen["Weapon"] = "Laser"
    weapon_col = len(NUMERIC) + CATEGORICAL.index("Weapon")
    assert prep.transform(unseen)[0, weapon_col] == -1, "unknown category must map to -1"


def check_split() -> None:
    df = _toy_frame(1000)
    df[TARGET_SEX] = ["Female" if i % 10 == 0 else "Male" for i in range(1000)]
    train, val, test = split(df)
    assert (len(train), len(val), len(test)) == (640, 160, 200)
    assert not (set(train.index) & set(val.index) or set(train.index) & set(test.index)
                or set(val.index) & set(test.index)), "folds must be disjoint"
    for fold in (train, val, test):  # stratification keeps the 10% minority share
        share = (fold[TARGET_SEX] == "Female").mean()
        assert abs(share - 0.10) < 0.01, share


def check_metric_formulas() -> None:
    y = np.array([0, 0, 1, 1])
    perfect = classification_metrics(y, y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert perfect["f1_macro"] == 1.0 and perfect["roc_auc"] == 1.0, perfect
    reg = regression_metrics(np.array([10.0, 20.0]), np.array([10.0, 24.0]))
    assert reg["mae_years"] == 2.0 and abs(reg["rmse_years"] - 8**0.5) < 1e-3, reg  # values are rounded to 4dp


def check_artifacts() -> None:
    import json
    import os
    from pathlib import Path

    import joblib

    # same env var shr.api honours (not imported from there: shr.api loads
    # the models at import time and this check must also run without them)
    MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))

    if not (MODELS_DIR / "sex_classifier.joblib").exists():
        print("  (skip) no artifacts in models/ - run `python -m shr.train` to cover this check")
        return
    clf = joblib.load(MODELS_DIR / "sex_classifier.joblib")
    reg = joblib.load(MODELS_DIR / "age_regressor.joblib")
    case = _toy_frame(1)
    p1, p2 = (float(clf.predict_proba(case)[0, 1]) for _ in range(2))
    assert p1 == p2, "prediction must be deterministic"
    assert 0.0 <= p1 <= 1.0, p1
    age = float(reg.predict(case)[0])
    assert 0.0 < age < 120.0, age
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
    assert metrics["classification_test"]["f1_macro"] > 0.47, "worse than the always-Male dummy"
    assert metrics["regression_test"]["r2"] > 0.0, "worse than predicting the mean age"


def main() -> None:
    for check in (check_clean_weapon, check_preprocessor, check_split, check_metric_formulas, check_artifacts):
        check()
        print(f"  {check.__name__} ok")
    print("selfcheck OK")


if __name__ == "__main__":
    main()
