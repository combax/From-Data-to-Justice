"""Train the production models and spend the held-out test fold exactly once.

The winning configurations were selected on the VALIDATION fold by
experiments/oversampling_sweep.py (full grid in experiments/results/).
This script refits them on train+val and reports the final numbers on the
untouched test fold — no synthetic sample ever crosses the split boundary.

Run from the repo root:
    python -m shr.train [--csv SHR65_22.csv] [--out models]
"""

import argparse
import json
import time
from pathlib import Path

import imblearn
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from xgboost import XGBClassifier, XGBRegressor

from shr import __version__
from shr.data import DEFAULT_CSV, FEATURES, TARGET_AGE, TARGET_SEX, load_solved, make_preprocessor, split

SEED = 42


def make_classifier() -> Pipeline:
    """Offender sex (1 = Female, the 11.7% minority class).

    SMOTEENN + XGBoost: best validation f1_macro in the sweep. The sampler
    only ever runs inside fit(), on training data — never at predict time.
    """
    return Pipeline([
        ("prep", make_preprocessor()),
        ("sampler", SMOTEENN(random_state=SEED)),
        ("model", XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            tree_method="hist", n_jobs=-1, random_state=SEED,
        )),
    ])


def make_regressor() -> Pipeline:
    """Offender age. No resampling: oversampling is a class-imbalance tool —
    the sweep shows 'SMOTE for regression' (each age treated as a class)
    hurts even when applied leak-free.

    XGBoost: best validation R^2 (HistGradientBoosting ties to the 3rd decimal).
    """
    return Pipeline([
        ("prep", make_preprocessor()),
        ("model", XGBRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            tree_method="hist", n_jobs=-1, random_state=SEED,
        )),
    ])


def classification_metrics(y_true, pred, proba) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, pred), 4),
        "f1_macro": round(f1_score(y_true, pred, average="macro"), 4),
        "f1_female": round(f1_score(y_true, pred, pos_label=1), 4),
        "precision_female": round(precision_score(y_true, pred, pos_label=1, zero_division=0), 4),
        "recall_female": round(recall_score(y_true, pred, pos_label=1), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "pr_auc_female": round(average_precision_score(y_true, proba), 4),
    }


def regression_metrics(y_true, pred) -> dict:
    return {
        "rmse_years": round(root_mean_squared_error(y_true, pred), 4),
        "mae_years": round(mean_absolute_error(y_true, pred), 4),
        "r2": round(r2_score(y_true, pred), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="path to SHR65_22.csv")
    parser.add_argument("--out", default="models", help="output directory for artifacts")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_solved(args.csv)
    train, val, test = split(df)
    trainval = pd.concat([train, val])
    print(f"training on {len(trainval)} rows (train+val), final evaluation on {len(test)} test rows", flush=True)

    X_fit, X_test = trainval[FEATURES], test[FEATURES]

    t0 = time.perf_counter()
    y_fit = (trainval[TARGET_SEX] == "Female").to_numpy(dtype=np.int8)
    y_test = (test[TARGET_SEX] == "Female").to_numpy(dtype=np.int8)
    clf = make_classifier().fit(X_fit, y_fit)
    # ponytail: the fitted sampler pickles its nearest-neighbour state (tens of
    # MB) but is never used at predict time — swap it out before serialising.
    clf.steps[1] = ("sampler", "passthrough")
    clf_metrics = classification_metrics(y_test, clf.predict(X_test), clf.predict_proba(X_test)[:, 1])
    print(f"classifier trained in {(time.perf_counter() - t0) / 60:.1f} min: {clf_metrics}", flush=True)

    t0 = time.perf_counter()
    reg = make_regressor().fit(X_fit, trainval[TARGET_AGE].to_numpy(dtype=np.float32))
    reg_metrics = regression_metrics(test[TARGET_AGE].to_numpy(dtype=np.float32), reg.predict(X_test))
    print(f"regressor trained in {(time.perf_counter() - t0) / 60:.1f} min: {reg_metrics}", flush=True)

    joblib.dump(clf, out / "sex_classifier.joblib", compress=3)
    joblib.dump(reg, out / "age_regressor.joblib", compress=3)
    metrics = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "shr_version": __version__,
        "versions": {
            "scikit-learn": sklearn.__version__,
            "imbalanced-learn": imblearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
        },
        "n_train": len(trainval),
        "n_test": len(test),
        "female_share_test": round(float(y_test.mean()), 4),
        "classifier": "SMOTEENN + XGBoost(n_estimators=300, lr=0.1, max_depth=6)",
        "regressor": "XGBoost(n_estimators=300, lr=0.1, max_depth=6), no resampling",
        "classification_test": clf_metrics,
        "regression_test": reg_metrics,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"artifacts written to {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
