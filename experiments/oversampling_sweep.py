"""Leak-free comparison of oversampling strategies x ML models.

Oversampling before the train/test split is the classic SMOTE mistake: the
test set fills with synthetic points interpolated from (what become)
training rows, scores inflate, and the 50/50 test distribution stops
matching reality. One deliberately leaky run is included for contrast.

This sweep does it properly:
  * split first (shr.data.split, stratified 64/16/20),
  * preprocessing fit on the training fold only,
  * samplers see the training fold only,
  * combos are ranked on the validation fold,
  * the test fold stays untouched — shr/train.py spends it once, on the winner.

Usage:
    python experiments/oversampling_sweep.py           # full run (takes hours)
    python experiments/oversampling_sweep.py --smoke   # 30k-row sanity run

One row per (strategy, model) is appended to experiments/results/*.csv as it
finishes; already-recorded combos are skipped on rerun, so the sweep is safe
to interrupt and resume.
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SMOTENC,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
    SVMSMOTE,
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from shr.data import CAT_IDX, FEATURES, TARGET_AGE, TARGET_SEX, load_solved, make_preprocessor, split

SEED = 42
SVM_CAP = 60_000  # SVMSMOTE fits an SVC (~O(n^2)); infeasible on 458k rows, so it gets a capped subsample


def clf_models(balanced: bool, scale_pos_weight: float) -> dict:
    cw = "balanced" if balanced else None
    return {
        "DecisionTree": DecisionTreeClassifier(random_state=SEED, class_weight=cw),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=SEED, class_weight=cw),
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED, class_weight=cw),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, random_state=SEED, class_weight=cw),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6, tree_method="hist",
            n_jobs=-1, random_state=SEED, scale_pos_weight=scale_pos_weight if balanced else 1.0,
        ),
    }


def reg_models() -> dict:
    return {
        "DecisionTree": DecisionTreeRegressor(random_state=SEED),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=SEED),
        "RandomForest": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=300, random_state=SEED),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=6, tree_method="hist", n_jobs=-1, random_state=SEED,
        ),
        "XGBoost-tuned": XGBRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, tree_method="hist", n_jobs=-1, random_state=SEED,
        ),
    }


def strategies() -> dict:
    """name -> sampler factory (None = no resampling)."""
    return {
        "none": None,
        "class_weight": None,  # cost-sensitive learning: the no-resampling competitor
        "RandomOverSampler": lambda: RandomOverSampler(random_state=SEED),
        "SMOTE": lambda: SMOTE(random_state=SEED),
        "SMOTENC": lambda: SMOTENC(categorical_features=CAT_IDX, random_state=SEED),
        "BorderlineSMOTE": lambda: BorderlineSMOTE(random_state=SEED),
        "ADASYN": lambda: ADASYN(random_state=SEED),
        "KMeansSMOTE": lambda: KMeansSMOTE(random_state=SEED),
        "SMOTETomek": lambda: SMOTETomek(random_state=SEED),
        "SMOTEENN": lambda: SMOTEENN(random_state=SEED),
        "SVMSMOTE": lambda: SVMSMOTE(random_state=SEED),  # last: slowest sampler, must not block the rest
    }


def clf_metrics(model, X, y) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "f1_female": f1_score(y, pred, pos_label=1),
        "precision_female": precision_score(y, pred, pos_label=1, zero_division=0),
        "recall_female": recall_score(y, pred, pos_label=1),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc_female": average_precision_score(y, proba),
    }


def reg_metrics(model, X, y) -> dict:
    pred = model.predict(X)
    return {
        "rmse": root_mean_squared_error(y, pred),
        "mae": mean_absolute_error(y, pred),
        "r2": r2_score(y, pred),
    }


CLF_COLUMNS = [
    "task", "strategy", "model", "n_train", "female_share_train", "sampler_s", "fit_s",
    "accuracy", "balanced_accuracy", "f1_macro", "f1_female", "precision_female",
    "recall_female", "roc_auc", "pr_auc_female", "note", "error",
]
REG_COLUMNS = ["task", "strategy", "model", "n_train", "sampler_s", "fit_s", "rmse", "mae", "r2", "note", "error"]


def append_row(path: Path, row: dict) -> None:
    # fixed schema: partial rows (e.g. sampler failures) must stay column-aligned
    columns = CLF_COLUMNS if row["task"] == "classification" else REG_COLUMNS
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).reindex(columns=columns).to_csv(path, mode="a", header=not path.exists(), index=False)


def done_combos(path: Path) -> set:
    if not path.exists():
        return set()
    prev = pd.read_csv(path)
    return set(zip(prev["strategy"], prev["model"]))


def run_classification(train, val, out_path: Path) -> None:
    done = done_combos(out_path)
    prep = make_preprocessor().fit(train[FEATURES])
    X_train = prep.transform(train[FEATURES]).astype(np.float32)
    X_val = prep.transform(val[FEATURES]).astype(np.float32)
    y_train = (train[TARGET_SEX] == "Female").to_numpy(dtype=np.int8)
    y_val = (val[TARGET_SEX] == "Female").to_numpy(dtype=np.int8)
    spw = float((y_train == 0).sum() / (y_train == 1).sum())

    for strat_name, factory in strategies().items():
        model_set = clf_models(balanced=(strat_name == "class_weight"), scale_pos_weight=spw)
        if strat_name == "none":
            model_set["Dummy-majority"] = DummyClassifier(strategy="most_frequent")
        todo = [m for m in model_set if (strat_name, m) not in done]
        if not todo:
            print(f"[clf] {strat_name}: all done, skipping", flush=True)
            continue

        note = ""
        t0 = time.perf_counter()
        try:
            if factory is None:
                X_res, y_res = X_train, y_train
            else:
                X_in, y_in = X_train, y_train
                if strat_name == "SVMSMOTE" and len(X_train) > SVM_CAP:
                    idx = train_test_split(
                        np.arange(len(X_train)), train_size=SVM_CAP, stratify=y_train, random_state=SEED,
                    )[0]
                    X_in, y_in = X_train[idx], y_train[idx]
                    note = f"sampler ran on {SVM_CAP} stratified subsample (SVC is O(n^2))"
                X_res, y_res = factory().fit_resample(X_in, y_in)
        except Exception as exc:  # noqa: BLE001 - record and move on, sweep must survive
            sampler_s = time.perf_counter() - t0
            print(f"[clf] {strat_name}: SAMPLER FAILED after {sampler_s:.0f}s: {exc!r}", flush=True)
            for model_name in todo:
                append_row(out_path, {
                    "task": "classification", "strategy": strat_name, "model": model_name,
                    "error": f"sampler: {exc!r}"[:300],
                })
            continue
        sampler_s = time.perf_counter() - t0
        print(f"[clf] {strat_name}: sampled to {len(y_res)} rows "
              f"({(y_res == 1).mean():.1%} female) in {sampler_s:.0f}s", flush=True)

        for model_name in todo:
            model = model_set[model_name]
            t0 = time.perf_counter()
            try:
                model.fit(X_res, y_res)
                fit_s = time.perf_counter() - t0
                row = {
                    "task": "classification", "strategy": strat_name, "model": model_name,
                    "n_train": len(y_res), "female_share_train": round(float((y_res == 1).mean()), 4),
                    "sampler_s": round(sampler_s, 1), "fit_s": round(fit_s, 1),
                    **{k: round(v, 4) for k, v in clf_metrics(model, X_val, y_val).items()},
                    "note": note, "error": "",
                }
            except Exception as exc:  # noqa: BLE001
                row = {"task": "classification", "strategy": strat_name, "model": model_name,
                       "note": note, "error": f"fit: {exc!r}"[:300]}
                print(f"[clf] {strat_name} x {model_name}: FAILED: {exc!r}", flush=True)
            else:
                print(f"[clf] {strat_name} x {model_name}: f1_macro={row['f1_macro']:.4f} "
                      f"bal_acc={row['balanced_accuracy']:.4f} ({fit_s:.0f}s)", flush=True)
            append_row(out_path, row)
            del model
            gc.collect()
        del X_res, y_res
        gc.collect()


def run_leak_replication(train, val, out_path: Path) -> None:
    """Run the classic mistake once, for contrast: encoder fit and SMOTE
    applied to all data, split done afterwards. The resulting metrics are
    meaningless by construction — that is the point of the row."""
    if ("LEAKY_smote_before_split", "XGBoost") in done_combos(out_path):
        print("[leak] demonstration already recorded, skipping", flush=True)
        return
    full = pd.concat([train, val])  # the real test fold stays out even here
    prep = make_preprocessor().fit(full[FEATURES])
    X = prep.transform(full[FEATURES]).astype(np.float32)
    y = (full[TARGET_SEX] == "Female").to_numpy(dtype=np.int8)
    t0 = time.perf_counter()
    X_res, y_res = SMOTE(random_state=SEED).fit_resample(X, y)
    X_tr, X_te, y_tr, y_te = train_test_split(X_res, y_res, test_size=0.2, random_state=SEED)
    model = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                          tree_method="hist", n_jobs=-1, random_state=SEED)
    model.fit(X_tr, y_tr)
    row = {
        "task": "classification", "strategy": "LEAKY_smote_before_split", "model": "XGBoost",
        "n_train": len(y_tr), "female_share_train": round(float((y_tr == 1).mean()), 4),
        "sampler_s": 0.0, "fit_s": round(time.perf_counter() - t0, 1),
        **{k: round(v, 4) for k, v in clf_metrics(model, X_te, y_te).items()},
        "note": "LEAKY protocol run for contrast: synthetic samples in the test set. Metrics are invalid.",
        "error": "",
    }
    append_row(out_path, row)
    print(f"[leak] SMOTE-before-split 'accuracy'={row['accuracy']:.4f} f1_macro={row['f1_macro']:.4f} "
          "<- inflated, meaningless by construction", flush=True)


def run_regression(train, val, out_path: Path) -> None:
    done = done_combos(out_path)
    prep = make_preprocessor().fit(train[FEATURES])
    X_train = prep.transform(train[FEATURES]).astype(np.float32)
    X_val = prep.transform(val[FEATURES]).astype(np.float32)
    y_train = train[TARGET_AGE].to_numpy(dtype=np.float32)
    y_val = val[TARGET_AGE].to_numpy(dtype=np.float32)

    model_set = dict(reg_models())
    model_set["Dummy-mean"] = DummyRegressor(strategy="mean")
    for model_name, model in model_set.items():
        if ("none", model_name) in done:
            continue
        t0 = time.perf_counter()
        try:
            model.fit(X_train, y_train)
            row = {
                "task": "regression", "strategy": "none", "model": model_name,
                "n_train": len(y_train), "sampler_s": 0.0, "fit_s": round(time.perf_counter() - t0, 1),
                **{k: round(v, 4) for k, v in reg_metrics(model, X_val, y_val).items()},
                "note": "", "error": "",
            }
        except Exception as exc:  # noqa: BLE001
            row = {"task": "regression", "strategy": "none", "model": model_name, "error": f"fit: {exc!r}"[:300]}
            print(f"[reg] none x {model_name}: FAILED: {exc!r}", flush=True)
        else:
            print(f"[reg] none x {model_name}: r2={row['r2']:.4f} rmse={row['rmse']:.2f} "
                  f"({row['fit_s']:.0f}s)", flush=True)
        append_row(out_path, row)
        del model
        gc.collect()

    # SMOTE applied to regression by treating every age value as a class is a
    # category error. Run it leak-free (train fold only) to show it does not
    # help either way.
    if ("SMOTE_age_as_classes", "XGBoost-tuned") not in done:
        t0 = time.perf_counter()
        try:
            counts = pd.Series(y_train.astype(int)).value_counts()
            keep = np.isin(y_train.astype(int), counts[counts >= 5].index.to_numpy())
            X_res, y_res = SMOTE(k_neighbors=4, random_state=SEED).fit_resample(X_train[keep], y_train[keep].astype(int))
            sampler_s = time.perf_counter() - t0
            print(f"[reg] SMOTE_age_as_classes: {len(y_res)} rows after resampling ({sampler_s:.0f}s)", flush=True)
            model = reg_models()["XGBoost-tuned"]
            t0 = time.perf_counter()
            model.fit(X_res, y_res.astype(np.float32))
            row = {
                "task": "regression", "strategy": "SMOTE_age_as_classes", "model": "XGBoost-tuned",
                "n_train": len(y_res), "sampler_s": round(sampler_s, 1), "fit_s": round(time.perf_counter() - t0, 1),
                **{k: round(v, 4) for k, v in reg_metrics(model, X_val, y_val).items()},
                "note": "methodologically invalid (age is not a class label); included to close the question",
                "error": "",
            }
            print(f"[reg] SMOTE_age_as_classes x XGBoost-tuned: r2={row['r2']:.4f} rmse={row['rmse']:.2f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            row = {"task": "regression", "strategy": "SMOTE_age_as_classes", "model": "XGBoost-tuned",
                   "error": f"{exc!r}"[:300]}
            print(f"[reg] SMOTE_age_as_classes: FAILED: {exc!r}", flush=True)
        append_row(out_path, row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="30k-row sanity run into results_smoke/")
    args = parser.parse_args()

    t_start = time.perf_counter()
    results_dir = ROOT / "experiments" / ("results_smoke" if args.smoke else "results")
    clf_path = results_dir / "classification_results.csv"
    reg_path = results_dir / "regression_results.csv"

    df = load_solved(os.environ.get("SHR_CSV", str(ROOT / "SHR65_22.csv")))
    if args.smoke:
        df = df.sample(n=30_000, random_state=SEED).reset_index(drop=True)
    train, val, test = split(df)
    print(f"rows: train={len(train)} val={len(val)} test={len(test)} (test untouched here)", flush=True)

    run_regression(train, val, reg_path)
    run_leak_replication(train, val, clf_path)
    run_classification(train, val, clf_path)

    print(f"\nTotal wall time: {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)
    clf = pd.read_csv(clf_path)
    ok = clf[clf["error"].fillna("") == ""].sort_values("f1_macro", ascending=False)
    cols = ["strategy", "model", "f1_macro", "balanced_accuracy", "accuracy", "recall_female", "roc_auc"]
    print("\nTop 10 classification combos by validation f1_macro:", flush=True)
    print(ok[cols].head(10).to_string(index=False), flush=True)
    reg = pd.read_csv(reg_path)
    ok = reg[reg["error"].fillna("") == ""].sort_values("r2", ascending=False)
    print("\nRegression by validation R^2:", flush=True)
    print(ok[["strategy", "model", "r2", "rmse", "mae"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
