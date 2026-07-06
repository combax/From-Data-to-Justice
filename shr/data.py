"""Load and clean the SHR (Supplementary Homicide Reports, 1976-2022) dataset.

Single source of truth for feature lists, cleaning rules and the
train/val/test split. Both the experiment sweep and production training
import from here so the two can never drift apart.
"""

import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Features available for UNSOLVED cases too — the deployment target is
# profiling unknown offenders, so offender attributes must never be inputs.
NUMERIC = ["VicAge", "Year"]
CATEGORICAL = ["Weapon", "VicSex", "VicRace", "State", "Agentype", "Month", "Homicide", "ActionType"]
FEATURES = NUMERIC + CATEGORICAL
TARGET_SEX = "OffSex"  # classification target; modelled as 1 = Female (minority class)
TARGET_AGE = "OffAge"  # regression target

# Column index of each categorical feature after make_preprocessor() runs
# (numeric block first, categorical block second) — needed by SMOTENC.
CAT_IDX = list(range(len(NUMERIC), len(FEATURES)))

DEFAULT_CSV = os.environ.get("SHR_CSV", "SHR65_22.csv")

_WEAPON_RENAMES = {
    "Narcotics or drugs, sleeping pills": "Narcotics",
    "Firearm, type not stated": "Firearm",
    "Knife or cutting instrument": "Knife",
    "Other or type unknown": "Unknown",
    "Personal weapons, includes beating": "Personal Weapon",
    "Pushed or thrown out window": "Pushed",
    "Weapon Not Reported": "Not Reported",
}


def clean_weapon(weapon: pd.Series) -> pd.Series:
    """Collapse weapon subtypes ("Strangulation - hanging" -> "Strangulation")."""
    return weapon.str.replace(r" - .*", "", regex=True).replace(_WEAPON_RENAMES)


def load_solved(csv_path: str = DEFAULT_CSV) -> pd.DataFrame:
    """Solved cases with a known offender: the supervised training population.

    999 is the SHR sentinel for unknown age.
    """
    df = pd.read_csv(csv_path, usecols=FEATURES + [TARGET_SEX, TARGET_AGE, "Solved"])
    df["Weapon"] = clean_weapon(df["Weapon"])
    df = df[
        (df["Solved"] == "Yes")
        & (df["VicAge"] != 999)
        & (df[TARGET_AGE] != 999)
        & (df[TARGET_SEX] != "Unknown")
        & (df["VicSex"] != "Unknown")
    ]
    return df.drop(columns=["Solved"]).reset_index(drop=True)


def split(df: pd.DataFrame, seed: int = 42):
    """64/16/20 train/val/test, stratified on offender sex.

    val ranks experiment combos; test is touched exactly once, by the final
    model in shr/train.py. Oversampling must happen strictly after this split
    and only on the training fold — resampling first would leak synthetic
    points into the test set.
    """
    trainval, test = train_test_split(df, test_size=0.20, stratify=df[TARGET_SEX], random_state=seed)
    train, val = train_test_split(trainval, test_size=0.20, stratify=trainval[TARGET_SEX], random_state=seed)
    return train, val, test


def make_preprocessor() -> ColumnTransformer:
    """Scaled numerics + ordinal-coded categoricals, fit on training data only.

    Ordinal codes (not one-hot) so tree models get a compact matrix and
    SMOTENC can be told which columns are categorical. Unseen categories at
    inference time become -1, which tree models handle as just another value.
    """
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
        ]
    )
