"""Shared data, feature sets, training mask and patient-level folds for every model.

    .venv/bin/python -m modeling.common     # (re)build results/folds.csv and print a summary

Models should import from here rather than redefine anything:

    from modeling.common import FEATURE_SET_1, load_feature_table, load_folds, training_mask

Feature sets (METHODOLOGY.md §9), each feature summarised across channels by
median, max and p75, exactly as in helsinki.features:
  FEATURE_SET_1  55 adapted Temko features      -> 165 columns
  FEATURE_SET_2  set 1 + 3 rhythmicity features -> 174 columns
QC flags, n_valid_channels and annotation counts are metadata, never predictors.

Training uses windows with label 0/1 that are eligible and not unusable (§8).
Ambiguous windows (label -1) are never trained on but keep their fold, so they
stay available for evaluation.

Folds are assigned per patient (subject_id), so no patient is in both train and
test. Seizure subjects (at least one label == 1 window) and non-seizure subjects
are each shuffled and split into 5 near-equal parts, and part i of both forms
fold i: every fold gets 9-10 of the 46 seizure subjects and 6-7 of the 33 others.
results/folds.csv is the shared split: teammates should call load_folds(), not
rebuild folds themselves.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from helsinki.features import FEATURE_SETS, SUMMARY_STATS
from helsinki.paths import DATA_ROOT, REPO_ROOT

FEATURE_TABLE = DATA_ROOT / "processed" / "feature_table.parquet"
RESULTS_DIR = REPO_ROOT / "results"
FOLDS_CSV = RESULTS_DIR / "folds.csv"

N_SPLITS = 5
RANDOM_STATE = 0


def _summary_columns(feature_set: str) -> list[str]:
    return [f"{name}_{stat}" for name, *_ in FEATURE_SETS[feature_set] for stat in SUMMARY_STATS]


FEATURE_SET_1 = _summary_columns("baseline")
FEATURE_SET_2 = _summary_columns("secondary")
FEATURE_SETS_BY_NAME = {"set1": FEATURE_SET_1, "set2": FEATURE_SET_2}


def load_feature_table() -> pd.DataFrame:
    """The window table from scripts/build_feature_table.py, checked for the expected columns."""
    if not FEATURE_TABLE.exists():
        raise FileNotFoundError(f"{FEATURE_TABLE} not found; run scripts/build_feature_table.py first")
    df = pd.read_parquet(FEATURE_TABLE)
    missing = [c for c in ["subject_id", "label", "eligible", "unusable", *FEATURE_SET_2] if c not in df]
    if missing:
        raise ValueError(f"feature table is missing columns: {missing}")
    return df


def training_mask(df: pd.DataFrame) -> pd.Series:
    """Windows a model may be trained on: label 0 or 1, eligible, not unusable."""
    return df["label"].isin([0, 1]) & df["eligible"] & ~df["unusable"]


def make_folds(df: pd.DataFrame) -> pd.DataFrame:
    """One row per subject: subject_id, fold (0..N_SPLITS-1), has_seizure and window counts.

    Seizure and non-seizure subjects are split separately with KFold (shuffled,
    RANDOM_STATE), so both are spread as evenly as possible; part i of each is fold i.
    """
    per_subject = df.groupby("subject_id").agg(n_label1_windows=("label", lambda s: int((s == 1).sum())))
    per_subject["has_seizure"] = per_subject["n_label1_windows"] > 0
    train = df[training_mask(df)]
    per_subject["n_train_windows"] = train.groupby("subject_id").size()
    per_subject["n_train_seizure_windows"] = train.groupby("subject_id")["label"].sum()
    per_subject = per_subject.fillna(0).astype({"n_train_windows": int, "n_train_seizure_windows": int})

    fold_of = {}
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for has_seizure in (True, False):
        subjects = np.sort(per_subject.index[per_subject["has_seizure"] == has_seizure].to_numpy())
        for fold, (_, part) in enumerate(cv.split(subjects)):
            for sid in subjects[part]:
                if int(sid) in fold_of:
                    raise AssertionError(f"subject {sid} assigned to two folds")
                fold_of[int(sid)] = fold

    folds = per_subject.reset_index().assign(subject_id=lambda d: d.subject_id.astype(int))
    folds.insert(1, "fold", folds["subject_id"].map(fold_of))
    if folds["fold"].isna().any() or folds["subject_id"].duplicated().any():
        raise AssertionError("every subject must be in exactly one fold")
    return folds[["subject_id", "fold", "has_seizure", "n_label1_windows", "n_train_windows",
                  "n_train_seizure_windows"]].sort_values("subject_id").reset_index(drop=True)


def load_folds() -> pd.DataFrame:
    """The shared fold assignment from results/folds.csv."""
    if not FOLDS_CSV.exists():
        raise FileNotFoundError(f"{FOLDS_CSV} not found; run `python -m modeling.common` first")
    return pd.read_csv(FOLDS_CSV)


def add_folds(df: pd.DataFrame, folds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Copy of df with a `fold` column for every window, including ambiguous ones."""
    folds = load_folds() if folds is None else folds
    out = df.merge(folds[["subject_id", "fold"]], on="subject_id", how="left", validate="many_to_one")
    if out["fold"].isna().any():
        raise ValueError("some windows belong to subjects missing from the fold assignment")
    return out


def main() -> None:
    df = load_feature_table()
    train = training_mask(df)
    print(f"feature table: {len(df)} windows, {df.subject_id.nunique()} subjects")
    print(f"feature set 1: {len(FEATURE_SET_1)} columns | feature set 2: {len(FEATURE_SET_2)} columns")
    print(f"labels, all windows:      {df.label.value_counts().sort_index(ascending=False).to_dict()}")
    print(f"labels, training windows: {df[train].label.value_counts().sort_index(ascending=False).to_dict()} "
          f"({train.sum()} windows, {df[train].subject_id.nunique()} subjects)")

    folds = make_folds(df)
    RESULTS_DIR.mkdir(exist_ok=True)
    folds.to_csv(FOLDS_CSV, index=False)
    print(f"\nsaved {FOLDS_CSV.relative_to(REPO_ROOT)}\n")

    print(f"seizure subjects: {int(folds.has_seizure.sum())} | non-seizure subjects: {int((~folds.has_seizure).sum())}\n")

    per_fold = folds.groupby("fold").agg(subjects=("subject_id", "size"), seizure_subjects=("has_seizure", "sum"),
                                         trainable_windows=("n_train_windows", "sum"),
                                         trainable_label1=("n_train_seizure_windows", "sum"),
                                         label1_windows=("n_label1_windows", "sum"))
    per_fold["seizure_share"] = (per_fold.trainable_label1 / per_fold.trainable_windows).round(3)
    print(per_fold.to_string())
    for fold, row in per_fold.iterrows():
        if row.seizure_subjects == 0:
            warnings.warn(f"fold {fold} has no seizure subjects", stacklevel=1)


if __name__ == "__main__":
    main()
