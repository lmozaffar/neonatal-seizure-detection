#!/usr/bin/env python3
"""
Classical ML baselines (LogReg, RandomForest, KNN, LinearSVC) for comparison
against the ConvLSTM deep model, trained on the same kfolds_all_cork_huh data.

Not part of the original neoSeer repo -- the paper's own baseline code
(backing notebooks/preds_df.pd) isn't published. This reimplements comparable
baselines on per-epoch QEEG features (no sequence windowing), using the same
PowerTransformer scaling approach as utils.fetch_training_objs.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc as sk_auc

KFOLDS_DIR = "kfolds_all_cork_huh"
RESULTS_DIR = "results_baselines"
EXCLUDE_COLS = ['subject_id', 'abs_time_idx', 'preictal', 'augmentation', 'snd_grp', 'pna_szf', 'pna']
NUM_KFOLDS = 10

MODELS = {
    "LogReg": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RF":     lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=42),
    "KNN":    lambda: KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
    "SVC":    lambda: LinearSVC(class_weight="balanced", max_iter=5000, dual="auto"),
}


def load_fold(kfold, split):
    df = pd.read_hdf(f"{KFOLDS_DIR}/{split}_kf{kfold}.hdf")
    # drop ictal/buffer rows (preictal==2); binary interictal(0) vs preictal(1)
    df = df[df.preictal.isin([0, 1])].copy()
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())
    return df, feat_cols


def score(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    return model.decision_function(X_test)


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_fn = f"{RESULTS_DIR}/baseline_results.csv"
    rows = []

    for kfold in range(NUM_KFOLDS):
        print(f"\n=== kfold {kfold} ===", flush=True)
        df_train, feat_cols = load_fold(kfold, "train")
        df_test, _ = load_fold(kfold, "test")

        scaler = PowerTransformer()
        X_train = scaler.fit_transform(df_train[feat_cols].values)
        y_train = df_train['preictal'].values.astype(int)
        X_test = scaler.transform(df_test[feat_cols].values)
        y_test = df_test['preictal'].values.astype(int)

        print(f"  train n={len(y_train)} (pos={y_train.sum()}), test n={len(y_test)} (pos={y_test.sum()})", flush=True)

        for name, make_model in MODELS.items():
            t0 = time.time()
            model = make_model()
            model.fit(X_train, y_train)

            y_score = score(model, X_test)
            y_pred = (y_score >= (0.5 if hasattr(model, "predict_proba") else 0.0)).astype(int)

            auroc = roc_auc_score(y_test, y_score)
            f1 = f1_score(y_test, y_pred)
            prec, rec, _ = precision_recall_curve(y_test, y_score)
            prauc = sk_auc(rec, prec)
            elapsed = time.time() - t0

            print(f"  {name:8s} AUROC={auroc:.4f} F1={f1:.4f} PR-AUC={prauc:.4f} ({elapsed:.1f}s)", flush=True)

            rows.append(dict(
                kfold=kfold, model=name,
                n_train=len(y_train), n_test=len(y_test),
                test_rocauc=auroc, test_f1=f1, test_prauc=prauc,
                train_time_s=elapsed,
            ))
            pd.DataFrame(rows).to_csv(out_fn, index=False)

    print(f"\nDone. Results written to {out_fn}")


if __name__ == "__main__":
    run()
