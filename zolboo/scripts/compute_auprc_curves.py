#!/usr/bin/env python3
"""Compute PR-curve data (not just scalar AUPRC) for LogReg and RF on:
  Task 1: per-epoch preictal==1 classification (pooled out-of-fold across 10 kfolds)
  Task 2: the paper's SPH3/SOP7 alarm-based aggregated evaluation
"""
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc as sk_auc

sys.path.insert(0, "notebooks")
from notebook_utils import run_sph_sop_exp  # noqa: E402

KFOLDS_DIR = "kfolds_all_cork_huh"
BASELINE_EXCLUDE = ['subject_id', 'abs_time_idx', 'preictal', 'augmentation', 'snd_grp', 'pna_szf', 'pna']
NUM_KFOLDS = 10

MODELS = {
    "LogReg": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RF":     lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=42),
}


def load_fold(kfold, split):
    df = pd.read_hdf(f"{KFOLDS_DIR}/{split}_kf{kfold}.hdf")
    df = df[df.preictal.isin([0, 1])].copy()
    feat_cols = [c for c in df.columns if c not in BASELINE_EXCLUDE]
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())
    return df, feat_cols


def task1_pooled_curves():
    pooled_y = {name: [] for name in MODELS}
    pooled_s = {name: [] for name in MODELS}

    for kfold in range(NUM_KFOLDS):
        df_train, feat_cols = load_fold(kfold, "train")
        df_test, _ = load_fold(kfold, "test")
        scaler = PowerTransformer()
        X_train = scaler.fit_transform(df_train[feat_cols].values)
        y_train = df_train['preictal'].values.astype(int)
        X_test = scaler.transform(df_test[feat_cols].values)
        y_test = df_test['preictal'].values.astype(int)

        for name, make_model in MODELS.items():
            m = make_model()
            m.fit(X_train, y_train)
            score = m.predict_proba(X_test)[:, 1]
            pooled_y[name].append(y_test)
            pooled_s[name].append(score)
        print(f"task1 kfold {kfold} done", flush=True)

    out = {}
    for name in MODELS:
        y = np.concatenate(pooled_y[name])
        s = np.concatenate(pooled_s[name])
        precision, recall, _ = precision_recall_curve(y, s)
        auprc = sk_auc(recall, precision)
        out[name] = dict(precision=precision.tolist(), recall=recall.tolist(), auprc=auprc)
        print(f"Task1 {name}: AUPRC={auprc:.4f}")
    return out


def task2_sph3_sop7_curves():
    out = {}
    for name in MODELS:
        df_model = pd.read_csv(f"results_sph_sop/{name}_continuous_preds.csv")
        df_model['feat'] = 'all'
        threshes = np.arange(0.0, 1.01, 0.01)
        SPH_results_dict, _, _, _ = run_sph_sop_exp(
            df_model, name, [3], [7], 3, 3, mp_mode=0, threshes=threshes
        )
        results_dict = SPH_results_dict[3][7]
        precision = np.array(results_dict['precision_scores'])
        recall = np.array(results_dict['recall_scores'])
        not_nan = ~np.isnan(precision) & ~np.isnan(recall)
        precision = precision[not_nan]
        recall = recall[not_nan]
        order = np.argsort(recall)
        recall_sorted = recall[order]
        precision_sorted = precision[order]
        auprc = sk_auc(recall_sorted, precision_sorted)
        out[name] = dict(precision=precision_sorted.tolist(), recall=recall_sorted.tolist(), auprc=auprc)
        print(f"Task2 {name}: AUPRC={auprc:.4f}")
    return out


if __name__ == "__main__":
    result = {
        "task1_per_epoch": task1_pooled_curves(),
        "task2_sph3_sop7": task2_sph3_sop7_curves(),
    }
    with open("results_sph_sop/auprc_curves.json", "w") as f:
        json.dump(result, f)
    print("Saved to results_sph_sop/auprc_curves.json")
