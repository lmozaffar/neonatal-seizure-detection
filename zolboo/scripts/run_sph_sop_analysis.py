#!/usr/bin/env python3
"""
Reproduce the paper's SPH/SOP (seizure prediction horizon / onset period) alarm-based
evaluation from notebooks/Figures_4.ipynb, for all 5 models (ConvLSTM + 4 baselines),
instead of just ConvLSTM as in the original notebook.

Unlike the raw per-epoch AUROC in run_baselines.py / run_tsai.py, this builds continuous
per-epoch prediction scores (keeping ictal rows in, since sz_mask needs them to find
seizure onsets/offsets) and runs them through the authors' own alarm-simulation code
in notebooks/notebook_utils.py.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "notebooks")
from notebook_utils import run_sph_sop_exp, get_sph_sop_exp  # noqa: E402

from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from tsai.all import LSTMPlus  # noqa: E402

KFOLDS_DIR = "kfolds_all_cork_huh"
MULTIRUN_DIR = "multirun/2026-10-05/09-36-54"
NUM_KFOLDS = 10
WINLEN = 5  # matches preictal_duration=5 used for the ConvLSTM full run

BASELINE_EXCLUDE = ['subject_id', 'abs_time_idx', 'preictal', 'augmentation', 'snd_grp', 'pna_szf', 'pna']
CONVLSTM_EXCLUDE = ['subject_id', 'abs_time_idx', 'preictal', 'augmentation', 'snd_grp']

BASELINE_MODELS = {
    "LogReg": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RF":     lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=42),
    "KNN":    lambda: KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
    "SVC":    lambda: LinearSVC(class_weight="balanced", max_iter=5000, dual="auto"),
}


def load_fold_raw(kfold, split, exclude_cols, fillna="median"):
    df = pd.read_hdf(f"{KFOLDS_DIR}/{split}_kf{kfold}.hdf")
    df = df.sort_values(['subject_id', 'abs_time_idx']).reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in exclude_cols]
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    if fillna == "median":
        med = df[feat_cols].median()
        df[feat_cols] = df[feat_cols].fillna(med.fillna(0))
    else:
        df[feat_cols] = df[feat_cols].fillna(0)
    return df, feat_cols


def score_baseline_model(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    s = model.decision_function(X)
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def build_rows(df, scores):
    """df sorted by subject_id,(abs_time_idx); scores aligned 1:1 with df rows."""
    out = []
    for subj, g in df.groupby('subject_id', sort=False):
        idx = g.index.values
        n = len(idx)
        out.append(pd.DataFrame({
            'subject_id': subj,
            't': np.arange(n),
            'sz_mask': (g['preictal'].values == 2),
            'y_preds_run0': scores[idx],
        }))
    return pd.concat(out, ignore_index=True)


def convlstm_window_scores(df, feat_cols, model):
    """Slide a window of WINLEN rows (stride 1) per subject, score with the loaded model.
    Returns a results dataframe aligned to the LAST row of each window (t = local 0-index
    within the subject's prediction series, which starts WINLEN-1 rows into the recording)."""
    X_scaled = df[feat_cols].values.astype(np.float32)
    rows = []
    for subj, g in df.groupby('subject_id', sort=False):
        idx = g.index.values
        n = len(idx)
        if n < WINLEN:
            continue
        X_subj = X_scaled[idx]  # (n, c_in)
        # build all sliding windows at once: (n-WINLEN+1, c_in, WINLEN)
        windows = np.stack([X_subj[i:i+WINLEN].T for i in range(n - WINLEN + 1)], axis=0)
        with torch.no_grad():
            logits = model(torch.from_numpy(windows))
            probs = F.softmax(logits, dim=1)[:, 1].numpy()
        sz_mask_full = (g['preictal'].values == 2)
        # prediction i corresponds to window ending at local row (i+WINLEN-1)
        sz_mask_aligned = sz_mask_full[WINLEN - 1:]
        rows.append(pd.DataFrame({
            'subject_id': subj,
            't': np.arange(len(probs)),
            'sz_mask': sz_mask_aligned,
            'y_preds_run0': probs,
        }))
    return pd.concat(rows, ignore_index=True)


def run():
    os.makedirs("results_sph_sop", exist_ok=True)
    per_model_dfs = {name: [] for name in list(BASELINE_MODELS.keys()) + ["ConvLSTM"]}

    for kfold in range(NUM_KFOLDS):
        print(f"\n=== kfold {kfold}: building continuous predictions ===", flush=True)
        t0 = time.time()

        # --- Baselines ---
        df_train_b, feat_cols_b = load_fold_raw(kfold, "train", BASELINE_EXCLUDE)
        df_test_b, _ = load_fold_raw(kfold, "test", BASELINE_EXCLUDE)

        scaler_b = PowerTransformer()
        X_train_b = scaler_b.fit_transform(df_train_b[feat_cols_b].values)
        y_train_b = (df_train_b['preictal'].values == 1).astype(int)  # binary: preictal vs rest
        X_test_b = scaler_b.transform(df_test_b[feat_cols_b].values)

        for name, make_model in BASELINE_MODELS.items():
            m = make_model()
            m.fit(X_train_b, y_train_b)
            scores = score_baseline_model(m, X_test_b)
            fold_df = build_rows(df_test_b, scores)
            fold_df['kfold'] = kfold
            per_model_dfs[name].append(fold_df)
            print(f"  {name} scored ({time.time()-t0:.1f}s elapsed)", flush=True)

        # --- ConvLSTM ---
        df_train_c, feat_cols_c = load_fold_raw(kfold, "train", CONVLSTM_EXCLUDE, fillna="zero")
        df_test_c, _ = load_fold_raw(kfold, "test", CONVLSTM_EXCLUDE, fillna="zero")

        scaler_c = PowerTransformer()
        scaler_c.fit(df_train_c[feat_cols_c].values)
        df_test_c = df_test_c.copy()
        df_test_c[feat_cols_c] = scaler_c.transform(df_test_c[feat_cols_c].values)

        model = LSTMPlus(c_in=len(feat_cols_c), c_out=2, seq_len=WINLEN, n_layers=2,
                          bidirectional=True, hidden_size=100, rnn_dropout=0, fc_dropout=0.2)
        ckpt = torch.load(f"{MULTIRUN_DIR}/{kfold}/models/model.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt, strict=True)
        model.eval()

        fold_df = convlstm_window_scores(df_test_c, feat_cols_c, model)
        fold_df['kfold'] = kfold
        per_model_dfs["ConvLSTM"].append(fold_df)
        print(f"  ConvLSTM scored ({time.time()-t0:.1f}s elapsed)", flush=True)

    # Concatenate across folds (every subject appears exactly once as test)
    master_dfs = {}
    for name, dfs in per_model_dfs.items():
        master = pd.concat(dfs, ignore_index=True)
        master['feat'] = 'all'  # run_sph_sop_exp reads this column but never uses its value
        master_dfs[name] = master
        master.to_csv(f"results_sph_sop/{name}_continuous_preds.csv", index=False)
        print(f"{name}: {len(master)} total rows across {master.subject_id.nunique()} subjects")

    # --- Run SPH/SOP alarm analysis per model ---
    SPHs = range(1, 8)
    SOPs = range(1, 8)
    GLUE_TIME = 3
    WINS_PER_MIN = 3
    threshes = np.arange(0.0, 1.01, 0.01)

    summary_rows = []
    all_auroc = {}
    for name, df_model in master_dfs.items():
        print(f"\n######## Running SPH/SOP grid for {name} ########", flush=True)
        t0 = time.time()
        SPH_results_dict, _, _, _ = run_sph_sop_exp(
            df_model, name, SPHs, SOPs, GLUE_TIME, WINS_PER_MIN, mp_mode=0, threshes=threshes
        )
        (_, auroc_scores, _, _, auprc_scores, *_rest) = get_sph_sop_exp(SPH_results_dict, plot_on=False, model_num=0)
        all_auroc[name] = auroc_scores

        vals = np.array(list(auroc_scores.values()))
        headline = auroc_scores.get("SPH3_SOP7", np.nan)
        summary_rows.append(dict(
            model=name, headline_SPH3_SOP7=headline,
            mean_auroc=vals.mean(), max_auroc=vals.max(),
            elapsed_s=time.time() - t0,
        ))
        print(f"{name}: headline SPH3/SOP7 AUROC={headline:.4f}, mean={vals.mean():.4f}, max={vals.max():.4f} ({time.time()-t0:.1f}s)", flush=True)

        with open(f"results_sph_sop/{name}_auroc_grid.pkl", "wb") as f:
            pickle.dump(auroc_scores, f)

    summary = pd.DataFrame(summary_rows).sort_values("headline_SPH3_SOP7", ascending=False)
    summary.to_csv("results_sph_sop/sph_sop_summary.csv", index=False)
    print("\n=== FINAL SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
