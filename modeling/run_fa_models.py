"""Further models on feature set 2 for false-alarm reduction, cross-validated like modeling/run_experiments.py.

    .venv/bin/python -m modeling.run_fa_models                                   # all three models
    .venv/bin/python -m modeling.run_fa_models --models RF_leaf20 --folds 0 --out-dir results/test

Models (all on the 174 set-2 columns of the old feature table, with the folds of results/folds.csv):
  LightGBM      gradient-boosted trees, class_weight="balanced"
  LightGBM_ctx  LightGBM on the 174 columns plus, per column, its causal context: the mean over the same
                baby's previous CONTEXT_WINDOWS windows (30 s at a 6 s stride), never the current or a later
                window. Ineligible windows have no features, so the mean is over the past windows that have
                them; the first window of a recording has no past and its context is missing.
  RF_leaf20     the RF of modeling/models.py with min_samples_leaf=20 instead of 5

Each model is a Pipeline whose median imputation is learned from the training folds only (LightGBM needs no
scaling). Training uses training_mask windows of the other folds' patients; every window of the held-out patients
is scored, as in run_experiments.py.

Outputs in --out-dir (default results/):
  cv_auroc_fa_reduction.csv          rows as in cv_auroc.csv (feature_set "set2"); rows of the models run replace
                                     earlier ones
  predictions_<model>_set2.parquet   subject_id, start_s, end_s, label, eligible, score, fold
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from modeling.common import FEATURE_SET_2, N_SPLITS, RESULTS_DIR, add_folds, load_feature_table, training_mask
from modeling.models import RANDOM_STATE, _pipeline, predict_scores
from modeling.run_experiments import PREDICTION_COLUMNS, evaluate, save_cv_rows

MODEL_ORDER = ["RF_leaf20", "LightGBM", "LightGBM_ctx"]
CV_FILE = "cv_auroc_fa_reduction.csv"
CONTEXT_WINDOWS = 5
N_JOBS = 4              # leaves cores for a concurrent ConvLSTM run; results do not depend on it
CONTEXT_COLUMNS = [f"{c}_past{CONTEXT_WINDOWS}" for c in FEATURE_SET_2]


def lightgbm() -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("model", LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                                              min_child_samples=50, subsample=0.8, subsample_freq=1,
                                              colsample_bytree=0.8, class_weight="balanced",
                                              random_state=RANDOM_STATE, n_jobs=N_JOBS, deterministic=True,
                                              force_row_wise=True, verbose=-1))])


def get_models() -> dict:
    """Fresh, unfitted models by name."""
    return {
        "RF_leaf20": _pipeline(RandomForestClassifier(n_estimators=300, min_samples_leaf=20, class_weight="balanced",
                                                      n_jobs=N_JOBS, random_state=RANDOM_STATE)),
        "LightGBM": lightgbm(),
        "LightGBM_ctx": lightgbm(),
    }


FEATURES = {"RF_leaf20": FEATURE_SET_2, "LightGBM": FEATURE_SET_2, "LightGBM_ctx": FEATURE_SET_2 + CONTEXT_COLUMNS}


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with CONTEXT_COLUMNS: per subject, the mean of each set-2 column over the previous
    CONTEXT_WINDOWS windows, excluding the current one."""
    if not (df.groupby("subject_id").window_index.diff().dropna() == 1).all():
        raise ValueError("feature table must hold every window of a subject, in order")
    past = df.groupby("subject_id")[FEATURE_SET_2].shift(1)
    context = (past.groupby(df.subject_id).rolling(CONTEXT_WINDOWS, min_periods=1).mean()
               .reset_index(level=0, drop=True).sort_index())
    context.columns = CONTEXT_COLUMNS
    return pd.concat([df, context], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(N_SPLITS)))
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    models = [m for m in MODEL_ORDER if m in args.models]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = add_folds(load_feature_table())
    if "LightGBM_ctx" in models:
        df = add_context(df)
    train_mask = training_mask(df)
    print(f"{len(df)} windows, {df.subject_id.nunique()} subjects | models {models} | folds {args.folds}\n",
          flush=True)

    for model_name in models:
        t_model = time.time()
        features = FEATURES[model_name]
        rows, predictions = [], []
        for fold in args.folds:
            train = df[train_mask & (df.fold != fold)]
            held_out = df[df.fold == fold]
            model = get_models()[model_name]
            t0 = time.time()
            model.fit(train[features], train.label)
            t_fit = time.time() - t0
            held_out = held_out.assign(score=predict_scores(model, held_out[features]))
            t_pred = time.time() - t0 - t_fit
            row = evaluate(held_out, "set2", model_name, fold)
            rows.append({**row, "fit_s": round(t_fit, 1), "predict_s": round(t_pred, 1)})
            predictions.append(held_out[PREDICTION_COLUMNS])
            print(f"{model_name:12s} fold {fold}: AUROC {row['auroc']:.3f}  AUPRC {row['auprc']:.3f}  "
                  f"({row['n_pos']} pos / {row['n_windows']} scored, {len(held_out)} predicted)  "
                  f"fit {t_fit:.1f} s, predict {t_pred:.1f} s", flush=True)
        out = args.out_dir / f"predictions_{model_name}_set2.parquet"
        pd.concat(predictions).sort_values(["subject_id", "start_s"]).to_parquet(out, index=False)
        save_cv_rows(rows, args.out_dir, CV_FILE)    # after each model, so finished results survive an interruption
        r = pd.DataFrame(rows)
        print(f"{model_name} done in {(time.time() - t_model) / 60:.1f} min: AUROC {r.auroc.mean():.3f} "
              f"± {r.auroc.std():.3f}, AUPRC {r.auprc.mean():.3f} ± {r.auprc.std():.3f}\n", flush=True)


if __name__ == "__main__":
    main()
