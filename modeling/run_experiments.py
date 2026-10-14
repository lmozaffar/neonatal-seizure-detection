"""Patient-level cross-validation of the classical models on both feature sets.

    .venv/bin/python -m modeling.run_experiments                                 # everything
    .venv/bin/python -m modeling.run_experiments --models LogReg --folds 0 --out-dir results/test

For each feature set, model and fold in results/folds.csv: fit on the training
windows (training_mask) of the other folds' patients, then score every window of
the held-out patients, including ambiguous (label -1) and ineligible ones.
Ineligible windows touch a recording gap; their missing features are filled by
the pipeline's imputer, so their scores are not meaningful. Filter on `eligible`.

Outputs in --out-dir (default results/):
  cv_auroc.csv                           feature_set, model, fold, auroc, auprc and counts, on
                                         held-out windows with label in {0, 1} and eligible
  predictions_<model>_set<N>.parquet     subject_id, start_s, end_s, label, eligible, score, fold
Rows of cv_auroc.csv for the (feature_set, model, fold) combinations that are run
replace any earlier ones; other rows are kept.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from modeling.common import FEATURE_SETS_BY_NAME, N_SPLITS, RESULTS_DIR, add_folds, load_feature_table, training_mask
from modeling.models import get_models, predict_scores

MODEL_ORDER = ["LogReg", "RF", "KNN", "SVC"]   # fast ones first
PREDICTION_COLUMNS = ["subject_id", "start_s", "end_s", "label", "eligible", "score", "fold"]
KEY = ["feature_set", "model", "fold"]


def evaluate(held_out: pd.DataFrame, feature_set: str, model_name: str, fold: int) -> dict:
    """AUROC and AUPRC on held-out windows with label 0/1 that are eligible; NaN if a class is missing."""
    scored = held_out[held_out.label.isin([0, 1]) & held_out.eligible]
    n_pos = int((scored.label == 1).sum())
    row = {"feature_set": feature_set, "model": model_name, "fold": fold,
           "n_windows": len(scored), "n_pos": n_pos, "auroc": float("nan"), "auprc": float("nan")}
    if n_pos == 0 or n_pos == len(scored):
        warnings.warn(f"{feature_set} {model_name} fold {fold}: held-out windows have only one class "
                      f"({n_pos} positive of {len(scored)}); AUROC/AUPRC skipped", stacklevel=2)
        return row
    row["auroc"] = roc_auc_score(scored.label, scored.score)
    row["auprc"] = average_precision_score(scored.label, scored.score)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER)
    parser.add_argument("--sets", nargs="+", default=list(FEATURE_SETS_BY_NAME), choices=list(FEATURE_SETS_BY_NAME))
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(N_SPLITS)))
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    models = [m for m in MODEL_ORDER if m in args.models]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = add_folds(load_feature_table())
    train_mask = training_mask(df)
    print(f"{len(df)} windows, {df.subject_id.nunique()} subjects | models {models} | sets {args.sets} "
          f"| folds {args.folds}\n", flush=True)

    rows = []
    for model_name in models:
        t_model = time.time()
        for set_name in args.sets:
            features = FEATURE_SETS_BY_NAME[set_name]
            predictions = []
            for fold in args.folds:
                train = df[train_mask & (df.fold != fold)]
                held_out = df[df.fold == fold]
                model = get_models()[model_name]
                t0 = time.time()
                model.fit(train[features], train.label)
                t_fit = time.time() - t0
                held_out = held_out.assign(score=predict_scores(model, held_out[features]))
                t_pred = time.time() - t0 - t_fit
                row = evaluate(held_out, set_name, model_name, fold)
                rows.append({**row, "fit_s": round(t_fit, 1), "predict_s": round(t_pred, 1)})
                predictions.append(held_out[PREDICTION_COLUMNS])
                print(f"{model_name:6s} {set_name} fold {fold}: AUROC {row['auroc']:.3f}  AUPRC {row['auprc']:.3f}  "
                      f"({row['n_pos']} pos / {row['n_windows']} scored, {len(held_out)} predicted)  "
                      f"fit {t_fit:.1f} s, predict {t_pred:.1f} s", flush=True)
            out = args.out_dir / f"predictions_{model_name}_{set_name}.parquet"
            pd.concat(predictions).sort_values(["subject_id", "start_s"]).to_parquet(out, index=False)
        print(f"{model_name} done in {(time.time() - t_model) / 60:.1f} min\n", flush=True)

        # Save after each model so finished results survive an interruption.
        cv_path = args.out_dir / "cv_auroc.csv"
        new = pd.DataFrame(rows)
        if cv_path.exists():
            old = pd.read_csv(cv_path)
            old = old[~old.set_index(KEY).index.isin(new.set_index(KEY).index)]
            new = pd.concat([old, new])
        new.sort_values(KEY).to_csv(cv_path, index=False)

    summary = pd.DataFrame(rows).groupby(["feature_set", "model"], sort=False)[["auroc", "auprc"]].agg(["mean", "std"])
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
