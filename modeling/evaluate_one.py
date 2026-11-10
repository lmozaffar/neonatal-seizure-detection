"""Event-level results of one model's saved predictions, in memory, without writing any results file.

    .venv/bin/python -m modeling.evaluate_one LightGBM --n-jobs 4
    .venv/bin/python -m modeling.evaluate_one SVC --protocol published --n-jobs 4
    .venv/bin/python -m modeling.evaluate_one LightGBM --out /tmp/lightgbm_events.csv   # optional copy

Same steps as modeling.evaluate_events.run (thresholds and postprocessing chosen on the other folds, both FA/h
targets, baseline and postprocessed), using modeling.events unchanged, and the same numbers. Only the threshold scan
(fold_curves) is split into one job per smoothing; the curves are identical in any order. No comparisons, bootstrap
or plots. A model missing from evaluate_events.MODELS is read from results/predictions_<model>_set2.parquet.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from modeling.common import RESULTS_DIR
from modeling.evaluate_events import MODELS
from modeling.events import (FA_TARGETS, PROTOCOLS, build_timelines, choose, evaluate, fold_curves,
                             load_all_annotations, summary)

RESULTS_FILES = {"event_results.csv", "event_selected.csv", "event_comparisons.csv", "event_fold_gains.csv"}


def run(model: str, protocol, n_jobs: int = 1) -> pd.DataFrame:
    """One row per FA/h target and operating point, with the columns of results/event_results.csv."""
    pred = pd.read_parquet(RESULTS_DIR / MODELS.get(model, f"predictions_{model}_set2.parquet"))
    timelines, events = build_timelines(pred, load_all_annotations(pred.subject_id.unique()), protocol)
    parts = Parallel(n_jobs=n_jobs)(delayed(fold_curves)({k: tl}, events, protocol) for k, tl in timelines.items())
    curves = pd.concat(parts, ignore_index=True)
    rows = []
    for target in FA_TARGETS:
        assignments, _ = choose(curves, target)
        for op, assignment in assignments.items():
            rows.append({"model": model, "protocol": protocol.name, "fa_target": target, "operating_point": op,
                         **summary(*evaluate(timelines, events, assignment, protocol))})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model")
    parser.add_argument("--protocol", default="majority", choices=[p.name for p in PROTOCOLS])
    parser.add_argument("--n-jobs", type=int, default=1, help="up to 4 (one job per smoothing); results do not depend on it")
    parser.add_argument("--out", type=Path, help="also save the rows here (never one of the shared results files)")
    args = parser.parse_args()
    if args.out is not None and args.out.name in RESULTS_FILES:
        parser.error(f"--out must not be named {args.out.name}; that is evaluate_events.py's output")

    t0 = time.time()
    table = run(args.model, next(p for p in PROTOCOLS if p.name == args.protocol), args.n_jobs)
    table["fa_per_day"] = table.fa_per_h * 24
    pd.set_option("display.width", 240)
    show = ["model", "protocol", "fa_target", "operating_point", "n_events", "n_detected", "sensitivity",
            "n_detected_short", "sensitivity_short", "n_false_alarms", "fa_per_h", "fa_per_day", "latency_median_s"]
    print(table[show].round(3).to_string(index=False))
    print(f"\n{(time.time() - t0) / 60:.1f} min")
    if args.out is not None:
        table.to_csv(args.out, index=False)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
