"""Demo II summary table of the event-level results, with paired patient-bootstrap CIs for two comparisons.

    .venv/bin/python -m modeling.demo2_summary --n-jobs 4

Table rows: majority protocol, postprocessed operating point, FA/h targets 1 and 0.5, read from
results/event_results.csv (SVC, ConvLSTM, RF) and results/event_results_<model>.csv (the models of
modeling/evaluate_one.py). Nothing in the table is recomputed.

Comparisons (COMPARISONS): the per-patient held-out tables are rebuilt from the saved predictions with the same steps
as modeling/evaluate_one.py, checked against the CSV numbers, and compared with evaluate_events.comparison_row (paired
patient bootstrap of modeling/events.py, 2,000 resamples, seed 0; no matched FA/h). The interval columns are filled
only on the arm's row at the compared target. Raw, not corrected for multiplicity, and chosen after seeing the results.

Output: results/demo2_summary.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from modeling.common import RESULTS_DIR
from modeling.evaluate_events import MODELS, comparison_row
from modeling.events import MAJORITY, build_timelines, choose, evaluate, fold_curves, load_all_annotations, summary

TABLE_MODELS = ["SVC", "ConvLSTM", "RF", "ConvLSTM_pw2", "RF_leaf20", "LightGBM", "LightGBM_ctx"]
SHARED = {"SVC", "ConvLSTM", "RF"}                    # rows in results/event_results.csv
COMPARISONS = [("LightGBM", "SVC", 1.0), ("LightGBM_ctx", "ConvLSTM", 0.5)]    # (arm, reference, FA/h target)
OUT = RESULTS_DIR / "demo2_summary.csv"


def load_rows() -> pd.DataFrame:
    shared = pd.read_csv(RESULTS_DIR / "event_results.csv")
    parts = [shared[shared.model.isin(SHARED)]]
    parts += [pd.read_csv(RESULTS_DIR / f"event_results_{m}.csv") for m in TABLE_MODELS if m not in SHARED]
    d = pd.concat(parts, ignore_index=True)
    return d[(d.protocol == MAJORITY.name) & (d.operating_point == "postprocessed")]


def held_out(model: str, target: float, n_jobs: int):
    """Postprocessed held-out (patients, events) of `model` at `target`, as evaluate_one.run computes them."""
    pred = pd.read_parquet(RESULTS_DIR / MODELS.get(model, f"predictions_{model}_set2.parquet"))
    timelines, events = build_timelines(pred, load_all_annotations(pred.subject_id.unique()), MAJORITY)
    parts = Parallel(n_jobs=n_jobs)(delayed(fold_curves)({k: tl}, events, MAJORITY) for k, tl in timelines.items())
    assignments, _ = choose(pd.concat(parts, ignore_index=True), target)
    return evaluate(timelines, events, assignments["postprocessed"], MAJORITY)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-jobs", type=int, default=1, help="up to 4 (one job per smoothing); results do not depend on it")
    args = parser.parse_args()

    rows = load_rows()
    table = pd.DataFrame({
        "model": rows.model, "target": rows.fa_target.map({1.0: "1/h", 0.5: "0.5/h"}),
        "sensitivity": rows.sensitivity, "detected_of_492": rows.n_detected, "fa_per_h": rows.fa_per_h,
        "fa_per_day": rows.fa_per_h * 24, "short_detected_of_278": rows.n_detected_short})
    if not ((rows.n_events == 492).all() and (rows.n_events_short == 278).all()):
        raise ValueError("expected 492 events and 278 short events in every row")
    table["model"] = pd.Categorical(table.model, TABLE_MODELS, ordered=True)
    table = table.sort_values(["target", "model"], ascending=[False, True]).reset_index(drop=True)
    table["model"] = table.model.astype(str)

    ci_cols = ["compared_with", "d_sensitivity", "d_sensitivity_ci_low", "d_sensitivity_ci_high",
               "d_fa_per_h", "d_fa_per_h_ci_low", "d_fa_per_h_ci_high", "net_detected"]
    for c in ci_cols:
        table[c] = pd.Series(dtype=object if c == "compared_with" else float)
    for arm, ref, target in COMPARISONS:
        tables = {m: held_out(m, target, args.n_jobs) for m in (arm, ref)}
        for m, (patients, ev) in tables.items():          # the rebuilt tables must give the CSV numbers
            s, r = summary(patients, ev), rows[(rows.model == m) & (rows.fa_target == target)].iloc[0]
            if not all(np.isclose(s[k], r[k], equal_nan=True) for k in s):
                raise AssertionError(f"{m} at {target}/h does not reproduce its event_results row")
        c = comparison_row(tables[arm], tables[ref])
        i = table.index[(table.model == arm) & (table.target == {1.0: "1/h", 0.5: "0.5/h"}[target])][0]
        table.loc[i, ci_cols] = [ref, c["d_sensitivity"], c["d_sensitivity_ci_low"], c["d_sensitivity_ci_high"],
                                 c["fa_per_h_arm"] - c["fa_per_h_ref"], c["d_fa_per_h_ci_low"],
                                 c["d_fa_per_h_ci_high"], c["net_detected"]]
        print(f"{arm} vs {ref} at {target}/h: sensitivity {c['d_sensitivity'] * 100:+.1f} points "
              f"(95% CI {c['d_sensitivity_ci_low'] * 100:+.1f} to {c['d_sensitivity_ci_high'] * 100:+.1f}), "
              f"FA/h {c['fa_per_h_arm'] - c['fa_per_h_ref']:+.3f} (95% CI {c['d_fa_per_h_ci_low']:+.3f} to "
              f"{c['d_fa_per_h_ci_high']:+.3f}), net {c['net_detected']:+d} events, "
              f"top patient {c['top_patient']} ({c['top_patient_net']:+d})", flush=True)

    table.to_csv(OUT, index=False)
    pd.set_option("display.width", 240)
    print(f"\n{table.round(3).to_string(index=False)}\n\nsaved {OUT}")


if __name__ == "__main__":
    main()
