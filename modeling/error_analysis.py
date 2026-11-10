"""Why seizures are missed: missed events by peak score, and sensitivity by seizure duration (METHODOLOGY.md §22).

    .venv/bin/python -m modeling.error_analysis

For ConvLSTM, SVC and RF (feature set 2), under the majority protocol at <= 1 FA/h, two operating points chosen on
the other folds (modeling.events.choose): baseline (no postprocessing) and postprocessed. Each missed event is put in
the first category it fits:
  crossed, too short   its score reached the threshold, but no run lasted the minimum duration
  near miss            peak score >= the 99th percentile of the held-out fold's background seconds
  elevated             peak between the 90th and 99th percentile
  low                  peak below the 90th percentile: the model scores the seizure like background
Background seconds are evaluable and marked by no expert. Using the held-out fold's own scores here is description
only; nothing is chosen with it.

Outputs in results/:
  missed_seizures.csv          model, operating_point, outcome, n_missed, share_of_missed
  sensitivity_by_duration.csv  model, operating_point, duration_bin, n_events, n_detected, sensitivity. Rows of
                               these models are replaced; rows of other models (E9, E14) are kept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modeling.common import RESULTS_DIR
from modeling.events import (DURATION_LABELS, MAJORITY, build_timelines, choose, duration_bin, fold_curves,
                             load_all_annotations, match)

MODELS = ["ConvLSTM", "SVC", "RF"]
FA_TARGET = 1.0
MISS_CATEGORIES = ["crossed, too short", "near miss (>= p99)", "elevated (p90-p99)", "low (< p90)"]


def main() -> None:
    rows = []
    for model in MODELS:
        pred = pd.read_parquet(RESULTS_DIR / f"predictions_{model}_set2.parquet")
        timelines, events = build_timelines(pred, load_all_annotations(pred.subject_id.unique()), MAJORITY)
        assignments, _ = choose(fold_curves(timelines, events, MAJORITY), FA_TARGET)
        for operating_point, assignment in assignments.items():
            for fold, (cfg, threshold) in assignment.items():
                tl = timelines[cfg.smooth]
                tl = tl.subset(tl.fold == fold)
                m = match(tl, threshold, len(events), cfg.min_duration_s, cfg.merge_gap_s)
                background = np.sort(tl.score[tl.evaluable & (tl.event < 0) & ~tl.ambiguous])
                in_event = tl.evaluable & (tl.event >= 0)
                peak = pd.Series(tl.score[in_event]).groupby(tl.event[in_event]).max()
                ids = events.index[(events.fold == fold) & (events.evaluable_s > 0)]
                e = events.loc[ids, ["duration_s"]].assign(detected=m["event_detected"][ids], peak=peak.reindex(ids))
                peak_pct = np.searchsorted(background, e.peak, side="left") / len(background)   # share strictly below
                e["outcome"] = np.select([e.detected, e.peak >= threshold, peak_pct >= 0.99, peak_pct >= 0.90],
                                         ["detected", *MISS_CATEGORIES[:3]], MISS_CATEGORIES[3])
                rows.append(e.assign(model=model, operating_point=operating_point))
        print(f"{model}: done", flush=True)
    ev = pd.concat(rows, ignore_index=True)
    ev["duration_bin"] = duration_bin(ev.duration_s)

    keys = ["model", "operating_point"]
    missed = (ev[~ev.detected].groupby([*keys, "outcome"]).size().rename("n_missed")
              .reindex(pd.MultiIndex.from_product([MODELS, list(assignments), MISS_CATEGORIES],
                                                  names=[*keys, "outcome"]), fill_value=0).reset_index())
    missed["share_of_missed"] = missed.n_missed / missed.groupby(keys).n_missed.transform("sum")
    missed.to_csv(RESULTS_DIR / "missed_seizures.csv", index=False)

    by_duration = (ev.groupby([*keys, "duration_bin"], observed=False)
                   .agg(n_events=("detected", "size"), n_detected=("detected", "sum")).reset_index())
    by_duration["sensitivity"] = by_duration.n_detected / by_duration.n_events
    path = RESULTS_DIR / "sensitivity_by_duration.csv"
    if path.exists():                              # keep the rows other scripts wrote
        old = pd.read_csv(path)
        by_duration = pd.concat([old[~old.model.isin(MODELS)], by_duration], ignore_index=True)
    by_duration.to_csv(path, index=False)

    pd.set_option("display.width", 200)
    print("\nShare of missed seizures\n" + missed.pivot_table(index=keys, columns="outcome", values="share_of_missed")
          [MISS_CATEGORIES].round(3).to_string())
    print("\nSensitivity by seizure duration\n" + by_duration[by_duration.model.isin(MODELS)].pivot_table(
        index=keys, columns="duration_bin", values="sensitivity", observed=False)[list(DURATION_LABELS)]
        .round(3).to_string())


if __name__ == "__main__":
    main()
