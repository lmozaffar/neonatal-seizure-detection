"""Are short seizures represented in the training labels? Label coverage of every reference event (METHODOLOGY.md §23).

    .venv/bin/python -m modeling.label_analysis

Uses the window labels and eligibility in results/predictions_SVC_set2.parquet (scores are not used) and the
reference events of modeling.events (majority protocol). A window is a positive training window when its label is 1
and it is eligible (label 1: at least 6 of its 12 s are consensus seizure, METHODOLOGY.md §7). For every event: the
windows overlapping it, how many are ambiguous (label -1), how many are positive training windows, and the share of
its seconds those positive windows cover.

Output: results/label_coverage.csv, one row per seizure-duration bin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modeling.common import RESULTS_DIR
from modeling.events import build_timeline, duration_bin, load_all_annotations


def main() -> None:
    windows = pd.read_parquet(RESULTS_DIR / "predictions_SVC_set2.parquet")
    _, events = build_timeline(windows, load_all_annotations(windows.subject_id.unique()))
    windows["positive_train"] = (windows.label == 1) & windows.eligible

    rows = []
    for sid, ev in events.groupby("subject_id"):
        w = windows[windows.subject_id == sid]
        for e in ev.itertuples():
            overlap = w[(w.start_s < e.end_s) & (w.end_s > e.start_s)]
            positive = overlap[overlap.positive_train]
            covered = np.zeros(e.duration_s, bool)
            for a, b in zip(positive.start_s, positive.end_s):
                covered[max(0, int(a - e.start_s)):max(0, int(b - e.start_s))] = True
            rows.append({"duration_s": e.duration_s, "n_windows": len(overlap),
                         "n_ambiguous": int((overlap.label == -1).sum()), "n_positive_train": len(positive),
                         "covered_frac": covered.mean()})
    ev = pd.DataFrame(rows)
    ev["duration_bin"] = duration_bin(ev.duration_s)

    g = ev.groupby("duration_bin", observed=False)
    out = pd.DataFrame({"n_events": g.size(),
                        "events_with_positive_train": g.n_positive_train.agg(lambda s: int((s > 0).sum())),
                        "events_without_positive_train": g.n_positive_train.agg(lambda s: int((s == 0).sum())),
                        "median_positive_train_per_event": g.n_positive_train.median(),
                        "mean_covered_frac": g.covered_frac.mean(),
                        "ambiguous_window_share": g.n_ambiguous.sum() / g.n_windows.sum()}).reset_index()
    out.to_csv(RESULTS_DIR / "label_coverage.csv", index=False)

    short = ev[ev.duration_s < 60]
    print(out.round(3).to_string(index=False))
    print(f"\nseizures < 60 s with at least one positive training window: {int((short.n_positive_train > 0).sum())} "
          f"of {len(short)}")


if __name__ == "__main__":
    main()
