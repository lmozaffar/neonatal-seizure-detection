"""Seizure labels for 12 s windows from the three experts' per-second annotations.

Consensus: a second is seizure when at least 2 of the 3 experts mark it. A
second is annotated only if all three experts have a value for it (NaN after a
recording ends); unannotated seconds never count as seizure or as non-seizure.

Primary training label (provisional), `label`:
   1  positive   >= 6 of the window's 12 s are consensus seizure
   0  negative   all 12 s annotated and no expert marks any of them
  -1  ambiguous  everything else (1-5 consensus seconds, expert-only marks,
                 missing annotation)
Two alternatives are kept for comparison: `label_any_overlap` (1 if >= 1
consensus second) and `label_binary50` (1 if >= 6 consensus seconds, else 0);
both are -1 when the window is not fully annotated.

Labels are independent of gap eligibility and artifact usability; no window is
removed here. The stored counts are enough to rebuild every label with
`strategy_labels`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from helsinki.paths import ANNOTATION_CSVS

EXPERTS = ("A", "B", "C")
CONSENSUS_MIN_VOTES = 2
POSITIVE_MIN_S = 6


def load_annotations(subject_id: int) -> np.ndarray:
    """(3, n_seconds) array, rows A/B/C: 1 seizure, 0 none, NaN not annotated."""
    return np.stack([pd.read_csv(ANNOTATION_CSVS[e], usecols=[str(subject_id)]).iloc[:, 0].to_numpy(float)
                     for e in EXPERTS])


def strategy_labels(consensus_s, any_expert_s, fully_annotated) -> dict[str, np.ndarray]:
    """Labels of the three strategies from per-window counts (arrays of equal length)."""
    consensus_s = np.asarray(consensus_s)
    any_expert_s = np.asarray(any_expert_s)
    full = np.asarray(fully_annotated, bool)
    return {
        "label": np.where(consensus_s >= POSITIVE_MIN_S, 1, np.where(full & (any_expert_s == 0), 0, -1)),
        "label_any_overlap": np.where(~full, -1, (consensus_s >= 1).astype(int)),
        "label_binary50": np.where(~full, -1, (consensus_s >= POSITIVE_MIN_S).astype(int)),
    }


def label_windows(meta: pd.DataFrame, annotations: np.ndarray) -> pd.DataFrame:
    """Copy of `meta` with per-window annotation summary and labels.

    Added columns:
      annotated_s       seconds annotated by all three experts (of 12)
      fully_annotated   annotated_s == window length
      seizure_s_A/B/C   seconds each expert marks as seizure
      any_expert_s      seconds marked by >= 1 expert
      consensus_s       seconds marked by >= 2 experts
      unanimous_s       seconds marked by all 3 experts
      consensus_frac    consensus_s / window length
      n_experts_marking experts marking at least one second in the window
      label, label_any_overlap, label_binary50   (1 / 0 / -1, see module doc)
    Seconds with 1, 2 and 3 votes are any - consensus, consensus - unanimous
    and unanimous.
    """
    start = meta.start_s.to_numpy()
    end = meta.end_s.to_numpy()
    if not (np.all(start == np.round(start)) and np.all(end == np.round(end))):
        raise ValueError("windows must start and end on whole seconds to align with annotations")
    start, end = start.astype(int), end.astype(int)
    length = end - start

    # Pad so windows reaching past the annotation matrix read as unannotated.
    n = max(annotations.shape[1], int(end.max(initial=0)))
    ann = np.full((len(EXPERTS), n), np.nan)
    ann[:, :annotations.shape[1]] = annotations
    annotated = np.isfinite(ann).all(axis=0)
    marks = np.where(annotated, ann, 0) == 1                 # (3, n) seizure marks on annotated seconds
    votes = marks.sum(axis=0)

    def per_window(per_second):
        c = np.r_[0, np.cumsum(per_second)]
        return c[end] - c[start]

    out = meta.copy()
    out["annotated_s"] = per_window(annotated)
    out["fully_annotated"] = out["annotated_s"].to_numpy() == length
    for e, row in zip(EXPERTS, marks):
        out[f"seizure_s_{e}"] = per_window(row)
    out["any_expert_s"] = per_window(votes >= 1)
    out["consensus_s"] = per_window(votes >= CONSENSUS_MIN_VOTES)
    out["unanimous_s"] = per_window(votes == len(EXPERTS))
    out["consensus_frac"] = out["consensus_s"] / length
    out["n_experts_marking"] = sum((out[f"seizure_s_{e}"] > 0).astype(int) for e in EXPERTS)
    for name, values in strategy_labels(out.consensus_s, out.any_expert_s, out.fully_annotated).items():
        out[name] = values
    return out
