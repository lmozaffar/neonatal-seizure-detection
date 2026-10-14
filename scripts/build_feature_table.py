"""Run Leila's pipeline (helsinki/) on every recording and save one window-level table.

    .venv/bin/python scripts/build_feature_table.py                  # all 79 recordings
    .venv/bin/python scripts/build_feature_table.py --subjects 1 57  # a subset

For each recording: load -> preprocess -> make_windows -> assess_windows (quality)
-> label_windows -> extract_features. Nothing in helsinki/ is changed or
reimplemented; this script only calls it and stacks the results.

Output: data/processed/feature_table.parquet, one row per 12 s window, and
data/processed/feature_columns.json listing the columns of each feature set.
Recordings that fail are logged to data/processed/build_log.txt and skipped.

Columns
  identity     subject_id, window_index, start_s, end_s
  labels       label (conservative, the training label), label_any_overlap,
               label_binary50; 1 / 0 / -1 (METHODOLOGY.md §7)
  usability    eligible (no gap samples, §5), unusable (>= 9 of 18 QC-bad channels, §6)
  metadata     gap_frac, filter_edge, n_flagged_channels, n_bad_channels,
               n_valid_channels, fully_annotated, consensus_s, any_expert_s.
               Not model predictors (§9).
  features     <feature>_median, <feature>_max, <feature>_p75 across valid channels.
    Feature set 1 (baseline, 165 columns): the 55 adapted Temko features.
    Feature set 2 (secondary, 174 columns): set 1 plus the 3 rhythmicity
      features acf_peak, acf_peak_lag, spectral_peak_prominence.

Features are computed once with Leila's "secondary" set. Its first 55 per-channel
features are the baseline set, computed by the same code, so set 1 is a column
subset of set 2 (checked against a separate baseline run with --check-baseline).
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from helsinki.features import FEATURE_SETS, SUMMARY_STATS, extract_features
from helsinki.io import load_recording
from helsinki.labeling import label_windows, load_annotations
from helsinki.paths import DATA_ROOT, SUBJECT_IDS
from helsinki.preprocessing import make_windows, preprocess
from helsinki.quality import assess_windows

OUT_DIR = DATA_ROOT / "processed"

ID_COLUMNS = ["subject_id", "window_index", "start_s", "end_s"]
LABEL_COLUMNS = ["label", "label_any_overlap", "label_binary50"]
USABILITY_COLUMNS = ["eligible", "unusable"]
METADATA_COLUMNS = ["gap_frac", "filter_edge", "n_flagged_channels", "n_bad_channels", "n_valid_channels",
                    "fully_annotated", "consensus_s", "any_expert_s"]


def summary_columns(feature_set: str) -> list[str]:
    """Window-level feature columns of a feature set, in Leila's column order."""
    return [f"{name}_{stat}" for name, *_ in FEATURE_SETS[feature_set] for stat in SUMMARY_STATS]


FEATURE_SET_1 = summary_columns("baseline")    # 55 Temko features x 3 stats = 165
FEATURE_SET_2 = summary_columns("secondary")   # + 3 rhythmicity features x 3 stats = 174


def process_recording(subject_id: int, check_baseline: bool = False) -> pd.DataFrame:
    """One recording -> one row per window with the columns described above."""
    raw = load_recording(subject_id, preload=True)
    x, segments = preprocess(raw)
    windows, meta = make_windows(x, subject_id)
    channel_qc, _, meta = assess_windows(raw, x, segments, meta)
    meta = label_windows(meta, load_annotations(subject_id))
    _, window_feats, _ = extract_features(windows, channel_qc, feature_set="secondary")

    if check_baseline:
        _, base_feats, _ = extract_features(windows, channel_qc, feature_set="baseline")
        pd.testing.assert_frame_equal(base_feats[FEATURE_SET_1], window_feats[FEATURE_SET_1])

    table = pd.concat([meta.reset_index(drop=True), window_feats], axis=1)
    return table[ID_COLUMNS + LABEL_COLUMNS + USABILITY_COLUMNS + METADATA_COLUMNS + FEATURE_SET_2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subjects", type=int, nargs="+", default=list(SUBJECT_IDS))
    parser.add_argument("--out", type=Path, default=OUT_DIR / "feature_table.parquet")
    parser.add_argument("--check-baseline", action="store_true",
                        help="also run the baseline feature set and assert it equals the set-1 columns")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out.parent / "build_log.txt"
    tables, failed = [], []
    t_all = time.time()
    with open(log_path, "w") as log:
        for i, sid in enumerate(args.subjects, 1):
            t0 = time.time()
            try:
                table = process_recording(sid, args.check_baseline)
            except Exception as e:
                failed.append(sid)
                msg = f"eeg{sid}: FAILED ({type(e).__name__}: {e})"
                print(f"[{i}/{len(args.subjects)}] {msg}", flush=True)
                log.write(msg + "\n" + traceback.format_exc() + "\n")
                continue
            tables.append(table)
            msg = (f"eeg{sid}: {len(table)} windows, {int(table.eligible.sum())} eligible, "
                   f"{int(table.unusable.sum())} unusable, {time.time() - t0:.1f} s")
            print(f"[{i}/{len(args.subjects)}] {msg}", flush=True)
            log.write(msg + "\n")

    if not tables:
        raise SystemExit("no recordings processed")
    df = pd.concat(tables, ignore_index=True)
    df["subject_id"] = df["subject_id"].astype(np.int16)
    for col in LABEL_COLUMNS:
        df[col] = df[col].astype(np.int8)
    df.to_parquet(args.out, index=False)

    columns = {
        "id": ID_COLUMNS, "labels": LABEL_COLUMNS, "usability": USABILITY_COLUMNS, "metadata": METADATA_COLUMNS,
        "feature_set_1": FEATURE_SET_1, "feature_set_2": FEATURE_SET_2,
        "rhythmicity_only": [c for c in FEATURE_SET_2 if c not in FEATURE_SET_1],
    }
    (args.out.parent / "feature_columns.json").write_text(json.dumps(columns, indent=1))

    print(f"\nsaved {args.out} ({len(df)} windows, {df.subject_id.nunique()} subjects) "
          f"in {(time.time() - t_all) / 60:.1f} min")
    if failed:
        print(f"failed: {['eeg%d' % s for s in failed]} (see {log_path})")


if __name__ == "__main__":
    main()
