"""Patient-level cross-validation of the spike-feature, per-channel and temporal SVC arms (METHODOLOGY.md §24, §25).

    .venv/bin/python -m modeling.run_channel_models                       # every arm
    .venv/bin/python -m modeling.run_channel_models --arms C1 --n-jobs 2  # one arm, two folds at a time

Needs data/processed/ from `scripts/build_feature_table.py --channel-features --spike-features`.

Arms (the RBF SVC of modeling.models everywhere, untuned; A0 is the existing SVC on feature set 2):
  A1  window level, set 2 + the 30 spike columns (204)
  S   window level, the spike columns only (30)
  C0  per channel, the 58 set-2 channel features
  C1  per channel, the 58 features + the 10 spike features (68)
  T-BD, T-B, T-D  C1 + causal temporal features (E14; 104, 80 and 92): T-BD is the pre-specified candidate, T-B and
      T-D are ablations. See temporal_features().

Per-channel arms train on (window, channel) rows of valid channels:
  1         window label 1 and seizure_per_channel_Dec_2017.xlsx (Tapani et al.) marks the channel in a seizure
            overlapping the window
  0         window label 0, every valid channel
  excluded  unmarked channels of label-1 windows, and label-1 windows no spreadsheet row overlaps
Training uses at most MAX_POSITIVE_ROWS random positive rows (seed 0) and, inside the SVC, at most
SVC_MAX_NEGATIVES negatives. A held-out window's score is the maximum over its valid channels.

As in run_experiments.py: folds from results/folds.csv, training windows from training_mask() of the other folds
only, imputation and scaling fitted inside the pipeline on those windows, and every held-out window scored.

Outputs in results/:
  predictions_SVC_<arm>.parquet   subject_id, start_s, end_s, label, eligible, score, fold
  cv_auroc.csv                    rows for feature_set <arm>, model SVC replace earlier ones; other rows are kept
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from numpy.lib.stride_tricks import sliding_window_view

from helsinki.features import FEATURE_SETS
from helsinki.paths import ANNOTATIONS_DIR, DATA_ROOT
from helsinki.preprocessing import BIPOLAR_NAMES
from helsinki.spike_features import SPIKE_COLUMNS, SPIKE_FEATURES
from modeling.common import FEATURE_SET_2, N_SPLITS, RESULTS_DIR, add_folds, training_mask
from modeling.models import RANDOM_STATE, get_models, predict_scores
from modeling.run_experiments import PREDICTION_COLUMNS, evaluate, save_cv_rows

PROCESSED_DIR = DATA_ROOT / "processed"
CHANNEL_ANNOTATIONS = ANNOTATIONS_DIR / "seizure_per_channel_Dec_2017.xlsx"
WINDOW_ARMS = {"A1": FEATURE_SET_2 + SPIKE_COLUMNS, "S": SPIKE_COLUMNS}
CHANNEL_ARMS = {"C0": False, "C1": True}        # arm -> add the spike features
TEMPORAL_ARMS = {"T-BD": ("B", "D2", "T4"), "T-B": ("B",), "T-D": ("D2", "T4")}   # C1 + these temporal blocks
ARMS = [*WINDOW_ARMS, *CHANNEL_ARMS, *TEMPORAL_ARMS]
MAX_POSITIVE_ROWS = 10_000

# Temporal features (E14), all chosen before any temporal result.
C1_FEATURE_NAMES = [name for name, *_ in FEATURE_SETS["secondary"]] + list(SPIKE_FEATURES)
BASE_FEATURES = ["line_length", "rms_amp", "snleo_mean", "spectral_entropy", "sef90", "hjorth_mobility",
                 "svd_entropy", "band_0_2_norm", "acf_peak", "spectral_peak_prominence", "spike_count",
                 "spike_corr_mean"]
LOG_FEATURES = ["line_length", "rms_amp", "snleo_mean"]
BLOCKS = ["B", "D2", "T4"]
HORIZON_WINDOWS, GUARD_WINDOWS, MIN_BASELINE_WINDOWS = 60, 10, 10
POOL_WINDOWS = HORIZON_WINDOWS - GUARD_WINDOWS
CHUNK_WINDOWS = 256                              # baseline windows sorted at once, to bound memory


def load_inputs():
    """Window table (with folds and spike columns), channel features, spike channel features, channel validity."""
    table = pd.read_parquet(PROCESSED_DIR / "feature_table.parquet")
    spike = pd.read_parquet(PROCESSED_DIR / "spike_window_features.parquet")
    index = pd.read_parquet(PROCESSED_DIR / "channel_index.parquet")
    channels = np.load(PROCESSED_DIR / "channel_features.npy", mmap_mode="r")
    spike_channels = np.load(PROCESSED_DIR / "spike_channel_features.npy", mmap_mode="r")
    valid = np.load(PROCESSED_DIR / "channel_valid.npy")
    if not (len(spike) == len(index) == len(channels) == len(spike_channels) == len(valid) == len(table)
            and (index.subject_id.to_numpy() == table.subject_id.to_numpy()).all()
            and (index.window_index.to_numpy() == table.window_index.to_numpy()).all()):
        raise ValueError(f"files in {PROCESSED_DIR} are not row-aligned; rebuild them together")
    return add_folds(pd.concat([table, spike], axis=1)), channels, spike_channels, valid


def load_channel_annotations() -> pd.DataFrame:
    """subject_id, start_s, stop_s and one 0/1 column per channel (BIPOLAR_NAMES), from the spreadsheet.

    Read with the standard library (no Excel dependency). Rows with stop <= start or no channel marked are data
    errors and are dropped."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(CHANNEL_ANNOTATIONS) as z:
        shared = ["".join(si.itertext()) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for r in sheet.iter(f"{{{ns['m']}}}row"):
        row = {}
        for c in r.findall("m:c", ns):
            value = c.find("m:v", ns)
            if value is not None:
                column = "".join(ch for ch in c.get("r") if ch.isalpha())
                row[column] = shared[int(value.text)] if c.get("t") == "s" else value.text
        rows.append(row)
    header = rows[0]
    channels = [c for c, name in header.items() if name in BIPOLAR_NAMES]
    if tuple(header[c] for c in channels) != BIPOLAR_NAMES:
        raise ValueError(f"unexpected channel columns in {CHANNEL_ANNOTATIONS}")
    table = pd.DataFrame([{"subject_id": int(r["A"]), "start_s": float(r["B"]), "stop_s": float(r["C"]),
                           **{header[c]: int(r.get(c, 0)) for c in channels}} for r in rows[1:] if r.get("A")])
    bad = (table.stop_s <= table.start_s) | (table[list(BIPOLAR_NAMES)].sum(axis=1) == 0)
    return table[~bad].reset_index(drop=True)


def channel_labels(table: pd.DataFrame, seizures: pd.DataFrame) -> np.ndarray:
    """(n_windows, 18): 1 positive, 0 negative, -1 excluded from per-channel training (module doc)."""
    labels = np.full((len(table), len(BIPOLAR_NAMES)), -1, np.int8)
    labels[table.label.to_numpy() == 0] = 0
    positives = table.label.to_numpy() == 1
    sid, start, end = table.subject_id.to_numpy(), table.start_s.to_numpy(), table.end_s.to_numpy()
    for subject, a, b, marked in zip(seizures.subject_id, seizures.start_s, seizures.stop_s,
                                     seizures[list(BIPOLAR_NAMES)].to_numpy(bool)):
        labels[np.ix_(positives & (sid == subject) & (start < b) & (end > a), marked)] = 1
    return labels


def trailing_median(f: np.ndarray) -> np.ndarray:
    """Per row t of `f` (NaN = unusable): the median of the usable rows t-60 to t-11, NaN with fewer than 10."""
    out = np.full(f.shape, np.nan)
    n = len(f)
    if n <= GUARD_WINDOWS + 1:
        return out
    pool = np.concatenate([np.full((POOL_WINDOWS - 1, *f.shape[1:]), np.nan), f[:n - GUARD_WINDOWS - 1]])
    n_rows = len(pool) - POOL_WINDOWS + 1            # pool row r belongs to window t = r + GUARD_WINDOWS + 1
    for start in range(0, n_rows, CHUNK_WINDOWS):
        stop = min(start + CHUNK_WINDOWS, n_rows)
        window = sliding_window_view(pool[start:stop + POOL_WINDOWS - 1], POOL_WINDOWS, axis=0)
        count = (~np.isnan(window)).sum(axis=-1)
        ordered = np.sort(window, axis=-1)           # NaN sorts last
        low = np.take_along_axis(ordered, np.maximum(count - 1, 0)[..., None] // 2, axis=-1)[..., 0]
        high = np.take_along_axis(ordered, np.minimum(count // 2, POOL_WINDOWS - 1)[..., None], axis=-1)[..., 0]
        out[start + GUARD_WINDOWS + 1:stop + GUARD_WINDOWS + 1] = np.where(count >= MIN_BASELINE_WINDOWS,
                                                                           (low + high) / 2, np.nan)
    return out


def patient_temporal(base: np.ndarray, window_index, eligible, valid) -> np.ndarray:
    """(n_rows, 18, 36) float32 temporal features of one patient's rows: blocks B, D2, T4 of the 12 base features."""
    window_index = np.asarray(window_index, int)
    if len(np.unique(window_index)) != len(window_index) or (window_index < 0).any():
        raise ValueError("window_index must be unique and non-negative within a patient")
    n_grid = int(window_index.max()) + 1
    f = np.full((n_grid, base.shape[1], len(BASE_FEATURES)), np.nan)    # absent windows behave like a gap
    f[window_index] = base
    logged = np.isin(BASE_FEATURES, LOG_FEATURES)
    with np.errstate(divide="ignore", invalid="ignore"):
        f[..., logged] = np.log(f[..., logged])
    f[~np.isfinite(f)] = np.nan                      # a non-positive energy becomes NaN, never inf
    eligible_grid = np.zeros(n_grid, bool)
    eligible_grid[window_index] = eligible
    valid_grid = np.zeros((n_grid, base.shape[1]), bool)
    valid_grid[window_index] = valid
    edge = np.r_[True, ~eligible_grid[:-1]]          # first window, or the one after an ineligible/absent window
    f[~((eligible_grid & ~edge)[:, None] & valid_grid)] = np.nan

    baseline = f - trailing_median(f)
    difference = np.full(f.shape, np.nan)
    difference[2:] = f[2:] - f[:-2]
    trend = np.full(f.shape, np.nan)
    trend[4:] = (f[4:] - f[:-4]) / 2
    trend[4:][np.isnan(f[2:-2])] = np.nan            # the slope over t, t-2, t-4 needs all three
    return np.concatenate([baseline, difference, trend], axis=2)[window_index].astype(np.float32)


def temporal_features(c1: np.ndarray, subject_id, window_index, eligible, valid) -> np.ndarray:
    """(n_windows, 18, 36) causal temporal features from the C1 channel features, each patient on its own.

    For one channel and base feature f(t) (line length, RMS and SNLEO mean logged):
      B   f(t) - median of f over windows t-60 to t-11 (at least 10 usable; t-1 to t-10 are a guard)
      D2  f(t) - f(t-2), the previous window with no shared data
      T4  (f(t) - f(t-4)) / 2, the least-squares slope over t, t-2, t-4
    A window is usable for a channel when it is eligible, the channel is valid, and it is not an edge window. Unusable
    values are NaN and are never filled. Only past windows are used; no labels, annotations or scores.
    """
    base = np.asarray(c1[:, :, [C1_FEATURE_NAMES.index(name) for name in BASE_FEATURES]])
    out = np.full((*base.shape[:2], 3 * len(BASE_FEATURES)), np.nan, np.float32)
    for subject in np.unique(subject_id):
        rows = np.flatnonzero(subject_id == subject)
        out[rows] = patient_temporal(base[rows], window_index[rows], eligible[rows], valid[rows])
    return out


def fit_predict_window(X, y, folds, trainable, fold) -> np.ndarray:
    """Scores of fold `fold`'s windows from an SVC trained on the other folds."""
    train = trainable & (folds != fold)
    model = get_models()["SVC"].fit(X[train], y[train])
    return predict_scores(model, X[folds == fold])


def fit_predict_channels(features, labels, valid, folds, trainable, eligible, fold) -> np.ndarray:
    """(n_held_out, 18) channel scores of fold `fold` (NaN where not scored) from an SVC trained on the other folds."""
    usable = (labels >= 0) & valid & (trainable & (folds != fold))[:, None]
    win, ch = np.nonzero(usable)
    y = labels[win, ch].astype(int)
    pos = np.flatnonzero(y == 1)
    if len(pos) > MAX_POSITIVE_ROWS:
        pos = np.random.default_rng(RANDOM_STATE).choice(pos, MAX_POSITIVE_ROWS, replace=False)
    keep = np.sort(np.r_[np.flatnonzero(y == 0), pos])
    model = get_models()["SVC"].fit(np.asarray(features[win[keep], ch[keep]], float), y[keep])

    held = np.flatnonzero(folds == fold)
    scored = valid[held] & eligible[held][:, None]
    out = np.full(scored.shape, np.nan, np.float32)
    w, c = np.nonzero(scored)
    if len(w):
        out[w, c] = predict_scores(model, np.asarray(features[held[w], c], float))
    return out


def max_over_channels(channel_scores: np.ndarray) -> np.ndarray:
    """Maximum over scored channels; a window with none gets the lowest score seen, so it never raises an alarm."""
    best = np.where(np.isfinite(channel_scores), channel_scores, -np.inf).max(axis=1)
    finite = best[np.isfinite(best)]
    return np.where(np.isfinite(best), best, finite.min() if finite.size else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    parser.add_argument("--n-jobs", type=int, default=1, help="folds trained in parallel; results do not depend on it")
    args = parser.parse_args()

    df, channels, spike_channels, valid = load_inputs()
    folds, trainable, eligible = df.fold.to_numpy(), training_mask(df).to_numpy(), df.eligible.to_numpy()
    fold_ids = list(range(N_SPLITS))
    print(f"{len(df)} windows, {df.subject_id.nunique()} subjects | arms {args.arms}\n", flush=True)

    rows = []
    for arm in args.arms:
        scores = np.full(len(df), np.nan)
        if arm in WINDOW_ARMS:
            X, y = df[WINDOW_ARMS[arm]].to_numpy(float), df.label.to_numpy()
            held_out = Parallel(n_jobs=args.n_jobs)(delayed(fit_predict_window)(X, y, folds, trainable, f)
                                                    for f in fold_ids)
            for f, s in zip(fold_ids, held_out):
                scores[folds == f] = s
        else:
            features = channels if arm == "C0" else np.concatenate([channels, spike_channels], axis=2)
            if arm in TEMPORAL_ARMS:
                temporal = temporal_features(features, df.subject_id.to_numpy(), df.window_index.to_numpy(),
                                             eligible, valid)
                n_base = len(BASE_FEATURES)
                columns = [BLOCKS.index(b) * n_base + i for b in TEMPORAL_ARMS[arm] for i in range(n_base)]
                features = np.concatenate([features, temporal[:, :, columns]], axis=2)
            labels = channel_labels(df, load_channel_annotations())
            held_out = Parallel(n_jobs=args.n_jobs)(delayed(fit_predict_channels)(
                features, labels, valid, folds, trainable, eligible, f) for f in fold_ids)
            for f, s in zip(fold_ids, held_out):
                scores[folds == f] = max_over_channels(s)

        pred = df.assign(score=scores)
        for f in fold_ids:
            row = evaluate(pred[pred.fold == f], arm, "SVC", f)
            rows.append(row)
            print(f"SVC {arm} fold {f}: AUROC {row['auroc']:.3f}  AUPRC {row['auprc']:.3f}", flush=True)
        pred[PREDICTION_COLUMNS].sort_values(["subject_id", "start_s"]).to_parquet(
            RESULTS_DIR / f"predictions_SVC_{arm}.parquet", index=False)
        save_cv_rows(rows, RESULTS_DIR)        # after each arm, so finished arms survive an interruption


if __name__ == "__main__":
    main()
