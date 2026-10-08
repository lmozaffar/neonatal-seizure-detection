#!/usr/bin/env python3
"""
Simplified SPH/SOP (Seizure Prediction Horizon / Seizure Occurrence Period)
alarm-based evaluation.

This is a cleaned-up, documented reimplementation of the core logic in
notebooks/notebook_utils.py (run_sz_alarm / run_sph_sop_exp / get_sph_sop_exp),
with the BSS/brier-score bookkeeping, FDR/FDE granular stats, postnatal-age
handling, and multiprocessing stripped out. It is NOT guaranteed to reproduce
the original's numbers to the last decimal (see "Known simplifications"
below) -- for the faithful reproduction, see run_sph_sop_analysis.py /
resume_sph_sop.py, which call the original notebook_utils functions directly.

----------------------------------------------------------------------------
THE ALGORITHM
----------------------------------------------------------------------------
For a seizure starting at epoch `onset`, an alarm only counts as a correct
prediction (TP) if it fires inside the window:

    [onset - SPH - SOP,  onset - SPH]

  - SPH (Seizure Prediction Horizon): a mandatory "blackout" right before
    the seizure. An alarm firing inside the last SPH minutes doesn't count
    -- clinically, there's no time left to act on it.
  - SOP (Seizure Occurrence Period): the valid alarm window before that
    blackout. An alarm anywhere in the SOP minutes immediately preceding
    the SPH blackout still counts as having caught this seizure.

Pipeline, per subject, per probability threshold:
  1. Smooth the raw per-epoch model probability with a moving average
     (reduces epoch-to-epoch noise into a stable "risk level").
  2. Threshold the smoothed signal -> binary alarm on/off per epoch.
  3. Extend each "on" epoch forward by SOP epochs (once an alarm fires, it
     stays live for the window it's claiming the seizure will occur in).
  4. For each seizure onset: did the extended alarm fire anywhere in its
     [onset-SPH-SOP, onset-SPH] window? Yes -> TP. No -> FN.
  5. Any alarm-on epoch *outside* every seizure's valid window, and not
     during a seizure itself (ictal epochs are excluded), counts as a false
     alarm (FP) epoch.
  6. Pool TP/FP/FN across every subject in the test set, then compute
     sensitivity/specificity/precision/recall from the pooled totals.
  7. Sweep step 2-6 across many thresholds to trace an AUROC/AUPRC curve.

Seizures within `glue_minutes` of each other are merged into one event
first, so a rapid cluster doesn't get scored as several independent targets.

----------------------------------------------------------------------------
KNOWN SIMPLIFICATIONS vs. the original notebook_utils.py
----------------------------------------------------------------------------
  - Precision here is a simple event-count ratio (TP events / (TP events +
    FP alarm events)). The original normalizes FP by elapsed SOP-hours with
    several unit conversions -- more precise but much harder to read.
  - No Brier skill score, no FDR/FDE "granular false-detection-event" stats,
    no postnatal-age-aware subject filtering, no multiprocessing.
  - Specificity uses wall-clock seizure-free hours directly; the original
    tracks a few additional edge cases around augmentation-segment and
    recording-boundary handling.

----------------------------------------------------------------------------
INPUT FORMAT
----------------------------------------------------------------------------
A long dataframe with one row per (subject, epoch), columns:
  - subject_id   : subject identifier
  - t            : epoch index within the subject, starting at 0, strictly
                    increasing by 1 (NOT a timestamp -- a position)
  - sz_mask      : bool, True while a seizure is actively occurring
  - y_pred       : float in [0, 1], the model's predicted probability

This matches results_sph_sop/{model}_continuous_preds.csv from this repo's
SPH/SOP reproduction (rename its y_preds_run0 column to y_pred, or pass
pred_col="y_preds_run0").
"""
import numpy as np
import pandas as pd
from sklearn.metrics import auc as sk_auc

EPOCH_SECONDS = 20  # each row/epoch = 20 seconds of EEG


def moving_average(x, window):
    """Centered moving average, edge-padded."""
    window = int(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    x_padded = np.pad(x, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(x_padded, kernel, mode="valid")


def glue_seizures(sz_mask, wins_per_min, glue_minutes):
    """Merge seizures separated by a gap shorter than glue_minutes into one."""
    sz_mask = np.asarray(sz_mask).astype(int).copy()
    glue_epochs = int(wins_per_min * glue_minutes)

    starts = np.where(np.diff(sz_mask) == 1)[0] + 1
    ends = np.where(np.diff(sz_mask) == -1)[0] + 1
    if sz_mask[0] == 1:
        starts = np.r_[0, starts]
    if sz_mask[-1] == 1:
        ends = np.r_[ends, len(sz_mask)]

    for end_of_prev, start_of_next in zip(ends[:-1], starts[1:]):
        if start_of_next - end_of_prev <= glue_epochs:
            sz_mask[end_of_prev:start_of_next] = 1
    return sz_mask


def seizure_onsets(sz_mask):
    """Epoch indices where a (glued) seizure begins."""
    sz_mask = np.asarray(sz_mask).astype(int)
    onsets = np.where(np.diff(sz_mask) == 1)[0] + 1
    if sz_mask[0] == 1:
        onsets = np.r_[0, onsets]
    return onsets


def extend_alarm(alarm, sop_epochs):
    """Hold each alarm 'on' epoch active for sop_epochs after it fires."""
    alarm = np.asarray(alarm).astype(int)
    extended = np.zeros_like(alarm)
    on_positions = np.where(alarm == 1)[0]
    for pos in on_positions:
        extended[pos: pos + sop_epochs + 1] = 1
    return extended[: len(alarm)]


def score_subject(sz_mask, y_pred, thresh, sph_epochs, sop_epochs, smooth_epochs):
    """TP/FN seizure counts and FP epoch-count for one subject at one threshold."""
    smoothed = moving_average(y_pred, smooth_epochs)
    alarm = extend_alarm(smoothed >= thresh, sop_epochs)

    onsets = seizure_onsets(sz_mask)
    # only score onsets with enough pre-ictal history for a full [onset-sph-sop, onset-sph] window
    onsets = onsets[onsets - sph_epochs - sop_epochs >= 0]

    tp, fn = 0, 0
    counted_as_valid = np.zeros_like(alarm, dtype=bool)
    for onset in onsets:
        win_start = onset - sph_epochs - sop_epochs
        win_end = onset - sph_epochs
        counted_as_valid[win_start:win_end] = True
        if alarm[win_start:win_end].any():
            tp += 1
        else:
            fn += 1

    false_alarm_epochs = alarm & ~counted_as_valid & ~sz_mask.astype(bool)
    fp_epochs = int(false_alarm_epochs.sum())
    seizure_free_epochs = int((~sz_mask.astype(bool)).sum())

    return tp, fn, fp_epochs, seizure_free_epochs


def evaluate_at_threshold(df, thresh, sph_min, sop_min, wins_per_min, smooth_min, pred_col):
    sph_epochs = int(sph_min * wins_per_min)
    sop_epochs = int(sop_min * wins_per_min)
    smooth_epochs = int(smooth_min * wins_per_min)

    total_tp = total_fn = total_fp_epochs = total_szfree_epochs = 0
    for _, g in df.groupby("subject_id", sort=False):
        g = g.sort_values("t")
        tp, fn, fp_epochs, szfree_epochs = score_subject(
            g["sz_mask"].values, g[pred_col].values, thresh, sph_epochs, sop_epochs, smooth_epochs
        )
        total_tp += tp
        total_fn += fn
        total_fp_epochs += fp_epochs
        total_szfree_epochs += szfree_epochs

    sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    fp_hours = total_fp_epochs * EPOCH_SECONDS / 3600
    szfree_hours = total_szfree_epochs * EPOCH_SECONDS / 3600
    specificity = 1 - fp_hours / szfree_hours if szfree_hours > 0 else np.nan

    precision = total_tp / (total_tp + total_fp_epochs) if (total_tp + total_fp_epochs) > 0 else np.nan
    recall = sensitivity

    return dict(tp=total_tp, fn=total_fn, fp_epochs=total_fp_epochs,
                sensitivity=sensitivity, specificity=specificity,
                precision=precision, recall=recall)


def run_sph_sop_curve(df, sph_min, sop_min, wins_per_min=3, glue_min=3, smooth_min=20,
                       thresholds=None, pred_col="y_pred"):
    """Sweep thresholds for one (SPH, SOP) cell. Returns a results dict with
    per-threshold arrays plus AUROC/AUPRC."""
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.01)

    df = df.copy()
    df["sz_mask"] = df.groupby("subject_id", sort=False)["sz_mask"].transform(
        lambda m: glue_seizures(m.values, wins_per_min, glue_min)
    ).astype(bool)

    rows = [evaluate_at_threshold(df, t, sph_min, sop_min, wins_per_min, smooth_min, pred_col)
            for t in thresholds]
    out = pd.DataFrame(rows)
    out["threshold"] = thresholds

    # ROC: sensitivity (TPR) vs 1-specificity (FPR), endpoints pinned for a clean curve.
    # Sort by fpr while keeping each tpr paired to its own fpr (sorting the two
    # arrays independently would silently decouple the pairing and corrupt the AUC).
    fpr = np.r_[1, 1 - out["specificity"].values, 0]
    tpr = np.r_[1, out["sensitivity"].values, 0]
    order = np.argsort(fpr)
    auroc = sk_auc(fpr[order], tpr[order])

    pr = out.dropna(subset=["precision"])
    if len(pr) > 1:
        order = np.argsort(pr["recall"].values)
        auprc = sk_auc(pr["recall"].values[order], pr["precision"].values[order])
    else:
        auprc = np.nan

    return dict(curve=out, auroc=auroc, auprc=auprc)


def run_sph_sop_grid(df, sph_values, sop_values, **kwargs):
    """Run run_sph_sop_curve over every (SPH, SOP) combination. Returns a
    DataFrame with one row per cell."""
    rows = []
    for sph in sph_values:
        for sop in sop_values:
            res = run_sph_sop_curve(df, sph, sop, **kwargs)
            rows.append(dict(sph=sph, sop=sop, auroc=res["auroc"], auprc=res["auprc"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    # Example: run on one of this repo's saved continuous-prediction CSVs.
    model = sys.argv[1] if len(sys.argv) > 1 else "ConvLSTM"
    path = f"results_sph_sop/{model}_continuous_preds.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"y_preds_run0": "y_pred"})
    df["sz_mask"] = df["sz_mask"].astype(bool)

    result = run_sph_sop_curve(df, sph_min=3, sop_min=7, pred_col="y_pred")
    print(f"{model}  SPH=3min SOP=7min  AUROC={result['auroc']:.4f}  AUPRC={result['auprc']:.4f}")
