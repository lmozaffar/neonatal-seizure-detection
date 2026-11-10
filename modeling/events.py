"""Event-level evaluation of held-out window predictions against the experts' annotations (METHODOLOGY.md §21).

    from modeling.events import MAJORITY, CONFIGS, load_all_annotations, build_timelines, fold_curves, choose, evaluate

Takes predictions in the format of results/predictions_<model>_<set>.parquet and the three annotation CSVs
(helsinki.labeling.load_annotations). Needs no EEG and retrains nothing.

1. Reference events (Protocol). A second is seizure when at least min_votes experts mark it, and it counts only if all
   three experts annotated it. A reference event is a run of consecutive seizure seconds.
     majority   2 of 3 experts (our labels). A false alarm overlaps no event; hours are all evaluable seconds.
     published  the scoring of the published Helsinki detector (Tapani et al. 2019): all 3 experts, runs of at least
                10 s (shorter all-3 runs become ambiguous); a false alarm overlaps no second marked by any expert; hours
                leave out seconds marked by only 1 or 2 experts.
   Ambiguous seconds are marked by an expert but are not reference seizure.

2. Score timeline (causal). Second t takes the score of the earliest-ending eligible window that contains it, and is
   decided at that window's end_s. A second is evaluable when it is annotated and an eligible window scores it.
   Postprocessing (PostProcessing), in this order:
     smooth k        each window's score becomes the mean of the last k eligible windows of its subject;
     threshold       detections are runs of evaluable seconds with score >= threshold;
     min duration D  runs shorter than D s are dropped; an alarm is raised at the decision time of the run's D-th second;
     merge gap G     a run starting at most G s after the previous one joins its detection (no new alarm).
   Alarm times only use windows that have ended, so they never use EEG recorded after the alarm.

3. Matching (any overlap). An event is detected when a detection flags one of its seconds; a detection that overlaps no
   event is a false alarm. Latency is the earliest alarm of the detections overlapping an event minus its onset,
   floored at 0. Events without an evaluable second are left out.

4. Threshold and postprocessing selection (choose). For each test fold, every configuration of CONFIGS gets the threshold
   chosen on the other 4 folds: scanning down from the highest threshold, the lowest one before the other folds' FA/h
   first exceeds the target. The configuration with the highest other-fold sensitivity is then chosen (ties: lower
   other-fold FA/h, then the simpler configuration). Nothing about the test fold is used. This is cross-fitted, NOT
   strictly nested: the other folds' scores come from models trained on the test fold's patients, and the threshold grid
   (QUANTILES of every fold's scores) pools all folds.

5. Statistics. Paired patient bootstrap: N_BOOT resamples of whole patients (seed SEED), both models on the same
   resample, nothing selected again. Matched FA/h: each model's thresholds are stepped along its grid around the chosen
   ones, and sensitivity is read off both curves at the lower of the two realised FA/h, inside every resample too.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from helsinki.labeling import load_annotations
from helsinki.preprocessing import STRIDE_S

FA_TARGETS = (1.0, 0.5)            # false alarms per hour
QUANTILES = np.unique(np.r_[np.linspace(0, 0.98, 50), 1 - np.geomspace(0.02, 1e-4, 50)])   # dense near the top
SMOOTHING = (1, 3, 5, 7)
MIN_DURATIONS_S = (0, 12, 18, 24)
MERGE_GAPS_S = (0, 6, 12, 30)
MATCH_STEPS = range(-40, 41)       # threshold grid steps around the chosen threshold; positive = stricter
SHORT_S = 60                       # short seizures: under 60 s
DURATION_BINS = (0, 12, 30, 60, 120, np.inf)
DURATION_LABELS = ("<12 s", "12-30 s", "30-60 s", "60-120 s", ">=120 s")
N_BOOT = 2000
SEED = 0


@dataclass(frozen=True)
class Protocol:
    name: str
    min_votes: int                 # experts needed for a seizure second
    min_event_s: int = 0           # shorter runs are ambiguous, not events
    fa_lenient: bool = False       # a false alarm must also miss every expert-marked second
    hours_agreed: bool = False     # FA/h denominator leaves out ambiguous seconds


MAJORITY = Protocol("majority", 2)
PUBLISHED = Protocol("published", 3, min_event_s=10, fa_lenient=True, hours_agreed=True)
PROTOCOLS = (MAJORITY, PUBLISHED)


@dataclass(frozen=True, order=True)
class PostProcessing:
    smooth: int = 1
    min_duration_s: int = 0
    merge_gap_s: int = 0

    @property
    def name(self) -> str:
        return f"smooth {self.smooth}, min {self.min_duration_s} s, merge {self.merge_gap_s} s"


BASELINE = PostProcessing()        # no postprocessing
CONFIGS = [PostProcessing(*c) for c in itertools.product(SMOOTHING, MIN_DURATIONS_S, MERGE_GAPS_S)]


def duration_bin(duration_s) -> pd.Categorical:
    return pd.cut(np.asarray(duration_s, float), DURATION_BINS, right=False, labels=DURATION_LABELS)


# ---- 1, 2. reference events and score timeline ----------------------------------------

def load_all_annotations(subject_ids) -> dict[int, np.ndarray]:
    return {int(s): load_annotations(int(s)) for s in subject_ids}


def runs(mask) -> tuple[np.ndarray, np.ndarray]:
    """(starts, ends) of the runs of True in a 1-D boolean array; ends are exclusive."""
    edges = np.diff(np.r_[0, np.asarray(mask, np.int8), 0])
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def expert_seconds(annotations: np.ndarray, protocol: Protocol = MAJORITY):
    """(annotated, seizure, ambiguous) per second from a (3, n) annotation array."""
    annotated = np.isfinite(annotations).all(axis=0)
    votes = (np.where(annotated, annotations, 0) == 1).sum(axis=0)
    seizure = votes >= protocol.min_votes
    if protocol.min_event_s > 1:
        start, end = runs(seizure)
        seizure = np.zeros_like(seizure)
        for a, b in zip(start, end):
            if b - a >= protocol.min_event_s:
                seizure[a:b] = True
    return annotated, seizure, (votes >= 1) & ~seizure


def smooth_scores(predictions: pd.DataFrame, k: int) -> pd.DataFrame:
    """Copy with each eligible window's score replaced by the mean of the last k eligible windows of its subject."""
    p = predictions.sort_values(["subject_id", "start_s"])
    if k == 1:
        return p
    if not (p.groupby("subject_id").start_s.diff().dropna() == STRIDE_S).all():
        raise ValueError(f"predictions must hold every window of a subject, {STRIDE_S} s apart")
    eligible = p.eligible.astype(bool)
    smoothed = p.score.where(eligible).groupby(p.subject_id).transform(lambda s: s.rolling(k, min_periods=1).mean())
    return p.assign(score=smoothed.where(eligible, p.score))


def score_timeline(windows: pd.DataFrame, n_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-second (score, decision_s) of one subject: the earliest-ending eligible window containing the second."""
    w = windows[windows.eligible.astype(bool)].sort_values("end_s", ascending=False)
    start, end = w.start_s.to_numpy(), w.end_s.to_numpy()
    if not (np.all(start == np.round(start)) and np.all(end == np.round(end))):
        raise ValueError("windows must start and end on whole seconds to align with annotations")
    score = np.full(n_seconds, np.nan)
    decision = np.full(n_seconds, np.inf)
    for a, b, s in zip(start.astype(int), end.astype(int), w.score.to_numpy()):   # later windows first, so
        score[a:b] = s                                                             # earlier ones overwrite them
        decision[a:b] = b
    return score, decision


@dataclass
class Timeline:
    """Per-second arrays of all subjects. Each subject ends with one non-evaluable second, so no run crosses subjects."""
    subject_id: np.ndarray
    fold: np.ndarray
    score: np.ndarray
    decision_s: np.ndarray
    evaluable: np.ndarray
    event: np.ndarray        # row of the events table, -1 outside events
    ambiguous: np.ndarray

    def subset(self, mask) -> Timeline:
        return Timeline(**{f.name: getattr(self, f.name)[mask] for f in fields(self)})


def build_timeline(predictions: pd.DataFrame, annotations: dict, protocol: Protocol = MAJORITY):
    """Timeline and reference events (subject_id, fold, event_id, start_s, end_s, duration_s, evaluable_s)."""
    parts, events, n_events = [], [], 0
    for sid, w in predictions.groupby("subject_id", sort=True):
        sid = int(sid)
        if w.fold.nunique() != 1:
            raise ValueError(f"subject {sid} is in more than one fold")
        fold = int(w.fold.iloc[0])
        ann = annotations[sid]
        n = max(ann.shape[1], int(np.ceil(w.end_s.max())))
        padded = np.full((ann.shape[0], n), np.nan)       # seconds past the annotations are not annotated
        padded[:, :ann.shape[1]] = ann
        annotated, seizure, ambiguous = expert_seconds(padded, protocol)
        score, decision = score_timeline(w, n)
        start, end = runs(seizure)
        event = np.full(n, -1)
        for i, (a, b) in enumerate(zip(start, end)):
            event[a:b] = n_events + i
        n_events += len(start)
        events.append(pd.DataFrame({"subject_id": sid, "fold": fold, "start_s": start, "end_s": end,
                                    "duration_s": end - start}))
        parts.append({"subject_id": np.full(n + 1, sid), "fold": np.full(n + 1, fold),
                      "score": np.r_[score, np.nan], "decision_s": np.r_[decision, np.inf],
                      "evaluable": np.r_[annotated & np.isfinite(score), False], "event": np.r_[event, -1],
                      "ambiguous": np.r_[ambiguous, False]})
    tl = Timeline(**{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
    events = pd.concat(events, ignore_index=True)
    events.insert(2, "event_id", events.groupby("subject_id").cumcount())
    events["evaluable_s"] = np.bincount(tl.event[tl.evaluable & (tl.event >= 0)], minlength=len(events))
    return tl, events


def build_timelines(predictions: pd.DataFrame, annotations: dict, protocol: Protocol, smoothings=SMOOTHING):
    """{smooth: Timeline} and the reference events (the same for every smoothing)."""
    timelines = {}
    for k in sorted(set(smoothings)):
        timelines[k], events = build_timeline(smooth_scores(predictions, k), annotations, protocol)
    return timelines, events


def threshold_grid(tl: Timeline) -> np.ndarray:
    return np.unique(np.quantile(tl.score[tl.evaluable], QUANTILES))


# ---- 3. detection and matching ---------------------------------------------------------

def match(tl: Timeline, threshold, n_events: int, min_duration_s: int = 0, merge_gap_s: int = 0) -> dict:
    """Detections at `threshold` matched to the reference events (module doc, steps 2 and 3).

    Per detection: det_start (timeline index of its first flagged second), det_alarm_s, det_n_events (events it
    overlaps), det_ambiguous (it overlaps ambiguous time). Per event: event_detected, event_alarm_s (inf if missed).
    `on` marks the flagged seconds.
    """
    start, end = runs(tl.evaluable & (tl.score >= threshold))
    need = max(min_duration_s, 1)
    keep = end - start >= need
    start, end = start[keep], end[keep]
    alarm = tl.decision_s[start + need - 1]             # decision times never decrease within a run

    new = np.r_[True, (start[1:] - end[:-1] > merge_gap_s)
                | (tl.subject_id[start[1:]] != tl.subject_id[start[:-1]])][:len(start)]
    group = np.cumsum(new) - 1                          # detection of each kept run
    first = np.flatnonzero(new)

    delta = np.zeros(len(tl.score) + 1, np.int64)
    delta[start] += 1
    delta[end] -= 1
    on = np.cumsum(delta)[:-1] > 0
    marker = np.zeros(len(on), np.int64)
    marker[start] = 1
    run_of = np.cumsum(marker) - 1                      # kept run of each second, valid where `on`

    idx = np.flatnonzero(on & (tl.event >= 0))
    n = max(n_events, 1)
    key = np.unique(group[run_of[idx]] * n + tl.event[idx])        # (detection, event) pairs
    det_of, event_of = key // n, key % n
    det_alarm = alarm[first]
    event_alarm = np.full(n_events, np.inf)
    np.minimum.at(event_alarm, event_of, det_alarm[det_of])
    amb = np.flatnonzero(on & tl.ambiguous)
    return {"det_start": start[first], "det_alarm_s": det_alarm,
            "det_n_events": np.bincount(det_of, minlength=len(first)),
            "det_ambiguous": np.bincount(group[run_of[amb]], minlength=len(first)) > 0,
            "event_detected": np.isfinite(event_alarm), "event_alarm_s": event_alarm, "on": on}


def patient_counts(tl: Timeline, events: pd.DataFrame, m: dict, protocol: Protocol) -> pd.DataFrame:
    """Per subject of `tl`: events and detections (all and short), false alarms and hours, as the protocol counts them."""
    subjects, first = np.unique(tl.subject_id, return_index=True)
    present = np.zeros(len(events), bool)
    present[tl.event[tl.event >= 0]] = True
    ev = np.flatnonzero(present & (events.evaluable_s.to_numpy() > 0))
    pos = np.searchsorted(subjects, events.subject_id.to_numpy()[ev])
    short = events.duration_s.to_numpy()[ev] < SHORT_S
    detected = m["event_detected"][ev]
    fa = m["det_n_events"] == 0
    if protocol.fa_lenient:
        fa &= ~m["det_ambiguous"]
    in_hours = tl.evaluable & ~tl.ambiguous if protocol.hours_agreed else tl.evaluable
    n = len(subjects)
    seconds = np.bincount(np.searchsorted(subjects, tl.subject_id[in_hours]), minlength=n)
    return pd.DataFrame({
        "subject_id": subjects, "fold": tl.fold[first],
        "n_events": np.bincount(pos, minlength=n), "n_detected": np.bincount(pos[detected], minlength=n),
        "n_events_short": np.bincount(pos[short], minlength=n),
        "n_detected_short": np.bincount(pos[detected & short], minlength=n),
        "n_false_alarms": np.bincount(np.searchsorted(subjects, tl.subject_id[m["det_start"]][fa]), minlength=n),
        "seconds": seconds, "hours": seconds / 3600})


def fold_hours(patients: pd.DataFrame) -> pd.Series:
    """Hours per fold from whole seconds, so sums over folds add the same numbers as the original implementation."""
    return patients.groupby("fold").seconds.sum() / 3600


# ---- 4. selection on the other folds --------------------------------------------------

def fold_curves(timelines: dict, events: pd.DataFrame, protocol: Protocol, configs=CONFIGS) -> pd.DataFrame:
    """Counts per configuration, threshold and fold; thresholds at QUANTILES of each smoothing's scores."""
    rows = []
    for k, tl in timelines.items():
        for t in threshold_grid(tl):
            for cfg in (c for c in configs if c.smooth == k):
                m = match(tl, t, len(events), cfg.min_duration_s, cfg.merge_gap_s)
                patients = patient_counts(tl, events, m, protocol)
                counts = patients.drop(columns=["subject_id", "seconds", "hours"]).groupby("fold").sum()
                counts["hours"] = fold_hours(patients)
                rows.append(counts.reset_index().assign(smooth=cfg.smooth, min_duration_s=cfg.min_duration_s,
                                                        merge_gap_s=cfg.merge_gap_s, threshold=t))
    return pd.concat(rows, ignore_index=True)


def choose(curves: pd.DataFrame, fa_target: float) -> tuple[dict, pd.DataFrame]:
    """Per test fold, chosen on the other folds (module doc, section 4).

    Returns ({"baseline": {fold: (config, threshold)}, "postprocessed": {...}}, the chosen rows). The baseline is
    BASELINE at its own other-fold threshold.
    """
    keys = ["smooth", "min_duration_s", "merge_gap_s"]
    folds = sorted(curves.fold.unique())
    rows = []
    for cfg, c in curves.groupby(keys):
        for fold in folds:
            other = c[c.fold != fold].groupby("threshold")[["n_detected", "n_events", "n_false_alarms", "hours"]].sum()
            other = other.sort_index(ascending=False)
            fa = (other.n_false_alarms / other.hours).to_numpy()
            ok = np.cumprod(fa <= fa_target).astype(bool)          # True until FA/h first exceeds the target
            if ok.any():
                i = np.flatnonzero(ok)[-1]
                pick = (other.index[i], other.n_detected.iloc[i] / other.n_events.iloc[i], fa[i])
            else:
                pick = (np.inf, 0.0, 0.0)                          # no threshold fits: no detections
            rows.append({"fold": int(fold), **dict(zip(keys, cfg)), "threshold": float(pick[0]),
                         "other_sensitivity": float(pick[1]), "other_fa_per_h": float(pick[2])})
    cand = pd.DataFrame(rows)
    best = (cand.sort_values(["fold", "other_sensitivity", "other_fa_per_h", *keys],
                             ascending=[True, False, True, True, True, True]).groupby("fold").head(1))
    base = cand[(cand[keys] == [1, 0, 0]).all(axis=1)]
    chosen = pd.concat([base.assign(method="baseline"), best.assign(method="postprocessed")], ignore_index=True)
    chosen = chosen.assign(fa_target=fa_target)
    assignments = {method: {int(r.fold): (PostProcessing(int(r.smooth), int(r.min_duration_s), int(r.merge_gap_s)),
                                          float(r.threshold)) for r in g.itertuples()}
                   for method, g in chosen.groupby("method")}
    return assignments, chosen


# ---- held-out results -----------------------------------------------------------------

def evaluate(timelines: dict, events: pd.DataFrame, assignment: dict, protocol: Protocol):
    """Held-out (patients, events) when test fold f uses assignment[f] = (config, threshold).

    patients: one row per subject with patient_counts, fold, config and threshold.
    events: one row per evaluable reference event with detected and latency_s (inf if missed).
    """
    patients, rows = [], []
    for fold, (cfg, threshold) in sorted(assignment.items()):
        tl = timelines[cfg.smooth]
        tl = tl.subset(tl.fold == fold)
        m = match(tl, threshold, len(events), cfg.min_duration_s, cfg.merge_gap_s)
        patients.append(patient_counts(tl, events, m, protocol).assign(config=cfg.name, threshold=threshold))
        ids = events.index[(events.fold == fold) & (events.evaluable_s > 0)]
        rows.append(events.loc[ids].assign(detected=m["event_detected"][ids],
                                           latency_s=np.maximum(0, m["event_alarm_s"][ids] - events.start_s[ids])))
    return pd.concat(patients, ignore_index=True), pd.concat(rows)


def summary(patients: pd.DataFrame, ev: pd.DataFrame) -> dict:
    """Pooled held-out metrics."""
    p = patients.sum(numeric_only=True)
    hours = fold_hours(patients).sum()
    latency = ev.latency_s[ev.detected]
    return {"n_events": int(p.n_events), "n_detected": int(p.n_detected), "sensitivity": p.n_detected / p.n_events,
            "n_events_short": int(p.n_events_short), "n_detected_short": int(p.n_detected_short),
            "sensitivity_short": p.n_detected_short / p.n_events_short,
            "n_false_alarms": int(p.n_false_alarms), "hours": hours, "fa_per_h": p.n_false_alarms / hours,
            "latency_median_s": float(np.median(latency)) if len(latency) else np.nan,
            "patients_none_detected": int(((patients.n_events > 0) & (patients.n_detected == 0)).sum())}


# ---- 5. statistics ---------------------------------------------------------------------

def bootstrap_indices(n_patients: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_patients, size=(n_boot, n_patients))


def paired_bootstrap(ref: pd.DataFrame, arm: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """95% intervals of arm - ref (sensitivity, short-seizure sensitivity, FA/h) from per-patient tables."""
    subjects = np.sort(ref.subject_id.to_numpy())
    r, a = ref.set_index("subject_id").loc[subjects], arm.set_index("subject_id").loc[subjects]
    idx = bootstrap_indices(len(subjects), n_boot, seed)

    def ratio(t, num, den):
        return t[num].to_numpy(float)[idx].sum(1) / t[den].to_numpy(float)[idx].sum(1)

    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for name, num, den in [("d_sensitivity", "n_detected", "n_events"),
                               ("d_sensitivity_short", "n_detected_short", "n_events_short"),
                               ("d_fa_per_h", "n_false_alarms", "hours")]:
            out[f"{name}_ci_low"], out[f"{name}_ci_high"] = np.nanpercentile(ratio(a, num, den) - ratio(r, num, den),
                                                                             [2.5, 97.5])
    return out


def step_threshold(grid, threshold: float, step: int) -> float:
    """The threshold `step` grid positions above `threshold` (clamped); inf, the top, means no detections."""
    g = np.r_[np.sort(np.asarray(grid, float)), np.inf]
    i = len(g) - 1 if not np.isfinite(threshold) else int(np.argmin(np.abs(g[:-1] - threshold)))
    return float(g[np.clip(i + step, 0, len(g) - 1)])


def threshold_curve(timelines: dict, events: pd.DataFrame, assignment: dict, protocol: Protocol,
                    subjects, steps=MATCH_STEPS) -> dict:
    """Per-patient counts with every fold's threshold moved `step` positions along its grid, strictest step first."""
    subjects = np.sort(np.asarray(subjects))
    steps = np.sort(np.asarray(steps))[::-1]
    grids = {fold: threshold_grid(timelines[cfg.smooth]) for fold, (cfg, _) in assignment.items()}
    per_step = ("n_detected", "n_detected_short", "n_false_alarms")      # change with the threshold
    fixed = ("n_events", "n_events_short", "hours")                      # do not
    curve = {k: np.zeros((len(steps), len(subjects))) for k in per_step} | {k: np.zeros(len(subjects)) for k in fixed}
    for s, step in enumerate(steps):
        for fold, (cfg, threshold) in assignment.items():
            tl = timelines[cfg.smooth]
            tl = tl.subset(tl.fold == fold)
            m = match(tl, step_threshold(grids[fold], threshold, int(step)), len(events), cfg.min_duration_s,
                      cfg.merge_gap_s)
            c = patient_counts(tl, events, m, protocol).set_index("subject_id").reindex(subjects, fill_value=0)
            for k in per_step:
                curve[k][s] += c[k].to_numpy()
            if s == 0:
                for k in fixed:
                    curve[k] += c[k].to_numpy()
    return {"steps": steps, "subjects": subjects, **curve}


def interpolate_sensitivity(fa: np.ndarray, sensitivity: np.ndarray, target: float) -> float:
    """Sensitivity at FA/h `target`, interpolated on log FA/h down to the FA/h peak; NaN outside the observed range.
    `fa` and `sensitivity` run from the strictest threshold down."""
    if not target > 0 or not np.isfinite(fa).any():
        return np.nan
    peak = int(np.nanargmax(fa))
    f, s = fa[:peak + 1], sensitivity[:peak + 1]
    keep = f > 0
    f, s = f[keep], s[keep]
    if not len(f) or target < f.min() or target > f.max():
        return np.nan
    x, inverse = np.unique(np.log(np.maximum.accumulate(f)), return_inverse=True)
    y = np.full(len(x), -np.inf)
    np.maximum.at(y, inverse, s)
    return float(np.interp(np.log(target), x, y))


def pooled_rates(curve: dict, cols: np.ndarray):
    """(FA/h, sensitivity, short-seizure sensitivity) per step of a threshold_curve(), pooled over patient columns
    `cols` (a bootstrap resample may repeat columns)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        fa = curve["n_false_alarms"][:, cols].sum(1) / curve["hours"][cols].sum()
        sensitivity = curve["n_detected"][:, cols].sum(1) / curve["n_events"][cols].sum()
        short = curve["n_detected_short"][:, cols].sum(1) / curve["n_events_short"][cols].sum()
    return fa, sensitivity, short


def matched_fa(arm: dict, ref: dict, exclude=(), n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Sensitivity (all, short) of arm and ref at the lower of their realised FA/h, with paired bootstrap CIs.

    `arm` and `ref` are threshold_curve()s over the same subjects; `exclude` drops patients before anything is computed.
    The matched FA/h is found again inside every resample, so the intervals include that step.
    """
    keep = np.flatnonzero(~np.isin(ref["subjects"], list(exclude)))
    zero = np.flatnonzero(arm["steps"] == 0)[0]                 # the chosen thresholds
    samples = [keep] + [keep[row] for row in bootstrap_indices(len(keep), n_boot, seed)]   # observed, then resamples

    rows = []                                                   # per sample: fa, arm all, ref all, arm short, ref short
    for cols in samples:
        fa_a, sens_a, short_a = pooled_rates(arm, cols)
        fa_r, sens_r, short_r = pooled_rates(ref, cols)
        fa = min(fa_a[zero], fa_r[zero])
        rows.append([fa, interpolate_sensitivity(fa_a, sens_a, fa), interpolate_sensitivity(fa_r, sens_r, fa),
                     interpolate_sensitivity(fa_a, short_a, fa), interpolate_sensitivity(fa_r, short_r, fa)])
    rows = np.array(rows)
    observed, boot = rows[0], rows[1:]

    out = {"matched_fa_per_h": float(observed[0]), "sensitivity_arm": observed[1], "sensitivity_ref": observed[2],
           "sensitivity_short_arm": observed[3], "sensitivity_short_ref": observed[4]}
    for name, a, r in [("d_sensitivity", 1, 2), ("d_sensitivity_short", 3, 4)]:
        out[name] = observed[a] - observed[r]
        d = boot[:, a] - boot[:, r]
        d = d[np.isfinite(d)]                                   # outside a curve's FA/h range: NaN
        out[f"{name}_ci_low"], out[f"{name}_ci_high"] = np.percentile(d, [2.5, 97.5]) if len(d) else (np.nan, np.nan)
    return out
