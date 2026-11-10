"""Event-level results of the saved held-out predictions, and the pre-specified comparisons (METHODOLOGY.md §21-25).

    .venv/bin/python -m modeling.evaluate_events
    .venv/bin/python -m modeling.evaluate_events --n-jobs 2       # results do not depend on it

Models: ConvLSTM, SVC and RF on feature set 2 (results/predictions_<model>_set2.parquet; SVC is A0, the original
detector) and the arms of modeling.run_channel_models (results/predictions_SVC_<arm>.parquet). For each model,
protocol (majority, published) and FA/h target (1, 0.5), thresholds and postprocessing are chosen on the other folds
(modeling.events.choose) and applied to the held-out fold, without postprocessing (baseline) and with it
(postprocessed). Like every event-level result here, this is cross-fitted, NOT strictly nested (modeling/events.py).

Comparisons, at the postprocessed operating points (every protocol and target; only the rows below are tests):
  E9, pre-specified: A1, C0 and C1 against A0 (SVC), majority, 1 FA/h. Success: a sensitivity gain of at least
      5 points whose paired patient-bootstrap 95% CI lies above 0, at a realised FA/h of at most 1.10. The three
      comparisons are not corrected for multiplicity. S against A0 and C1 against A1 and C0 are descriptive.
  E14, pre-specified: T-BD against C1, majority, 1 FA/h, at matched FA/h (the lower of the two realised FA/h;
      modeling.events.matched_fa). Success: a short-seizure (< 60 s) gain of at least 3 points with a 95% CI above
      0, T-BD's realised FA/h at most 1.10, and the lower bound of the overall difference at least -2 points.
      T-B and T-D are descriptive ablations.
  Post hoc (evaluation only; nothing is chosen again): C1 against A0 without patient 41, and T-BD against C1
      without patient 16, the largest contributors.
Every row gives the net gain in detected events, the patient with the largest net gain and that gain. Majority rows
also give the net gain in events no second of which all three experts marked (net_detected_2of3_only).

Outputs in results/:
  event_selected.csv           per model, protocol, target, operating point and test fold: configuration, threshold
                               and the other folds' sensitivity and FA/h
  event_results.csv            held-out results per model, protocol, target and operating point
  event_comparisons.csv        the comparisons above
  event_fold_gains.csv         detected events per fold of each comparison (majority, 1 FA/h, all patients)
  sensitivity_by_duration.csv  majority, 1 FA/h: rows of DURATION_MODELS replace earlier ones; other rows are kept
  sensitivity_by_duration.png  sensitivity by seizure duration, postprocessed
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from modeling.common import RESULTS_DIR
from modeling.events import (DURATION_LABELS, FA_TARGETS, MAJORITY, PROTOCOLS, Protocol, build_timelines, choose,
                             duration_bin, evaluate, expert_seconds, fold_curves, load_all_annotations, matched_fa,
                             paired_bootstrap, summary, threshold_curve)
from modeling.plot_results import GRID, SURFACE, TEXT, TEXT_MUTED
from modeling.run_channel_models import ARMS

MODELS = {"ConvLSTM": "predictions_ConvLSTM_set2.parquet", "SVC": "predictions_SVC_set2.parquet",
          "RF": "predictions_RF_set2.parquet", **{arm: f"predictions_SVC_{arm}.parquet" for arm in ARMS},
          "RF_leaf20": "predictions_RF_leaf20_set2.parquet", "LightGBM": "predictions_LightGBM_set2.parquet",
          "LightGBM_ctx": "predictions_LightGBM_ctx_set2.parquet"}
COMPARISONS = [("E9", "A1", "SVC", "pre-specified"), ("E9", "C0", "SVC", "pre-specified"),
               ("E9", "C1", "SVC", "pre-specified"), ("E9", "S", "SVC", "descriptive"),
               ("E9", "C1", "A1", "descriptive"), ("E9", "C1", "C0", "descriptive"),
               ("E14", "T-BD", "C1", "pre-specified"), ("E14", "T-B", "C1", "descriptive"),
               ("E14", "T-D", "C1", "descriptive")]
POST_HOC = [("E9", "C1", "SVC", 41), ("E14", "T-BD", "C1", 16)]    # (experiment, arm, reference, patient left out)
MATCHED_EXPERIMENTS = {"E14"}                      # compared at matched FA/h as well
CURVE_MODELS = {"C1", "T-BD", "T-B", "T-D"}
E9_MIN_GAIN, E14_MIN_SHORT_GAIN, E14_MIN_OVERALL, MAX_REALISED_FA = 0.05, 0.03, -0.02, 1.10
ALL_THREE = Protocol("all three experts", 3)
DURATION_MODELS = ["C1", "T-BD"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]         # categorical slots 1-3, as in plot_results.py


def run(model: str, protocol) -> dict:
    """{fa_target: (chosen rows, {operating point: (patients, events)}, threshold curve or None)}."""
    pred = pd.read_parquet(RESULTS_DIR / MODELS[model])
    annotations = load_all_annotations(pred.subject_id.unique())
    timelines, events = build_timelines(pred, annotations, protocol)
    if protocol == MAJORITY:                       # does any second of the event have all three experts?
        all_three = {s: expert_seconds(a, ALL_THREE)[1] for s, a in annotations.items()}
        events["all_three"] = [all_three[s][a:b].any() for s, a, b in zip(events.subject_id, events.start_s,
                                                                            events.end_s)]
    curves = fold_curves(timelines, events, protocol)
    out = {}
    for target in FA_TARGETS:
        assignments, chosen = choose(curves, target)
        tables = {op: evaluate(timelines, events, a, protocol) for op, a in assignments.items()}
        curve = (threshold_curve(timelines, events, assignments["postprocessed"], protocol, pred.subject_id.unique())
                 if model in CURVE_MODELS else None)
        out[target] = chosen, tables, curve
    return out


def comparison_row(arm_tables, ref_tables, exclude=(), curves=None) -> dict:
    """Arm against reference from their held-out (patients, events); `exclude` drops patients from both;
    `curves` (arm, reference threshold curves) adds the matched-FA comparison."""
    (pa, ea), (pr, er) = [(p[~p.subject_id.isin(exclude)], e[~e.subject_id.isin(exclude)])
                          for p, e in (arm_tables, ref_tables)]
    a, r = summary(pa, ea), summary(pr, er)
    if len(exclude):                               # hours summed per patient, as the original code did here
        a["fa_per_h"], r["fa_per_h"] = (p.n_false_alarms.sum() / p.hours.sum() for p in (pa, pr))
    net = pa.set_index("subject_id").n_detected - pr.set_index("subject_id").n_detected
    row = {"sensitivity_arm": a["sensitivity"], "sensitivity_ref": r["sensitivity"],
           "d_sensitivity": a["sensitivity"] - r["sensitivity"],
           "sensitivity_short_arm": a["sensitivity_short"], "sensitivity_short_ref": r["sensitivity_short"],
           "d_sensitivity_short": a["sensitivity_short"] - r["sensitivity_short"],
           "fa_per_h_arm": a["fa_per_h"], "fa_per_h_ref": r["fa_per_h"],
           "latency_median_arm_s": a["latency_median_s"], "latency_median_ref_s": r["latency_median_s"],
           "patients_none_detected_arm": a["patients_none_detected"],
           "patients_none_detected_ref": r["patients_none_detected"],
           **paired_bootstrap(pr, pa),
           "net_detected": int(net.sum()), "top_patient": int(net.idxmax()), "top_patient_net": int(net.max())}
    if "all_three" in ea:
        only = ~ea.all_three.astype(bool)
        row["net_detected_2of3_only"] = int(ea.detected[only].sum() - er.detected.loc[ea.index][only].sum())
    if curves is not None:
        m = matched_fa(*curves, exclude=exclude)
        row.update({"matched_fa_per_h": m["matched_fa_per_h"], **{f"{k}_matched": m[k] for k in [
            "d_sensitivity", "d_sensitivity_ci_low", "d_sensitivity_ci_high",
            "d_sensitivity_short", "d_sensitivity_short_ci_low", "d_sensitivity_short_ci_high"]}})
    return row


def success(experiment: str, row: dict) -> bool:
    if experiment == "E9":
        return bool(row["d_sensitivity"] >= E9_MIN_GAIN and row["d_sensitivity_ci_low"] > 0
                    and row["fa_per_h_arm"] <= MAX_REALISED_FA)
    return bool(row["d_sensitivity_short_matched"] >= E14_MIN_SHORT_GAIN - 1e-12
                and row["d_sensitivity_short_ci_low_matched"] > 0 and row["fa_per_h_arm"] <= MAX_REALISED_FA
                and row["d_sensitivity_ci_low_matched"] >= E14_MIN_OVERALL)


def plot_durations(table: pd.DataFrame, path) -> None:
    """Postprocessed sensitivity by seizure duration: E4's models, then A0 against the improved arms."""
    t = table[table.operating_point == "postprocessed"]
    panels = [("Original models (E4)", ["ConvLSTM", "SVC", "RF"]),
              ("A0 and the new arms", ["SVC", *[m for m in DURATION_MODELS if m in set(t.model)]])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE, sharey=True)
    x = np.arange(len(DURATION_LABELS))
    for ax, (title, models) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        width = 0.8 / len(models)
        for i, model in enumerate(models):
            s = t[t.model == model].set_index("duration_bin").reindex(DURATION_LABELS).sensitivity
            label = "SVC (A0)" if model == "SVC" and title.startswith("A0") else model
            ax.bar(x + (i - (len(models) - 1) / 2) * width, 100 * s, width, color=COLORS[i], label=label, zorder=2)
        ax.set_xticks(x, DURATION_LABELS, color=TEXT)
        ax.tick_params(colors=TEXT_MUTED)
        ax.set_xlabel("Seizure duration", color=TEXT)
        ax.set_title(title, color=TEXT, loc="left", fontsize=11)
        ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=TEXT)
    axes[0].set_ylabel("Seizure events detected (%)", color=TEXT)
    axes[0].set_ylim(0, 100)
    fig.text(0.01, 0.01, "Majority protocol, postprocessed, thresholds chosen on the other folds for <= 1 FA/h.",
             fontsize=7.5, color=TEXT_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    jobs = [(m, p) for m in MODELS for p in PROTOCOLS]
    done = dict(zip(jobs, Parallel(n_jobs=args.n_jobs)(delayed(run)(m, p) for m, p in jobs)))
    held = {(m, p.name, t): tables for (m, p), out in done.items() for t, (_, tables, _) in out.items()}
    curves = {(m, p.name, t): c for (m, p), out in done.items() for t, (_, _, c) in out.items()}

    selected, results, durations = [], [], []
    for (model, protocol), out in done.items():
        for target, (chosen, tables, _) in out.items():
            key = {"model": model, "protocol": protocol.name, "fa_target": target}
            selected.append(chosen.rename(columns={"method": "operating_point"}).assign(**key))
            for op, (patients, ev) in tables.items():
                results.append({**key, "operating_point": op, **summary(patients, ev)})
                if model in DURATION_MODELS and protocol == MAJORITY and target == 1.0:
                    d = (ev.assign(duration_bin=duration_bin(ev.duration_s)).groupby("duration_bin", observed=False)
                         .detected.agg(n_events="size", n_detected="sum").reset_index())
                    durations.append(d.assign(model=model, operating_point=op))

    front = ["model", "protocol", "fa_target", "operating_point"]
    selected = pd.concat(selected, ignore_index=True)
    selected[front + [c for c in selected if c not in front]].to_csv(RESULTS_DIR / "event_selected.csv", index=False)
    pd.DataFrame(results).to_csv(RESULTS_DIR / "event_results.csv", index=False)

    comparisons, fold_gains = [], []
    for experiment, arm, ref, role in COMPARISONS:
        for protocol in PROTOCOLS:
            for target in FA_TARGETS:
                k_arm, k_ref = (arm, protocol.name, target), (ref, protocol.name, target)
                a, r = held[k_arm]["postprocessed"], held[k_ref]["postprocessed"]
                pair = (curves[k_arm], curves[k_ref]) if experiment in MATCHED_EXPERIMENTS else None
                row = {"experiment": experiment, "arm": arm, "reference": ref, "protocol": protocol.name,
                       "fa_target": target, "analysis": "all patients", **comparison_row(a, r, curves=pair)}
                primary = role == "pre-specified" and protocol == MAJORITY and target == 1.0
                row["role"] = role if primary else "descriptive"
                row["success"] = success(experiment, row) if primary else None
                comparisons.append(row)
                if protocol == MAJORITY and target == 1.0:
                    f = a[0].groupby("fold")[["n_events", "n_detected", "n_events_short", "n_detected_short"]].sum()
                    f[["n_detected_ref", "n_detected_short_ref"]] = r[0].groupby("fold")[
                        ["n_detected", "n_detected_short"]].sum()
                    fold_gains.append(f.assign(experiment=experiment, arm=arm, reference=ref,
                                               net_detected=f.n_detected - f.n_detected_ref).reset_index())
    for experiment, arm, ref, patient in POST_HOC:
        k_arm, k_ref = (arm, MAJORITY.name, 1.0), (ref, MAJORITY.name, 1.0)
        pair = (curves[k_arm], curves[k_ref]) if experiment in MATCHED_EXPERIMENTS else None
        comparisons.append({"experiment": experiment, "arm": arm, "reference": ref, "protocol": MAJORITY.name,
                            "fa_target": 1.0, "analysis": f"without patient {patient} (post hoc)",
                            **comparison_row(held[k_arm]["postprocessed"], held[k_ref]["postprocessed"],
                                             exclude=[patient], curves=pair),
                            "role": "post hoc", "success": None})
    pd.DataFrame(comparisons).to_csv(RESULTS_DIR / "event_comparisons.csv", index=False)
    front = ["experiment", "arm", "reference", "fold"]
    fold_gains = pd.concat(fold_gains, ignore_index=True)
    fold_gains[front + [c for c in fold_gains if c not in front]].to_csv(RESULTS_DIR / "event_fold_gains.csv",
                                                                         index=False)

    path = RESULTS_DIR / "sensitivity_by_duration.csv"
    new = pd.concat(durations, ignore_index=True).assign(sensitivity=lambda d: d.n_detected / d.n_events)
    new = new[["model", "operating_point", "duration_bin", "n_events", "n_detected", "sensitivity"]]
    if path.exists():                              # keep the rows other scripts wrote (E4)
        old = pd.read_csv(path)
        new = pd.concat([old[~old.model.isin(new.model.unique())], new], ignore_index=True)
    new.to_csv(path, index=False)
    plot_durations(new, RESULTS_DIR / "sensitivity_by_duration.png")

    pd.set_option("display.width", 240)
    show = ["experiment", "arm", "reference", "protocol", "fa_target", "analysis", "sensitivity_arm",
            "sensitivity_ref", "d_sensitivity_ci_low", "d_sensitivity_ci_high", "fa_per_h_arm",
            "d_sensitivity_short_matched", "d_sensitivity_short_ci_low_matched", "role", "success", "net_detected",
            "net_detected_2of3_only", "top_patient", "top_patient_net"]
    print(pd.DataFrame(comparisons)[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
