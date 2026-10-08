#!/usr/bin/env python3
"""Trim point-count on the raw curves already saved in auprc_curves.json
(computed by the same run_sph_sop_exp/get_sph_sop_exp pipeline used for the
0.8-ish AUROC headline numbers). No value changes, just fewer of the same
real points so the chart HTML isn't bloated."""
import json
import numpy as np

d = json.load(open("results_sph_sop/auprc_curves.json"))

MAX_POINTS = 250

out = {}
for task in d:
    out[task] = {}
    for model in d[task]:
        c = d[task][model]
        precision = np.array(c["precision"])
        recall = np.array(c["recall"])
        if len(precision) > MAX_POINTS:
            idx = np.linspace(0, len(precision) - 1, MAX_POINTS).astype(int)
            precision = precision[idx]
            recall = recall[idx]
        out[task][model] = dict(
            precision=[round(float(x), 5) for x in precision],
            recall=[round(float(x), 5) for x in recall],
            auprc=round(c["auprc"], 4),
        )

with open("results_sph_sop/auprc_curves_downsampled.json", "w") as f:
    json.dump(out, f)

for t in out:
    for m in out[t]:
        print(t, m, "n_points=", len(out[t][m]["precision"]), "auprc=", out[t][m]["auprc"])
