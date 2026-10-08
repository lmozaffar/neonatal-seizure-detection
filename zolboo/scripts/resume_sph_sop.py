#!/usr/bin/env python3
"""Resume the SPH/SOP grid analysis from the already-saved continuous prediction CSVs,
skipping the ~6-minute prediction-rebuild phase."""

import sys
import time
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, "notebooks")
from notebook_utils import run_sph_sop_exp, get_sph_sop_exp  # noqa: E402

MODEL_NAMES = ["LogReg", "RF", "KNN", "SVC", "ConvLSTM"]

SPHs = range(1, 8)
SOPs = range(1, 8)
GLUE_TIME = 3
WINS_PER_MIN = 3
threshes = np.arange(0.0, 1.01, 0.01)


def run():
    summary_rows = []
    for name in MODEL_NAMES:
        df_model = pd.read_csv(f"results_sph_sop/{name}_continuous_preds.csv")
        df_model['feat'] = 'all'

        print(f"\n######## Running SPH/SOP grid for {name} ########", flush=True)
        t0 = time.time()
        SPH_results_dict, _, _, _ = run_sph_sop_exp(
            df_model, name, SPHs, SOPs, GLUE_TIME, WINS_PER_MIN, mp_mode=0, threshes=threshes
        )
        (_, auroc_scores, _, _, auprc_scores, *_rest) = get_sph_sop_exp(SPH_results_dict, plot_on=False, model_num=0)

        vals = np.array(list(auroc_scores.values()))
        headline = auroc_scores.get("SPH3_SOP7", np.nan)
        summary_rows.append(dict(
            model=name, headline_SPH3_SOP7=headline,
            mean_auroc=vals.mean(), max_auroc=vals.max(),
            elapsed_s=time.time() - t0,
        ))
        print(f"{name}: headline SPH3/SOP7 AUROC={headline:.4f}, mean={vals.mean():.4f}, max={vals.max():.4f} ({time.time()-t0:.1f}s)", flush=True)

        with open(f"results_sph_sop/{name}_auroc_grid.pkl", "wb") as f:
            pickle.dump(auroc_scores, f)

        pd.DataFrame(summary_rows).to_csv("results_sph_sop/sph_sop_summary.csv", index=False)

    summary = pd.DataFrame(summary_rows).sort_values("headline_SPH3_SOP7", ascending=False)
    print("\n=== FINAL SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
