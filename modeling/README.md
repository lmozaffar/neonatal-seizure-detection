# Modeling: how to rerun everything

Five models (LogReg, RF, KNN, SVC, ConvLSTM) on two feature sets, evaluated with the
same patient-level folds. Results are already committed in `results/`; rerun only if
the features, folds or models change. Commands are run from the repo root.

## 1. Environment

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]" scikit-learn xgboost matplotlib pyarrow
.venv/bin/python -m pip install torch==2.2.2     # only needed for the ConvLSTM
```

PyTorch 2.2.2 is the newest version available for Intel Macs, and it was built for
numpy 1.x while this project uses numpy 2. `run_convlstm.py` works around this by
passing arrays through raw bytes, so do not downgrade numpy. On Apple Silicon or Linux
a newer torch also works.

## 2. Data

Download the Helsinki dataset (Zenodo record 4940267) into:

```
data/raw/eeg1.edf ... eeg79.edf
data/annotations/annotations_2017_A_fixed.csv, annotations_2017_B.csv, annotations_2017_C.csv
data/clinical/clinical_information.csv
```

`helsinki/paths.py` looks for the data one folder above the repo by default. If yours
is in the repo's `data/` folder, set `export HELSINKI_DATA_ROOT="$PWD/data"` first.
`data/` is git-ignored.

## 3. Feature table (~20 min)

```bash
.venv/bin/python scripts/build_feature_table.py --channel-features
```

Runs Leila's `helsinki/` pipeline on all 79 recordings and writes to `data/processed/`:

- `feature_table.parquet`: one row per 12 s window (67,025 windows) with labels and summary features.
- `channel_features.npy`, `channel_valid.npy`, `channel_index.parquet`: per-channel features
  (windows x 18 channels x 58 features, ~270 MB). These are only needed for the ConvLSTM, and
  only written with `--channel-features`. Make sure about 1 GB of disk space is free.

## 4. Folds (do not rebuild)

`results/folds.csv` is the shared patient-level split. Everyone should use it as is, via
`modeling.common.load_folds()` / `add_folds()`. `python -m modeling.common` rebuilds it
deterministically, but only do that if the team agrees to change the split.

## 5. Models

```bash
.venv/bin/python -m modeling.run_experiments    # LogReg, RF, KNN, SVC (~22 min; SVC is most of it)
.venv/bin/python -m modeling.run_convlstm       # ConvLSTM (~18 min on CPU; needs step 3's channel files)
```

Both write the same format to `results/`: rows in `cv_auroc.csv` (AUROC/AUPRC per
feature set, model and fold, on held-out eligible windows with label 0/1; rerunning a
model replaces only its rows) and `predictions_<model>_set<N>.parquet` (a score for
every window of the held-out patients). The ConvLSTM also writes
`convlstm_history.csv` and `convlstm_curves.png`.

For a quick check, both scripts accept `--folds 0 --out-dir results/test` (and
`run_convlstm` also `--max-epochs 5`). Delete `results/test/` afterwards.

Leave the Mac awake and the lid open during long runs (e.g. prefix commands with
`caffeinate -i`); sleep pauses the run.
