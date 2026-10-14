# How our seizure-detection models work

This is a walkthrough of the machine learning part of our project: what we predict,
what we train on, how we test it, and why we made each choice. The preprocessing,
labeling and feature extraction it builds on are documented in `METHODOLOGY.md` §1–11;
the modeling details are in §12–20. Numbers below come from
`data/processed/feature_table.parquet`, `results/folds.csv` and `results/cv_auroc.csv`.

## 1. The question

Given a 12-second stretch of a newborn's EEG, is it a seizure? We answer this for every
12 s window of a recording, with a new window starting every 6 s. A detector like this
could flag stretches for a clinician to review instead of reading hours of EEG by hand.

Right now we evaluate **window by window**. The question clinicians care about more,
whether we catch each seizure *event* and how many false alarms we raise per hour, is
the next step and is not done yet (see §8).

## 2. The data

- **Recordings.** The Helsinki neonatal EEG dataset (Zenodo record 4940267): 79
  newborns, 0.87–4.28 hours each, 111.9 hours in total.
- **Signals.** Each recording becomes 18 bipolar EEG channels at 32 Hz, cut into
  12 s windows with 50% overlap: **67,025 windows** in all.
- **Labels.** Three experts marked seizures second by second. A second counts as
  seizure when at least 2 of the 3 agree. Each window then gets one label:
  - **1 (seizure):** at least 6 of its 12 seconds are agreed seizure. 8,516 windows.
  - **0 (no seizure):** no expert marked anything in it. 53,789 windows.
  - **−1 (ambiguous):** everything in between, such as seizure onsets and offsets, or
    marks only one expert made. 4,720 windows.
- **Seizure babies.** 46 babies have at least one seizure window; 33 have none.
  (`METHODOLOGY.md` §1 counts 47 babies marked by two or more experts. The extra one,
  eeg8, was marked by two experts but never in the same second, so it has no agreed
  seizure time.)
- **Features.** Each window is described by numbers computed from its EEG, such as
  power in frequency bands, line length, entropy and rhythmicity. They are computed per
  channel, then summarised across the 18 channels by their median, maximum and 75th
  percentile. We compare two feature sets:
  - **Set 1:** 55 features from a published neonatal seizure detector (Temko et al.,
    2011), giving 165 columns.
  - **Set 2:** set 1 plus 3 rhythmicity features (`acf_peak`, `acf_peak_lag`,
    `spectral_peak_prominence`), giving 174 columns. Seizures are often rhythmic, so we
    wanted to test whether measuring rhythm directly helps.

## 3. What we train on, and why

We train only on windows where we trust the label and the signal. That's **60,885
windows**: 8,501 seizure and 52,384 non-seizure, from all 79 babies.

- **Ambiguous windows (−1) are left out of training.** Their true label is uncertain,
  and training on uncertain labels teaches the model noise. We still predict them and
  keep them for evaluation later, because in real use they are the hardest cases.
- **Windows that touch a recording gap are left out.** Part of their signal is missing,
  so their features are incomplete.
- **Unusable windows** (half or more channels technically corrupted) are also excluded.
  There is only one, and it also touches a gap.

## 4. Patient-level splitting

We use 5-fold cross-validation **split by baby, not by window**: all windows of one
baby are in the same fold. In each round we train on 4 folds and test on the fifth.

**Why by baby.** Neighbouring windows overlap by 6 s and come from the same brain, so
they look very alike. If one baby's windows were in both training and test, the model
could recognise *that baby* rather than seizures, and the score would be too optimistic.
Splitting by baby measures what we actually care about: performance on a new patient.
We checked that no baby is in both training and test in any fold.

**How the folds are built.** We split the 46 seizure babies and the 33 non-seizure
babies into 5 groups each, separately, then combine them. Every fold gets 9–10 seizure
babies and 6–7 others.

We first tried scikit-learn's `StratifiedGroupKFold`. It balances the *share of seizure
windows* per fold, but a few babies have hundreds of seizure windows, so one fold ended
up with only 5 seizure babies and another with 15. Because we will later evaluate
whole seizure events, the number of seizure babies per fold matters more to us.

The trade-off is that the share of seizure windows now varies by fold, from 8.8% to
21.9%. The split is saved in `results/folds.csv`, and all models use exactly this file.

## 5. The five models, and why we chose them

| Model | What it does | Why it's here |
|---|---|---|
| **Logistic regression** | A weighted sum of the features, turned into a probability | The simplest sensible baseline. If complex models don't beat it, they aren't worth it. |
| **Random forest** | 300 decision trees on random samples of the data, averaged | Captures non-linear patterns and interactions between features, and copes well with features on very different scales. |
| **k-nearest neighbours** | Looks at the 25 most similar training windows and takes their vote | A different kind of model that makes almost no assumptions. It tells us whether "similar-looking windows" is already enough. |
| **SVM (RBF kernel)** | Finds a flexible boundary between the classes | The original Temko detector, whose features we use, was an SVM, so this is the closest comparison to published work. |
| **ConvLSTM** (neural network) | Reads the last 30 s of EEG features channel by channel | Our only model that uses time and space: seizures build up over seconds and spread between neighbouring electrodes. |

The settings are in `modeling/models.py` and `modeling/run_convlstm.py`. **We fixed
them before seeing results and did not tune them.** Some details:

- **KNN uses 25 neighbours.** With the default of 5, scores can only take 6 different
  values, which makes them crude for ranking windows.
- **Random forest** uses at least 5 windows per leaf, so trees can't memorise single
  windows.
- **The ConvLSTM** looks at each target window together with the 5 windows before it
  (past only, never the future, and never another baby's windows).
  - Each window enters as an 18-channel × feature grid.
  - A small convolution looks across neighbouring channels, and an LSTM carries
    information from one window to the next.
  - It is deliberately small, one layer and about 14,000 parameters, because each
    fold's training data contains only 36–37 seizure babies.
  - Training stops when performance on a held-out group of *training* babies stops
    improving: 12 babies per fold, 7 of them with seizures, which leaves 29–30
    seizure babies to fit the network on. The test fold is never used for that
    decision.

## 6. Class imbalance and missing values

**Imbalance.** Only about 1 in 7 training windows is a seizure (8,501 of 60,885). A
model can look accurate by mostly saying "no seizure", so we make seizure errors count
more:

- Logistic regression, random forest and SVM use `class_weight="balanced"`, which weighs
  each class inversely to its size.
- The ConvLSTM weighs seizure windows in the loss by the ratio of non-seizure to
  seizure windows.
- KNN has no such option in scikit-learn, so it sees the raw imbalance. That's one
  reason we judge models by ranking metrics (§7), which are less affected by it.

**SVM subsampling.** The SVM's training time grows roughly with the square of the data
size: we measured 8 s at 10,000 windows and 35 s at 20,000, which extrapolates to 3–4
minutes per fold. So we train it on all seizure windows plus 20,000 randomly chosen
non-seizure windows, about 25,000–27,000 windows per fold. It still scores every test
window.

**Missing values and scaling.**
- Training windows have no missing feature values.
- Windows that touch a gap do have missing values. We still predict them, so the
  classical models fill gaps with the training data's median and then standardise each
  feature.
- The ConvLSTM sets missing channels to a neutral value and adds a "this channel is
  missing" flag, so the network can tell missing data from quiet EEG. Its scaling is a
  rank-based transform, because some features (like band power) span several orders of
  magnitude.
- **All of these are learned from the training babies only**, in every fold. If the
  test babies' values were used, information about the test set would leak into
  training.

Scores for windows that touch a gap aren't meaningful; anyone using the prediction files
should filter on `eligible`.

## 7. Evaluation metrics and their limits

For each fold we compute two metrics on the test babies' windows that are labelled
0 or 1 and don't touch a gap:

- **AUROC:** the probability that a random seizure window gets a higher score than a
  random non-seizure window. 0.5 is chance and 1.0 is perfect. It doesn't depend on a
  threshold or on how common seizures are.
- **AUPRC** (average precision): focuses on how well seizure windows rise to the top of
  the ranking. Its chance level equals the share of seizure windows, so it is more
  sensitive to how rare seizures are.

We report the mean and standard deviation across the 5 folds.

**Limits we're aware of:**
- **Window-level only.** We don't yet measure whether each seizure *event* is detected,
  or how many false alarms happen per hour.
- **Ambiguous windows are excluded from these scores**, so they leave out the hardest
  cases and are probably optimistic.
- **AUPRC can't be compared between folds**, because the share of seizure windows
  differs from 8.8% to 21.9%. It is fine for comparing models within a fold.
- **No threshold is chosen** and the scores aren't calibrated. The SVM's scores aren't
  probabilities at all.

## 8. Results

Mean ± standard deviation across the 5 folds (`results/cv_auroc.csv`; chart in
`results/auroc_bar_chart.png`):

| Model | AUROC set 1 | AUROC set 2 | AUPRC set 1 | AUPRC set 2 |
|---|---|---|---|---|
| Logistic regression | 0.754 ± 0.065 | 0.761 ± 0.066 | 0.419 ± 0.148 | 0.430 ± 0.153 |
| Random forest | 0.782 ± 0.053 | 0.792 ± 0.055 | 0.492 ± 0.100 | 0.511 ± 0.090 |
| KNN | 0.754 ± 0.038 | 0.760 ± 0.038 | 0.444 ± 0.077 | 0.457 ± 0.069 |
| SVM | 0.792 ± 0.038 | 0.798 ± 0.041 | 0.523 ± 0.087 | 0.530 ± 0.086 |
| ConvLSTM | **0.805 ± 0.063** | **0.813 ± 0.059** | **0.569 ± 0.105** | **0.585 ± 0.096** |

What we take from this:

- **All models are clearly better than chance**, but none is close to perfect at the
  window level.
- **The ConvLSTM has the best average.** It beats the SVM in 7 of 10 fold comparisons
  and the random forest in 9 of 10. It isn't best everywhere: the SVM wins on fold 1
  (both feature sets) and on fold 3 (set 2).
- **Which babies are tested matters more than which model we use.** Fold 0 is the
  easiest and fold 3 the hardest for most models. The spread across folds (0.04–0.07
  AUROC) is larger than most gaps between models.
- **The rhythmicity features help a little.** Set 2 has a higher average than set 1 for
  every model (+0.006 to +0.010 AUROC). That's small compared with the fold-to-fold
  spread, and we haven't tested it for statistical significance.
- **The ConvLSTM overfits quickly.** Its training loss keeps falling, but its loss on
  the held-out training babies ended higher than at the first epoch in 9 of 10 runs.
  Early stopping keeps the best epoch (between epoch 2 and 16), which limits the damage
  (`results/convlstm_curves.png`).

## 9. Limitations and next steps

**Limitations:**
- **Not a like-for-like comparison.** The ConvLSTM sees 30 s of context and every
  channel separately, while the other models see one window's channel summaries. Some
  of its advantage may come from the extra context rather than the network. We haven't
  tested this yet.
- **No tuning.** The settings were not tuned, so every model might do better. Picking
  the best of five models on the same folds is also slightly optimistic.
- **SVM and KNN differ in training.** The SVM is trained on fewer non-seizure windows
  than the others, and KNN is not class-weighted.
- **Small seizure population.** 46 seizure babies in total and 9–10 per test fold.
  Results could shift with different babies.
- **Reproducibility is only partly checked.** Seeds are fixed everywhere. We confirmed
  one ConvLSTM fold reproduces exactly; the classical runs haven't been repeated to
  confirm identical results.

**Next steps:**
1. Event-level evaluation: seizures detected, missed, and false alarms per hour.
2. Decide how ambiguous windows count in evaluation, and report results with them.
3. Give the classical models the previous windows' features too, to separate the effect
   of context from the ConvLSTM architecture.
4. Reduce ConvLSTM overfitting (lower learning rate, more dropout).
5. Tune settings with nested cross-validation, calibrate the scores, and choose a
   detection threshold.
6. Try XGBoost as a further model.

## How to rerun

### Requirements

- **Python 3.11** (results produced with 3.11.9).
- **PyTorch 2.2.2.** You need this to rerun the ConvLSTM. The other four models don't
  use it. 2.2.2 is the newest PyTorch for Intel Macs, and it was built for numpy 1.x.
  `run_convlstm.py` works around that, so **keep numpy 2.x and don't downgrade it**. On
  Apple Silicon or Linux, a newer torch also works.
- About 5 GB of free disk space.

Other package versions used for the committed results: numpy 2.4.6, scipy 1.17.1,
pandas 3.0.6, scikit-learn 1.9.1, mne 1.13.2, PyWavelets 1.9.0, pyarrow 26.0.0,
matplotlib 3.11.2.

### Commands, in order (from the repo root)

```bash
# 1. Environment
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pip install numpy==2.4.6 scipy==1.17.1 pandas==3.0.6 scikit-learn==1.9.1 \
    mne==1.13.2 PyWavelets==1.9.0 pyarrow==26.0.0 matplotlib==3.11.2 torch==2.2.2

# 2. Data from Zenodo (slow: about 3-4 h; rerun the last line to resume after a dropped transfer)
mkdir -p data/raw data/annotations data/clinical
Z="https://zenodo.org/records/4940267/files"
for f in annotations_2017_A_fixed.csv annotations_2017_B.csv annotations_2017_C.csv; do
  curl -L -C - -o "data/annotations/$f" "$Z/$f?download=1"; done
curl -L -C - -o data/clinical/clinical_information.csv "$Z/clinical_information.csv?download=1"
seq 1 79 | xargs -P 4 -I{} curl -sL --retry 5 -C - -o data/raw/eeg{}.edf "$Z/eeg{}.edf?download=1"

# 3. Tell the code where the data is (in every new terminal) and check it
export HELSINKI_DATA_ROOT="$PWD/data"
.venv/bin/python scripts/check_setup.py

# 4. Feature table plus per-channel features for the ConvLSTM (about 20 min)
.venv/bin/python scripts/build_feature_table.py --channel-features

# 5. Models (folds come from the committed results/folds.csv; don't rebuild it)
.venv/bin/python -m modeling.run_experiments    # LogReg, RF, KNN, SVM: about 22 min
.venv/bin/python -m modeling.run_convlstm       # ConvLSTM: about 18 min, needs torch 2.2.2

# 6. Chart and summary table
.venv/bin/python -m modeling.plot_results
```

Run times are from one Intel Core i5-1038NG7 laptop, CPU only. On a Mac, put
`caffeinate -i` before long commands and keep the lid open, because sleep pauses the run.
