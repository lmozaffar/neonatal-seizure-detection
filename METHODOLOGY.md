# Methodology

Living record of methodological decisions for the neonatal EEG seizure-detection
pipeline on the Helsinki dataset. Update this file whenever a meaningful decision
is made; do not create separate documentation files.

**Status legend.** *Implemented* = in code and tested. *Planned* = agreed but not
yet built. *Final* = not expected to change. *Provisional* = to be revisited,
usually once all 79 recordings have been checked.

**Validation scope so far.** Everything marked "validated" was checked on two
recordings only: **eeg1** (seizures annotated by all three experts) and
**eeg57** (no seizures annotated). All other recordings are untested.

Code: `helsinki/paths.py`, `helsinki/io.py`, `helsinki/preprocessing.py`,
`helsinki/quality.py`, `helsinki/labeling.py`, `helsinki/features.py`; tests in
`tests/`.

---

## 1. Dataset

*Implemented (loading and paths). Facts below verified as stated.*

- **Recordings.** 79 EDF files `data/raw/eeg<ID>.edf`, IDs 1–79. The
  `ID` / `EEG file` columns of `data/clinical/clinical_information.csv` map one-to-one
  to these files.
- **Channels** (headers read for eeg1 and eeg57 only): 21 channels, 256 Hz,
  1 s EDF records. 19 EEG electrodes of the 10–20 system (Fp1, Fp2, F3, F4, F7,
  F8, Fz, C3, C4, Cz, T3, T4, T5, T6, P3, P4, Pz, O1, O2), one ECG and one
  respiration channel.
  - **Referential, not bipolar.** Electrodes are recorded against a common reference.
  - **Units:** µV in the header, physical range ±4999.84 µV, quantisation step 0.15 µV.
  - **No hardware filter on the EEG.** In eeg57, ECG and respiration carry their own
    filters (1–70 Hz and 0.5–20 Hz); in eeg1 nothing is filtered.
- **Naming and order differ between files**, for example `EEG Fp1-REF` (eeg1) and
  `EEG Fp1-Ref` (eeg57), with different channel orders. Channels are therefore
  always selected by normalised name, never by position.
- **Near-zero electrode.** One electrode per recording (Cz in eeg1, Fz in eeg57)
  is close to zero throughout (std about 0.3 µV), consistent with it being the
  recording reference. This is an interpretation, not confirmed from dataset
  documentation. Bipolar derivations remain valid either way.
- **Annotations.** Three experts (A, B, C) annotated independently, one value per
  second:
  - **Files:** `data/annotations/annotations_2017_{A_fixed,B,C}.csv` (we use
    `A_fixed`). `annotations_2017.mat` holds identical data (verified for all 79
    subjects and all three experts).
  - **Layout:** each CSV is a 15416 × 79 matrix. Columns are subject IDs; row *i*
    covers seconds [i, i+1) from the start of the recording; 1 = seizure, 0 = none.
  - **NaN** marks the time after a recording ends. It is identical across experts
    and means "not annotated", which is different from 0.
  - **Alignment:** the annotated length matches the EDF duration (verified for eeg1
    and eeg57).
  - **Agreement between experts:**

    | Subjects marked as having seizures by… | Count |
    |---|---|
    | No expert | 22 |
    | 1 expert | 10 |
    | 2 experts | 7 |
    | All 3 experts | 40 |

    These counts match the clinical table's "Number of Reviewers Annotating
    Seizure" column for all 79 subjects.
  - **No official consensus label.** We build one by 2-of-3 majority per second (§7).
- **Total duration:** 0.87–4.28 h per recording, 111.9 h in all, gaps included.

## 2. Channel selection and bipolar montage

*Implemented. Final.*

- **Electrodes kept:** the 19 EEG electrodes, matched by name after removing the
  `EEG ` prefix and the `-REF`/`-Ref` suffix, ignoring case. ECG and respiration
  are dropped. A missing or duplicated electrode raises an error.
- **Montage:** the 18-channel longitudinal bipolar ("double banana") montage, in a
  fixed order on every recording.
  - Right parasagittal: Fp2-F4, F4-C4, C4-P4, P4-O2
  - Left parasagittal: Fp1-F3, F3-C3, C3-P3, P3-O1
  - Right temporal: Fp2-F8, F8-T4, T4-T6, T6-O2
  - Left temporal: Fp1-F7, F7-T3, T3-T5, T5-O1
  - Midline: Fz-Cz, Cz-Pz
- **Why bipolar:** derivations between neighbouring electrodes are independent of
  the recording reference, and this is the standard montage for clinical review.
- **Alternative considered:** a 16-channel montage without the midline pair. We
  rejected it to keep the midline coverage.

## 3. Filtering and downsampling

*Implemented. Final unless validation on all recordings shows otherwise.*

- **Band-pass, 0.5–12.8 Hz.** Butterworth, order 4, applied forward and backward
  (zero-phase, `scipy.signal.sosfiltfilt`).
  - Running the filter both ways squares the magnitude response: each cutoff ends
    up at −6 dB and the roll-off is twice as steep.
  - The signal is filtered after the bipolar derivation. Both steps are linear, so
    the order makes no difference to the result.
- **Edge padding: mirror (even) reflection, 3 s per side.**
  - scipy's default is point (odd) reflection over about 0.1 s. Odd reflection
    around a value near a peak leaves a large offset in the pad, and the high-pass
    then rings from it.
  - On rhythmic test signals, even padding gave 5–7× less edge error than odd.
  - On real EEG it was better than the default beyond 0.25 s from an edge.
  - Constant padding was slightly better on typical real EEG but 2–4× worse on
    rhythmic activity. Rhythmic activity matters most for seizures, so we chose even.
- **Downsampling, 256 → 32 Hz.** `scipy.signal.resample_poly` (factor 8), whose
  linear-phase FIR low-pass at 16 Hz acts as the anti-aliasing filter.
  - The band-pass alone is not enough: it is only about 18 dB down at 16 Hz.
  - Combined, the two filters attenuate frequencies that would alias into
    0–12.8 Hz (19.2–128 Hz) by at least 86 dB (measured).
- **Units:** output is in µV. No normalisation or z-scoring at any stage so far.

## 4. Recording gaps

*Implemented. Detection rule: final. Short-stretch reliability threshold: provisional.*

- **What a gap looks like.** In both checked recordings there are stretches where
  all 19 electrodes hold exactly the same constant value (0.0035 µV, apparently
  the EDF's digital zero).
  - eeg1: 8 gaps, 919.9 s (13.2% of the recording).
  - eeg57: 3 gaps, 17.6 s (0.6%).
  - In eeg57 the ECG and respiration channels keep recording during the gaps, so
    these are pauses of the EEG amplifier specifically.
  - Annotators A and C marked no seizure seconds inside gaps. Annotator B marked
    7 s, all inside eeg1's final gap.
- **Detection rule.** A gap is any stretch of **at least 16 samples (62.5 ms)** where
  **all 19 raw electrodes are unchanged**, checked on the 256 Hz signal before any
  filtering.
- **Why this is not genuine low-amplitude EEG.** Outside gaps, all 19 electrodes
  never stayed unchanged for even 2 consecutive samples. The quietest real second
  in eeg1 (median electrode std 1.1 µV) still had 30–50 distinct values per
  electrode. Tracé discontinu, tracé alternant and burst suppression are
  low-amplitude but not digitally constant, so the rule does not touch them. A
  single near-zero reference electrode does not trigger it either.
- **Representation.** Gaps are NaN in the preprocessed signal, rounded outward to
  whole 32 Hz samples (at most 31 ms per edge). The original timeline is kept, so
  the annotation alignment does not change.
- **Each valid stretch is filtered and downsampled on its own.**
  - Filtering across a gap turns its edges into ringing of up to 864 µV (255× the
    typical signal level), reaching about 5 s into valid EEG. The cause is the step
    from the electrode's normal DC offset down to the constant fill value.
  - Per-stretch processing removes this. Beyond 10 s from any edge, the output is
    identical to whole-recording processing (difference ≤ 0.003 µV).
  - Each stretch is trimmed inward to whole 32 Hz samples, so it lands exactly on
    the shared timeline.
- **Short stretches** are always processed and kept, never treated as missing.
  - A stretch shorter than **4 s** is flagged `reliable = False`, because at that
    length it is almost entirely filter edge.
  - A stretch is only left unprocessed if it covers no whole 32 Hz sample (31 ms).
  - Examples: eeg1 6991.0–6992.5 s (1.5 s) and eeg57 3129.0–3132.9 s (4.0 s).
- **Remaining edge effect.** Measured on 20 artificial gaps inserted into
  continuous EEG from eeg57; error relative to the channel's typical signal level:

  | Distance from gap edge | 0–0.25 s | 0.25–1 s | 1–2 s | 2–5 s | > 5 s |
  |---|---|---|---|---|---|
  | Typical | 0.89× | 0.41× | 0.17× | 0.05× | 0.002× |
  | Worst | 4.6× | 2.6× | 0.97× | 0.31× | 0.009× |

  This is inherent: nothing beyond the edge is known. The same effect exists at
  every recording's start and end. It is handled by the `filter_edge` flag (§6).
- **Real artifacts near gaps.** Some gap edges carry genuine high-amplitude
  artifacts in the raw signal (up to 2248 µV peak-to-peak after eeg1's 688–695 s
  gap). The processed output never exceeds the raw signal, so these are not
  created by filtering.

## 5. Windowing and eligibility

*Implemented. Final (window length and stride), provisional (eligibility rule).*

- **Windows:** 12 s long with a 6 s stride (50% overlap). Each window is
  (18, 384) at 32 Hz.
- **Timing:** window *i* covers [6i, 6i + 12) s, so it spans exactly annotation rows
  6i to 6i + 11. A trailing incomplete window is dropped (losing at most 11.97 s);
  windows are never padded.
- **Storage:** windows are a read-only NumPy view of the preprocessed signal (no
  copy).
- **Metadata per window:** `subject_id`, `window_index`, `start_s`, `end_s`,
  `gap_frac` (fraction of samples in a gap) and **`eligible` = (`gap_frac` == 0)**.
  - Windows touching a gap stay in the array but are not eligible for training.
  - Excluding even a partial overlap is conservative and costs about 0.5% of valid
    EEG.
- **Alternative considered:** restarting the window grid at each valid stretch. It
  gave the same window counts but broke the exact alignment with whole annotation
  seconds, so we rejected it.

| | eeg1 | eeg57 |
|---|---|---|
| Windows | 1164 | 521 |
| Fully inside a gap | 144 | 0 |
| Partly in a gap | 20 | 5 |
| Eligible | 1000 (85.9%) | 516 (99.0%) |
| Share of valid EEG covered by eligible windows | 99.5% | 99.8% |

## 6. Artifact assessment

*Implemented. All thresholds provisional.*

**Principle:** flag suspicious channels and windows; reject only clear technical
corruption. Unusual but real neonatal EEG, such as suppression, discontinuity,
high-amplitude seizures or movement, is kept and flagged, so the classifier can
learn to tell artifacts from seizures.

- **Gaps are not re-detected here.** Gap samples are left out of every metric: no
  flatlines inside gaps, no jumps computed across a gap.
- **Where things are measured.** Flatlines, jumps and clipping use the raw 256 Hz
  signal; filtering would hide them. Amplitude uses the preprocessed 32 Hz windows,
  which is what the classifier sees.

### Thresholds

The calibration numbers come from eeg1 (173 eligible windows containing at least
one second all three experts marked as seizure) and eeg57.

| Check | Metric | Flag | Counts as bad? | Why this threshold |
|---|---|---|---|---|
| Low amplitude | std of preprocessed channel | `low_amp`: < 0.5 µV | **Never**, descriptive only | Neonatal suppression is real EEG. Std in seizure windows went as low as 0.72 µV. |
| Technical flatline | `flat_frac`: fraction of the window's valid samples inside runs where the raw bipolar channel is unchanged for ≥ 1 s | `flatline`: > 0 | **Yes, if ≥ 0.5** | In real bipolar EEG, including the quietest 1% of channel-windows, the longest unchanged run was 7 samples (27 ms). 1 s is 36× longer. |
| Clipping | `clip_frac`: fraction of samples where either electrode is within one quantisation step of that file's EDF limits | `clipped`: > 0 | **Yes, if ≥ 0.05** (≥ 0.6 s) | Limits are read from each file's header, not assumed. |
| Electrode pop / jump | `max_jump_uv`: largest change between consecutive raw bipolar samples | `pop`: > 500 µV | Never | About 10× the largest jump inside annotated seizures (53 µV). |
| High amplitude | peak-to-peak of preprocessed channel | `high_amp`: > 500 µV | Never | Largest inside annotated seizures: 296 µV. |
| Extreme amplitude | same | `extreme`: > 1000 µV | Never | Likely movement or other non-cerebral activity, but still interpretable. |
| Filter edge (per window) | window overlaps the first or last 2 s of a valid stretch, including recording start and end | `filter_edge` | Never | Beyond 2 s the edge error is about 5% of signal level or less (§4). |

### Details

- **Flatlines are found on the whole recording,** so a run that crosses a window
  boundary counts in every window it touches.
- **Attribution to electrodes.** Bipolar channels share electrodes, so flatline,
  clipping and pop are also reported per electrode (`electrode_qc`).
  - A flat derivation is attributed to whichever of its electrodes is itself
    constant. If neither is (bridged, identical electrodes), both are blamed.
  - Clipping is measured directly on each electrode.
  - Pops are attributed to any electrode whose own raw jump exceeds 500 µV.
- **Unusable window:** **at least 9 of 18 channels bad** (provisional). Bad means
  only substantial flatline or clipping. `unusable` is kept separate from the
  gap-based `eligible`; how they combine for training is not decided yet.
- **Stored for feature extraction:**
  - `channel_qc`: (n_windows, 18) metrics `std_uv`, `ptp_uv`, `max_jump_uv`,
    `clip_frac`, `flat_frac`, and flags `low_amp`, `flatline`, `clipped`, `pop`,
    `high_amp`, `extreme`, `bad`.
  - `electrode_qc`: (n_windows, 19) metrics `clip_frac`, `flat_frac`,
    `max_jump_uv`, and flags `clipped`, `flatline`, `pop`.
  - Window columns added to `meta`: `filter_edge`, `n_flagged_channels`,
    `n_bad_channels`, `unusable`.

### Validation (eligible windows only)

| | eeg1 (1000) | eeg57 (516) |
|---|---|---|
| Unusable | 0 | 0 |
| Bad channels | 0 | 0 |
| Flatline | 0 | 0 |
| `low_amp` (channel-windows / windows) | 2 / 2 | 0 / 0 |
| Clipped (channel-windows / windows) | 4 / 2 (P3, ≤ 0.07% of samples) | 0 / 0 |
| Pop (channel-windows / windows) | 18 / 9 | 49 / 19 (mostly T4, T3) |
| High amplitude (channel-windows / windows) | 38 / 15 | 38 / 18 |
| Extreme (channel-windows / windows) | 21 / 11 | 0 / 0 |
| `filter_edge` (windows) | 7 | 3 |

**None of these flags fell in any of the 173 eligible eeg1 windows containing
seizure seconds agreed by all three experts.**

## 7. Seizure labels

*Implemented. Provisional: to be re-evaluated on all 79 recordings and against
model performance.*

### Consensus per second
- A second is **consensus seizure** when **at least 2 of the 3 experts** mark it.
- A second counts as **annotated** only if all three experts have a value for it.
  Unannotated seconds (NaN, after a recording ends) never count as seizure or as
  non-seizure.

### Training label: conservative (Strategy 1)
Stored as `label`, one per 12 s window:

| Label | Rule |
|---|---|
| **1, positive** | ≥ 6 of the window's 12 s are consensus seizure |
| **0, negative** | all 12 s annotated, and **no expert** marks any of them as seizure |
| **−1, ambiguous** | everything else: 1–5 consensus seconds, seizure marks without consensus, missing annotation |

A window with missing seconds can still be positive if 6 or more of its annotated
seconds are consensus seizure. It can never be negative.

### Why conservative
- **Experts disagree a lot.** In eeg1, 343 windows contain seizure marks without
  consensus; 280 of them are marked by expert B alone. B marks 3141 s, against
  1602 (A) and 997 (C). Labeling these windows negative would add uncertain
  negatives.
- **Onset/offset windows are ambiguous.** A window with 1–5 consensus seconds is
  neither clearly seizure nor clearly background. The other strategies force it
  into one class or the other.
- **Clean classes for training.** Every positive is mostly seizure, and every
  negative is free of seizure marks by all three experts.

**Cost:** in eeg1, **388 of 1000 eligible windows (39%) are ambiguous** and leave
the training set. They include the hardest cases: onsets, offsets and disputed
activity. They are kept in the data and must stay in evaluation (§8).

### Alternatives considered
Both are stored for comparison; −1 means the window is not fully annotated.
- **Any-overlap** (`label_any_overlap`): positive if ≥ 1 consensus second,
  otherwise negative.
- **50% binary** (`label_binary50`): positive if ≥ 6 consensus seconds, otherwise
  negative.

### Comparison on eeg1, eligible and usable windows (1000)

| | Conservative | Any-overlap | 50% binary |
|---|---|---|---|
| Positive | 262 | 309 | 262 |
| Negative | 350 | 691 | 738 |
| Ambiguous / excluded | 388 | 0 | 0 |
| Windows with consensus seizure labeled negative | 0 | 0 | 47 |
| Negatives containing any expert's seizure marks | 0 | 341 | 341 |
| Positives with fewer than 6 s of seizure | 0 | 47 | 0 |
| Consensus events with ≥ 1 positive window | 29/29 | 29/29 | 29/29 |

- **eeg57** has no seizure marks: all 516 eligible windows are negative under every
  strategy.
- **eeg1 events:** 29 consensus events, lasting 7–228 s. The shortest break between
  two events is 8 s, so no merging is needed for this recording.
- **Short events:** because windows start every 6 s, any event of 6 s or more fully
  covers 6 s of at least one window. Conservative and 50% binary labeling can only
  miss events shorter than 6 s; eeg1 has none.

### What is stored per window (`label_windows`)
- **Annotation counts:** `annotated_s`, `fully_annotated`, `seizure_s_A/B/C`,
  `any_expert_s`, `consensus_s`, `unanimous_s`, `consensus_frac`,
  `n_experts_marking`.
- **Labels for all three strategies.**
- **Rebuilding labels:** `strategy_labels` recomputes every label from
  `consensus_s`, `any_expert_s` and `fully_annotated` (verified identical on eeg1
  and eeg57).
- **Vote distribution:** seconds with 1, 2 and 3 votes are `any_expert_s −
  consensus_s`, `consensus_s − unanimous_s` and `unanimous_s`.

### Independence from eligibility
- **No window is removed** by labeling. Gap eligibility (§5) and artifact
  usability (§6) are separate columns.
- **Gap windows look negative by annotation.** Experts mark gaps as 0, so the 144
  eeg1 windows entirely inside a gap are labeled negative. That label is
  meaningless; they are excluded only through `eligible`. **Labels must always be
  combined with `eligible`.**
- **Expert B's 7 s inside eeg1's final gap** have no consensus. Those windows are
  ineligible anyway.

## 8. Evaluation approach

*Planned, not implemented.*

- **Training** uses conservative labels on windows that are eligible and not
  unusable.
- **Held-out evaluation keeps the ambiguous windows.** Dropping them would remove
  the hardest cases and inflate performance.
- **Primary metric: event-level detection** against the per-second consensus
  annotations. Did the detector find each consensus seizure event, and how many
  false detections per hour did it make?
- **Secondary: window-level binary metrics**, for example using `label_binary50` on
  all eligible windows.
- **Not yet decided:** event-matching rules, smoothing or post-processing of
  detections, and how windows flagged as artifacts enter evaluation.

## 9. Features: adapted Temko baseline and rhythmicity extension

*Implemented. Provisional. This is an **adapted Temko baseline**, not an exact
reproduction.*

### Sources
- **Paper:** Temko A, Thomas E, Marnane W, Lightbody G, Boylan G. *EEG-based
  neonatal seizure detection with Support Vector Machines.* Clin Neurophysiol
  2011;122:464–473 (PMC3036797), Table 2.
  - It lists the 55 features by name and gives a few parameters: 8 s epochs with
    50% overlap; 32 Hz with an anti-aliasing filter at 12.8 Hz; a 256-point FFT;
    0–12 Hz power in 2 Hz bands; and the wavelet (Daubechies-4, "fifth of eight
    coefficients", 1–2 Hz).
  - For exact formulae it refers to another publication, which we have not
    reviewed.
- **Related paper:** Mathieson et al. 2016 (Clin Neurophysiol 127:2246–2256,
  PMC4840013) describes the same algorithm by feature category only.
- **Public MATLAB implementation:** github.com/ktapani/Neonatal_Seizure_Detection
  (MIT licence; Tapani and Stevenson, 2018), detector `SDA_T`.
  - Files: `ch_temko.m`, `feats_temko.m`, `feature_scripts/*.m`.
  - Its `feature_list.mat` names the same 55 features, matching the paper's Table 2
    one-to-one.
  - It is a public implementation. We do not assume it is identical to Temko's
    original code.
- **How definitions were chosen:** the paper where it is explicit, otherwise the
  MATLAB code. MATLAB built-ins (`bandpower`, `periodogram`, `pyulear`, `pwelch`,
  `findpeaks`, `kurtosis`, `var`) are reproduced from their documented defaults.
  **Numerical agreement with MATLAB has not been verified**, because MATLAB was not
  available.

### Pipeline context: MATLAB `SDA_T` vs ours

| | MATLAB `SDA_T` | Ours |
|---|---|---|
| Montage | 18-channel double banana, same order | same (§2) |
| Units | µV | µV |
| Filtering | causal 50 Hz IIR notch at 256 Hz, then MATLAB `resample` to 32 Hz | zero-phase 0.5–12.8 Hz band-pass (§3) |
| Per-epoch high-pass | first-order 0.5 Hz, zero initial state | **not applied** (our band-pass already removes < 0.5 Hz, and per-window filtering adds a start-up transient) |
| Epochs | 8 s (256 samples), 4 s overlap | **12 s (384 samples), 6 s stride** (§5) |
| Spectrum length | periodogram, 256 points | **FFT length = window length (384)**, 1/12 Hz bins; 0.5, 12 and every band edge fall exactly on bins |
| Missing values | all features NaN if mean \|x\| < 10⁻⁴ µV in either half | same rule, plus NaN for gap samples and for any undefined operation |

### The 55 baseline features (per channel and window)
x is the 12 s preprocessed channel in µV; Δ and ΔΔ are the first and second
differences. **Source:** P = paper, M = MATLAB code, C = corrected from MATLAB,
O = our choice.

| Features | Definition | Unit | Source |
|---|---|---|---|
| `line_length` | Σ\|Δx\| over the **whole window** | µV | P ("curve length"), C |
| `wavelet_energy` | Σ cD4²: db4 detail coefficients cD4 (1–2 Hz at 32 Hz), symmetric extension; the 5th of 8 coefficient sets in a level-7 decomposition | µV² | P (wavelet, band), O (energy) |
| `min_max` | number of local maxima plus minima; a flat peak counts once | count | M |
| `rms_amp` | root mean square | µV | M |
| `shannon_entropy` | Moddemeijer unbiased histogram estimator: ⌈√N⌉ cells over [min − d/2, max + d/2], d = range/(N−1), natural log | nats | M |
| `peak_freq` | frequency of the periodogram maximum, **in Hz**, lowest if tied | Hz | P, C |
| `hjorth_activity`, `hjorth_mobility`, `hjorth_complexity` | population variance; std(Δx)/std(x); [std(ΔΔx)/std(Δx)] / mobility | µV², –, – | M |
| `nonlinear_energy` | mean(x[n]² − x[n−1]·x[n+1]) | µV² | M |
| `spectral_entropy` | log₂ Shannon entropy of the periodogram normalised to sum 1, over all bins 0–16 Hz | bits | M |
| `zero_crossings` | number of n with x[n]·x[n+1] < 0 | count | M |
| `ar_error_1` … `ar_error_9` | 100 · mean\|P_AR − P_W\| / max(mean P_AR, mean P_W). P_AR: Yule-Walker AR(p) PSD of the **first half** (biased autocorrelation, Levinson recursion, 256-point). P_W: Welch PSD of the **second half** (MATLAB defaults: 8 segments of ⌊N/4.5⌋ samples, 50% overlap, symmetric Hamming, 256-point). An AR PSD that is NaN is replaced by zeros (as in MATLAB) | % | P (orders 1–9), M |
| `total_power` | `bandpower` 0–12 Hz: Hamming-window periodogram, rectangle sum from the last bin ≤ the lower edge to the first bin ≥ the upper edge | µV² | P, M |
| `band_0_2` … `band_10_12` | the same, in 2 Hz bands stepped by 1 Hz | µV² | P, M |
| `band_*_norm` | band / `total_power` | – | P, M |
| `sef90`, `sef95`, `sef80` | first periodogram bin in 0.5–12 Hz where cumulative power reaches 90 / 95 / 80% | Hz | P, C |
| `kurtosis`, `skewness` | biased; kurtosis not excess (normal = 3) | – | M |
| `svd_entropy`, `fisher_info` | delay embedding with dimension 20, lag 1, N − 20 vectors; normalised singular values s: −Σ s log₂ s and Σ (sᵢ₊₁ − sᵢ)² / sᵢ | bits, – | M |
| `zero_crossings_d1`, `zero_crossings_d2` | zero crossings of Δx and ΔΔx | count | M |
| `variance_d1`, `variance_d2` | variance (N − 1) of Δx and ΔΔx | (µV/sample)² | M |

### Corrections to the public MATLAB code
We describe these as corrections relative to the paper's definitions, not as
proven errors in Temko's original algorithm.
1. **Line length:** MATLAB's `nonlin.m` sums \|Δx\| only over the first `fs`
   samples (1 s). We use the whole window, matching "curve length".
2. **Peak frequency:** MATLAB returns the periodogram bin index (1 + 8·f for an 8 s
   epoch). We return Hz. This is a monotone relabelling.
3. **Spectral edge:** MATLAB's `spectral_edge.m` returns `freq(i1+i)`, one bin above
   the bin where the cumulative power is reached. We return that bin itself.
4. **Wavelet:** MATLAB uses db8 and the mean absolute value of coefficients 9–16 of
   the `wavedec` output, selected by dyadic index arithmetic (`sub_pos.m`). With
   db8 under MATLAB's default symmetric extension, a 256-sample epoch gives
   coefficient lengths 16, 16, 18, 22, 30, 45, 75, 135 (confirmed with
   PyWavelets). Those indices therefore fall inside the level-7 approximation
   (0–0.125 Hz), not the paper's 1–2 Hz band. We follow the paper.

### Wavelet verification
- **Indexing:** `wavedec` returns 8 coefficient sets for 7 levels, ordered cA7,
  cD7, cD6, cD5, **cD4**, …. The paper's "fifth" set is therefore cD4. cD4 from a
  7-level decomposition is identical to cD4 from a 4-level one (verified), so the
  "level too high" warning for 7 levels on 384 samples (PyWavelets maximum: 5)
  does not affect it.
- **Band:** the energy gain of cD4 for unit sines at 32 Hz is 0.50 at 1.0 Hz, 0.88
  at 1.5 Hz and 0.53 at 2.0 Hz, falling below 0.02 at 0.5 Hz and at 3 Hz and above.
  The passband is about 1–2 Hz, as the paper states.
- **Boundary effects:** cD4 has 30 coefficients on a 384-sample window. **12 of
  them, 6 at each end, depend on how the edge is extended**, and total cD4 energy
  varied by up to about 40% across extension modes on noise. We use symmetric
  extension, MATLAB's default, over all 30 coefficients. *Provisional.* The
  alternatives are interior coefficients only, or computing cD4 on the continuous
  recording.
- **Energy definition:** the paper says "energy" without a formula. **Σ c² is our
  choice.** It scales with amplitude squared (tested).
- **On real data:** `wavelet_energy` correlates with `band_1_3` (Spearman 0.86 for
  eeg1, 0.80 for eeg57) more than with `band_6_8` (0.76 and 0.41).

### Effect of 12 s windows (vs Temko's 8 s)
- **Grow with duration (×1.5 for stationary signals):** `line_length`, `min_max`,
  zero crossings, and `wavelet_energy` (more coefficients). These values are **not
  numerically comparable** to published 8 s values.
- **Independent of duration:** amplitude, power, Hjorth, kurtosis/skewness and
  normalised bands. Spectral estimates get finer resolution (1/12 Hz instead of
  1/8 Hz).
- **AR error:** each half is 6 s (192 samples) instead of 4 s. The Welch segments
  are 42 samples instead of 28, still 256-point FFT.
- **SVD:** 364 embedding vectors instead of 236.
- **Temporal resolution:** a longer window smooths short events. Seizure events
  last at least 7 s in eeg1, and the labeling rule (≥ 6 consensus seconds) already
  sets a minimum.

### Secondary set: rhythmicity (3 features, new)
- `acf_peak`: the mean-removed autocorrelation is normalised so r(0) = 1. Take the
  highest local maximum of r(k) between r's **first zero crossing** and **3 s**
  (96 samples). **NaN** if r doesn't cross zero, or has no local maximum, within
  3 s. A rhythmic discharge repeats and gives a strong second autocorrelation peak;
  background activity decays. Range −1 to 1, independent of amplitude.
- `acf_peak_lag`: the lag of that peak, in s, which is the dominant period. 3 s
  covers rhythms down to about 0.33 Hz.
- `spectral_peak_prominence`: Welch PSD with 4 s periodic Hann segments, 50%
  overlap and each segment's mean removed (0.25 Hz resolution). Take the
  topographic prominence (`scipy.signal.peak_prominences`) of the highest PSD peak
  in 0.5–12 Hz, divided by the mean PSD in 0.5–12 Hz. **NaN** if there is no peak
  in the band. Independent of amplitude.

### Output and missing values
- **Channel level:** an array of shape (n_windows, 18, n_features), with
  `feature_table()` giving each feature's name, unit, source and definition.
- **Window summaries:** `<feature>_median`, `<feature>_max` and `<feature>_p75`
  across valid channels.
  - A channel counts for a feature's summary if that feature is finite there and
    the channel is not QC-`bad` (§6).
  - QC-bad channels keep their channel-level values.
  - `n_valid_channels` and the per-channel `channel_valid` mask are metadata, as are
    the §6 QC flags. **None of these are model predictors.**
- **Counts:**

  | | Per channel | Channel-level per window | Summaries per window |
  |---|---|---|---|
  | Baseline | 55 | 990 | 165 |
  | Secondary | 58 | 1044 | 174 |

- **Missing values:** undefined values are NaN, never ±inf. This covers gap
  samples, the MATLAB near-zero rule, zero variance or zero power, an empty
  histogram range, and no autocorrelation zero crossing. Values are left NaN for
  imputation inside each training fold later.
- **No transformation:** no z-scoring, scaling, imputation or feature selection is
  applied. Amplitudes stay in µV.

### Validation (eeg1, eeg57)
- **Analytic checks on sines** match: RMS, power (Parseval), Hjorth mobility,
  kurtosis 1.5, peak frequency and SEF exactly at the sine frequency, NLE, zero
  crossings, ACF lag equal to the period, and a reference re-implementation of the
  Moddemeijer entropy.
- **Real data, eligible windows** (18,000 channel-windows in eeg1, 9,288 in eeg57):
  - **0% missing** for every feature, including `acf_peak`.
  - No infinity anywhere.
  - Gap windows are entirely NaN.
- **Runtime:** about 0.2 ms per channel-window, so 4–5 s for eeg1 (2 h) on one CPU
  core.

## 10. Known limitations

- **Two-recording validation.** Every threshold and every result above comes from
  eeg1 and eeg57 alone; 77 recordings are still cloud-only in iCloud. Seizure
  calibration rests on one patient.
- **Channel naming.** An earlier audit reported three naming conventions; only two
  have been seen. An unrecognised name raises an error rather than being skipped.
- **A single stuck electrode is not detected as a flatline.** If only F3 is stuck,
  Fp1-F3 still carries Fp1's signal and is not flat. Detecting it at electrode
  level would also flag the near-zero reference electrode, which in eeg57 stays
  unchanged for up to about 10 s.
- **Pop threshold vs noisy recordings.** eeg57 has many large jumps outside
  seizures (99th percentile 309 µV per sample), so the pop flag fires more often
  there.
- **Labels are calibrated on one seizure patient.** All labeling comparisons rest
  on eeg1. Expert disagreement may look very different in other recordings.
- **Features not verified against MATLAB.** The Temko features follow the paper
  and the public MATLAB code, but nothing has been compared numerically with
  MATLAB output. Exact `bandpower`, `pyulear` and `pwelch` behaviour is reproduced
  from documentation only.
- **Wavelet boundary effects.** 12 of 30 cD4 coefficients depend on edge
  extension (§9).
- **Filter edges inside stretches.** The first and last ~1 s of every valid stretch
  carries edge error of 0.4–0.9× signal level (typical). It is flagged, not
  removed.

## 11. Decisions to revisit or still open

**Revisit once all 79 recordings are available:**
1. All Stage 4 thresholds (§6) and the ≥ 9/18 unusable rule.
2. Clip-limit handling: confirm every file's EDF limits.
3. Gap-rule minimum of 16 samples, and the 4 s short-stretch reliability threshold.
4. Whether the near-zero electrode is the reference in every recording.
5. Whether `filter_edge` windows should be weighted or excluded in training.

6. **Labeling (§7):** the conservative strategy and the 2-of-3 consensus. Across
   all recordings, seizure time differs by almost 2× depending on the rule: 39,259 s
   agreed by all three experts, 50,612 by at least two, 73,842 by at least one. Also
   check for events shorter than 6 s, which positive labeling cannot represent, and
   whether nearby events need merging.

7. **Features (§9):**
   - wavelet boundary handling and the energy definition;
   - the 3 s autocorrelation lag limit and the 4 s Welch segments for prominence;
   - whether to summarise over QC-flagged (not just bad) channels;
   - whether duration-dependent features should be normalised per second.

**Planned, not yet decided or implemented:**
8. **Evaluation (§8):** event-matching rules and post-processing.
9. **Model training,** including imputation and scaling inside training folds.
10. **Final eligibility for training:** how `eligible`, `unusable` and the
   per-channel flags combine.

---

**Modeling (sections 12–20, Esther).** These sections document the window-level models built on the features above.
Code: `scripts/build_feature_table.py`, `modeling/common.py`, `modeling/models.py`,
`modeling/run_experiments.py`, `modeling/run_convlstm.py`, `modeling/plot_results.py`;
results in `results/`. Rerun instructions: `modeling/README.md`. Counts below were
computed from `data/processed/feature_table.parquet` and the files in `results/`.

## 12. Window table and training set

*Implemented. Provisional (training mask follows §8 and §11.10).*

- **Window table.** `scripts/build_feature_table.py` runs §2–§9 unchanged on all 79
  recordings and stacks one row per 12 s window into
  `data/processed/feature_table.parquet`. No recording failed. The eeg1 and eeg57 window
  and eligibility counts match §5 (1164 / 1000 and 521 / 516). eeg1's eligible labels
  match §7 (262 positive, 350 negative, 388 ambiguous).
- **Counts:**

  | | Windows |
  |---|---|
  | All windows | 67,025 (79 subjects) |
  | `label` 1 / 0 / −1 | 8,516 / 53,789 / 4,720 |
  | Eligible (§5) | 65,504 |
  | Ineligible | 1,521 (15 of them `label` 1) |
  | Unusable (§6) | 1, and it is also ineligible (eeg28, window 957, `gap_frac` 0.83) |
  | **Training mask** | **60,885** (8,501 positive, 52,384 negative), all 79 subjects |
  | Eligible ambiguous (`label` −1) | 4,619 |

- **Training mask:** `label` ∈ {0, 1} **and** `eligible` **and not** `unusable`
  (`modeling.common.training_mask`). Since no eligible window is unusable, this is the
  same set as eligible windows with label 0/1. Ambiguous and ineligible windows are
  never training targets but are kept and predicted (§17).
- **Every window is fully annotated:** `label_any_overlap` and `label_binary50` are
  never −1.
- **No missing feature values in training windows** (0 NaN over the 174 feature
  columns). All 1,521 ineligible windows have at least one NaN feature.
- **Seizure subjects:** 46 subjects have at least one `label` 1 window and 33 have none.
  §1 counts 47 subjects marked by at least two experts. The difference is **eeg8**:
  expert A marks 32 s and expert B 73 s, but never the same second, so eeg8 has no
  consensus second (§7) and no positive window.

## 13. Feature sets for modeling

*Implemented.*

- **Set 1 (baseline):** the 55 Temko features (§9), each summarised across channels
  by median, max and p75 → **165 columns**.
- **Set 2 (secondary):** set 1 plus the 3 rhythmicity features `acf_peak`,
  `acf_peak_lag` and `spectral_peak_prominence` → **174 columns**.
- **Column lists** are built from `helsinki.features.FEATURE_SETS` and `SUMMARY_STATS`
  (`modeling.common.FEATURE_SET_1` / `FEATURE_SET_2`), so they follow §9 exactly. They
  are also written to `data/processed/feature_columns.json`.
- **Not predictors:** QC flags, `n_valid_channels`, `gap_frac`, `filter_edge` and the
  annotation counts are metadata (§9).
- **One feature run.** Features were extracted once with the secondary set. Set 1's
  columns from that run were checked to be identical to a separate baseline run on
  eeg1 and eeg57. The 3 extra features are appended after the 55, so the shared ones
  are computed by the same code.
- **Per-channel features (ConvLSTM only).** With `--channel-features`, the
  `(n_windows, 18, 58)` array returned by `extract_features` is also saved
  (`channel_features.npy`, float32), with `channel_valid` and a row index. On all
  eligible windows, the median across valid channels reproduces the table's
  `line_length_median`.

## 14. Patient-level folds

*Implemented. Final for this round of experiments.*

- **Split by patient.** Every window of a subject is in the same fold, so no
  patient is in both training and test.
- **Design.** The 46 seizure subjects and the 33 non-seizure subjects are each sorted,
  shuffled and split into 5 near-equal parts with `KFold(shuffle=True,
  random_state=0)`, and part *i* of both groups forms fold *i*. The result is stored in
  `results/folds.csv` and is shared by every model and teammate.
- **Why not `StratifiedGroupKFold`.** It was tried first. It balances the *share of
  seizure windows* per fold, which left between 5 and 15 seizure subjects per fold
  because a few subjects carry many positives. Event-level evaluation (§8) depends on
  the number of seizure patients, so each fold now gets 9–10.
- **Folds:**

  | Fold | Subjects | Seizure / non-seizure | Training-mask windows | Positives | Positive share | Ambiguous windows |
  |---|---|---|---|---|---|---|
  | 0 | 17 | 10 / 7 | 13,106 | 1,361 | 10.4% | 874 |
  | 1 | 16 | 9 / 7 | 12,059 | 1,067 | 8.8% | 853 |
  | 2 | 16 | 9 / 7 | 11,614 | 1,093 | 9.4% | 756 |
  | 3 | 15 | 9 / 6 | 14,173 | 3,099 | 21.9% | 727 |
  | 4 | 15 | 9 / 6 | 9,933 | 1,881 | 18.9% | 1,510 |

- **Cost:** the positive share differs by fold (8.8–21.9%), so AUPRC is not
  comparable across folds (§17).
- **Leakage check.** The training loops were instrumented. For every model, feature
  set and fold, zero subjects overlap between training and test, training targets
  contain no −1, ineligible or unusable windows, and every held-out window is
  predicted.

## 15. Classical models

*Implemented. Settings chosen in advance, not tuned.*

- **Pipeline.** Every model is `SimpleImputer(strategy="median")` →
  `StandardScaler()` → classifier (`modeling/models.py`). Imputation and scaling are
  fit inside each training fold. On fold 0, the stored medians and means were checked
  to equal the training rows' values and to differ from the full-data values.
- **Settings:**

  | Model | Classifier and settings |
  |---|---|
  | LogReg | `LogisticRegression(class_weight="balanced", max_iter=2000)` |
  | RF | `RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=0)` |
  | KNN | `KNeighborsClassifier(n_neighbors=25, n_jobs=-1)`. There is no `class_weight` option, so it trains on the raw ~1:6 imbalance. |
  | SVC | `SVC(kernel="rbf", class_weight="balanced")`, other parameters at scikit-learn defaults, with negative subsampling (below) |

- **Training data:** training-mask windows of the 4 other folds' subjects, 46,712–50,952
  windows per fold.
- **Scores.** `predict_scores` returns `predict_proba[:, 1]` where it exists. SVC has no
  `probability=True` (about 5× slower), so it returns `decision_function`, an unbounded
  score. Both are valid for AUROC and AUPRC, but SVC scores are not probabilities.
- **SVC negative subsampling.** RBF-SVC fit time grew roughly with the square of the
  training size. Measured on the development laptop (Intel Core i5-1038NG7): 8 s at
  10,000 windows and 35 s at 20,000, extrapolating to about 3–4 min per fold.
  `NegativeSubsampler` therefore fits each fold on **all positives plus 20,000 random
  negatives** (`SVC_MAX_NEGATIVES`, NumPy generator seed 0), 25,402–27,434 windows per
  fold. Only training is subsampled; every held-out window is scored.
  `class_weight="balanced"` is computed on the subsample.

## 16. ConvLSTM

*Implemented. Provisional: architecture and settings chosen in advance, not tuned.*

- **Input.** Each target window is a sequence of the **5 previous windows of the same
  subject plus the target** (6 windows, 30 s of past context at a 6 s stride), oldest
  first.
  - Only past windows are used, and a sequence never crosses subjects or folds
    (checked on all 67,025 windows).
  - Each step is the per-channel feature grid, 18 bipolar channels × 55 (set 1) or 58
    (set 2) features, plus one mask input per channel.
  - Steps before a recording's first window are padding (zeros, mask 0). 395 windows
    have padded history.
- **Missing context rule.**
  - A channel without features (gap, or QC-`bad` per §6) is set to 0 with mask 0.
  - Ambiguous (−1) windows have real features and are used as context.
  - Labels are never inputs. Targets are training-mask windows only.
  - Every held-out window is predicted.
- **Scaling.** Per feature, a `QuantileTransformer(n_quantiles=1000,
  output_distribution="normal", subsample=200000, random_state=0)` is fit on the valid
  channels of the training targets only, then applied to all windows. 0 is therefore the
  training median. Chosen because powers and energies span orders of magnitude.
- **Network.**
  - One ConvLSTM layer; each gate is a `Conv1d` across the 18 channels in montage order
    (kernel 3, padding 1), with 16 hidden channels.
  - The last hidden state goes through mean and max pooling over channels, dropout 0.3
    and a linear output.
  - 13,921 parameters (set 1) or 14,497 (set 2).
  - The kernel also spans the joins between montage chains (e.g. P4-O2 next to
    Fp1-F3), which are not physical neighbours.
- **Training.**
  - `BCEWithLogitsLoss` with `pos_weight` = negatives / positives among training targets.
  - Adam (lr 1e-3, weight decay 1e-4), batch 256, at most 30 epochs.
  - Early stopping with patience 5 on validation AUROC, restoring the best epoch.
  - Validation is 20% of the training folds' seizure subjects and 20% of their
    non-seizure subjects: 12 patients per fold (7 with seizures), never the test fold.
    These patients are also excluded from the scaling fit.
  - Seeds: `torch.manual_seed(fold)`, a shuffling generator seeded with the fold, and
    the NumPy seed for the validation split.
- **Software.** PyTorch 2.2.2 on CPU (4 threads), the newest PyTorch for Intel Macs.
  It cannot exchange arrays with numpy 2 directly, so arrays pass through raw bytes
  (`to_torch` / `to_numpy`). numpy was not downgraded.
- **Reproducibility.** Verified once: the 5-epoch test run and the full run gave the
  same set 2 fold 0 result (best epoch 2, AUROC 0.882). Other folds have not been
  rerun.

## 17. Evaluation

*Implemented: window-level metrics. Planned: event-level metrics (§8).*

- **Predictions.** For each feature set, model and fold, the model is fit on the
  training folds and scores **every window of the held-out subjects**, including
  ambiguous and ineligible ones (`results/predictions_<model>_set<N>.parquet`:
  `subject_id`, `start_s`, `end_s`, `label`, `eligible`, `score`, `fold`; 67,025 rows
  each, no missing scores).
- **Scores of ineligible windows are not meaningful.** Their missing features are
  imputed (classical models) or masked (ConvLSTM), so filter on `eligible`.
- **Metrics** (`results/cv_auroc.csv`): AUROC (`roc_auc_score`) and AUPRC
  (`average_precision_score`) per fold, on held-out windows with `label` ∈ {0, 1} and
  `eligible`.
  - **Excluded:** ambiguous (−1) windows and ineligible windows. Unusable windows are not
    filtered separately; the only one is ineligible.
  - A fold with only one class would be skipped with a warning. None was.
- **Summary** = mean and sample standard deviation (ddof 1) across the 5 folds.
  `modeling/plot_results.py` plots it as `results/auroc_bar_chart.png`.
- **Why AUPRC is per fold only.** AUPRC depends on the positive share, which varies by
  fold (8.8–21.9%, §14), so it is compared between models within a fold, not across
  folds.
- **Not done:** event-level detection against per-second consensus, false detections
  per hour, smoothing or thresholding, and evaluation on ambiguous windows.

## 18. Results (window level)

*Provisional: window-level only; to be complemented by event-level results.*

Mean ± SD across 5 folds (`results/cv_auroc.csv`):

| Model | AUROC set 1 | AUROC set 2 | AUPRC set 1 | AUPRC set 2 |
|---|---|---|---|---|
| LogReg | 0.754 ± 0.065 | 0.761 ± 0.066 | 0.419 ± 0.148 | 0.430 ± 0.153 |
| RF | 0.782 ± 0.053 | 0.792 ± 0.055 | 0.492 ± 0.100 | 0.511 ± 0.090 |
| KNN | 0.754 ± 0.038 | 0.760 ± 0.038 | 0.444 ± 0.077 | 0.457 ± 0.069 |
| SVC | 0.792 ± 0.038 | 0.798 ± 0.041 | 0.523 ± 0.087 | 0.530 ± 0.086 |
| ConvLSTM | 0.805 ± 0.063 | 0.813 ± 0.059 | 0.569 ± 0.105 | 0.585 ± 0.096 |

AUROC per fold, set 2:

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| LogReg | 0.857 | 0.727 | 0.798 | 0.694 | 0.728 |
| RF | 0.867 | 0.763 | 0.781 | 0.725 | 0.822 |
| KNN | 0.784 | 0.755 | 0.778 | 0.695 | 0.788 |
| SVC | 0.863 | 0.794 | 0.775 | 0.754 | 0.803 |
| ConvLSTM | 0.882 | 0.767 | 0.828 | 0.738 | 0.851 |

- **Fold matters more than model.** Fold 0 is easiest and fold 3 hardest for most
  models, and the fold-to-fold SD (0.04–0.07) exceeds most differences between models.
- **ConvLSTM has the highest mean** AUROC and AUPRC on both sets. It beats SVC in 7 of
  10 feature-set × fold pairs and RF in 9 of 10. SVC is higher on fold 1 (both sets) and
  on fold 3 (set 2).
- **Set 2 vs set 1:** set 2 has the higher mean for every model (+0.006 to +0.010 AUROC).
  Per fold, set 2 is higher in 4/5 (LogReg, KNN, SVC), 5/5 (RF) and 3/5 (ConvLSTM).
  The gain is small relative to the SD and has not been tested for significance.
- **ConvLSTM training.**
  - Best epochs ranged from 2 to 16 (7–21 epochs run).
  - Validation loss ended higher than at epoch 1 in 9 of 10 runs, while training loss fell
    throughout. The network becomes overconfident on unseen patients; early stopping on
    validation AUROC keeps the best epoch (`results/convlstm_curves.png`).
  - Validation AUROC is noisy with 12 patients.
- **Runtime** (one Intel Core i5-1038NG7 laptop, all 10 fits per model): LogReg 0.8 min,
  RF 5.9 min, KNN 0.7 min, SVC 14.7 min, ConvLSTM 17.9 min (9.2 + 8.7). Times from the
  run logs.

## 19. Modeling caveats

- **Window-level only.** AUROC/AUPRC on clean windows are not the primary metric (§8).
  Event-level results may rank the models differently.
- **Unequal inputs.** ConvLSTM sees 30 s of past context and all 18 channels
  separately; the classical models see one window's channel summaries. Part of its
  advantage may come from the context, not the architecture. Not tested.
- **Settings not tuned.** All hyperparameters (§15, §16) were fixed in advance. The
  ConvLSTM fold 0 test result was seen before the full run; nothing was changed after
  it. Picking the best of 5 models on the same folds is still mildly optimistic.
- **SVC is trained on a subsample** (all positives + 20,000 negatives), so it uses fewer
  negatives than the other models.
- **KNN is not class-weighted.**
- **Calibration not assessed.** SVC outputs margins, and the other models' probabilities
  have not been calibrated. Thresholds for detection are not set.
- **Small seizure population.** 46 seizure subjects in total and 9–10 per test fold.
  Per-fold results depend strongly on which patients are held out.
- **QC thresholds** (§6) were calibrated on eeg1 and eeg57. eeg1 is a test subject in
  fold 4. The flags are not predictors and affect only eligibility and channel
  validity.
- **Reproducibility of classical runs not re-verified.** Seeds are fixed (RF
  `random_state=0`, SVC subsample seed 0), but no repeat run was compared.
- **Earlier `cv_auroc.csv` rows.** Before a fix to `save_cv_rows` (exact float
  parsing), rewriting the file could change kept values in the 17th significant digit.
  The committed file was rebuilt so the 40 classical rows are byte-identical to their
  first commit.

## 20. Modeling: open items

1. Event-level evaluation (§8): event matching, smoothing and false detections per hour,
   using the saved predictions.
2. Decide how ambiguous windows enter evaluation, and report results with them.
3. A fair context comparison: give the classical models the previous windows'
   features, to separate the effect of context from the ConvLSTM architecture.
4. ConvLSTM regularisation: validation loss rises early, so try a lower learning rate,
   more dropout, or fewer inputs.
5. Hyperparameter tuning with nested cross-validation, if time allows.
6. Probability calibration and an operating threshold.
7. XGBoost (installed, not yet used) as a further classical model.
