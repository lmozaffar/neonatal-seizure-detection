#!/usr/bin/env python3
"""Build a column-by-column metadata/data-dictionary file for
data/tprocessed_dataset.csv, grounded in the actual qeegfeats feature-generation
code (utils/preproc.py: BasicStats, AsymmetryFeatures, rqa_channel,
relative_log_power) and neoSeer's preproc.py/preproc_utils.py labeling code.
"""
import re
import pandas as pd

FEATURE_COLS = open("data/feature_columns.txt").read().splitlines()

ASYMM_STAT_DESC = {
    "mean": "mean signal amplitude",
    "stddev": "standard deviation of signal amplitude",
    "kurtosis": "kurtosis of signal amplitude distribution",
    "skew": "skewness of signal amplitude distribution",
    "90th_percentile": "90th percentile of signal amplitude",
    "10th_percentile": "10th percentile of signal amplitude",
}

RQA_METRIC_DESC = {
    "RR": "Recurrence Rate - density of recurrence points in the phase-space recurrence plot",
    "DET": "Determinism - fraction of recurrence points forming diagonal lines (signal predictability)",
    "LAM": "Laminarity - fraction of recurrence points forming vertical lines (signal intermittency/laminar states)",
    "Lmax": "length of the longest diagonal line in the recurrence plot",
    "Lentr": "Shannon entropy of the diagonal line length distribution",
    "Lmean": "mean diagonal line length",
    "TT": "Trapping Time - mean length of vertical line structures",
}

LABEL_META = [
    ("preictal", "label",
     "Training label: 0=interictal/background, 1=preictal (last 15 min before a qualifying seizure onset, "
     "see preproc_utils.get_lbls_sz_epochs), 2=ictal/buffer (during a seizure; excluded from binary classification)."),
    ("subject_id", "identifier",
     "Subject ID. 3-digit numeric strings (e.g. '001'-'079') are Helsinki University Hospital (HUH) subjects; "
     "4-digit IDs starting with '97' are Cork University Maternity Hospital subjects (remapped from 'ID...' by preproc.get_df_cork)."),
    ("abs_time_idx", "identifier",
     "Running epoch index within a subject's recording. Each step = one 20-second QEEG epoch "
     "(computed over a 60-second sliding window internally; see qeegfeats baruto.yml len_epoch=60 / Windower window=20)."),
    ("augmentation", "bookkeeping",
     "Data augmentation applied to this row's epoch: 'none', 'vertflip', 'horizflip', or 'horizvertflip' "
     "(left-right / polarity flips of the EEG montage used to expand the training set)."),
    ("ii_safe", "bookkeeping",
     "1 if this epoch falls in a 'safe' interictal stretch (>=20 min away from any seizure) usable for clean "
     "training labels; set via preproc_utils.get_lbls_sz_epochs. -2 marks augmentation-segment boundaries."),
    ("snd_grp", "bookkeeping",
     "Segment group ID grouping epochs into interictal-to-seizure segments; encodes subject_id and a running "
     "segment counter (see preproc_utils.py snd_id = subject_id*1e6 + snd_grp)."),
    ("pna_szf", "bookkeeping",
     "Postnatal age (minutes) excluding time after the first seizure. Only populated for Cork subjects with known "
     "age-at-EEG metadata; NaN/0 for HUH subjects (age reference not available in this pipeline)."),
    ("pna", "bookkeeping",
     "Postnatal age (minutes) at each epoch. Only populated for Cork subjects with known age-at-EEG metadata; "
     "NaN/0 for HUH subjects."),
]


def describe_feature(col):
    if col.startswith("asymm_"):
        m = re.match(r"asymm_(.+?)_([a-z0-9]+-[a-z0-9]+)_vs_([a-z0-9]+-[a-z0-9]+)", col)
        stat_key, right_ch, left_ch = m.group(1), m.group(2), m.group(3)
        stat_desc = ASYMM_STAT_DESC.get(stat_key, stat_key)
        return (
            "asymm",
            f"Left-right hemisphere asymmetry of {stat_desc}, computed as "
            f"(right[{right_ch}] - left[{left_ch}]) / (right[{right_ch}] + left[{left_ch}]) "
            f"over the 20-second epoch (AsymmetryFeatures in qeegfeats/utils/preproc.py)."
        )
    if col.startswith("rqa_"):
        m = re.match(r"rqa_([A-Za-z]+)_([a-z0-9]+-[a-z0-9]+)", col)
        metric_key, ch = m.group(1), m.group(2)
        metric_desc = RQA_METRIC_DESC.get(metric_key, metric_key)
        return (
            "rqa",
            f"Recurrence Quantification Analysis metric on bipolar channel {ch}: {metric_desc} "
            f"(pyrqa-based rqa_channel class in qeegfeats/utils/preproc.py, derived from Brinkmann et al. 2016 "
            f"Kaggle seizure-prediction features)."
        )
    if col.startswith("rellogpower_"):
        idx = int(re.match(r"rellogpower_feat(\d+)", col).group(1))
        return (
            "rellogpower",
            f"Relative log-power spectral feature #{idx} (of 56). Computed via Welch's method "
            "(qeegfeats/utils/preproc.py relative_log_power) as log10(band_power / total_power) per channel. "
            "56 = likely 14 bipolar channels (same montage as the rqa_* features) x 4 usable frequency bands "
            "(delta 0.1-4Hz, theta 4-8Hz, alpha 8-12Hz, beta 12-30Hz -- the default function supports 6 bands "
            "but the data is bandpass-filtered to 20Hz and resampled to 40Hz per qeegfeats configs/baruto.yml, "
            "which excludes the two higher gamma bands). The exact feat-index-to-(channel,band) ordering is NOT "
            "confirmed from the available code/config -- qeegfeats' own readable-name mapping file "
            "(metadata/human_readable_feat_names_ucsf.yml, referenced in scripts/generate_features.py) is not "
            "present in the cloned repo."
        )
    return ("other", "")


def build():
    rows = []
    for col, coltype, desc in LABEL_META:
        if col in ("subject_id", "abs_time_idx", "augmentation"):
            family = "identifier"
        elif coltype == "label":
            family = "label"
        else:
            family = "bookkeeping"
        rows.append(dict(column=col, type=coltype, family=family, description=desc))

    for col in FEATURE_COLS:
        family, desc = describe_feature(col)
        rows.append(dict(column=col, type="feature", family=family, description=desc))

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = build()
    df.to_csv("data/tprocessed_dataset_metadata.csv", index=False)
    print("rows:", len(df))
    print(df.family.value_counts())
    print("saved to data/tprocessed_dataset_metadata.csv")
