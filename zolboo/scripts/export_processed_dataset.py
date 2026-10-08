#!/usr/bin/env python3
"""Export the single canonical labeled dataset (all HUH + Cork subjects, every
epoch, with the final preictal 0/1/2 labels and the ~190 QEEG features) that
preproc.py's preproc_df_noTD() builds internally before splitting into kfolds.

This replicates preproc.py lines ~372-456 up to (but not including) the
gen_kf_dfs() fold-splitting step, so what's saved here is the exact same
dataframe every train/valid/test kfold file is carved out of.
"""
import numpy as np
import pandas as pd
import preproc
import preproc_utils as utils

OUT_CSV = "data/tprocessed_dataset.csv"


def build_master_dataset():
    df_huh_train = preproc.get_huh_training_df()

    feature_column_names = [c for c in df_huh_train.columns if preproc.valid_feat_name(c)]

    df_huh_train["preictal"] = df_huh_train["preictal"].astype(str)
    df_huh_train["seizing"] = df_huh_train["seizing"].astype(str)

    df_master_train = df_huh_train.copy()
    df_master_train = df_master_train.sort_values(['subject_id', 'abs_time_idx'])

    grp_lbls, snd_lbls, newPI_lbls, ii_safe_subjs, postnatal_age_lbls, postnatal_time_without_sz_lbls = \
        utils.get_lbls_sz_epochs(df_master_train, one_seizure=False)

    df_master_train['ii_safe'] = np.zeros((len(df_master_train),))
    for grp, lbls in grp_lbls.items():
        df_master_train.loc[df_master_train['subject_id'] == grp, 'ii_safe'] = lbls

    df_master_train['snd_grp'] = np.zeros((len(df_master_train),))
    for grp, lbls in snd_lbls.items():
        df_master_train.loc[df_master_train['subject_id'] == grp, 'snd_grp'] = lbls

    for grp, lbls in newPI_lbls.items():
        df_master_train.loc[df_master_train['subject_id'] == grp, 'preictal'] = lbls

    df_master_train['pna_szf'] = np.zeros((len(df_master_train),))
    for grp, lbls in postnatal_time_without_sz_lbls.items():
        df_master_train.loc[df_master_train['subject_id'] == grp, 'pna_szf'] = lbls

    df_master_train['pna'] = np.zeros((len(df_master_train),))
    for grp, lbls in postnatal_age_lbls.items():
        df_master_train.loc[df_master_train['subject_id'] == grp, 'pna'] = lbls

    numel_df_master_train = len(df_master_train)
    df_cork = preproc.get_df_cork()
    df_master_train = pd.concat([df_master_train, df_cork], ignore_index=True)
    df_master_train.loc[numel_df_master_train:, ['ii_safe']] = 1

    df_master_train['preictal'] = df_master_train['preictal'].astype(int)
    df_master_train['subject_id'] = df_master_train['subject_id'].astype(str)

    final_cols = ['subject_id', 'abs_time_idx', 'augmentation'] + feature_column_names_dedup(feature_column_names) + \
                 ['preictal', 'ii_safe', 'snd_grp', 'pna_szf', 'pna']
    final_cols = [c for c in dict.fromkeys(final_cols) if c in df_master_train.columns]

    return df_master_train[final_cols], feature_column_names


def feature_column_names_dedup(cols):
    # feature_column_names (from valid_feat_name) also includes subject_id/abs_time_idx
    # since those aren't in the exclusion list -- drop them here, they're added back
    # explicitly (as leading identifier columns) in build_master_dataset().
    return [c for c in cols if c not in ('subject_id', 'abs_time_idx', 'augmentation')]


if __name__ == "__main__":
    df, feature_cols = build_master_dataset()
    print("shape:", df.shape)
    print("subjects:", df.subject_id.nunique())
    print("preictal counts:\n", df.preictal.value_counts())
    df.to_csv(OUT_CSV, index=False)
    print(f"saved to {OUT_CSV}")

    with open("data/feature_columns.txt", "w") as f:
        f.write("\n".join(feature_column_names_dedup(feature_cols)))
    print(f"n_feature_columns: {len(feature_column_names_dedup(feature_cols))}")
