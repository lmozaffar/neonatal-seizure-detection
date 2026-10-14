"""Check that the environment, data paths and EDF loading work.

    .venv.nosync/bin/python scripts/check_setup.py [subject_id]
"""

import sys

import mne
import numpy as np

from helsinki.io import is_dataless, load_recording
from helsinki.paths import ANNOTATION_CSVS, CLINICAL_CSV, DATA_ROOT, SUBJECT_IDS, edf_path


def main(subject_id: int = 1) -> None:
    print(f"mne {mne.__version__}, numpy {np.__version__}")
    print(f"data root: {DATA_ROOT}")

    missing = [p for p in [*map(edf_path, SUBJECT_IDS), *ANNOTATION_CSVS.values(), CLINICAL_CSV] if not p.exists()]
    if missing:
        sys.exit(f"missing files: {missing}")
    n_dataless = sum(is_dataless(edf_path(s)) for s in SUBJECT_IDS)
    print(f"found 79 EDFs ({n_dataless} not downloaded from iCloud), 3 annotation CSVs, clinical table")

    raw = load_recording(subject_id, preload=True)
    data = raw.get_data()
    print(f"\n{edf_path(subject_id).name}: {len(raw.ch_names)} channels, {raw.info['sfreq']:g} Hz, "
          f"{raw.times[-1] + 1 / raw.info['sfreq']:.0f} s, data {data.shape} in volts")
    print("channels:", ", ".join(raw.ch_names))
    print(f"median per-channel std: {np.median(data.std(axis=1)) * 1e6:.1f} uV, all finite: {np.isfinite(data).all()}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
