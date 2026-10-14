"""Locations of the Helsinki dataset files.

The data lives outside the repo, in Desktop/Helsinki/data (a sibling of
main/). Set HELSINKI_DATA_ROOT to point somewhere else, e.g. a copy
outside iCloud. Nothing here writes to the data directory.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("HELSINKI_DATA_ROOT", REPO_ROOT.parent / "data")).resolve()

RAW_DIR = DATA_ROOT / "raw"
ANNOTATIONS_DIR = DATA_ROOT / "annotations"
CLINICAL_CSV = DATA_ROOT / "clinical" / "clinical_information.csv"

# Expert A's file was corrected by the dataset authors; the "_fixed"
# version is the one to use. All three are identical to annotations_2017.mat.
ANNOTATION_CSVS = {
    "A": ANNOTATIONS_DIR / "annotations_2017_A_fixed.csv",
    "B": ANNOTATIONS_DIR / "annotations_2017_B.csv",
    "C": ANNOTATIONS_DIR / "annotations_2017_C.csv",
}

SUBJECT_IDS = tuple(range(1, 80))


def edf_path(subject_id: int) -> Path:
    """Path to eeg<subject_id>.edf. Subject IDs match the annotation CSV
    column headers and the clinical table's ID column."""
    if subject_id not in SUBJECT_IDS:
        raise ValueError(f"subject_id must be 1-79, got {subject_id}")
    return RAW_DIR / f"eeg{subject_id}.edf"
