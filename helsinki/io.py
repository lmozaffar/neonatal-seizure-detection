"""Read-only loading of Helsinki EDF recordings."""

from __future__ import annotations

import os
import stat
import warnings
from pathlib import Path

import mne

from helsinki.paths import edf_path

# macOS flag for iCloud placeholders whose content is not on disk.
_SF_DATALESS = getattr(stat, "SF_DATALESS", 0x40000000)


def is_dataless(path: Path) -> bool:
    """True if iCloud has evicted the file; reading it triggers a full download."""
    return bool(getattr(os.stat(path), "st_flags", 0) & _SF_DATALESS)


def load_recording(subject_id: int, preload: bool = False) -> mne.io.BaseRaw:
    """Load eeg<subject_id>.edf unchanged (all 21 channels, referential, 256 Hz).

    MNE converts the EDF's microvolts to volts. Channel names, types and
    order are left as stored; they differ between recordings, so select
    channels by name, never by index.
    """
    path = edf_path(subject_id)
    if not path.exists():
        raise FileNotFoundError(path)
    if is_dataless(path):
        warnings.warn(f"{path.name} is an iCloud placeholder; reading it downloads the file", stacklevel=2)
    return mne.io.read_raw_edf(path, preload=preload, verbose="error")
