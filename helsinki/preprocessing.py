"""EEG preprocessing: referential EDF -> 18-channel bipolar montage, 0.5-12.8 Hz, 32 Hz, µV.

Steps, in order:
  1. Keep the 19 EEG electrodes, matched by normalised name ("EEG Fp1-REF",
     "EEG Fp1-Ref" -> "Fp1"); ECG and respiration channels are dropped.
     Recordings store channels in different orders, so never index by position.
  2. Find recording gaps on the raw 256 Hz electrodes (find_gaps): stretches
     where all 19 electrodes hold exactly the same value. Genuine EEG, even
     very low-amplitude, never stays unchanged on all 19 at once; a reference
     electrode that is ~0 on its own does not count.
  3. Double-banana bipolar montage, 18 derivations in BIPOLAR_NAMES order.
  4. Zero-phase Butterworth band-pass 0.5-12.8 Hz (order 4, applied forward
     and backward). Forward-backward squares the magnitude response, so each
     cutoff sits at -6 dB rather than -3 dB and the roll-off is twice as steep.
  5. Downsample 256 -> 32 Hz with resample_poly, whose linear-phase FIR
     low-pass at the new Nyquist (16 Hz) is the anti-aliasing filter. The
     band-pass alone is not enough: it is only ~18 dB down at 16 Hz.

Steps 4-5 run separately on each valid stretch between gaps. Filtering across
a gap turns its edges into ringing of hundreds of µV that reaches ~5 s into
valid EEG; per-stretch filtering leaves only the ordinary edge effect seen at
any recording start or end.

Output is a float64 array (18, n_samples) in µV at 32 Hz on the original
timeline, NaN inside gaps (rounded outward to whole 32 Hz samples, <= 31 ms).
No normalisation. make_windows then cuts it into 12 s windows with a 6 s stride.
"""
from __future__ import annotations

import re

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt

FS_IN = 256.0
FS_OUT = 32.0
BANDPASS_HZ = (0.5, 12.8)
FILTER_ORDER = 4
# Edge padding for the band-pass: mirror (even) reflection, 3 s per side. On
# rhythmic activity this has 5-7x less edge error than scipy's default (odd,
# 0.1 s), whose point reflection can leave a large offset in the pad.
EDGE_PAD_S = 3.0

WINDOW_S = 12.0
STRIDE_S = 6.0
WINDOW_SAMPLES = int(WINDOW_S * FS_OUT)  # 384
STRIDE_SAMPLES = int(STRIDE_S * FS_OUT)  # 192

# All 19 electrodes unchanged for >= 16 samples (62.5 ms) = recording gap. In
# real EEG not even 2 consecutive samples are unchanged on all 19 at once.
MIN_GAP_SAMPLES = 16
# A valid stretch shorter than this is still filtered and kept, but flagged
# unreliable: it is all filter edge (0.5 Hz high-pass settles over ~2 s per side).
MIN_RELIABLE_S = 4.0

ELECTRODES = ("Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "C3", "C4", "Cz",
              "T3", "T4", "T5", "T6", "P3", "P4", "Pz", "O1", "O2")

BIPOLAR_PAIRS = (
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),  # right parasagittal
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),  # left parasagittal
    ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),  # right temporal
    ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),  # left temporal
    ("Fz", "Cz"), ("Cz", "Pz"),                                # midline
)
BIPOLAR_NAMES = tuple(f"{a}-{b}" for a, b in BIPOLAR_PAIRS)

_CANONICAL = {e.upper(): e for e in ELECTRODES}


def normalize_channel_name(name: str) -> str | None:
    """"EEG Fp1-REF" / "EEG FP1-Ref" -> "Fp1"; None for non-EEG channels (ECG, Resp)."""
    core = re.sub(r"^EEG\s+", "", name.strip(), flags=re.IGNORECASE)
    core = re.sub(r"-REF$", "", core, flags=re.IGNORECASE)
    return _CANONICAL.get(core.upper())


def select_eeg(raw: mne.io.BaseRaw) -> np.ndarray:
    """(19, n) referential EEG in µV, rows in ELECTRODES order."""
    index: dict[str, int] = {}
    for i, name in enumerate(raw.ch_names):
        electrode = normalize_channel_name(name)
        if electrode is None:
            continue
        if electrode in index:
            raise ValueError(f"electrode {electrode} appears twice: {raw.ch_names}")
        index[electrode] = i
    missing = [e for e in ELECTRODES if e not in index]
    if missing:
        raise ValueError(f"missing electrodes {missing} in channels {raw.ch_names}")
    return raw.get_data(picks=[index[e] for e in ELECTRODES]) * 1e6


def find_gaps(eeg: np.ndarray, min_samples: int = MIN_GAP_SAMPLES) -> list[tuple[int, int]]:
    """Recording gaps in raw (19, n) referential EEG, as half-open sample ranges
    [start, end) where every electrode holds one constant value."""
    still = np.all(np.diff(eeg, axis=1) == 0, axis=0)  # sample i equals sample i + 1
    edges = np.diff(np.r_[0, still.astype(np.int8), 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(int(a), int(b) + 1) for a, b in zip(starts, stops) if b + 1 - a >= min_samples]


def bipolar_montage(eeg: np.ndarray) -> np.ndarray:
    """(19, n) referential, ELECTRODES order -> (18, n), BIPOLAR_NAMES order."""
    pos = {e: i for i, e in enumerate(ELECTRODES)}
    return np.stack([eeg[pos[a]] - eeg[pos[b]] for a, b in BIPOLAR_PAIRS])


def bandpass(x: np.ndarray, fs: float = FS_IN, band: tuple[float, float] = BANDPASS_HZ) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis."""
    sos = butter(FILTER_ORDER, band, btype="bandpass", fs=fs, output="sos")
    # Pad length is shortened for stretches too short for EDGE_PAD_S.
    return sosfiltfilt(sos, x, axis=-1, padtype="even", padlen=min(int(EDGE_PAD_S * fs), x.shape[-1] - 1))


def downsample(x: np.ndarray, fs_in: float = FS_IN, fs_out: float = FS_OUT) -> np.ndarray:
    """Anti-aliased integer-factor downsampling along the last axis."""
    factor = fs_in / fs_out
    if factor != int(factor):
        raise ValueError(f"{fs_in} Hz -> {fs_out} Hz is not an integer factor")
    return resample_poly(x, 1, int(factor), axis=-1)


def preprocess(raw: mne.io.BaseRaw) -> tuple[np.ndarray, pd.DataFrame]:
    """Full chain for one recording.

    Returns `x`, (18, n_samples) µV at FS_OUT with NaN in gaps, and `segments`,
    one row per gap or valid stretch in time order: kind ("gap" / "valid"),
    start_s, end_s, duration_s, processed, reliable. A valid stretch is
    processed unless it covers no whole 32 Hz sample, and reliable if it is
    also at least MIN_RELIABLE_S long. Gaps are never processed.
    """
    if raw.info["sfreq"] != FS_IN:
        raise ValueError(f"expected {FS_IN} Hz, got {raw.info['sfreq']} Hz")
    eeg = select_eeg(raw)
    gaps = find_gaps(eeg)
    bip = bipolar_montage(eeg)
    factor = int(FS_IN / FS_OUT)
    n = bip.shape[1]
    x = np.full((bip.shape[0], -(-n // factor)), np.nan)

    bounds = [0, *(t for gap in gaps for t in gap), n]
    rows = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if a == b:
            continue
        kind = "valid" if i % 2 == 0 else "gap"
        processed = False
        if kind == "valid":
            # Only whole output samples fully inside the stretch, so every
            # stretch lands on the shared 32 Hz timeline.
            a_out, b_out = -(-a // factor), b // factor
            if b_out > a_out:
                x[:, a_out:b_out] = downsample(bandpass(bip[:, a_out * factor:b_out * factor]))
                processed = True
        duration = (b - a) / FS_IN
        rows.append({"kind": kind, "start_s": a / FS_IN, "end_s": b / FS_IN, "duration_s": duration,
                     "processed": processed, "reliable": processed and duration >= MIN_RELIABLE_S})
    return x, pd.DataFrame(rows)


def make_windows(x: np.ndarray, subject_id: int, fs: float = FS_OUT) -> tuple[np.ndarray, pd.DataFrame]:
    """Cut preprocessed EEG (18, n) into 12 s windows with a 6 s stride.

    Returns `windows`, a read-only view of shape (n_windows, 18, 384), and
    `meta`, one row per window: subject_id, window_index, start_s, end_s,
    gap_frac (fraction of the window's samples that are NaN, i.e. in a gap)
    and eligible (gap_frac == 0). Windows touching a gap stay in the array
    but are not eligible for training.
    Window i covers [start_s, end_s) = [6 i, 6 i + 12) seconds from recording
    start. Only complete windows are kept; a trailing partial window is dropped.

    Windows start on whole seconds and span 12 whole seconds, so window i
    covers exactly annotation rows start_s .. end_s - 1.
    """
    if x.ndim != 2 or x.shape[0] != len(BIPOLAR_NAMES):
        raise ValueError(f"expected ({len(BIPOLAR_NAMES)}, n_samples), got {x.shape}")
    if fs != FS_OUT:
        raise ValueError(f"expected {FS_OUT} Hz, got {fs} Hz")
    n_windows = max(0, (x.shape[1] - WINDOW_SAMPLES) // STRIDE_SAMPLES + 1)
    if n_windows:
        windows = np.lib.stride_tricks.sliding_window_view(x, WINDOW_SAMPLES, axis=1)[:, ::STRIDE_SAMPLES]
        windows = windows.transpose(1, 0, 2)  # (n_windows, 18, 384)
    else:
        windows = np.empty((0, x.shape[0], WINDOW_SAMPLES), dtype=x.dtype)
        windows.flags.writeable = False
    start_samples = np.arange(n_windows) * STRIDE_SAMPLES
    nan_count = np.r_[0, np.cumsum(np.isnan(x).any(axis=0))]
    gap_frac = (nan_count[start_samples + WINDOW_SAMPLES] - nan_count[start_samples]) / WINDOW_SAMPLES
    meta = pd.DataFrame({
        "subject_id": subject_id,
        "window_index": np.arange(n_windows),
        "start_s": start_samples / fs,
        "end_s": (start_samples + WINDOW_SAMPLES) / fs,
        "gap_frac": gap_frac,
        "eligible": gap_frac == 0,
    })
    return windows, meta
