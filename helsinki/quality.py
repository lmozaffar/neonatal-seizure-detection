"""Artifact assessment of 12 s windows: flag suspicious channels, reject only technical corruption.

Inputs are the raw recording (for the 256 Hz signals, before filtering) and the
outputs of preprocessing (x, segments, meta). Recording gaps come from
`segments`; they are never re-detected here, and gap samples are left out of
every metric (no flatlines inside gaps, no jumps across them).

Checks per bipolar channel and window (thresholds below):
  low_amp    std < 0.5 µV (preprocessed)              descriptive only
  flatline   raw bipolar unchanged for >= 1 s          bad if >= 50% of window
  clipped    either electrode at the EDF's limits      bad if >= 5% of window
  pop        raw jump > 500 µV between samples         flag only
  high_amp   peak-to-peak > 500 µV (preprocessed)      flag only
  extreme    peak-to-peak > 1000 µV (preprocessed)     flag only
and per window: filter_edge (within 2 s of a valid stretch's start or end).

Only flatline and clipping make a channel bad, and a window is unusable only
when >= 9 of 18 channels are bad. Low or suppressed neonatal EEG never makes
real samples identical for long: in eeg1/eeg57 a raw bipolar channel never
stayed unchanged for more than 7 samples, even in the quietest 1% of windows.

Flatlines are found on the whole recording, so a run spanning two windows
counts in both. Bipolar channels share electrodes, so flatline, clipping and
pops are also attributed to the responsible electrodes (electrode_qc).
"""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd

from helsinki.preprocessing import (BIPOLAR_PAIRS, ELECTRODES, FS_IN, FS_OUT, WINDOW_S, bipolar_montage,
                                    normalize_channel_name, select_eeg)

LOW_AMP_STD_UV = 0.5
FLATLINE_MIN_S = 1.0          # 36x the longest unchanged run seen in real bipolar EEG
FLATLINE_BAD_FRAC = 0.5
CLIP_BAD_FRAC = 0.05
POP_JUMP_UV = 500.0           # ~10x the largest jump inside annotated seizures (53 µV)
HIGH_AMP_PTP_UV = 500.0       # largest peak-to-peak inside annotated seizures: 296 µV
EXTREME_PTP_UV = 1000.0
FILTER_EDGE_S = 2.0
UNUSABLE_MIN_BAD_CHANNELS = 9  # provisional

_UNIT_TO_UV = {"uv": 1.0, "µv": 1.0, "mv": 1e3, "v": 1e6}


def edf_clip_limits(path) -> dict[str, tuple[float, float]]:
    """Per-electrode (low, high) saturation levels in µV from an EDF header:
    the physical value of the digital min/max, minus one quantisation step."""
    with open(path, "rb") as f:
        head = f.read(256)
        ns = int(head[252:256])
        h = f.read(256 * ns)

    def field(offset, width):  # offset in units of ns-wide blocks, as in the EDF spec
        return [h[offset * ns + i * width: offset * ns + (i + 1) * width].decode("latin-1").strip() for i in range(ns)]

    labels, dims = field(0, 16), field(96, 8)
    pmin, pmax, dmin, dmax = (list(map(float, field(o, 8))) for o in (104, 112, 120, 128))
    limits = {}
    for i, label in enumerate(labels):
        electrode = normalize_channel_name(label)
        if electrode is None:
            continue
        scale = _UNIT_TO_UV[dims[i].lower()]
        step = (pmax[i] - pmin[i]) / (dmax[i] - dmin[i])
        limits[electrode] = ((pmin[i] + step) * scale, (pmax[i] - step) * scale)
    return limits


def _runs_at_least(still: np.ndarray, min_len: int) -> np.ndarray:
    """Samples belonging to runs of >= min_len identical consecutive samples,
    given still[..., i] = (sample i == sample i + 1). Output has one more column."""
    n = still.shape[-1] + 1
    out = np.zeros(still.shape[:-1] + (n,), bool)
    for idx in np.ndindex(still.shape[:-1]):
        edges = np.diff(np.r_[0, still[idx].astype(np.int8), 0])
        a, b = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        long = b + 1 - a >= min_len  # steps a..b-1 unchanged -> samples a..b
        for lo, hi in zip(a[long], b[long]):
            out[idx][lo:hi + 1] = True
    return out


def _window_frac(mask: np.ndarray, valid: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    """Fraction of valid samples that are in `mask`, per window -> (n_windows, n_rows); NaN if none valid."""
    def wsum(m):
        c = np.concatenate([np.zeros(m.shape[:-1] + (1,)), np.cumsum(m, axis=-1)], axis=-1)
        return c[..., starts + length] - c[..., starts]
    n_valid = wsum(valid[None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        return (wsum(mask & valid) / n_valid).T


def _window_max(v: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    """Max per window over the last axis, ignoring NaN -> (n_windows, n_rows); NaN if all NaN."""
    v = np.where(np.isnan(v), -np.inf, v)
    m = np.stack([v[:, s:s + length].max(axis=-1) for s in starts]) if len(starts) else np.empty((0, v.shape[0]))
    return np.where(np.isneginf(m), np.nan, m)


def assess_windows(raw: mne.io.BaseRaw, x: np.ndarray, segments: pd.DataFrame, meta: pd.DataFrame,
                   clip_limits: dict[str, tuple[float, float]] | None = None):
    """Artifact flags and quality metrics for every window in `meta`.

    Returns (channel_qc, electrode_qc, meta):
      channel_qc   dict of (n_windows, 18) arrays in BIPOLAR_NAMES order:
                   std_uv, ptp_uv, max_jump_uv, clip_frac, flat_frac (metrics),
                   low_amp, flatline, clipped, pop, high_amp, extreme, bad (flags)
      electrode_qc dict of (n_windows, 19) arrays in ELECTRODES order:
                   clip_frac, flat_frac, max_jump_uv (metrics), clipped, flatline, pop (flags)
      meta         copy with filter_edge, n_flagged_channels, n_bad_channels, unusable
    Metrics are NaN and flags False where a window has no valid samples.
    `clip_limits` (µV per electrode) defaults to the limits in the EDF header.
    """
    if clip_limits is None:
        clip_limits = edf_clip_limits(raw.filenames[0])
    eeg = select_eeg(raw)                       # (19, n) µV at 256 Hz
    bip = bipolar_montage(eeg)                  # (18, n)
    n = eeg.shape[1]
    factor = int(FS_IN / FS_OUT)
    win_in = int(WINDOW_S * FS_IN)
    starts_in = np.round(meta.start_s.to_numpy() * FS_IN).astype(int)
    starts_out = starts_in // factor
    win_out = win_in // factor

    valid = np.ones(n, bool)                    # raw samples outside recording gaps
    for g in segments[segments.kind == "gap"].itertuples():
        valid[round(g.start_s * FS_IN):round(g.end_s * FS_IN)] = False
    step_ok = valid[:-1] & valid[1:]            # consecutive pairs inside one valid stretch

    # Flatline: runs of identical samples, on the whole recording, never through gaps.
    min_run = int(FLATLINE_MIN_S * FS_IN)
    bip_flat = _runs_at_least((np.diff(bip, axis=1) == 0) & step_ok, min_run)
    el_const = _runs_at_least((np.diff(eeg, axis=1) == 0) & step_ok, min_run)
    # Attribute each flat derivation to its constant electrode(s); if neither
    # electrode is constant on its own (identical/bridged electrodes), to both.
    pos = {e: i for i, e in enumerate(ELECTRODES)}
    el_flat = np.zeros_like(el_const)
    for d, (a, b) in enumerate(BIPOLAR_PAIRS):
        ia, ib = pos[a], pos[b]
        neither = ~el_const[ia] & ~el_const[ib]
        el_flat[ia] |= bip_flat[d] & (el_const[ia] | neither)
        el_flat[ib] |= bip_flat[d] & (el_const[ib] | neither)

    # Clipping at each electrode's own recording limits.
    low = np.array([clip_limits[e][0] for e in ELECTRODES])[:, None]
    high = np.array([clip_limits[e][1] for e in ELECTRODES])[:, None]
    el_clip = (eeg <= low) | (eeg >= high)
    bip_clip = np.stack([el_clip[pos[a]] | el_clip[pos[b]] for a, b in BIPOLAR_PAIRS])

    # Jumps between consecutive samples, only inside valid stretches.
    def jumps(sig):
        j = np.abs(np.diff(sig, axis=1))
        j[:, ~step_ok] = np.nan
        return _window_max(j, starts_in, win_in - 1)

    # Amplitude on the preprocessed windows (NaN in gaps).
    w = np.lib.stride_tricks.sliding_window_view(x, win_out, axis=1)[:, starts_out]  # (18, n_windows, 384)
    has = ~np.isnan(w).all(axis=-1)
    with np.errstate(invalid="ignore"):
        std = np.where(has, np.nanstd(np.where(has[..., None], w, 0), axis=-1), np.nan).T
    ptp = (_window_max(x, starts_out, win_out) + _window_max(-x, starts_out, win_out))

    cq = {
        "std_uv": std,
        "ptp_uv": ptp,
        "max_jump_uv": jumps(bip),
        "clip_frac": _window_frac(bip_clip, valid, starts_in, win_in),
        "flat_frac": _window_frac(bip_flat, valid, starts_in, win_in),
    }
    with np.errstate(invalid="ignore"):
        cq["low_amp"] = cq["std_uv"] < LOW_AMP_STD_UV
        cq["flatline"] = cq["flat_frac"] > 0
        cq["clipped"] = cq["clip_frac"] > 0
        cq["pop"] = cq["max_jump_uv"] > POP_JUMP_UV
        cq["high_amp"] = cq["ptp_uv"] > HIGH_AMP_PTP_UV
        cq["extreme"] = cq["ptp_uv"] > EXTREME_PTP_UV
        cq["bad"] = (cq["flat_frac"] >= FLATLINE_BAD_FRAC) | (cq["clip_frac"] >= CLIP_BAD_FRAC)

    eq = {
        "clip_frac": _window_frac(el_clip, valid, starts_in, win_in),
        "flat_frac": _window_frac(el_flat, valid, starts_in, win_in),
        "max_jump_uv": jumps(eeg),
    }
    with np.errstate(invalid="ignore"):
        eq["clipped"] = eq["clip_frac"] > 0
        eq["flatline"] = eq["flat_frac"] > 0
        eq["pop"] = eq["max_jump_uv"] > POP_JUMP_UV

    # Filter edge: window overlaps the first or last FILTER_EDGE_S of a processed valid stretch.
    start, end = meta.start_s.to_numpy(), meta.end_s.to_numpy()
    edge = np.zeros(len(meta), bool)
    for s in segments[(segments.kind == "valid") & segments.processed].itertuples():
        for lo, hi in ((s.start_s, min(s.start_s + FILTER_EDGE_S, s.end_s)),
                       (max(s.end_s - FILTER_EDGE_S, s.start_s), s.end_s)):
            edge |= (start < hi) & (end > lo)

    flagged = cq["flatline"] | cq["clipped"] | cq["pop"] | cq["high_amp"] | cq["extreme"]
    meta = meta.copy()
    meta["filter_edge"] = edge
    meta["n_flagged_channels"] = flagged.sum(axis=1)
    meta["n_bad_channels"] = cq["bad"].sum(axis=1)
    meta["unusable"] = meta["n_bad_channels"] >= UNUSABLE_MIN_BAD_CHANNELS
    return cq, eq, meta
