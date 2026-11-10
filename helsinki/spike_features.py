"""SNLEO and spike-correlation features of Tapani et al. (2019), on a 64 Hz wide-band signal.

Ten features per channel-window, ported from the public MATLAB implementation
(github.com/ktapani/Neonatal_Seizure_Detection, feats_sda.m, nlin_energy.m,
regularity.m, spike_analysis.m, xcorr_ns.m; MIT licence):

  snleo_skew         biased skewness of the smoothed nonlinear energy (SNLEO)
  snleo_regularity   SD of the SNLEO skewness over 4 s sub-windows with a 1 s step. This is
                     the second output of regularity.m, the one feats_sda.m keeps; the
                     Spearman histogram correlation (first output) is discarded there.
  spike_count        spikes: runs longer than 3 samples above an adaptive SNLEO threshold
  spike_width_ms     median spike width (NaN without spikes)
  spike_gap_ms       median gap between spikes (half the window with fewer than 2 spikes)
  spike_corr_mean    mean of the spike-to-spike correlations (0 with <= 3 spikes)
  spike_corr_std     SD of the spike-to-spike correlations (0 with <= 3 spikes)
  snleo_mean, snleo_std
  spikiness          mean spike peak / local background; max / mean of SNLEO with <= 3 spikes

The MATLAB code runs on 64 Hz signals with content up to about 30 Hz. Our main
signal (32 Hz, 0.5-12.8 Hz) would change what its sample-based constants mean, so
preprocess_wideband makes a second signal with the same electrodes, gap handling
and montage as helsinki.preprocessing, band-passed 0.5-30 Hz and downsampled to
64 Hz, cut on the same 12 s / 6 s grid.

Adapted from MATLAB: 12 s windows instead of 32 s. Reproduced from MATLAB as
written: the epoch-wise first-order 1 Hz high-pass (zero initial state), no lag-0
term in the spike correlation (xcorr_ns starts at lag 1), MATLAB quantile
(numpy "hazen") and rounding, biased skewness, N-1 standard deviations.

Undefined values are NaN, never inf; a channel-window with a gap sample or a
half-window mean |x| below 1e-4 µV gets NaN for all features, as in
helsinki.features. Input windows are preprocessed EEG in µV.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import mne
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from helsinki.features import NEAR_ZERO_UV, SUMMARY_STATS
from helsinki.preprocessing import FS_IN, STRIDE_S, WINDOW_S, bandpass, bipolar_montage, downsample, find_gaps, select_eeg

FS_WIDE = 64.0
WIDEBAND_HZ = (0.5, 30.0)
WIDE_WINDOW_SAMPLES = int(WINDOW_S * FS_WIDE)  # 768
WIDE_STRIDE_SAMPLES = int(STRIDE_S * FS_WIDE)  # 384
# Gap rounding can leave the 64 Hz signal a few samples short of the 32 Hz window grid.
MAX_SHORTFALL_SAMPLES = 4

HIGHPASS_HZ = 1.0
SNLEO_SMOOTH_SAMPLES = 7
SUBWINDOW_S, SUBSTEP_S = 4, 1
N_THRESHOLDS = 50
MIN_SPIKE_SAMPLES = 3         # a spike is a run strictly longer than this
MIN_CORR_SAMPLES = 5          # correlations use strictly more samples than this, else 0
FEW_SPIKES = 3                # at most this many spikes: correlations are 0
ALL_PAIRS_MAX_MIDPOINTS = 5   # above this many inter-spike midpoints, only segments up to 5 apart are compared
MAX_SEGMENT_DISTANCE = 5
CHUNK_ROWS = 512

SPIKE_FEATURES = ("snleo_skew", "snleo_regularity", "spike_count", "spike_width_ms", "spike_gap_ms",
                  "spike_corr_mean", "spike_corr_std", "snleo_mean", "snleo_std", "spikiness")
SPIKE_COLUMNS = [f"{name}_{stat}" for name in SPIKE_FEATURES for stat in SUMMARY_STATS]


# ---- wide-band signal --------------------------------------------------------------

def preprocess_wideband(raw: mne.io.BaseRaw) -> np.ndarray:
    """(18, n) µV at 64 Hz, 0.5-30 Hz, NaN in gaps; each valid stretch filtered on its own."""
    if raw.info["sfreq"] != FS_IN:
        raise ValueError(f"expected {FS_IN} Hz, got {raw.info['sfreq']} Hz")
    eeg = select_eeg(raw)
    gaps = find_gaps(eeg)
    bip = bipolar_montage(eeg)
    factor = int(FS_IN / FS_WIDE)
    n = bip.shape[1]
    x = np.full((bip.shape[0], -(-n // factor)), np.nan)
    bounds = [0, *(t for gap in gaps for t in gap), n]
    for i in range(0, len(bounds) - 1, 2):     # even intervals are the valid stretches
        a_out, b_out = -(-bounds[i] // factor), bounds[i + 1] // factor
        if b_out > a_out:
            stretch = bandpass(bip[:, a_out * factor:b_out * factor], band=WIDEBAND_HZ)
            x[:, a_out:b_out] = downsample(stretch, fs_out=FS_WIDE)
    return x


def wide_windows(x: np.ndarray, n_windows: int) -> np.ndarray:
    """(n_windows, 18, 768) on the grid of the 32 Hz windows; a last window a few samples short is NaN-padded."""
    if n_windows == 0:
        return np.empty((0, x.shape[0], WIDE_WINDOW_SAMPLES))
    short = (n_windows - 1) * WIDE_STRIDE_SAMPLES + WIDE_WINDOW_SAMPLES - x.shape[1]
    if short > MAX_SHORTFALL_SAMPLES:
        raise ValueError(f"64 Hz signal is {short} samples short of {n_windows} windows")
    if short > 0:
        x = np.pad(x, ((0, 0), (0, short)), constant_values=np.nan)
    view = np.lib.stride_tricks.sliding_window_view(x, WIDE_WINDOW_SAMPLES, axis=1)
    return view[:, ::WIDE_STRIDE_SAMPLES][:, :n_windows].transpose(1, 0, 2)


# ---- SNLEO --------------------------------------------------------------------------

def snleo(X: np.ndarray) -> np.ndarray:
    """(M, N) -> (M, N) smoothed nonlinear energy (nlin_energy.m): |x[n+1] x[n+2] - x[n] x[n+3]|,
    zero-extended at the end, 7-sample centred moving average, minimum subtracted."""
    M, N = X.shape
    d = np.concatenate([X, np.zeros((M, 3))], axis=1)
    energy = np.abs(d[:, 1:N + 1] * d[:, 2:N + 2] - d[:, :N] * d[:, 3:N + 3])
    extended = np.concatenate([energy, np.zeros((M, 3))], axis=1)
    smoothed = lfilter(np.ones(SNLEO_SMOOTH_SAMPLES) / SNLEO_SMOOTH_SAMPLES, 1.0, extended, axis=1)[:, 3:N + 3]
    return smoothed - smoothed.min(axis=1, keepdims=True)


def _skewness(x: np.ndarray) -> np.ndarray:
    """Biased skewness along the last axis; NaN for constant input."""
    d = x - x.mean(axis=-1, keepdims=True)
    return (d ** 3).mean(axis=-1) / (d ** 2).mean(axis=-1) ** 1.5


def _regularity(S: np.ndarray, fs: float) -> np.ndarray:
    """SD (N-1) over finite values of the skewness of 4 s sub-windows, 1 s apart; one finite value gives 0."""
    win, step = int(SUBWINDOW_S * fs), int(SUBSTEP_S * fs)
    sub = np.lib.stride_tricks.sliding_window_view(S, win, axis=1)[:, ::step]
    sk = _skewness(sub)
    sk[~np.isfinite(sk)] = np.nan
    count = np.isfinite(sk).sum(axis=1)
    out = np.full(len(S), np.nan)
    out[count == 1] = 0.0
    if (count > 1).any():
        out[count > 1] = np.nanstd(sk[count > 1], axis=1, ddof=1)
    return out


# ---- spike detection and correlation (spike_analysis.m) ------------------------------

def _crossing_counts(S: np.ndarray, th: np.ndarray) -> np.ndarray:
    """Number of runs above each of the first 49 thresholds -> (M, 49)."""
    above = S[:, None, :] > th[:, :N_THRESHOLDS - 1, None]
    return above[:, :, 0].astype(int) + np.count_nonzero(above[:, :, 1:] & ~above[:, :, :-1], axis=2)


def _adaptive_threshold(counts: np.ndarray, th: np.ndarray) -> float:
    """The level where the run count stays constant longest, among levels with more than one run."""
    falls = np.flatnonzero(np.r_[counts, 0] < np.r_[0, counts])
    if falls.size < 2:
        return float(th[-1])
    longest_first = falls[np.argsort(-np.diff(falls), kind="stable")]
    usable = longest_first[counts[longest_first] > 1]
    return float(th[usable[0]]) if usable.size else float(th[-1])


def _lagged_correlations(a: np.ndarray, b: np.ndarray, max_lag: int) -> np.ndarray:
    """Pearson correlation of a[z + i] with b[i] over all overlapping samples, z = 1..max_lag.
    0 where 5 or fewer samples overlap, NaN for constant input (xcorr_ns.m, one direction)."""
    z = np.arange(1, max_lag + 1)
    m = np.minimum(len(a) - z, len(b))
    out = np.zeros(max_lag)
    ok = m > MIN_CORR_SAMPLES
    if not ok.any():
        return out
    lag, n = z[ok], m[ok]
    ab = np.correlate(a, b, "full")[len(b) - 1 + lag]
    ca, caa = np.r_[0, np.cumsum(a)], np.r_[0, np.cumsum(a * a)]
    cb, cbb = np.r_[0, np.cumsum(b)], np.r_[0, np.cumsum(b * b)]
    sa, saa = ca[lag + n] - ca[lag], caa[lag + n] - caa[lag]
    sb, sbb = cb[n], cbb[n]
    denominator = (n * saa - sa * sa) * (n * sbb - sb * sb)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[ok] = np.where(denominator > 0, (n * ab - sa * sb) / np.sqrt(denominator), np.nan)
    return out


def _max_correlation(d1: np.ndarray, d2: np.ndarray, max_lag: int) -> float:
    """Maximum over both directions, ignoring NaN (all NaN gives NaN)."""
    return float(np.fmax.reduce(np.r_[_lagged_correlations(d1, d2, max_lag), _lagged_correlations(d2, d1, max_lag)]))


def _segment_pairs(n_midpoints: int) -> list[tuple[int, int]]:
    """Pairs of inter-spike segments compared: the first five diagonals of r in spike_analysis.m, or all pairs."""
    n_segments = n_midpoints - 1
    reach = MAX_SEGMENT_DISTANCE if n_midpoints > ALL_PAIRS_MAX_MIDPOINTS else n_segments
    return [(a, b) for a in range(n_segments) for b in range(a + 1, min(a + reach + 1, n_segments))]


def _spike_correlation_summary(values: np.ndarray, n_midpoints: int) -> tuple[float, float]:
    """(mean, SD N-1). With few midpoints MATLAB drops NaN first; otherwise a NaN propagates."""
    if n_midpoints <= ALL_PAIRS_MAX_MIDPOINTS:
        values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan, np.nan
    return float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else 0.0


def _spike_stats(s: np.ndarray, x: np.ndarray, counts: np.ndarray, th: np.ndarray, fs: float) -> list[float]:
    """[count, width ms, gap ms, correlation mean, correlation SD, spikiness] of one channel-window.
    `s` is its SNLEO, `x` the high-passed EEG."""
    level = _adaptive_threshold(counts, th)
    edges = np.diff(np.r_[0, (s > level).astype(np.int8), 0])
    start, stop = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)   # stop is one past the last sample
    keep = stop - start > MIN_SPIKE_SAMPLES
    start, stop = start[keep], stop[keep]
    n = len(start)
    width = float(np.median(stop - start)) if n else np.nan
    gap = float(np.median(start[1:] - stop[:-1])) if n > 1 else len(x) / 2
    ms = 1000.0 / fs
    if n <= FEW_SPIKES:
        return [n, width * ms, gap * ms, 0.0, 0.0, float(s.max() / s.mean()) if s.mean() else np.nan]

    # Midpoints between consecutive spikes, as 1-based MATLAB indices.
    mid = np.floor(stop[:-1] + 1 + (start[1:] - stop[:-1]) / 2 + 0.5).astype(int)
    ratios = []
    for k in range(1, len(mid)):
        background = np.r_[s[mid[k - 1] - 1:start[k] + 1], s[stop[k]:mid[k]]]
        ratios.append(s[start[k]:stop[k] + 1].max() / background.mean() if background.mean() else np.inf)
    spikiness = float(np.mean(ratios))

    segments = [x[mid[a] - 1:mid[a + 1]] for a in range(len(mid) - 1)]
    max_lag = int(np.floor(width + gap))
    values = np.array([_max_correlation(segments[a], segments[b], max_lag) for a, b in _segment_pairs(len(mid))])
    mean, sd = _spike_correlation_summary(values, len(mid))
    return [n, width * ms, gap * ms, mean, sd, spikiness]


# ---- the block ----------------------------------------------------------------------

def _block(X: np.ndarray, fs: float) -> np.ndarray:
    """(M, N) channel-windows -> (M, 10), SPIKE_FEATURES order."""
    out = np.full((len(X), len(SPIKE_FEATURES)), np.nan)
    half = X.shape[1] // 2
    ok = (np.isfinite(X).all(axis=1)
          & (np.mean(np.abs(X[:, :half]), axis=1) >= NEAR_ZERO_UV)
          & (np.mean(np.abs(X[:, half:]), axis=1) >= NEAR_ZERO_UV))
    if not ok.any():
        return out
    pole = 1 - HIGHPASS_HZ * 2 * np.pi / fs
    x = lfilter([1.0, -1.0], [1.0, -pole], X[ok], axis=1)
    S = snleo(x)
    with np.errstate(all="ignore"):
        lo, hi = np.quantile(S, [0.1, 0.9], axis=1, method="hazen")
        th = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, N_THRESHOLDS)
        counts = _crossing_counts(S, th)
        spikes = np.array([_spike_stats(S[i], x[i], counts[i], th[i], fs) for i in range(len(S))])
        out[ok] = np.column_stack([_skewness(S), _regularity(S, fs), spikes[:, :5], S.mean(axis=1),
                                   S.std(axis=1, ddof=1), spikes[:, 5]])
    out[~np.isfinite(out)] = np.nan
    return out


def spike_block(X: np.ndarray, fs: float = FS_WIDE, n_jobs: int = 1) -> np.ndarray:
    """(M, N) µV channel-windows -> (M, 10), SPIKE_FEATURES order; NaN where undefined, never inf."""
    X = np.asarray(X, float)
    chunks = [X[i:i + CHUNK_ROWS] for i in range(0, len(X), CHUNK_ROWS)]
    if not chunks:
        return np.empty((0, len(SPIKE_FEATURES)))
    if n_jobs == 1:
        return np.concatenate([_block(c, fs) for c in chunks])
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return np.concatenate(list(pool.map(partial(_block, fs=fs), chunks)))


def spike_window_summary(spike: np.ndarray, channel_valid: np.ndarray) -> pd.DataFrame:
    """Median, max and p75 of each feature across valid channels; (n_windows, 18, 10) -> SPIKE_COLUMNS."""
    masked = np.where(channel_valid[:, :, None], spike, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN windows -> NaN
        stats = {"median": np.nanmedian(masked, axis=1), "max": np.nanmax(masked, axis=1),
                 "p75": np.nanpercentile(masked, 75, axis=1)}
    return pd.DataFrame({f"{name}_{stat}": stats[stat][:, j] for j, name in enumerate(SPIKE_FEATURES)
                         for stat in SUMMARY_STATS})
