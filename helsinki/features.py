"""Window features: an adapted Temko baseline (55) and a secondary set (55 + 3 rhythmicity).

Baseline = the 55 features of Temko et al. (2011, Clin Neurophysiol 122:464-473),
with definitions taken from the paper where it is explicit and otherwise from the
public MATLAB implementation (github.com/ktapani/Neonatal_Seizure_Detection,
detector SDA_T: feats_temko.m and feature_scripts/). It is an adaptation, not an
exact reproduction:
  * 12 s windows at 32 Hz (384 samples) instead of 8 s; spectra use an FFT of the
    window length (1/12 Hz bins) instead of 256 points.
  * our zero-phase 0.5-12.8 Hz band-pass replaces the MATLAB notch, resample and
    per-epoch first-order 0.5 Hz high-pass (not applied here).
  * corrections to the MATLAB code: line length over the whole window (MATLAB sums
    only the first fs samples), peak frequency in Hz (MATLAB returns a bin index),
    spectral edge at the bin where the cumulative power is reached (MATLAB returns
    the next bin).
  * wavelet energy follows the paper (db4, 5th of 8 coefficient sets = detail cD4,
    1-2 Hz at 32 Hz) as the sum of squared coefficients; MATLAB uses db8 and the
    mean absolute value of a slice that falls in the level-7 approximation.
MATLAB built-ins (bandpower, pyulear, pwelch, findpeaks, kurtosis, var) are
reproduced from their documented defaults; numerical identity with MATLAB has
not been verified.

Undefined values are NaN, never inf: a channel-window that contains a gap sample
or whose mean |x| is below 1e-4 µV in either half (MATLAB rule) gets NaN for all
features, and any individual undefined operation (zero variance, zero power,
no autocorrelation zero crossing, ...) gives NaN for that feature.

Input windows are preprocessed EEG in µV; no normalisation is applied.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pywt
from scipy.signal import find_peaks, peak_prominences, welch
from scipy.signal.windows import hamming, hann

from helsinki.preprocessing import BIPOLAR_NAMES, FS_OUT

NEAR_ZERO_UV = 1e-4          # MATLAB ch_temko: all features NaN below this mean |x| in a half window
BANDS_HZ = tuple((lo, lo + 2) for lo in range(0, 11))   # 0-2, 1-3, ..., 10-12 Hz
TOTAL_POWER_HZ = (0.0, 12.0)
SEF_RANGE_HZ = (0.5, 12.0)
SEF_PERCENTS = (90, 95, 80)  # MATLAB feature order
AR_MAX_ORDER = 9
AR_NFFT = 256                # MATLAB pyulear / pwelch default for these lengths
SVD_EMBED_DIM = 20
WAVELET, WAVELET_LEVEL, WAVELET_BAND = "db4", 7, 4   # coeffs [cA7, cD7, cD6, cD5, cD4, ...]: index 4 = cD4
ACF_MAX_LAG_S = 3.0
PROMINENCE_SEGMENT_S = 4.0
PROMINENCE_RANGE_HZ = (0.5, 12.0)
SUMMARY_STATS = ("median", "max", "p75")

# name, unit, source, definition
_TEMKO = [
    ("line_length", "µV", "paper (curve length); corrected from MATLAB", "sum |x[n+1]-x[n]| over the whole window"),
    ("wavelet_energy", "µV²", "paper (db4, 1-2 Hz); energy definition ours", "sum of squared db4 detail coefficients cD4 (level 4 of 7, symmetric extension)"),
    ("min_max", "count", "MATLAB", "number of local maxima plus local minima (plateaus counted once)"),
    ("rms_amp", "µV", "MATLAB", "root mean square amplitude"),
    ("shannon_entropy", "nats", "MATLAB", "Moddemeijer unbiased histogram entropy, ceil(sqrt(N)) cells over [min-d/2, max+d/2], d=(max-min)/(N-1)"),
    ("peak_freq", "Hz", "paper; corrected from MATLAB (bin index)", "frequency of the periodogram maximum (lowest if tied)"),
    ("hjorth_activity", "µV²", "MATLAB", "population variance"),
    ("hjorth_mobility", "-", "MATLAB", "std(dx)/std(x), population std"),
    ("hjorth_complexity", "-", "MATLAB", "(std(ddx)/std(dx)) / mobility"),
    ("nonlinear_energy", "µV²", "MATLAB", "mean(x[n]^2 - x[n-1] x[n+1])"),
    ("spectral_entropy", "bits", "MATLAB", "log2 Shannon entropy of the periodogram normalised to sum 1, all bins 0-16 Hz"),
    ("zero_crossings", "count", "MATLAB", "number of n with x[n] x[n+1] < 0"),
    *[(f"ar_error_{p}", "%", "MATLAB", f"100 mean|P_ar - P_welch| / max of their means; Yule-Walker AR({p}) PSD of the "
       "first half vs Welch PSD of the second half, 256-point, MATLAB defaults") for p in range(1, AR_MAX_ORDER + 1)],
    ("total_power", "µV²", "paper/MATLAB", "bandpower 0-12 Hz: Hamming-window periodogram, rectangle sum over bins in range"),
    *[(f"band_{lo}_{hi}", "µV²", "paper/MATLAB", f"bandpower {lo}-{hi} Hz") for lo, hi in BANDS_HZ],
    *[(f"band_{lo}_{hi}_norm", "-", "paper/MATLAB", f"band_{lo}_{hi} / total_power") for lo, hi in BANDS_HZ],
    *[(f"sef{p}", "Hz", "paper; corrected from MATLAB (one bin high)", f"frequency where cumulative periodogram power in 0.5-12 Hz first reaches {p}%")
      for p in SEF_PERCENTS],
    ("kurtosis", "-", "MATLAB", "biased, non-excess kurtosis (normal = 3)"),
    ("skewness", "-", "MATLAB", "biased skewness"),
    ("svd_entropy", "bits", "MATLAB", "log2 entropy of normalised singular values of the delay embedding (dim 20, lag 1, N-20 vectors)"),
    ("fisher_info", "-", "MATLAB", "sum_i (s[i+1]-s[i])^2 / s[i] over the normalised singular values"),
    ("zero_crossings_d1", "count", "MATLAB", "zero crossings of the first difference"),
    ("zero_crossings_d2", "count", "MATLAB", "zero crossings of the second difference"),
    ("variance_d1", "(µV/sample)²", "MATLAB", "variance (N-1) of the first difference"),
    ("variance_d2", "(µV/sample)²", "MATLAB", "variance (N-1) of the second difference"),
]
_RHYTHM = [
    ("acf_peak", "-", "new", "highest local maximum of the normalised autocorrelation between its first zero crossing and 3 s; NaN if none"),
    ("acf_peak_lag", "s", "new", "lag of acf_peak"),
    ("spectral_peak_prominence", "-", "new", "prominence of the highest Welch-PSD peak in 0.5-12 Hz (4 s Hann, 50% overlap, segment mean removed) / mean PSD in 0.5-12 Hz; NaN if no peak"),
]
FEATURE_SETS = {"baseline": _TEMKO, "secondary": _TEMKO + _RHYTHM}


def feature_table(feature_set: str = "baseline") -> pd.DataFrame:
    """Name, unit, source and definition of every per-channel feature, in column order."""
    return pd.DataFrame(FEATURE_SETS[feature_set], columns=["name", "unit", "source", "definition"])


# ---- per-row helpers: X is (M, N), one channel-window per row ------------------

def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b != 0, a / np.where(b != 0, b, 1), np.nan)


def _xlogx(p, log):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > 0, p * log(np.where(p > 0, p, 1)), 0.0)


def _zero_crossings(X):
    return np.sum(X[:, :-1] * X[:, 1:] < 0, axis=1).astype(float)


def _psd(X, fs, window):
    """One-sided PSD, FFT length N, MATLAB periodogram scaling (µV²/Hz); window None = rectangular."""
    n = X.shape[1]
    w = window(n) if window is not None else np.ones(n)
    P = np.abs(np.fft.rfft(X * w, axis=1)) ** 2 / (fs * np.sum(w ** 2))
    P[:, 1:-1 if n % 2 == 0 else None] *= 2
    return np.arange(n // 2 + 1) * fs / n, P   # exact: 12 Hz is bin 144 of 384 at 32 Hz


def _bandpower(f, P, lo, hi):
    """MATLAB bandpower(Pxx, F, [lo hi], 'psd'): rectangle widths diff(F) (last 0), bins
    from the last F <= lo to the first F >= hi."""
    width = np.r_[np.diff(f), 0.0]
    i1 = np.flatnonzero(f <= lo)[-1]
    i2 = np.flatnonzero(f >= hi)[0]
    return P[:, i1:i2 + 1] @ width[i1:i2 + 1]


def _moddemeijer_entropy(X):
    m, n = X.shape
    ncell = int(np.ceil(np.sqrt(n)))
    lo, hi = X.min(axis=1), X.max(axis=1)
    delta = (hi - lo) / (n - 1)
    lower, upper = lo - delta / 2, hi + delta / 2
    rng = upper - lower
    ok = rng > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        idx = np.floor((X - lower[:, None]) / rng[:, None] * ncell + 0.5).astype(np.int64)  # MATLAB round, values > 0
    idx = np.where(ok[:, None], idx, 1)
    inside = (idx >= 1) & (idx <= ncell)
    h = np.zeros((m, ncell))
    rows = np.repeat(np.arange(m), n)
    np.add.at(h, (rows[inside.ravel()], idx.ravel()[inside.ravel()] - 1), 1)
    count = h.sum(axis=1)
    est = -_xlogx(h, np.log).sum(axis=1) / count + np.log(count) + np.log(_safe_div(rng, ncell))
    est = est + (ncell - 1) / (2 * count)  # unbiased: remove N-bias -(ncell-1)/(2 count)
    return np.where(ok, est, np.nan)


def _count_extrema(X):
    return np.array([len(find_peaks(x)[0]) + len(find_peaks(-x)[0]) for x in X], float)


def _levinson(r, order):
    """Yule-Walker via Levinson-Durbin on rows of autocorrelations r (M, order+1).
    Returns [(a (M, p+1), noise variance (M,)) for p = 1..order]."""
    m = r.shape[0]
    a = np.zeros((m, order + 1))
    a[:, 0] = 1
    e = r[:, 0].copy()
    out = []
    for k in range(1, order + 1):
        acc = r[:, k] + np.sum(a[:, 1:k] * r[:, k - 1:0:-1], axis=1)
        kk = _safe_div(-acc, e)
        a_prev = a.copy()
        a[:, k] = kk
        a[:, 1:k] = a_prev[:, 1:k] + kk[:, None] * a_prev[:, k - 1:0:-1]
        e = e * (1 - kk ** 2)
        out.append((a[:, :k + 1].copy(), e.copy()))
    return out


def _ar_errors(X):
    """MATLAB ar_prediction_error_newer: AR(p) Yule-Walker PSD (pyulear) of the first
    half vs Welch PSD (pwelch defaults) of the second half, for p = 1..9."""
    n = X.shape[1]
    model, test = X[:, :n // 2], X[:, -(-n // 2):]
    # pwelch default: 8 segments, 50% overlap, symmetric Hamming, nfft max(256, 2^nextpow2(L))
    seg = test.shape[1] * 2 // 9   # fix(N / 4.5)
    nfft = max(AR_NFFT, 2 ** int(np.ceil(np.log2(seg))))
    _, p_welch = welch(test, fs=2 * np.pi, window=hamming(seg, sym=True), nperseg=seg, noverlap=seg // 2,
                       nfft=nfft, detrend=False, scaling="density", axis=1)
    # pyulear: biased autocorrelation, Levinson, PSD = e / |A|^2 per rad/sample, one-sided
    nm = model.shape[1]
    r = np.stack([np.sum(model[:, :nm - k] * model[:, k:], axis=1) / nm for k in range(AR_MAX_ORDER + 1)], axis=1)
    errors = []
    for a, e in _levinson(r, AR_MAX_ORDER):
        A = np.abs(np.fft.rfft(a, n=nfft, axis=1)) ** 2
        p_ar = _safe_div(e[:, None], A) / (2 * np.pi)
        p_ar[:, 1:-1] *= 2
        p_ar = np.where(np.isnan(p_ar).any(axis=1, keepdims=True), 0.0, p_ar)  # MATLAB: NaN PSD -> zeros
        err = np.mean(np.abs(p_ar - p_welch), axis=1)
        errors.append(100 * _safe_div(err, np.maximum(p_ar.mean(axis=1), p_welch.mean(axis=1))))
    return errors


def _svd_features(X, chunk=2048):
    n = X.shape[1]
    n_vec = n - SVD_EMBED_DIM
    ent, fisher = [], []
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk]
        Y = np.lib.stride_tricks.sliding_window_view(x, SVD_EMBED_DIM, axis=1)[:, :n_vec]   # (m, n_vec, 20)
        s = np.linalg.svd(Y, compute_uv=False)
        s = s / s.sum(axis=1, keepdims=True)
        ent.append(-_xlogx(s, np.log2).sum(axis=1))
        fisher.append(np.sum(_safe_div((s[:, 1:] - s[:, :-1]) ** 2, s[:, :-1]), axis=1))
    return np.concatenate(ent), np.concatenate(fisher)


def _wavelet_energy(X):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # level 7 > max level; cD4 itself is unaffected
        coeffs = pywt.wavedec(X, WAVELET, mode="symmetric", level=WAVELET_LEVEL, axis=1)
    return np.sum(coeffs[WAVELET_BAND] ** 2, axis=1)


def _temko(X, fs):
    n = X.shape[1]
    d1, d2 = np.diff(X, axis=1), np.diff(X, n=2, axis=1)
    std0, std1, std2 = X.std(axis=1), d1.std(axis=1), d2.std(axis=1)
    mobility = _safe_div(std1, std0)

    f, P = _psd(X, fs, None)                     # periodogram, rectangular window
    f_h, P_h = _psd(X, fs, lambda k: hamming(k, sym=True))   # bandpower's modified periodogram
    peak = f[np.argmax(P, axis=1)]
    pdf = _safe_div(P, P.sum(axis=1, keepdims=True))
    spec_ent = -_xlogx(pdf, np.log2).sum(axis=1)
    total = _bandpower(f_h, P_h, *TOTAL_POWER_HZ)
    bands = [_bandpower(f_h, P_h, lo, hi) for lo, hi in BANDS_HZ]
    sel = (f >= SEF_RANGE_HZ[0]) & (f <= SEF_RANGE_HZ[1])
    cum = np.cumsum(P[:, sel], axis=1)
    sefs = [np.where(cum[:, -1] > 0, f[sel][np.argmax(cum >= p / 100 * cum[:, -1:], axis=1)], np.nan)
            for p in SEF_PERCENTS]

    xc = X - X.mean(axis=1, keepdims=True)
    m2 = np.mean(xc ** 2, axis=1)
    svd_ent, fisher = _svd_features(X)
    cols = [
        np.abs(d1).sum(axis=1),
        _wavelet_energy(X),
        _count_extrema(X),
        np.sqrt(np.mean(X ** 2, axis=1)),
        _moddemeijer_entropy(X),
        peak,
        std0 ** 2,
        mobility,
        _safe_div(_safe_div(std2, std1), mobility),
        np.mean(X[:, 1:-1] ** 2 - X[:, :-2] * X[:, 2:], axis=1),
        spec_ent,
        _zero_crossings(X),
        *_ar_errors(X),
        total,
        *bands,
        *[_safe_div(b, total) for b in bands],
        *sefs,
        _safe_div(np.mean(xc ** 4, axis=1), m2 ** 2),
        _safe_div(np.mean(xc ** 3, axis=1), m2 ** 1.5),
        svd_ent,
        fisher,
        _zero_crossings(d1),
        _zero_crossings(d2),
        d1.var(axis=1, ddof=1),
        d2.var(axis=1, ddof=1),
    ]
    return np.stack(cols, axis=1)


def _rhythm(X, fs):
    m, n = X.shape
    xc = X - X.mean(axis=1, keepdims=True)
    max_lag = int(ACF_MAX_LAG_S * fs)
    spec = np.fft.rfft(xc, n=2 * n, axis=1)
    acf = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :max_lag + 1]
    acf = _safe_div(acf, acf[:, :1])
    acf_peak = np.full(m, np.nan)
    acf_lag = np.full(m, np.nan)
    for i, r in enumerate(acf):
        below = np.flatnonzero(r <= 0)
        if not np.isfinite(r).all() or below.size == 0:
            continue
        k0 = below[0]
        peaks, _ = find_peaks(r[k0:])
        if peaks.size:
            best = peaks[np.argmax(r[k0:][peaks])]
            acf_peak[i], acf_lag[i] = r[k0 + best], (k0 + best) / fs

    seg = int(PROMINENCE_SEGMENT_S * fs)
    f, P = welch(X, fs=fs, window=hann(seg, sym=False), nperseg=seg, noverlap=seg // 2, nfft=seg,
                 detrend="constant", scaling="density", axis=1)
    band = (f >= PROMINENCE_RANGE_HZ[0]) & (f <= PROMINENCE_RANGE_HZ[1])
    prominence = np.full(m, np.nan)
    for i, p in enumerate(P):
        peaks, _ = find_peaks(p)
        peaks = peaks[band[peaks]]
        mean_p = p[band].mean()
        if peaks.size and mean_p > 0:
            top = peaks[np.argmax(p[peaks])]
            prominence[i] = peak_prominences(p, [top])[0][0] / mean_p
    return np.stack([acf_peak, acf_lag, prominence], axis=1)


def channel_features(X: np.ndarray, feature_set: str = "baseline", fs: float = FS_OUT) -> np.ndarray:
    """(M, N) µV channel-windows -> (M, n_features); NaN where undefined, never inf."""
    X = np.asarray(X, float)
    n_feat = len(FEATURE_SETS[feature_set])
    out = np.full((X.shape[0], n_feat), np.nan)
    half = X.shape[1] // 2
    ok = (np.isfinite(X).all(axis=1)
          & (np.mean(np.abs(X[:, :half]), axis=1) >= NEAR_ZERO_UV)
          & (np.mean(np.abs(X[:, half:]), axis=1) >= NEAR_ZERO_UV))
    if ok.any():
        Xo = X[ok]
        with np.errstate(all="ignore"):
            feats = [_temko(Xo, fs)]
            if feature_set == "secondary":
                feats.append(_rhythm(Xo, fs))
        out[ok] = np.concatenate(feats, axis=1)
    out[~np.isfinite(out)] = np.nan
    return out


def extract_features(windows: np.ndarray, channel_qc: dict | None = None, feature_set: str = "baseline"):
    """Features for every window and channel.

    windows     (n_windows, 18, 384) preprocessed µV (from make_windows)
    channel_qc  optional output of assess_windows; channels flagged `bad` are left
                out of the window summaries (their channel-level values are kept)

    Returns (channel_feats, window_feats, channel_valid):
      channel_feats  (n_windows, 18, n_features) array, columns as feature_table(feature_set)
      window_feats   DataFrame: <feature>_median, <feature>_max, <feature>_p75 across valid
                     channels, plus n_valid_channels (metadata, not a predictor)
      channel_valid  (n_windows, 18) bool: channel has features and is not QC-bad
    A channel is valid for a summary of a given feature only if that feature is finite.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {list(FEATURE_SETS)}")
    n_w, n_ch, n_s = windows.shape
    if n_ch != len(BIPOLAR_NAMES):
        raise ValueError(f"expected {len(BIPOLAR_NAMES)} channels, got {n_ch}")
    names = [f[0] for f in FEATURE_SETS[feature_set]]
    feats = channel_features(windows.reshape(n_w * n_ch, n_s), feature_set).reshape(n_w, n_ch, len(names))

    channel_valid = np.isfinite(feats).any(axis=2)
    if channel_qc is not None:
        channel_valid &= ~np.asarray(channel_qc["bad"], bool)
    masked = np.where(channel_valid[:, :, None], feats, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN windows -> NaN
        stats = {"median": np.nanmedian(masked, axis=1), "max": np.nanmax(masked, axis=1),
                 "p75": np.nanpercentile(masked, 75, axis=1)}
    window_feats = pd.DataFrame({f"{name}_{stat}": stats[stat][:, j] for j, name in enumerate(names)
                                 for stat in SUMMARY_STATS})
    window_feats["n_valid_channels"] = channel_valid.sum(axis=1)
    return feats, window_feats, channel_valid
