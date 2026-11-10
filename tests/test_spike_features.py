import mne
import numpy as np
import pytest
from scipy.signal import lfilter

from helsinki.preprocessing import ELECTRODES, FS_IN, make_windows, preprocess
from helsinki.spike_features import (FS_WIDE, SPIKE_COLUMNS, SPIKE_FEATURES, WIDE_WINDOW_SAMPLES, _crossing_counts,
                                     _max_correlation, _regularity, _spike_stats, preprocess_wideband, snleo, spike_block,
                                     spike_window_summary, wide_windows)

FS = int(FS_WIDE)
N = WIDE_WINDOW_SAMPLES


# ---- a loop-by-loop transliteration of the MATLAB, 1-based, as the reference ---------

def _m(x, i, j):
    """MATLAB x(i:j)."""
    return x[i - 1:j]


def _pearson(a, b):
    return np.nan if a.std() == 0 or b.std() == 0 else np.corrcoef(a, b)[0, 1]


def _matlab_xcorr_ns(d1, d2, swidth):
    p1 = np.zeros(swidth)
    for z in range(1, swidth + 1):
        dum1 = d2[z:]
        n = min(len(dum1), len(d1))
        if n > 5:
            p1[z - 1] = _pearson(dum1[:n], d1[:n])
    p2 = np.zeros(swidth)
    for z in range(1, swidth + 1):
        dum1 = d1[z:]
        n = min(len(dum1), len(d2))
        if n > 5:
            p2[z - 1] = _pearson(dum1[:n], d2[:n])
    return np.r_[p2[::-1], p1]


def _matlab_spike_analysis(snleo_x, dat, fs):
    n_s = len(snleo_x)
    th = np.linspace(*np.quantile(snleo_x, [0.1, 0.9], method="hazen"), 50)
    erf1 = np.zeros(49, int)
    for ii in range(49):
        dum = (snleo_x > th[ii]).astype(int)
        erf1[ii] = np.sum(np.diff(np.r_[0, dum, 0]) == 1)
    q1 = np.flatnonzero(np.r_[erf1, 0] < np.r_[0, erf1]) + 1          # 1-based
    th1 = th[49]
    if q1.size:
        q3 = np.argsort(-np.diff(q1), kind="stable") + 1
        candidates = [k for k, q in enumerate(q3) if erf1[q1[q - 1] - 1] > 1]
        if candidates:
            th1 = th[q1[q3[candidates[0]] - 1] - 1]
    dum = (snleo_x > th1).astype(int)
    d = np.diff(np.r_[0, dum, 0])
    r1, r2 = np.flatnonzero(d == 1) + 1, np.flatnonzero(d == -1) + 1
    qq = r2 - r1 > 3
    r1, r2 = r1[qq], r2[qq]
    swidth = np.median(r2 - r1) if len(r1) else np.nan
    snum = len(r1)
    sgap = np.median(r1[1:] - r2[:-1]) if snum > 1 else len(dat) / 2
    if len(r1) <= 3:
        return [snum, swidth / fs * 1000, sgap / fs * 1000, 0.0, 0.0, snleo_x.max() / snleo_x.mean()]
    rx1 = np.floor(r2[:-1] + (r1[1:] - r2[:-1]) / 2 + 0.5).astype(int)
    s = []
    for i in range(1, len(rx1)):
        peaks = _m(snleo_x, r1[i], r2[i]).max()
        bg = np.r_[_m(snleo_x, rx1[i - 1], r1[i]), _m(snleo_x, r2[i], rx1[i])]
        s.append(peaks / bg.mean())
    spikiness = np.mean(s)
    L = len(rx1)
    r = np.full((L, L - 2), np.nan)
    for z1 in range(1, L):
        d1 = _m(dat, rx1[z1 - 1], rx1[z1])
        lim1 = z1 + 11 if L > z1 + 11 else L
        for z2 in range(z1 + 1, lim1):
            d2 = _m(dat, rx1[z2 - 1], rx1[z2])
            r[z1 - 1, z2 - 2] = np.fmax.reduce(_matlab_xcorr_ns(d1, d2, int(np.floor(swidth + sgap))))
    if L > 5:
        rr = np.concatenate([np.diag(r, k) for k in range(5)])
    else:
        rr = r[~np.isnan(r)]
    return [snum, swidth / fs * 1000, sgap / fs * 1000, rr.mean(), rr.std(ddof=1) if rr.size > 1 else 0.0, spikiness]


def _signal(kind, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N)
    if kind == "train":
        for c in range(30, N - 30, rng.integers(25, 60)):
            x[c - 4:c + 5] += 40 * np.hanning(9)
    if kind == "few":
        for c in [150, 400, 650]:
            x[c - 4:c + 5] += 60 * np.hanning(9)
    if kind == "four":
        for c in [100, 280, 470, 650, 720]:
            x[c - 4:c + 5] += 60 * np.hanning(9)
    return lfilter([1.0, -1.0], [1.0, -(1 - 2 * np.pi / FS)], x)


@pytest.mark.parametrize("kind", ["train", "few", "four", "noise"])
@pytest.mark.parametrize("seed", range(4))
def test_spike_statistics_match_the_matlab_loops(kind, seed):
    x = _signal(kind, seed)
    s = snleo(x[None])[0]
    lo, hi = np.quantile(s, [0.1, 0.9], method="hazen")
    th = lo + (hi - lo) * np.linspace(0, 1, 50)
    counts = _crossing_counts(s[None], th[None])[0]
    np.testing.assert_allclose(_spike_stats(s, x, counts, th, FS), _matlab_spike_analysis(s, x, FS), atol=1e-9)


def test_lagged_correlation_matches_xcorr_ns_including_short_overlaps_and_long_lags():
    rng = np.random.default_rng(0)
    for la, lb, lag in [(40, 25, 30), (25, 40, 30), (12, 9, 20), (6, 40, 10)]:
        d1, d2 = rng.standard_normal(la), rng.standard_normal(lb)
        expected = np.fmax.reduce(_matlab_xcorr_ns(d1, d2, lag))
        np.testing.assert_allclose(_max_correlation(d1, d2, lag), expected, atol=1e-9)


# ---- SNLEO, regularity ---------------------------------------------------------------

def test_snleo_by_hand():
    x = np.arange(1.0, 13.0)
    extended = np.r_[x, 0, 0, 0]
    energy = np.abs(extended[1:13] * extended[2:14] - extended[:12] * extended[3:15])
    expected = np.convolve(energy, np.ones(7) / 7)[3:15]
    np.testing.assert_allclose(snleo(x[None])[0], expected - expected.min())
    assert snleo(x[None])[0].min() == 0


def test_regularity_is_sd_of_subwindow_skewness():
    s = snleo(np.random.default_rng(0).standard_normal((1, N)))
    sk = []
    for start in range(0, N - 4 * FS + 1, FS):
        v = s[0, start:start + 4 * FS]
        d = v - v.mean()
        sk.append((d ** 3).mean() / (d ** 2).mean() ** 1.5)
    assert len(sk) == 9
    np.testing.assert_allclose(_regularity(s, FS)[0], np.std(sk, ddof=1))


# ---- behaviour on signals with a known answer ----------------------------------------

def _spike_train(period, n=N, seed=0, noise=0.05):
    x = noise * np.random.default_rng(seed).standard_normal(n)
    template = 50 * np.hanning(11)
    for c in range(20, n - 20, period):
        x[c - 5:c + 6] += template
    return x


def test_regular_identical_spikes_are_detected_with_the_right_timing():
    out = spike_block(_spike_train(48)[None])[0]
    f = dict(zip(SPIKE_FEATURES, out))
    assert f["spike_count"] == pytest.approx(16, abs=1)     # (768 - 40) / 48
    assert 60 < f["spike_width_ms"] < 400
    width, gap = f["spike_width_ms"], f["spike_gap_ms"]
    assert width + gap == pytest.approx(48 / FS_WIDE * 1000, rel=0.1)
    assert f["spike_corr_mean"] > 0.9


def test_noise_has_low_spike_correlation_compared_with_a_spike_train():
    train = spike_block(_spike_train(48)[None])[0][SPIKE_FEATURES.index("spike_corr_mean")]
    noise = spike_block(np.random.default_rng(1).standard_normal((1, N)) * 20)[0][SPIKE_FEATURES.index("spike_corr_mean")]
    assert noise < train


def test_nan_rules_and_batches():
    rng = np.random.default_rng(0)
    good = rng.standard_normal(N) * 10
    gap = good.copy()
    gap[100] = np.nan
    flat_half = good.copy()
    flat_half[N // 2:] = 0.0
    out = spike_block(np.stack([good, gap, flat_half]))
    assert np.isfinite(out[0]).sum() >= 8
    assert np.isnan(out[1:]).all()
    assert out.shape == (3, 10) and not np.isinf(out).any()
    assert spike_block(np.empty((0, N))).shape == (0, 10)


def test_parallel_equals_serial():
    X = np.stack([_spike_train(40 + i, seed=i) for i in range(6)])
    np.testing.assert_array_equal(spike_block(X, n_jobs=2), spike_block(X, n_jobs=1))


def test_window_summary_uses_valid_channels_only():
    spike = np.zeros((1, 18, 10))
    spike[0, :, 0] = 1.0
    spike[0, 0, 0] = 100.0
    valid = np.ones((1, 18), bool)
    valid[0, 0] = False
    summary = spike_window_summary(spike, valid)
    assert list(summary.columns) == SPIKE_COLUMNS
    assert summary.snleo_skew_max.iloc[0] == 1.0
    assert len(SPIKE_COLUMNS) == 30


# ---- wide-band signal ----------------------------------------------------------------

def _raw(seconds, freq=20.0, gaps=()):
    t = np.arange(int(seconds * FS_IN)) / FS_IN
    data = np.zeros((19, len(t)))
    data[ELECTRODES.index("Fp2")] = 50e-6 * np.sin(2 * np.pi * freq * t)
    data += 1e-6 * np.random.default_rng(0).standard_normal(data.shape)
    for a, b in gaps:
        data[:, int(a * FS_IN):int(b * FS_IN)] = 0.0
    names = [f"EEG {e}-REF" for e in ELECTRODES]
    return mne.io.RawArray(data, mne.create_info(names, FS_IN, "eeg"), verbose="error")


def test_wideband_keeps_20hz_that_the_main_signal_removes():
    raw = _raw(60)
    wide = preprocess_wideband(raw)
    main, _ = preprocess(raw)
    assert wide.shape == (18, 3840)
    middle = (wide[0, 640:3200])
    assert np.sqrt(np.mean(middle ** 2)) > 25
    assert np.sqrt(np.mean(main[0, 320:1600] ** 2)) < 5


def test_wideband_gap_is_nan_and_windows_share_the_32hz_grid():
    raw = _raw(60, gaps=[(20.0, 21.0)])
    wide = preprocess_wideband(raw)
    assert np.isnan(wide[:, 20 * FS:21 * FS]).all()
    main, _ = preprocess(raw)
    windows, meta = make_windows(main, 1)
    cut = wide_windows(wide, len(meta))
    assert cut.shape == (len(meta), 18, N)
    # a window is NaN-free at 64 Hz whenever it is eligible at 32 Hz
    assert np.isfinite(cut[meta.eligible.to_numpy()]).all()


@pytest.mark.parametrize("seconds", [60.0, 60.03, 66.1])
def test_wide_windows_match_the_32hz_window_count(seconds):
    raw = _raw(seconds)
    _, meta = make_windows(preprocess(raw)[0], 1)
    assert len(wide_windows(preprocess_wideband(raw), len(meta))) == len(meta)


def test_wide_windows_refuse_a_large_shortfall_and_pad_a_small_one():
    x = np.ones((18, N + 384 - 2))
    padded = wide_windows(x, 2)
    assert np.isnan(padded[1, 0, -2:]).all() and np.isfinite(padded[0]).all()
    with pytest.raises(ValueError, match="short"):
        wide_windows(np.ones((18, N)), 2)
