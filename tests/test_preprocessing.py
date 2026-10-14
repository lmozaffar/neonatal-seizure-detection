import mne
import numpy as np
import pytest

from helsinki.preprocessing import (BIPOLAR_NAMES, ELECTRODES, FS_IN, FS_OUT, MIN_RELIABLE_S, WINDOW_SAMPLES,
                                    bandpass, bipolar_montage, downsample, find_gaps, make_windows,
                                    normalize_channel_name, preprocess, select_eeg)


def _raw(names, data=None, sfreq=FS_IN):
    data = np.zeros((len(names), 512)) if data is None else data
    return mne.io.RawArray(data, mne.create_info(list(names), sfreq, "eeg"), verbose="error")


def _sine(f, fs, seconds=60.0):
    t = np.arange(int(seconds * fs)) / fs
    return np.sin(2 * np.pi * f * t)


def _rms(x):
    return np.sqrt(np.mean(x**2))


def test_channel_mapping_by_name_not_position():
    assert normalize_channel_name("EEG Fp1-REF") == "Fp1"
    assert normalize_channel_name("EEG FP1-Ref") == "Fp1"
    assert normalize_channel_name("ECG EKG-REF") is None
    assert normalize_channel_name("Resp Effort") is None

    # Shuffled order, mixed suffixes, extra non-EEG channels; each electrode
    # carries its ELECTRODES index (in volts) so we can check the row order.
    rng = np.random.default_rng(0)
    order = rng.permutation(len(ELECTRODES))
    names = [f"EEG {ELECTRODES[i]}-{'REF' if k % 2 else 'Ref'}" for k, i in enumerate(order)] + ["ECG EKG", "Resp Effort"]
    data = np.vstack([np.full((1, 512), i * 1e-6) for i in order] + [np.full((2, 512), 99e-6)])
    eeg = select_eeg(_raw(names, data))
    assert eeg.shape == (19, 512)
    np.testing.assert_allclose(eeg[:, 0], np.arange(19))  # µV, ELECTRODES order

    with pytest.raises(ValueError, match="missing"):
        select_eeg(_raw([f"EEG {e}-REF" for e in ELECTRODES if e != "Cz"]))


def test_bipolar_derivations():
    values = np.arange(1, 20, dtype=float) ** 2  # distinct per electrode
    eeg = np.repeat(values[:, None], 10, axis=1)
    bip = bipolar_montage(eeg)
    assert bip.shape == (18, 10)
    v = dict(zip(ELECTRODES, values))
    for row, name in zip(bip, BIPOLAR_NAMES):
        a, b = name.split("-")
        np.testing.assert_allclose(row, v[a] - v[b])
    assert BIPOLAR_NAMES[-2:] == ("Fz-Cz", "Cz-Pz")


def test_bandpass_passes_5hz_without_phase_shift_and_rejects_out_of_band():
    mid = slice(int(10 * FS_IN), int(50 * FS_IN))  # skip edge transients
    x = _sine(5, FS_IN)
    y = bandpass(x)
    np.testing.assert_allclose(y[mid], x[mid], atol=0.02)  # unity gain, zero lag
    assert _rms(bandpass(_sine(0.1, FS_IN))[mid]) < 0.05
    assert _rms(bandpass(_sine(30, FS_IN))[mid]) < 1e-3


def test_downsample_length_and_anti_aliasing():
    x = _sine(5, FS_IN)
    y = downsample(x)
    assert y.shape == (len(x) // 8,)
    mid = slice(int(10 * FS_OUT), int(50 * FS_OUT))
    np.testing.assert_allclose(y[mid], _sine(5, FS_OUT)[mid], atol=0.01)
    # 20 Hz would alias to 12 Hz at 32 Hz without a proper anti-alias filter.
    assert _rms(downsample(_sine(20, FS_IN))[mid]) < 1e-3


def test_preprocess_output_shape_and_units():
    n = int(30 * FS_IN)
    names = [f"EEG {e}-REF" for e in ELECTRODES] + ["ECG EKG-REF", "Resp Effort-REF"]
    data = np.zeros((21, n))
    data[ELECTRODES.index("Fz")] = 50e-6 * _sine(5, FS_IN, 30)  # 50 µV at Fz only
    out, segments = preprocess(_raw(names, data))
    assert list(segments.kind) == ["valid"] and segments.reliable.all()
    assert out.shape == (18, n // 8)
    mid = slice(int(10 * FS_OUT), int(20 * FS_OUT))
    assert _rms(out[BIPOLAR_NAMES.index("Fz-Cz"), mid]) == pytest.approx(50 / np.sqrt(2), rel=0.02)
    assert np.abs(out[BIPOLAR_NAMES.index("Fp1-F3")]).max() == 0


def _ramp(n_samples):
    """(18, n) array whose values encode (channel, sample) for content checks."""
    return np.arange(18)[:, None] * 1e6 + np.arange(n_samples)[None, :]


def test_windows_shape_count_and_trailing_drop():
    n = int(100 * FS_OUT) + 5  # 100 s plus a fraction: (3200 + 5 - 384) // 192 + 1 = 15
    windows, meta = make_windows(_ramp(n), subject_id=7)
    assert windows.shape == (15, 18, WINDOW_SAMPLES)
    assert len(meta) == 15
    assert meta["end_s"].iloc[-1] == 96.0  # next window would end at 102 s > 100.2 s
    assert not windows.flags.writeable


def test_windows_times_and_ids():
    _, meta = make_windows(_ramp(int(60 * FS_OUT)), subject_id=42)
    assert list(meta.columns) == ["subject_id", "window_index", "start_s", "end_s", "gap_frac", "eligible"]
    assert (meta["gap_frac"] == 0).all() and meta["eligible"].all()
    assert (meta["subject_id"] == 42).all()
    np.testing.assert_array_equal(meta["window_index"], np.arange(9))
    np.testing.assert_array_equal(meta["start_s"], np.arange(9) * 6.0)
    np.testing.assert_array_equal(meta["end_s"], np.arange(9) * 6.0 + 12.0)


def test_windows_content_matches_input_slices():
    x = _ramp(int(60 * FS_OUT))
    windows, meta = make_windows(x, subject_id=1)
    for w, start in zip(windows, meta["start_s"]):
        s = int(start * FS_OUT)
        np.testing.assert_array_equal(w, x[:, s:s + WINDOW_SAMPLES])


def test_windows_edge_cases():
    windows, meta = make_windows(_ramp(WINDOW_SAMPLES - 1), subject_id=3)
    assert windows.shape == (0, 18, WINDOW_SAMPLES)
    assert meta.empty
    windows, _ = make_windows(_ramp(WINDOW_SAMPLES), subject_id=3)
    assert windows.shape == (1, 18, WINDOW_SAMPLES)
    with pytest.raises(ValueError):
        make_windows(np.zeros((19, 1000)), subject_id=3)


# ---- recording gaps -------------------------------------------------------

GAP_VALUE_V = 0.0035e-6  # what the EDFs store in gaps (µV 0.0035)
NAMES_21 = [f"EEG {e}-REF" for e in ELECTRODES] + ["ECG EKG-REF", "Resp Effort-REF"]


def _eeg_with_gaps(seconds, gaps_s, seed=0):
    """(21, n) volts: 5 Hz sine on every electrode with a different amplitude
    (so every derivation carries it), small noise and a DC offset like real
    electrodes, reference electrode Cz ~0, and all-electrode constants over
    each [start, end) in gaps_s."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS_IN)
    amp = np.linspace(10, 40, 19)[:, None]
    data = np.zeros((21, n))
    offset = np.linspace(-200, 200, 19)[:, None]
    data[:19] = (offset + amp * _sine(5, FS_IN, seconds) + rng.normal(0, 1, (19, n))) * 1e-6
    data[ELECTRODES.index("Cz")] = rng.normal(0, 0.3, n) * 1e-6  # reference electrode
    for a, b in gaps_s:
        data[:19, int(a * FS_IN):int(b * FS_IN)] = GAP_VALUE_V
    return data


def test_find_gaps_separates_gaps_from_low_amplitude_eeg():
    eeg = select_eeg(_raw(NAMES_21, _eeg_with_gaps(60, [(20.3, 31.0)])))
    assert find_gaps(eeg) == [(int(20.3 * FS_IN), int(31.0 * FS_IN))]

    # Genuine low-amplitude EEG (0.5 µV noise, quantised to the EDF's 0.15 µV
    # step) and a ~0 reference electrode are not gaps.
    rng = np.random.default_rng(1)
    quiet = np.round(rng.normal(0, 0.5, (19, 60 * 256)) / 0.1526) * 0.1526
    quiet[ELECTRODES.index("Cz")] = 0.0035
    assert find_gaps(quiet) == []
    # Shorter than MIN_GAP_SAMPLES: not a gap.
    quiet[:, 1000:1010] = 0.0035
    assert find_gaps(quiet) == []


@pytest.mark.parametrize("gap", [(20.3, 31.0), (20.15, 30.6), (19.9, 31.37)])
def test_gaps_are_nan_on_original_timeline_and_valid_eeg_is_undistorted(gap):
    x, segments = preprocess(_raw(NAMES_21, _eeg_with_gaps(60, [gap])))
    assert x.shape == (18, 60 * 32)
    nan = np.isnan(x)
    assert (nan.all(axis=0) == nan.any(axis=0)).all()  # gaps are NaN on all channels together
    # NaN exactly over the gap, rounded outward to whole 32 Hz samples.
    a, b = int(gap[0] * FS_IN), int(gap[1] * FS_IN)
    assert np.flatnonzero(nan[0]).tolist() == list(range(a // 8, -(-b // 8)))
    assert list(segments.kind) == ["valid", "gap", "valid"]
    np.testing.assert_allclose(segments.start_s, [0, a / FS_IN, b / FS_IN])

    # Valid EEG up to the gap matches the same signal processed without any
    # gap. Fz-Cz carries Fz's 5 Hz sine almost alone.
    clean, _ = preprocess(_raw(NAMES_21, _eeg_with_gaps(60, [])))
    ch = BIPOLAR_NAMES.index("Fz-Cz")
    amp = np.sqrt(2) * np.std(clean[ch])
    err = np.abs(x[ch] - clean[ch])
    t = np.arange(x.shape[1]) / 32
    dist = np.minimum(np.abs(t - a / FS_IN), np.abs(t - b / FS_IN))
    valid = ~nan[ch]
    assert np.isfinite(x[:, valid]).all()
    assert err[valid & (dist < 0.25)].max() < 0.5 * amp   # unavoidable at the very edge
    assert err[valid & (dist >= 0.25) & (dist < 1)].max() < 0.15 * amp
    assert err[valid & (dist >= 1)].max() < 0.02 * amp
    # Filtering across the gap (the old behaviour) rings far more.
    old = downsample(bandpass(bipolar_montage(select_eeg(_raw(NAMES_21, _eeg_with_gaps(60, [gap]))))))
    assert np.abs(old[ch] - clean[ch])[valid & (dist < 2)].max() > 3 * err[valid & (dist < 2)].max()


def test_short_stretches_are_kept_and_flagged():
    # 2 s of valid EEG between two gaps: too short to be reliable, but kept.
    x, segments = preprocess(_raw(NAMES_21, _eeg_with_gaps(40, [(10.0, 20.0), (22.0, 30.0)])))
    short = segments[(segments.kind == "valid") & (segments.start_s == 20.0)].iloc[0]
    assert short.duration_s == 2.0 and short.processed and not short.reliable
    assert MIN_RELIABLE_S > 2.0
    assert np.isfinite(x[:, 20 * 32:22 * 32]).all()
    assert segments[segments.kind == "valid"].processed.all()
    assert not segments[segments.kind == "gap"].processed.any()


def test_gap_frac_and_eligibility():
    x = _ramp(int(60 * FS_OUT)).astype(float)
    x[:, int(20 * FS_OUT):int(26 * FS_OUT)] = np.nan  # gap [20, 26) s
    _, meta = make_windows(x, subject_id=1)
    expected = {0: 0, 6: 0, 12: 4 / 12, 18: 6 / 12, 24: 2 / 12, 30: 0, 36: 0}  # overlap with [20, 26)
    got = dict(zip(meta.start_s.astype(int), meta.gap_frac))
    for start, frac in expected.items():
        assert got[start] == pytest.approx(frac)
    assert (meta.eligible == (meta.gap_frac == 0)).all()
    assert meta.eligible.sum() == 6  # windows starting at 0, 6, 30, 36, 42, 48
