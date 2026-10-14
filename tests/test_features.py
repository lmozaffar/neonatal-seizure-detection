import numpy as np
import pytest

from helsinki.features import (FEATURE_SETS, NEAR_ZERO_UV, channel_features, extract_features, feature_table)

FS, N = 32.0, 384
T = np.arange(N) / FS
RNG = np.random.default_rng(0)


def _sine(f, amp=10.0, phase=0.0):
    return amp * np.sin(2 * np.pi * f * T + phase)


def _feats(X, feature_set="baseline"):
    names = list(feature_table(feature_set).name)
    F = channel_features(np.atleast_2d(X), feature_set)
    return [dict(zip(names, row)) for row in F]


def test_feature_tables_and_counts():
    base, sec = feature_table("baseline"), feature_table("secondary")
    assert len(base) == 55 and len(sec) == 58
    assert list(sec.name[:55]) == list(base.name)
    assert list(sec.name[55:]) == ["acf_peak", "acf_peak_lag", "spectral_peak_prominence"]
    assert base.name.is_unique and base.unit.notna().all() and base.source.notna().all()
    w = np.stack([np.stack([_sine(3) + RNG.normal(0, 1, N) for _ in range(18)]) for _ in range(2)])
    for fs_name, n in (("baseline", 55), ("secondary", 58)):
        feats, summary, valid = extract_features(w, feature_set=fs_name)
        assert feats.shape == (2, 18, n) and valid.shape == (2, 18)
        assert summary.shape == (2, 3 * n + 1)
        assert list(summary.columns[:3]) == ["line_length_median", "line_length_max", "line_length_p75"]


def test_sine_matches_analytic_values():
    f = _feats(_sine(3))[0]
    assert f["rms_amp"] == pytest.approx(10 / np.sqrt(2), rel=1e-3)
    assert f["hjorth_activity"] == pytest.approx(50, rel=1e-3)
    assert f["total_power"] == pytest.approx(50, rel=1e-3)          # Parseval: power of a 10 µV sine
    assert f["band_2_4_norm"] == pytest.approx(1, abs=1e-3)
    assert f["band_6_8"] == pytest.approx(0, abs=1e-6)
    assert f["peak_freq"] == 3.0                                     # Hz, not a bin index
    assert f["sef80"] == f["sef90"] == f["sef95"] == 3.0             # corrected: not one bin high
    assert f["hjorth_mobility"] == pytest.approx(2 * np.sin(np.pi * 3 / FS), rel=2e-3)
    assert f["kurtosis"] == pytest.approx(1.5, abs=1e-3)
    assert f["skewness"] == pytest.approx(0, abs=1e-6)
    assert f["zero_crossings"] in (71, 72)                            # 36 cycles in 12 s
    assert f["line_length"] == pytest.approx(np.abs(np.diff(_sine(3))).sum())   # whole window
    assert f["nonlinear_energy"] == pytest.approx(100 * np.sin(2 * np.pi * 3 / FS) ** 2, rel=1e-2)
    assert f["spectral_entropy"] == pytest.approx(0, abs=1e-6)        # one spectral line
    assert f["variance_d1"] == pytest.approx(np.var(np.diff(_sine(3)), ddof=1))


def test_wavelet_energy_is_the_1_to_2_hz_detail_band():
    e = {f: np.mean([_feats(_sine(f, phase=p))[0]["wavelet_energy"] for p in np.linspace(0, np.pi, 6)])
         for f in (0.5, 1.5, 4.0, 8.0)}
    assert e[1.5] > 10 * max(e[0.5], e[4.0], e[8.0])
    # Sum of squared coefficients: scales with amplitude squared.
    assert _feats(_sine(1.5, amp=20))[0]["wavelet_energy"] == pytest.approx(4 * _feats(_sine(1.5))[0]["wavelet_energy"])


def test_entropy_svd_and_ar_behave_sensibly():
    sine, noise = _feats(np.stack([_sine(3), RNG.normal(0, 10, N)]))
    assert np.isfinite(sine["shannon_entropy"]) and np.isfinite(noise["shannon_entropy"])
    assert sine["svd_entropy"] < noise["svd_entropy"]               # sine spans ~2 dimensions
    assert sine["spectral_entropy"] < noise["spectral_entropy"]
    for p in range(1, 10):
        assert np.isfinite(noise[f"ar_error_{p}"]) and noise[f"ar_error_{p}"] >= 0


def test_moddemeijer_entropy_matches_reference_formula():
    x = RNG.normal(0, 5, N)
    lo, hi = x.min(), x.max()
    d = (hi - lo) / (N - 1)
    lower, upper, ncell = lo - d / 2, hi + d / 2, int(np.ceil(np.sqrt(N)))
    idx = np.floor((x - lower) / (upper - lower) * ncell + 0.5).astype(int)
    h = np.bincount(idx[(idx >= 1) & (idx <= ncell)] - 1, minlength=ncell).astype(float)
    nz = h[h > 0]
    ref = -np.sum(nz * np.log(nz)) / h.sum() + np.log(h.sum()) + np.log((upper - lower) / ncell) + (ncell - 1) / (2 * h.sum())
    assert _feats(x)[0]["shannon_entropy"] == pytest.approx(ref)


def test_rhythm_features():
    sine, noise = _feats(np.stack([_sine(2), RNG.normal(0, 10, N)]), "secondary")
    assert sine["acf_peak"] > 0.9 and sine["acf_peak_lag"] == pytest.approx(0.5, abs=1 / FS)
    assert noise["acf_peak"] < 0.4
    assert sine["spectral_peak_prominence"] > 10 * noise["spectral_peak_prominence"]
    # Rhythm slower than the 3 s lag limit: no autocorrelation peak -> NaN, not 0.
    slow = _feats(_sine(0.2), "secondary")[0]
    assert np.isnan(slow["acf_peak"]) and np.isnan(slow["acf_peak_lag"])
    # Amplitude-independent.
    big = _feats(_sine(2, amp=100), "secondary")[0]
    assert big["acf_peak"] == pytest.approx(sine["acf_peak"])
    assert big["spectral_peak_prominence"] == pytest.approx(sine["spectral_peak_prominence"])


def test_undefined_values_are_nan_never_inf():
    X = np.stack([
        np.zeros(N),                                   # flat: MATLAB near-zero rule
        np.r_[np.zeros(N // 2), _sine(3)[N // 2:]],    # half flat
        np.full(N, 5.0),                               # constant offset: zero variance
        np.r_[_sine(3)[:100], np.full(N - 100, np.nan)],   # gap samples
        _sine(3),
    ])
    F = channel_features(X, "secondary")
    assert not np.isinf(F).any()
    assert np.isnan(F[0]).all() and np.isnan(F[1]).all() and np.isnan(F[3]).all()
    names = list(feature_table("secondary").name)
    const = dict(zip(names, F[2]))
    for n in ("hjorth_mobility", "hjorth_complexity", "kurtosis", "skewness", "shannon_entropy", "sef90",
              "acf_peak", "spectral_peak_prominence"):
        assert np.isnan(const[n]), n
    assert np.isfinite(F[4][:55]).all()
    tiny = channel_features(np.full((1, N), NEAR_ZERO_UV / 2))
    assert np.isnan(tiny).all()


def test_window_summaries_use_valid_channels_only():
    w = np.stack([np.stack([_sine(3, amp=a) for a in np.arange(1, 19)])])     # rms grows with channel
    w[0, 17] = np.nan                                                       # gap-like channel
    qc = {"bad": np.zeros((1, 18), bool)}
    qc["bad"][0, 16] = True                                                 # QC-bad channel
    feats, summary, valid = extract_features(w, qc)
    rms = feats[0, :, list(feature_table().name).index("rms_amp")]
    assert np.isnan(rms[17]) and np.isfinite(rms[16])                       # channel values kept
    assert valid[0].sum() == 16 and not valid[0, 16] and not valid[0, 17]
    used = rms[:16]
    assert summary.rms_amp_median[0] == pytest.approx(np.median(used))
    assert summary.rms_amp_max[0] == pytest.approx(used.max())
    assert summary.rms_amp_p75[0] == pytest.approx(np.percentile(used, 75))
    assert summary.n_valid_channels[0] == 16
    # A window with no valid channel gives NaN summaries, no warning-driven inf.
    feats, summary, valid = extract_features(np.full((1, 18, N), np.nan))
    assert summary.drop(columns="n_valid_channels").isna().all(axis=None) and valid.sum() == 0


def test_no_scaling_amplitude_preserved():
    f1, f2 = _feats(np.stack([_sine(3), _sine(3, amp=20)]))
    assert f2["rms_amp"] == pytest.approx(2 * f1["rms_amp"])
    assert f2["total_power"] == pytest.approx(4 * f1["total_power"])
    assert f2["band_2_4_norm"] == pytest.approx(f1["band_2_4_norm"])


def test_rejects_unknown_feature_set():
    with pytest.raises(ValueError):
        extract_features(np.zeros((1, 18, N)), feature_set="other")
    assert set(FEATURE_SETS) == {"baseline", "secondary"}
