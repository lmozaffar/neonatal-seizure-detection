import numpy as np
import pytest

import modeling.run_channel_models as rc

N_BASE = len(rc.BASE_FEATURES)
SEF = rc.BASE_FEATURES.index("sef90")              # not logged
LL = rc.BASE_FEATURES.index("line_length")         # logged
OFFSET = {"B": 0, "D2": N_BASE, "T4": 2 * N_BASE}


def _series(values, channels=2):
    """(n, channels, 12) with `values` in every base column."""
    values = np.asarray(values, float)
    return np.broadcast_to(values[:, None, None], (len(values), channels, N_BASE)).copy()


def _block(base, eligible=None, valid=None, window_index=None):
    n = len(base)
    return rc.patient_temporal(base, np.arange(n) if window_index is None else np.asarray(window_index),
                               np.ones(n, bool) if eligible is None else np.asarray(eligible, bool),
                               np.ones(base.shape[:2], bool) if valid is None else np.asarray(valid, bool))


def _col(block, name, feature=SEF, channel=0):
    return block[:, channel, OFFSET[name] + feature]


def test_base_features_are_c1_columns_and_energies_are_logged():
    assert all(name in rc.C1_FEATURE_NAMES for name in rc.BASE_FEATURES) and len(rc.C1_FEATURE_NAMES) == 68
    out = _block(_series(np.arange(1, 9.0)))
    assert _col(out, "D2", SEF)[5] == pytest.approx(2.0)                       # (t) - (t-2) on the raw scale
    assert _col(out, "D2", LL)[5] == pytest.approx(np.log(6) - np.log(4))      # on the log scale
    assert np.isnan(_col(_block(_series([1, 2, 0, 4, 5])), "D2", LL)[4])      # log(0): NaN, never inf


def test_difference_and_trend_definitions():
    x = np.array([1.0, 2, 4, 7, 11, 16, 22])
    out = _block(_series(x))
    assert np.allclose(_col(out, "D2")[3:], x[3:] - x[1:-2])
    assert np.allclose(_col(out, "T4")[5:], (x[5:] - x[1:-4]) / 2)
    assert np.isnan(_col(out, "D2")[:3]).all() and np.isnan(_col(out, "T4")[:5]).all()   # window 0 is an edge


def test_baseline_pool_is_t60_to_t11_and_needs_ten_windows():
    rng = np.random.default_rng(0)
    x = rng.lognormal(size=150) + 1
    out = _col(_block(_series(x)), "B")
    for t in range(150):
        pool = x[max(1, t - 60):max(1, t - 10)]                                  # window 0 is an edge window
        expected = x[t] - np.median(pool) if len(pool) >= 10 else np.nan
        assert np.isnan(expected) and np.isnan(out[t]) or out[t] == pytest.approx(expected)


def test_guard_windows_never_enter_the_baseline():
    x = np.ones(100)
    x[80:90] = 1000.0                                                            # windows t-10 to t-1 of t = 90
    assert _col(_block(_series(x)), "B")[90] == pytest.approx(0.0)


def test_window_after_a_gap_is_an_edge_and_nothing_is_filled():
    eligible = np.ones(20, bool)
    eligible[8] = False
    out = _block(_series(np.arange(1, 21.0)), eligible=eligible)
    assert np.isnan(out[8]).all() and np.isnan(out[9]).all()                   # the gap and the window after it
    assert np.isnan(_col(out, "D2")[10]) and np.isnan(_col(out, "T4")[12])      # references are never filled


def test_invalid_channel_is_masked_on_that_channel_only():
    valid = np.ones((10, 2), bool)
    valid[4, 1] = False
    out = _block(_series(np.arange(1, 11.0)), valid=valid)
    assert np.isnan(out[4, 1]).all() and not np.isnan(_col(out, "D2", channel=0)[4])


@pytest.mark.parametrize("cut", [5, 40, 100])
def test_future_windows_cannot_change_past_features(cut):
    base = np.random.default_rng(1).lognormal(size=(150, 2, N_BASE))
    changed = base.copy()
    changed[cut:] = np.random.default_rng(2).lognormal(size=changed[cut:].shape) * 100
    eligible = np.ones(150, bool)
    eligible[cut + 3] = False                                                   # a future gap as well
    a, b = _block(base), _block(changed, eligible=eligible)
    assert np.array_equal(a[:cut], b[:cut], equal_nan=True)


def test_patients_are_computed_independently():
    rng = np.random.default_rng(3)
    c1 = rng.lognormal(size=(120, 2, 68)).astype(np.float32)
    subject = np.r_[np.full(60, 1), np.full(60, 2)]
    window = np.r_[np.arange(60), np.arange(60)]
    eligible, valid = np.ones(120, bool), np.ones((120, 2), bool)
    together = rc.temporal_features(c1, subject, window, eligible, valid)
    alone = rc.temporal_features(c1[:60], subject[:60], window[:60], eligible[:60], valid[:60])
    assert np.array_equal(together[:60], alone, equal_nan=True)
