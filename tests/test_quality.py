import mne
import numpy as np
import pytest

from helsinki.preprocessing import BIPOLAR_NAMES, ELECTRODES, FS_IN, make_windows, preprocess
from helsinki.quality import assess_windows, edf_clip_limits

LIMIT_UV = 5000.0
LIMITS = {e: (-LIMIT_UV, LIMIT_UV) for e in ELECTRODES}
SECONDS = 120


def _eeg(seed=0, std=10.0):
    """(19, n) µV: independent continuous noise with a DC offset per electrode."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, std, (19, int(SECONDS * FS_IN))) + np.linspace(-100, 100, 19)[:, None]


def _assess(eeg_uv):
    names = [f"EEG {e}-REF" for e in ELECTRODES] + ["ECG EKG-REF", "Resp Effort-REF"]
    data = np.vstack([eeg_uv, np.zeros((2, eeg_uv.shape[1]))]) * 1e-6
    raw = mne.io.RawArray(data, mne.create_info(names, FS_IN, "eeg"), verbose="error")
    x, segments = preprocess(raw)
    _, meta = make_windows(x, subject_id=1)
    return assess_windows(raw, x, segments, meta, clip_limits=LIMITS)


def _sl(t0, t1):
    return slice(int(t0 * FS_IN), int(t1 * FS_IN))


def _win(meta, start_s):
    return int(np.flatnonzero(meta.start_s.to_numpy() == start_s)[0])


def _ch(*names):
    return [BIPOLAR_NAMES.index(n) for n in names]


def test_clean_eeg_has_no_flags():
    cq, eq, meta = _assess(_eeg())
    for flag in ("flatline", "clipped", "pop", "high_amp", "extreme", "bad", "low_amp"):
        assert not cq[flag].any(), flag
    assert not meta.unusable.any() and (meta.n_flagged_channels == 0).all()


def test_suppressed_neonatal_eeg_is_descriptive_only():
    # Very low-amplitude background quantised to the EDF's 0.15 µV step, with
    # brief low bursts, and a reference electrode stuck at one value throughout.
    rng = np.random.default_rng(1)
    eeg = np.round(rng.normal(0, 0.3, (19, SECONDS * 256)) / 0.1526) * 0.1526
    for t in range(5, SECONDS - 5, 15):
        eeg[:, _sl(t, t + 2)] += rng.normal(0, 3, (19, 512))
    eeg[ELECTRODES.index("Cz")] = 0.0035
    cq, eq, meta = _assess(eeg)
    assert cq["low_amp"].any()
    assert not cq["flatline"].any() and not cq["bad"].any() and not meta.unusable.any()
    assert not eq["flatline"].any()  # the constant reference electrode is not blamed


def test_flatline_is_found_across_windows_and_attributed_to_the_stuck_electrodes():
    eeg = _eeg()
    f3, c3 = ELECTRODES.index("F3"), ELECTRODES.index("C3")
    eeg[[f3, c3], _sl(30, 38)] = [[12.3], [-7.1]]        # both stuck for 8 s
    eeg[[f3, c3], _sl(71.5, 72.7)] = [[-4.0], [3.0]]     # 1.2 s, across the 72 s window boundary
    cq, eq, meta = _assess(eeg)
    flat = _ch("F3-C3")
    others = [c for c in range(18) if c not in flat]
    assert cq["flat_frac"][_win(meta, 24), flat] == pytest.approx(6 / 12, abs=0.01)
    assert cq["bad"][_win(meta, 24), flat].all()         # 6 of 12 s
    assert cq["bad"][_win(meta, 30), flat].all()         # 8 of 12 s
    assert cq["flatline"][_win(meta, 36), flat].all()    # 2 of 12 s: flagged...
    assert not cq["bad"][_win(meta, 36), flat].any()     # ...but not bad
    # The 1.2 s run counts in every window it touches, even though each holds < 1 s.
    for start in (60, 66, 72):
        assert cq["flatline"][_win(meta, start), flat].all()
    assert not cq["flatline"][:, others].any()
    assert eq["flatline"][:, [f3, c3]].any(axis=0).all()
    assert not np.delete(eq["flatline"], [f3, c3], axis=1).any()


def test_single_stuck_electrode_is_not_a_bipolar_flatline():
    # Known limitation: Fp1-F3 still carries Fp1 when only F3 is stuck.
    eeg = _eeg()
    eeg[ELECTRODES.index("F3"), _sl(30, 38)] = 12.3
    cq, eq, _ = _assess(eeg)
    assert not cq["flatline"].any() and not eq["flatline"].any()


def test_identical_electrodes_blame_both():
    eeg = _eeg()
    t3, t5 = ELECTRODES.index("T3"), ELECTRODES.index("T5")
    eeg[t5, _sl(40, 50)] = eeg[t3, _sl(40, 50)]  # bridged: T3-T5 exactly 0, both still moving
    cq, eq, meta = _assess(eeg)
    assert cq["flatline"][:, _ch("T3-T5")].any()
    assert eq["flatline"][:, [t3, t5]].any(axis=0).all()


def test_clipping_uses_limits_and_marks_electrode():
    eeg = _eeg()
    o2 = ELECTRODES.index("O2")
    eeg[o2, _sl(50, 50.9)] = LIMIT_UV      # 0.9 s at the limit: 7.5% of a window
    eeg[o2, _sl(100, 100.3)] = -LIMIT_UV   # 0.3 s: 2.5%
    cq, eq, meta = _assess(eeg)
    clip = _ch("P4-O2", "T6-O2")
    assert cq["bad"][_win(meta, 42), clip].all()
    assert cq["clipped"][_win(meta, 90), clip].all() and not cq["bad"][_win(meta, 90), clip].any()
    assert not cq["clipped"][:, [c for c in range(18) if c not in clip]].any()
    assert eq["clipped"][:, o2].any() and not np.delete(eq["clipped"], o2, axis=1).any()
    assert not cq["flatline"].any()        # 0.9 s is shorter than the flatline minimum


def test_pop_is_a_flag_only():
    eeg = _eeg()
    eeg[ELECTRODES.index("C4"), _sl(60, SECONDS)] += 700   # electrode pop: 700 µV step
    cq, eq, meta = _assess(eeg)
    assert cq["pop"][:, _ch("F4-C4", "C4-P4")].any(axis=0).all()
    assert not np.delete(cq["pop"], _ch("F4-C4", "C4-P4"), axis=1).any()
    assert eq["pop"][:, ELECTRODES.index("C4")].any() and not np.delete(eq["pop"], ELECTRODES.index("C4"), axis=1).any()
    assert not cq["bad"].any() and not meta.unusable.any()


@pytest.mark.parametrize("amplitude, high_amp", [(200, False), (300, True)])  # 400 / 600 µV p-p
def test_seizure_like_rhythm_is_never_bad(amplitude, high_amp):
    eeg = _eeg()
    t = np.arange(int(10 * FS_IN)) / FS_IN
    eeg[ELECTRODES.index("T3"), _sl(20, 30)] += amplitude * np.sin(2 * np.pi * 3 * t)
    cq, _, meta = _assess(eeg)
    burst = _ch("F7-T3", "T3-T5")
    assert cq["high_amp"][_win(meta, 18), burst].all() == high_amp
    assert not cq["extreme"].any() and not cq["pop"].any()
    assert not cq["bad"].any() and not meta.unusable.any()


@pytest.mark.parametrize("electrodes, unusable", [(("Fp1", "Fp2", "O1", "O2"), False),          # 8 channels
                                                  (("Fp1", "Fp2", "O1", "O2", "Fz"), True)])   # 9 channels
def test_unusable_needs_nine_bad_channels(electrodes, unusable):
    eeg = _eeg()
    for e in electrodes:
        eeg[ELECTRODES.index(e), _sl(50, 50.9)] = LIMIT_UV
    cq, _, meta = _assess(eeg)
    w = _win(meta, 42)
    assert meta.n_bad_channels[w] == (9 if unusable else 8)
    assert meta.unusable[w] == unusable
    assert meta.unusable.sum() == (2 if unusable else 0)  # windows at 42 and 48 s


def test_gaps_are_not_flatlines_and_filter_edges_are_flagged():
    eeg = _eeg()
    eeg[:, _sl(40, 70)] = 0.0035           # recording gap [40, 70) s
    cq, eq, meta = _assess(eeg)
    assert not cq["flatline"].any() and not eq["flatline"].any()
    assert not cq["pop"].any()             # no jumps across the gap edges
    inside = _win(meta, 48)                # [48, 60) s: entirely gap
    assert np.isnan(cq["std_uv"][inside]).all() and np.isnan(cq["flat_frac"][inside]).all()
    assert not cq["bad"][inside].any() and meta.n_flagged_channels[inside] == 0
    # Edge regions: [0, 2), [38, 40), [70, 72), [118, 120).
    edge_starts = set(meta.start_s[meta.filter_edge].astype(int))
    assert edge_starts == {0, 30, 36, 60, 66, 108}


def test_edf_clip_limits_reads_header(tmp_path):
    # Minimal 2-signal EDF header: one EEG channel (µV) and one ECG channel.
    ns, labels = 2, ["EEG Fp1-REF", "ECG EKG-REF"]
    def f(vals, w):
        return "".join(str(v).ljust(w)[:w] for v in vals)
    head = "0".ljust(8) + " " * 80 + " " * 80 + "01.01.01" + "00.00.00" + str(256 + 256 * ns).ljust(8) + " " * 44 + "1".ljust(8) + "1".ljust(8) + str(ns).ljust(4)
    sig = (f(labels, 16) + f([""] * ns, 80) + f(["uV", "mV"], 8) + f([-100, -1], 8) + f([100, 1], 8)
           + f([-32768, -32768], 8) + f([32767, 32767], 8) + f([""] * ns, 80) + f([256, 256], 8) + f([""] * ns, 32))
    path = tmp_path / "x.edf"
    path.write_bytes((head + sig).encode("latin-1"))
    step = 200 / 65535
    assert edf_clip_limits(path) == {"Fp1": pytest.approx((-100 + step, 100 - step))}
