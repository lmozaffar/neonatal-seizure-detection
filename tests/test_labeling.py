import numpy as np
import pandas as pd
import pytest

from helsinki.labeling import label_windows, load_annotations, strategy_labels


def _meta(n_windows):
    start = np.arange(n_windows) * 6.0
    return pd.DataFrame({"subject_id": 1, "window_index": np.arange(n_windows), "start_s": start, "end_s": start + 12.0,
                         "eligible": True})


def _ann(n_seconds, **marks):
    """(3, n) zeros; marks like A=[(10, 20)] set [10, 20) to 1 for expert A."""
    ann = np.zeros((3, n_seconds))
    for e, spans in marks.items():
        for a, b in spans:
            ann["ABC".index(e), a:b] = 1
    return ann


def _row(df, start):
    return df[df.start_s == start].iloc[0]


def test_conservative_label_definitions():
    # Consensus (A+B) seizure on [24, 33): 9 s. C alone marks [60, 63).
    out = label_windows(_meta(13), _ann(84, A=[(24, 33)], B=[(24, 33)], C=[(60, 63)]))
    assert _row(out, 24).consensus_s == 9 and _row(out, 24).label == 1   # [24, 36)
    assert _row(out, 18).consensus_s == 6 and _row(out, 18).label == 1   # [18, 30): exactly 6
    assert _row(out, 12).consensus_s == 0 and _row(out, 12).label == 0   # [12, 24): clean
    assert _row(out, 30).consensus_s == 3 and _row(out, 30).label == -1  # [30, 42): onset/offset 1-5 s
    assert _row(out, 54).consensus_s == 0 and _row(out, 54).label == -1  # [54, 66): C only
    assert _row(out, 54).n_experts_marking == 1 and _row(out, 54).seizure_s_C == 3


def test_majority_vote_and_agreement_counts():
    # A on [0, 6), B on [3, 9), C on [4, 5): votes 1,1,1,2,3,2,1,1,1 for s 0..8
    out = label_windows(_meta(1), _ann(12, A=[(0, 6)], B=[(3, 9)], C=[(4, 5)]))
    r = out.iloc[0]
    assert (r.seizure_s_A, r.seizure_s_B, r.seizure_s_C) == (6, 6, 1)
    assert (r.any_expert_s, r.consensus_s, r.unanimous_s) == (9, 3, 1)
    assert r.consensus_frac == pytest.approx(3 / 12) and r.n_experts_marking == 3
    assert (r.label, r.label_any_overlap, r.label_binary50) == (-1, 1, 0)


def test_missing_annotations_are_never_negative():
    ann = _ann(30, A=[(0, 8)], B=[(0, 8)])
    ann[:, 20:] = np.nan                       # recording annotated for 20 s only
    out = label_windows(_meta(4), ann)        # windows end at 12, 18, 24, 30
    w0, w1, w2, w3 = (out.iloc[i] for i in range(4))
    assert w0.label == 1 and w0.fully_annotated            # 8 consensus seconds
    assert w1.label == -1 and w1.label_binary50 == 0        # [6, 18): 2 consensus s, fully annotated
    assert not w2.fully_annotated and w2.annotated_s == 8   # [12, 24)
    assert (w2.label, w2.label_any_overlap, w2.label_binary50) == (-1, -1, -1)
    assert w3.annotated_s == 2 and w3.label == -1
    # A window entirely past the annotation matrix is unannotated, not negative.
    out = label_windows(_meta(6), _ann(20))
    assert (out.annotated_s.iloc[-1], out.label.iloc[-1]) == (0, -1)
    # Partially missing (one expert NaN) counts as unannotated for that second.
    ann = _ann(12)
    ann[1, 5] = np.nan
    r = label_windows(_meta(1), ann).iloc[0]
    assert r.annotated_s == 11 and r.label == -1


def test_positive_despite_missing_seconds_under_conservative_only():
    ann = _ann(18, A=[(6, 18)], B=[(6, 18)])
    ann[:, 15:] = np.nan
    r = label_windows(_meta(2), ann).iloc[1]   # [6, 18): 9 consensus s, 3 unannotated
    assert r.consensus_s == 9 and not r.fully_annotated
    assert (r.label, r.label_any_overlap, r.label_binary50) == (1, -1, -1)


def test_alternative_strategies():
    out = label_windows(_meta(5), _ann(36, A=[(10, 14)], B=[(10, 14)], C=[(30, 31)]))
    # [6,18) 4 consensus s; [0,12) 2; [24,36) C only; [12,24) 2; [18,30) none
    assert list(out.label_any_overlap) == [1, 1, 1, 0, 0]
    assert list(out.label_binary50) == [0, 0, 0, 0, 0]
    assert list(out.label) == [-1, -1, -1, 0, -1]


def test_labels_rebuild_from_stored_counts_and_meta_is_kept():
    rng = np.random.default_rng(0)
    ann = (rng.random((3, 400)) < 0.3).astype(float)
    ann[:, 380:] = np.nan
    out = label_windows(_meta(65), ann)
    rebuilt = strategy_labels(out.consensus_s, out.any_expert_s, out.fully_annotated)
    for name, values in rebuilt.items():
        np.testing.assert_array_equal(out[name], values)
    assert len(out) == 65 and out.eligible.all()   # no rows dropped, existing columns kept


def test_rejects_windows_not_on_whole_seconds():
    meta = _meta(2)
    meta.loc[0, "start_s"] = 0.5
    with pytest.raises(ValueError):
        label_windows(meta, _ann(30))


def test_load_annotations_shape():
    ann = load_annotations(57)
    assert ann.shape[0] == 3
    finite = np.isfinite(ann).all(axis=0)
    assert finite[:3133].all() and not finite[3133:].any()   # eeg57: 3133 s annotated
