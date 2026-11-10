import numpy as np
import pandas as pd

import modeling.events as ev


def _windows(subject, fold, scores):
    start = np.arange(len(scores)) * 6
    return pd.DataFrame({"subject_id": subject, "fold": fold, "start_s": start, "end_s": start + 12, "label": 0,
                         "eligible": True, "score": scores})


def _ann(n_seconds, seizures=()):
    """(3, n) zeros; seizures like [(start, end, n_experts)] mark [start, end) for the first n_experts experts."""
    ann = np.zeros((3, n_seconds))
    for a, b, n_experts in seizures:
        ann[:n_experts, a:b] = 1
    return ann


def test_alarm_is_causal_and_ignores_later_windows():
    scores = np.zeros(20)
    scores[5] = 1.0                                # window [30, 42), decided at 42 s
    ann = {1: _ann(130, [(36, 60, 3)])}
    tl, events = ev.build_timeline(_windows(1, 0, scores), ann)
    m = ev.match(tl, 0.5, len(events))
    assert m["event_detected"][0] and m["det_alarm_s"][0] >= 42
    later = scores.copy()
    later[8:] = 5.0                                # windows that end after the alarm
    tl2, _ = ev.build_timeline(_windows(1, 0, later), ann)
    assert ev.match(tl2, 0.5, len(events))["det_alarm_s"][0] == m["det_alarm_s"][0]


def test_min_duration_delays_alarm_and_merge_joins_runs():
    scores = np.zeros(30)
    scores[[4, 5, 6]] = 1.0                        # flagged [30, 48)
    scores[[10, 11]] = 1.0                         # flagged [66, 78)
    tl, events = ev.build_timeline(_windows(1, 0, scores), {1: _ann(200)})
    assert len(ev.match(tl, 0.5, len(events))["det_start"]) == 2
    assert len(ev.match(tl, 0.5, len(events), merge_gap_s=18)["det_start"]) == 1
    m = ev.match(tl, 0.5, len(events), min_duration_s=13)
    assert len(m["det_start"]) == 1 and m["det_alarm_s"][0] == tl.decision_s[m["det_start"][0] + 12]


def test_published_protocol_drops_short_and_partial_events():
    ann = _ann(200, [(10, 15, 3), (40, 80, 3), (100, 130, 2)])
    _, seizure, ambiguous = ev.expert_seconds(ann, ev.PUBLISHED)
    assert list(zip(*ev.runs(seizure))) == [(40, 80)]
    assert ambiguous[10:15].all() and ambiguous[100:130].all()
    _, seizure, _ = ev.expert_seconds(ann, ev.MAJORITY)
    assert len(ev.runs(seizure)[0]) == 3


def test_choice_for_a_fold_never_uses_that_fold():
    rng = np.random.default_rng(0)
    preds, ann = [], {}
    for sid in range(10):
        scores = rng.normal(size=300)
        scores[100:110] += 3
        preds.append(_windows(sid, sid % 5, scores))
        ann[sid] = _ann(300 * 6 + 12, [(600, 660, 3)])
    timelines, events = ev.build_timelines(pd.concat(preds, ignore_index=True), ann, ev.MAJORITY, {1, 3})
    curves = ev.fold_curves(timelines, events, ev.MAJORITY, [ev.BASELINE, ev.PostProcessing(3, 12, 6)])
    # The threshold grid pools all folds (a documented limitation). Given the grid, fold 0's choice may depend only
    # on the other folds' counts, so changing fold 0's own counts must leave it unchanged.
    changed = curves.copy()
    own = changed.fold == 0
    changed.loc[own, ["n_detected", "n_false_alarms"]] = rng.integers(0, 50, size=(own.sum(), 2))
    _, chosen = ev.choose(curves, 1.0)
    _, chosen_changed = ev.choose(changed, 1.0)
    pd.testing.assert_frame_equal(chosen[chosen.fold == 0].reset_index(drop=True),
                                  chosen_changed[chosen_changed.fold == 0].reset_index(drop=True))


def test_bootstrap_of_identical_models_is_zero():
    t = pd.DataFrame({"subject_id": range(6), "n_events": [3, 0, 2, 5, 1, 4], "n_detected": [1, 0, 2, 3, 0, 1],
                      "n_events_short": [1, 0, 1, 2, 1, 2], "n_detected_short": [0, 0, 1, 1, 0, 1],
                      "n_false_alarms": [1, 2, 0, 3, 1, 0], "hours": [1.0, 2, 1.5, 3, 1, 2]})
    assert all(v == 0 for v in ev.paired_bootstrap(t, t).values())
