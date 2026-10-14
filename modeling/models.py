"""The classical models, each a Pipeline: median imputation -> standard scaling -> classifier.

    from modeling.models import get_models, predict_scores

Imputation and scaling sit inside the pipeline, so in cross-validation their
medians, means and standard deviations are learned from the training folds only.

Use predict_scores(model, X) rather than predict_proba: SVC has no probabilities
(probability=True would make it ~5x slower), so it returns decision_function
scores instead. Both are fine for ranking metrics such as AUROC; only
predict_proba outputs are probabilities.

SVC fits scale roughly with the square of the training size (measured: 8 s at
10,000 windows, 35 s at 20,000, ~3-4 min extrapolated to a ~50,000-window
training fold). It is therefore trained on all positives plus a random
SVC_MAX_NEGATIVES negatives; see NegativeSubsampler.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import _safe_indexing

RANDOM_STATE = 0
SVC_MAX_NEGATIVES = 20_000


def _pipeline(classifier) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", classifier)])


class NegativeSubsampler(ClassifierMixin, BaseEstimator):
    """Fit `estimator` on all positives plus at most `max_negatives` random negatives.

    Only training is subsampled; prediction uses every window it is given.
    The subsample is drawn with `random_state`, so fits are reproducible.
    """

    def __init__(self, estimator, max_negatives: int = SVC_MAX_NEGATIVES, random_state: int = RANDOM_STATE):
        self.estimator = estimator
        self.max_negatives = max_negatives
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        neg = np.flatnonzero(y == 0)
        if len(neg) > self.max_negatives:
            neg = np.random.default_rng(self.random_state).choice(neg, self.max_negatives, replace=False)
        keep = np.sort(np.r_[np.flatnonzero(y == 1), neg])
        self.n_fit_samples_ = len(keep)
        self.estimator_ = clone(self.estimator).fit(_safe_indexing(X, keep), y[keep])
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def decision_function(self, X):
        return self.estimator_.decision_function(X)


def get_models() -> dict[str, BaseEstimator]:
    """Fresh, unfitted models by name."""
    return {
        # Linear baseline: a weighted sum of features through a sigmoid; fast and interpretable.
        "LogReg": _pipeline(LogisticRegression(class_weight="balanced", max_iter=2000)),
        # Many decision trees on bootstrap samples, averaged; captures non-linear feature interactions.
        "RF": _pipeline(RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                               n_jobs=-1, random_state=RANDOM_STATE)),
        # Votes of the 25 nearest training windows in scaled feature space. KNeighborsClassifier has no
        # class_weight option, so it sees the raw ~1:6 imbalance; it is only used as a ranker here.
        "KNN": _pipeline(KNeighborsClassifier(n_neighbors=25, n_jobs=-1)),
        # Non-linear max-margin boundary with an RBF kernel; trained on a negative subsample for speed.
        "SVC": NegativeSubsampler(_pipeline(SVC(kernel="rbf", class_weight="balanced"))),
    }


def predict_scores(model, X) -> np.ndarray:
    """Seizure score per window, higher = more likely seizure.

    predict_proba[:, 1] when the model has it (a probability in [0, 1]),
    otherwise decision_function (an unbounded score, e.g. SVC's signed margin).
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, list(model.classes_).index(1)]
    return model.decision_function(X)
