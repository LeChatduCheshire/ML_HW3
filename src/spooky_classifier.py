"""Model definition for the Spooky Author Identification baseline."""

from __future__ import annotations

from collections.abc import Sequence

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier


class SpookyClassifier:
    """Word n-gram TF-IDF with a multiclass logistic-loss linear classifier."""

    def __init__(
        self,
        seed: int = 42,
        min_df: int = 2,
        max_features: int = 100_000,
        alpha: float = 1e-5,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
        )
        self.estimator = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            learning_rate="optimal",
            max_iter=1,
            tol=None,
            random_state=seed,
            average=True,
        )

    def fit_vectorizer(self, texts: Sequence[str]) -> csr_matrix:
        """Fit vocabulary on training passages only and return their features."""
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: Sequence[str]) -> csr_matrix:
        return self.vectorizer.transform(texts)

    def partial_fit(
        self, features: csr_matrix, labels: Sequence[str], classes: Sequence[str] | None = None
    ) -> SpookyClassifier:
        self.estimator.partial_fit(features, labels, classes=classes)
        return self

    @property
    def classes_(self):
        return self.estimator.classes_

    def predict_proba(self, features: csr_matrix):
        return self.estimator.predict_proba(features)

    def predict(self, features: csr_matrix):
        return self.estimator.predict(features)
