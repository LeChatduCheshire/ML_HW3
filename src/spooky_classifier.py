"""Model definition for the Spooky Author Identification baseline."""

from __future__ import annotations

from collections.abc import Sequence

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification


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


class SpookyTextCNN(nn.Module):
    """Embedding-based text CNN with parallel word n-gram convolutions."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 128,
        filters_per_kernel: int = 128,
        dropout: float = 0.4,
        kernel_sizes: tuple[int, ...] = (2, 3, 4),
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, filters_per_kernel, kernel_size, padding=kernel_size // 2)
            for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(filters_per_kernel * len(kernel_sizes) * 2, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return author logits for a batch of padded token sequences."""
        mask = token_ids.ne(0)
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooled_features = []
        for convolution in self.convolutions:
            features = torch.relu(convolution(embedded))
            # Even-sized kernels with symmetric padding can add one extra position.
            features = features[..., : token_ids.shape[1]]
            valid_positions = mask[:, : features.shape[2]].unsqueeze(1)
            max_features = features.masked_fill(~valid_positions, torch.finfo(features.dtype).min)
            max_features = max_features.max(dim=2).values
            mean_features = (features * valid_positions).sum(dim=2) / valid_positions.sum(
                dim=2
            ).clamp_min(1)
            pooled_features.extend((max_features, mean_features))
        return self.output(self.dropout(torch.cat(pooled_features, dim=1)))


class SpookyDistilBERTClassifier(nn.Module):
    """Pretrained DistilBERT encoder fine-tuned for the three Spooky authors."""

    def __init__(self, pretrained_path: str) -> None:
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_path,
            num_labels=3,
            id2label={index: label for index, label in enumerate(("EAP", "HPL", "MWS"))},
            label2id={label: index for index, label in enumerate(("EAP", "HPL", "MWS"))},
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits
