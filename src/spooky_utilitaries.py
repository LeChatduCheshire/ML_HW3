"""Data, training, evaluation, and graph utilities for the Spooky task."""

from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from spooky_classifier import SpookyClassifier, SpookyDistilBERTClassifier, SpookyTextCNN


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset" / "spooky-author-identification"
SPOOKY_BUILD_DIR = ROOT / "build" / "spooky"
RESULTS_DIR = ROOT / "results"
AUTHOR_LABELS = ("EAP", "HPL", "MWS")
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]", re.IGNORECASE)
PAD_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"


def load_labeled_csv(path: Path) -> tuple[list[str], list[str], list[str]]:
    """Read passage IDs, text, and author labels from the labeled CSV."""
    ids, texts, labels = [], [], []
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "text", "author"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Expected CSV columns {sorted(required)} in {path}")
        for row in reader:
            ids.append(row["id"])
            texts.append(row["text"] or "")
            labels.append(row["author"])
    if not texts:
        raise ValueError(f"No labeled passages found in {path}")
    return ids, texts, labels


def load_test_csv(path: Path) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "text"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Expected CSV columns {sorted(required)} in {path}")
        for row in reader:
            ids.append(row["id"])
            texts.append(row["text"] or "")
    return ids, texts


def stratified_validation_split(
    texts: list[str], labels: list[str], validation_fraction: float, seed: int
) -> tuple[list[str], list[str], list[str], list[str]]:
    return train_test_split(
        texts,
        labels,
        test_size=validation_fraction,
        random_state=seed,
        stratify=labels,
    )


def tokenize_passage(text: str) -> list[str]:
    """Lowercase and split text into words, numbers, and punctuation marks."""
    return TOKEN_PATTERN.findall(text.lower())


def build_word_vocabulary(
    training_texts: list[str], max_vocabulary_size: int
) -> tuple[dict[str, int], int]:
    """Build a training-only vocabulary and return it with the 95th length percentile."""
    tokenized = [tokenize_passage(text) for text in training_texts]
    counts = Counter(token for passage in tokenized for token in passage)
    # Counter preserves first-seen order for frequency ties, making IDs reproducible.
    vocabulary = {PAD_TOKEN: 0, UNKNOWN_TOKEN: 1}
    for token, _ in counts.most_common(max(0, max_vocabulary_size - 2)):
        vocabulary[token] = len(vocabulary)
    lengths = np.asarray([len(passage) for passage in tokenized], dtype=np.int32)
    max_sequence_length = max(1, int(np.ceil(np.percentile(lengths, 95))))
    return vocabulary, max_sequence_length


def encode_passages(
    texts: list[str], vocabulary: dict[str, int], max_sequence_length: int
) -> np.ndarray:
    """Map words to IDs, truncate long passages, and pad short passages with ID zero."""
    encoded = np.zeros((len(texts), max_sequence_length), dtype=np.int64)
    unknown_id = vocabulary[UNKNOWN_TOKEN]
    for row, text in enumerate(texts):
        token_ids = [
            vocabulary.get(token, unknown_id)
            for token in tokenize_passage(text)[:max_sequence_length]
        ]
        encoded[row, : len(token_ids)] = token_ids
    return encoded


def _text_cnn_probabilities(
    model: SpookyTextCNN,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    all_probabilities = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(features[start : start + batch_size], device=device)
            all_probabilities.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(all_probabilities, axis=0)


def text_cnn_metrics(
    model: SpookyTextCNN,
    features: np.ndarray,
    labels: list[str],
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, Any]:
    probabilities = _text_cnn_probabilities(model, features, device, batch_size)
    predictions = np.asarray(AUTHOR_LABELS)[probabilities.argmax(axis=1)]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "log_loss": float(log_loss(labels, probabilities, labels=AUTHOR_LABELS)),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=AUTHOR_LABELS
        ).tolist(),
        "class_order": list(AUTHOR_LABELS),
    }


def train_text_cnn_with_history(
    model: SpookyTextCNN,
    train_features: np.ndarray,
    train_labels: list[str],
    validation_features: np.ndarray,
    validation_labels: list[str],
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> tuple[SpookyTextCNN, list[dict[str, float]], int]:
    """Train the embedding CNN, stopping when validation log loss stalls."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    label_to_id = {label: index for index, label in enumerate(AUTHOR_LABELS)}
    train_dataset = TensorDataset(
        torch.from_numpy(train_features),
        torch.tensor([label_to_id[label] for label in train_labels]),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for token_ids, targets in loader:
            token_ids, targets = token_ids.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(token_ids), targets)
            loss.backward()
            optimizer.step()

        train_metrics = text_cnn_metrics(
            model, train_features, train_labels, device
        )
        validation_metrics = text_cnn_metrics(
            model, validation_features, validation_labels, device
        )
        row = {
            "epoch": epoch,
            "train_accuracy": train_metrics["accuracy"],
            "validation_accuracy": validation_metrics["accuracy"],
            "train_loss": train_metrics["log_loss"],
            "validation_loss": validation_metrics["log_loss"],
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train accuracy {row['train_accuracy']:.4%} "
            f"log loss {row['train_loss']:.4f} | "
            f"validation accuracy {row['validation_accuracy']:.4%} "
            f"log loss {row['validation_loss']:.4f}"
        )
        if row["validation_loss"] < best_loss:
            best_loss = row["validation_loss"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Text CNN training did not produce a best model")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def _distilbert_probabilities(
    model: SpookyDistilBERTClassifier,
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(input_ids), batch_size):
            ids = input_ids[start : start + batch_size].to(device)
            masks = attention_masks[start : start + batch_size].to(device)
            probabilities.append(
                torch.softmax(model(ids, masks), dim=1).cpu().numpy()
            )
    return np.concatenate(probabilities, axis=0)


def distilbert_metrics(
    model: SpookyDistilBERTClassifier,
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    labels: list[str],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    probabilities = _distilbert_probabilities(
        model, input_ids, attention_masks, device, batch_size
    )
    predictions = np.asarray(AUTHOR_LABELS)[probabilities.argmax(axis=1)]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "log_loss": float(log_loss(labels, probabilities, labels=AUTHOR_LABELS)),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=AUTHOR_LABELS
        ).tolist(),
        "class_order": list(AUTHOR_LABELS),
    }


def train_distilbert_with_history(
    model: SpookyDistilBERTClassifier,
    train_ids: torch.Tensor,
    train_masks: torch.Tensor,
    train_labels: list[str],
    validation_ids: torch.Tensor,
    validation_masks: torch.Tensor,
    validation_labels: list[str],
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> tuple[SpookyDistilBERTClassifier, list[dict[str, float]], int]:
    """Fine-tune DistilBERT and restore the checkpoint with best validation loss."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    label_to_id = {label: index for index, label in enumerate(AUTHOR_LABELS)}
    targets = torch.tensor([label_to_id[label] for label in train_labels])
    loader = DataLoader(
        TensorDataset(train_ids, train_masks, targets),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for ids, masks, batch_targets in loader:
            ids, masks = ids.to(device), masks.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(ids, masks), batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_metrics = distilbert_metrics(
            model, train_ids, train_masks, train_labels, device, batch_size
        )
        validation_metrics = distilbert_metrics(
            model,
            validation_ids,
            validation_masks,
            validation_labels,
            device,
            batch_size,
        )
        row = {
            "epoch": epoch,
            "train_accuracy": train_metrics["accuracy"],
            "validation_accuracy": validation_metrics["accuracy"],
            "train_loss": train_metrics["log_loss"],
            "validation_loss": validation_metrics["log_loss"],
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train accuracy {row['train_accuracy']:.4%} "
            f"log loss {row['train_loss']:.4f} | "
            f"validation accuracy {row['validation_accuracy']:.4%} "
            f"log loss {row['validation_loss']:.4f}"
        )
        if row["validation_loss"] < best_loss:
            best_loss = row["validation_loss"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
                break

    if best_state is None:
        raise RuntimeError("DistilBERT training did not produce a best model")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def classification_metrics(
    model: SpookyClassifier, features: csr_matrix, labels: list[str]
) -> dict[str, Any]:
    probabilities = model.predict_proba(features)
    predictions = model.classes_[probabilities.argmax(axis=1)]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "log_loss": float(log_loss(labels, probabilities, labels=model.classes_)),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=model.classes_
        ).tolist(),
        "class_order": [str(label) for label in model.classes_],
    }


def train_with_history(
    model: SpookyClassifier,
    train_features: csr_matrix,
    train_labels: list[str],
    validation_features: csr_matrix,
    validation_labels: list[str],
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
) -> tuple[SpookyClassifier, list[dict[str, float]], int]:
    """Train in shuffled minibatches and retain the best validation-log-loss model."""
    rng = np.random.default_rng(seed)
    classes = np.unique(train_labels)
    history: list[dict[str, float]] = []
    best_model = None
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_labels))
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            first_update = not hasattr(model.estimator, "classes_")
            model.partial_fit(
                train_features[batch_indices],
                [train_labels[index] for index in batch_indices],
                classes=classes if first_update else None,
            )

        train_metrics = classification_metrics(model, train_features, train_labels)
        validation_metrics = classification_metrics(
            model, validation_features, validation_labels
        )
        row = {
            "epoch": epoch,
            "train_accuracy": train_metrics["accuracy"],
            "validation_accuracy": validation_metrics["accuracy"],
            "train_loss": train_metrics["log_loss"],
            "validation_loss": validation_metrics["log_loss"],
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train accuracy {row['train_accuracy']:.4%} "
            f"log loss {row['train_loss']:.4f} | "
            f"validation accuracy {row['validation_accuracy']:.4%} "
            f"log loss {row['validation_loss']:.4f}"
        )

        if row["validation_loss"] < best_validation_loss:
            best_validation_loss = row["validation_loss"]
            best_epoch = epoch
            best_model = deepcopy(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
                break

    if best_model is None:
        raise RuntimeError("Training did not produce a best model")
    return best_model, history, best_epoch


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def write_submission(
    path: Path, ids: list[str], model: SpookyClassifier, features: csr_matrix
) -> None:
    probabilities = model.predict_proba(features)
    fieldnames = ["id", *[str(label) for label in model.classes_]]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row_id, scores in zip(ids, probabilities):
            writer.writerow(
                {"id": row_id, **dict(zip(fieldnames[1:], scores.tolist()))}
            )


def _read_run_histories(build_dir: Path) -> list[tuple[str, list[dict[str, float]]]]:
    runs = []
    for history_path in sorted(build_dir.glob("*/history.csv")):
        if history_path.parent.name.startswith("_"):
            continue
        with history_path.open("r", newline="", encoding="utf-8") as stream:
            rows = [
                {key: float(value) for key, value in row.items() if value}
                for row in csv.DictReader(stream)
            ]
        if rows:
            runs.append((history_path.parent.name, rows))
    return runs


def _comparison_svg(
    runs: list[tuple[str, list[dict[str, float]]]],
    title: str,
    train_key: str,
    validation_key: str,
    y_min: float,
    y_max: float,
    y_label: str,
) -> str:
    width, height = 1120, 620
    left, top, plot_width, plot_height = 90, 75, 730, 465
    x_max = max(row["epoch"] for _, rows in runs for row in rows)
    colors = ("#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c", "#0891b2")

    def point(epoch: float, value: float) -> tuple[float, float]:
        x = left + (epoch - 1) / max(1, x_max - 1) * plot_width
        y = top + (y_max - value) / (y_max - y_min) * plot_height
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.grid{stroke:#d1d5db;stroke-width:1}.axis{stroke:#374151;stroke-width:1.5}</style>',
        f'<text x="{width / 2}" y="35" text-anchor="middle" font-size="21" font-weight="bold">{html.escape(title)}</text>',
    ]
    for index in range(6):
        value = y_min + (y_max - y_min) * index / 5
        y = top + plot_height - plot_height * index / 5
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-size="12">{value:.2f}</text>')
    epoch_ticks = sorted({max(1, round(1 + (x_max - 1) * index / 5)) for index in range(6)})
    for epoch in epoch_ticks:
        x, _ = point(epoch, y_min)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_height + 23}" text-anchor="middle" font-size="12">{epoch}</text>')
    parts.extend((
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
        f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
        f'<text x="{left + plot_width / 2}" y="{height - 28}" text-anchor="middle" font-size="14">Epoch</text>',
        f'<text x="24" y="{top + plot_height / 2}" text-anchor="middle" font-size="14" transform="rotate(-90 24 {top + plot_height / 2})">{html.escape(y_label)}</text>',
    ))
    for index, (name, rows) in enumerate(runs):
        color = colors[index % len(colors)]
        for metric_key, dash in ((train_key, ""), (validation_key, "8 5")):
            points = " ".join(
                f"{point(row['epoch'], row[metric_key])[0]:.1f},{point(row['epoch'], row[metric_key])[1]:.1f}"
                for row in rows
            )
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5"{dash_attr} points="{points}"/>')
        legend_y = top + 18 + index * 42
        parts.append(f'<line x1="850" y1="{legend_y}" x2="884" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<line x1="850" y1="{legend_y + 13}" x2="884" y2="{legend_y + 13}" stroke="{color}" stroke-width="3" stroke-dasharray="8 5"/>')
        parts.append(f'<text x="892" y="{legend_y + 4}" font-size="12">{html.escape(name)} train</text>')
        parts.append(f'<text x="892" y="{legend_y + 17}" font-size="12">{html.escape(name)} val</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def generate_comparison_graphs(
    build_dir: Path = SPOOKY_BUILD_DIR, results_dir: Path = RESULTS_DIR
) -> None:
    runs = _read_run_histories(build_dir)
    if not runs:
        raise FileNotFoundError(f"No Spooky histories found under {build_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)
    accuracy_values = [
        row[key] for _, rows in runs for row in rows
        for key in ("train_accuracy", "validation_accuracy")
    ]
    accuracy_min = max(0.0, min(accuracy_values) - 0.02)
    accuracy_max = min(1.0, max(accuracy_values) + 0.02)
    loss_max = max(
        row[key] for _, rows in runs for row in rows
        for key in ("train_loss", "validation_loss")
    )
    graphs = {
        "spooky_accuracy_comparison.svg": _comparison_svg(
            runs, "Spooky training and validation accuracy", "train_accuracy",
            "validation_accuracy", accuracy_min, accuracy_max, "Accuracy",
        ),
        "spooky_loss_comparison.svg": _comparison_svg(
            runs, "Spooky training and validation log loss", "train_loss",
            "validation_loss", 0.0, loss_max * 1.05, "Log loss",
        ),
    }
    for filename, svg in graphs.items():
        graph_path = results_dir / filename
        graph_path.write_text(svg, encoding="utf-8")
        print(f"Saved comparison graph: {graph_path}")


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
