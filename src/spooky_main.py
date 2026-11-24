"""Train a simple probabilistic TF-IDF classifier for Spooky author identification."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoTokenizer

from spooky_classifier import SpookyClassifier, SpookyDistilBERTClassifier, SpookyTextCNN
from spooky_utilitaries import (
    AUTHOR_LABELS,
    DATASET_DIR,
    ROOT,
    SPOOKY_BUILD_DIR,
    classification_metrics,
    distilbert_metrics,
    build_word_vocabulary,
    encode_passages,
    generate_comparison_graphs,
    load_labeled_csv,
    load_test_csv,
    stratified_validation_split,
    text_cnn_metrics,
    train_text_cnn_with_history,
    train_distilbert_with_history,
    train_with_history,
    write_history,
    write_metrics,
    write_submission,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=SPOOKY_BUILD_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument(
        "--model-type",
        choices=("tfidf", "textcnn", "distilbert"),
        default="tfidf",
        help="Choose the TF-IDF baseline or the word-ID embedding CNN.",
    )
    parser.add_argument("--max-vocabulary-size", type=int, default=50_000)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--filters-per-kernel", type=int, default=128)
    parser.add_argument(
        "--pretrained-model",
        type=Path,
        default=ROOT / "build" / "spooky" / "foundation_models" / "distilbert-base-uncased",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_only:
        generate_comparison_graphs(args.output_dir)
        return
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch size, and patience must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in (0, 1)")
    if args.alpha <= 0 or args.max_features < 1:
        raise ValueError("alpha and max features must be positive")

    train_ids, texts, labels = load_labeled_csv(args.data_dir / "train.csv")
    test_ids, test_texts = load_test_csv(args.data_dir / "test.csv")
    train_texts, validation_texts, train_labels, validation_labels = (
        stratified_validation_split(
            texts, labels, args.validation_fraction, args.seed
        )
    )
    print(f"Labeled passages: {len(texts)}; validation passages: {len(validation_texts)}")
    print(f"Training author counts: {dict(Counter(train_labels))}")
    print(f"Official test passages: {len(test_texts)}")

    if args.model_type == "textcnn":
        run_text_cnn(
            args,
            texts,
            labels,
            train_texts,
            validation_texts,
            train_labels,
            validation_labels,
            test_ids,
            test_texts,
        )
        return
    if args.model_type == "distilbert":
        run_distilbert(
            args,
            texts,
            labels,
            train_texts,
            validation_texts,
            train_labels,
            validation_labels,
            test_ids,
            test_texts,
        )
        return

    model = SpookyClassifier(seed=args.seed, alpha=args.alpha, max_features=args.max_features)
    train_features = model.fit_vectorizer(train_texts)
    validation_features = model.transform(validation_texts)
    best_model, history, best_epoch = train_with_history(
        model,
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        args.epochs,
        args.batch_size,
        args.patience,
        args.seed,
    )

    architecture = "tfidf_word_1-2gram_sgd_logistic"
    run_dir = args.output_dir / f"{architecture}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model.joblib"
    history_path = run_dir / "history.csv"
    metrics_path = run_dir / "metrics.json"
    submission_path = run_dir / "submission.csv"
    joblib.dump(best_model, model_path)
    write_history(history_path, history)

    training_metrics = classification_metrics(
        best_model, train_features, train_labels
    )
    validation_metrics = classification_metrics(
        best_model, validation_features, validation_labels
    )
    test_features = best_model.transform(test_texts)
    write_submission(submission_path, test_ids, best_model, test_features)
    author_counts = dict(Counter(labels))
    metrics = {
        "architecture": {
            "name": "TF-IDF word unigram/bigram + SGD logistic loss",
            "vocabulary_size": len(best_model.vectorizer.vocabulary_),
            "classes": [str(label) for label in best_model.classes_],
        },
        "settings": {
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "early_stopping_patience": args.patience,
            "alpha": args.alpha,
            "max_features": args.max_features,
            "word_ngram_range": [1, 2],
            "min_df": 2,
        },
        "dataset": {
            "labeled_rows": len(texts),
            "train_rows": len(train_texts),
            "validation_rows": len(validation_texts),
            "official_test_rows": len(test_texts),
            "author_counts": author_counts,
        },
        "train": training_metrics,
        "validation": validation_metrics,
        "artifacts": {
            "model": str(model_path.relative_to(ROOT)),
            "history": str(history_path.relative_to(ROOT)),
            "submission": str(submission_path.relative_to(ROOT)),
        },
    }
    write_metrics(metrics_path, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {model_path}")
    print(f"Saved training history: {history_path}")
    print(f"Saved Kaggle probability submission: {submission_path}")
    generate_comparison_graphs(args.output_dir)


def run_text_cnn(
    args: argparse.Namespace,
    all_texts: list[str],
    all_labels: list[str],
    train_texts: list[str],
    validation_texts: list[str],
    train_labels: list[str],
    validation_labels: list[str],
    test_ids: list[str],
    test_texts: list[str],
) -> None:
    """Train, evaluate, and save the word-ID embedding CNN variant."""
    if args.max_vocabulary_size < 2:
        raise ValueError("--max-vocabulary-size must be at least 2")
    if args.embedding_dim < 1 or args.filters_per_kernel < 1:
        raise ValueError("embedding dimension and convolution filters must be positive")

    vocabulary, max_sequence_length = build_word_vocabulary(
        train_texts, args.max_vocabulary_size
    )
    train_features = encode_passages(train_texts, vocabulary, max_sequence_length)
    validation_features = encode_passages(
        validation_texts, vocabulary, max_sequence_length
    )
    test_features = encode_passages(test_texts, vocabulary, max_sequence_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"TextCNN vocabulary: {len(vocabulary):,} tokens; "
        f"sequence length: {max_sequence_length}; device: {device}"
    )
    model = SpookyTextCNN(
        vocabulary_size=len(vocabulary),
        embedding_dim=args.embedding_dim,
        filters_per_kernel=args.filters_per_kernel,
    )
    best_model, history, best_epoch = train_text_cnn_with_history(
        model,
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        args.epochs,
        args.batch_size,
        args.patience,
        args.seed,
        device,
    )

    architecture = "embedding_textcnn"
    run_dir = args.output_dir / (
        f"{architecture}_vocab{args.max_vocabulary_size}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model.pt"
    history_path = run_dir / "history.csv"
    metrics_path = run_dir / "metrics.json"
    vocabulary_path = run_dir / "vocabulary.json"
    submission_path = run_dir / "submission.csv"
    torch.save(
        {key: value.detach().cpu() for key, value in best_model.state_dict().items()},
        model_path,
    )
    vocabulary_path.write_text(json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8")
    write_history(history_path, history)
    training_metrics = text_cnn_metrics(best_model, train_features, train_labels, device)
    validation_metrics = text_cnn_metrics(
        best_model, validation_features, validation_labels, device
    )
    probabilities = best_model_probabilities(best_model, test_features, device)
    with submission_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", *AUTHOR_LABELS])
        writer.writeheader()
        for row_id, scores in zip(test_ids, probabilities):
            writer.writerow({"id": row_id, **dict(zip(AUTHOR_LABELS, scores.tolist()))})

    metrics = {
        "architecture": {
            "name": "Word IDs + trainable embeddings + parallel TextCNN",
            "vocabulary_size": len(vocabulary),
            "max_sequence_length": max_sequence_length,
            "padding_id": 0,
            "unknown_id": 1,
            "classes": list(AUTHOR_LABELS),
        },
        "settings": {
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "early_stopping_patience": args.patience,
            "embedding_dim": args.embedding_dim,
            "filters_per_kernel": args.filters_per_kernel,
            "kernel_sizes": [2, 3, 4],
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "device": str(device),
        },
        "dataset": {
            "labeled_rows": len(all_texts),
            "train_rows": len(train_texts),
            "validation_rows": len(validation_texts),
            "official_test_rows": len(test_texts),
            "author_counts": dict(Counter(all_labels)),
        },
        "train": training_metrics,
        "validation": validation_metrics,
        "artifacts": {
            "model": str(model_path.relative_to(ROOT)),
            "vocabulary": str(vocabulary_path.relative_to(ROOT)),
            "history": str(history_path.relative_to(ROOT)),
            "submission": str(submission_path.relative_to(ROOT)),
        },
    }
    write_metrics(metrics_path, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Saved TextCNN model: {model_path}")
    print(f"Saved training history: {history_path}")
    print(f"Saved Kaggle probability submission: {submission_path}")
    generate_comparison_graphs(args.output_dir)


def best_model_probabilities(
    model: SpookyTextCNN, features: np.ndarray, device: torch.device
) -> np.ndarray:
    """Compute test probabilities in batches to keep device memory use low."""
    model.eval()
    results = []
    with torch.inference_mode():
        for start in range(0, len(features), 512):
            batch = torch.as_tensor(features[start : start + 512], device=device)
            results.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(results, axis=0)


def run_distilbert(
    args: argparse.Namespace,
    all_texts: list[str],
    all_labels: list[str],
    train_texts: list[str],
    validation_texts: list[str],
    train_labels: list[str],
    validation_labels: list[str],
    test_ids: list[str],
    test_texts: list[str],
) -> None:
    """Fine-tune the downloaded pretrained DistilBERT model on Spooky."""
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch size, and patience must be positive")
    if args.max_length < 8:
        raise ValueError("--max-length must be at least 8")
    if not args.pretrained_model.is_dir():
        raise FileNotFoundError(f"Pretrained model directory not found: {args.pretrained_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, local_files_only=True)

    def tokenize(text_batch: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = tokenizer(
            text_batch,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    train_ids, train_masks = tokenize(train_texts)
    validation_ids, validation_masks = tokenize(validation_texts)
    test_input_ids, test_attention_masks = tokenize(test_texts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Pretrained model: {args.pretrained_model}; max tokens: {args.max_length}; "
        f"device: {device}"
    )
    model = SpookyDistilBERTClassifier(str(args.pretrained_model))
    model, history, best_epoch = train_distilbert_with_history(
        model,
        train_ids,
        train_masks,
        train_labels,
        validation_ids,
        validation_masks,
        validation_labels,
        args.epochs,
        args.batch_size,
        args.patience,
        args.seed,
        device,
    )

    run_dir = args.output_dir / f"distilbert_base_uncased_seed{args.seed}"
    model_dir = run_dir / "model"
    run_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    history_path = run_dir / "history.csv"
    metrics_path = run_dir / "metrics.json"
    submission_path = run_dir / "submission.csv"
    write_history(history_path, history)
    train_metrics = distilbert_metrics(
        model, train_ids, train_masks, train_labels, device, args.batch_size
    )
    validation_metrics = distilbert_metrics(
        model,
        validation_ids,
        validation_masks,
        validation_labels,
        device,
        args.batch_size,
    )
    test_probabilities = _main_distilbert_probabilities(
        model, test_input_ids, test_attention_masks, device, args.batch_size
    )
    with (run_dir / "submission.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", *AUTHOR_LABELS])
        writer.writeheader()
        for row_id, scores in zip(test_ids, test_probabilities):
            writer.writerow({"id": row_id, **dict(zip(AUTHOR_LABELS, scores.tolist()))})

    metrics = {
        "architecture": {
            "name": "DistilBERT base uncased fine-tuned for three authors",
            "pretrained_model": "distilbert/distilbert-base-uncased",
            "classes": list(AUTHOR_LABELS),
            "max_sequence_length": args.max_length,
        },
        "settings": {
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "early_stopping_patience": args.patience,
            "optimizer": "AdamW",
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "device": str(device),
        },
        "dataset": {
            "labeled_rows": len(all_texts),
            "train_rows": len(train_texts),
            "validation_rows": len(validation_texts),
            "official_test_rows": len(test_texts),
            "author_counts": dict(Counter(all_labels)),
        },
        "train": train_metrics,
        "validation": validation_metrics,
        "artifacts": {
            "model": str(model_dir.relative_to(ROOT)),
            "history": str(history_path.relative_to(ROOT)),
            "submission": str(submission_path.relative_to(ROOT)),
        },
    }
    write_metrics(metrics_path, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Saved fine-tuned model and tokenizer: {model_dir}")
    print(f"Saved training history: {history_path}")
    print(f"Saved Kaggle probability submission: {submission_path}")
    generate_comparison_graphs(args.output_dir)


def _main_distilbert_probabilities(
    model: SpookyDistilBERTClassifier,
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(input_ids), batch_size):
            ids = input_ids[start : start + batch_size].to(device)
            masks = attention_masks[start : start + batch_size].to(device)
            batches.append(torch.softmax(model(ids, masks), dim=1).cpu().numpy())
    return np.concatenate(batches, axis=0)


if __name__ == "__main__":
    main()
