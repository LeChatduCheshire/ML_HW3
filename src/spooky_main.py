"""Train a simple probabilistic TF-IDF classifier for Spooky author identification."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib

from spooky_classifier import SpookyClassifier
from spooky_utilitaries import (
    DATASET_DIR,
    ROOT,
    SPOOKY_BUILD_DIR,
    classification_metrics,
    generate_comparison_graphs,
    load_labeled_csv,
    load_test_csv,
    stratified_validation_split,
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


if __name__ == "__main__":
    main()
