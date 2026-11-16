"""Command line entry point for training MNIST classifier variants."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch import nn

from MNIST_classifier import MNISTCNN, MNISTDeepCNN, MNISTMLP, MNISTResNet
from MNIST_utilitaries import (
    DATASET_DIR,
    MNIST_BUILD_DIR,
    ROOT,
    evaluate,
    generate_comparison_graphs,
    make_data_loaders,
    probability_summary,
    train_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=MNIST_BUILD_DIR)
    parser.add_argument("--model-type", choices=("mlp", "cnn", "deepcnn", "resnet"), default="mlp")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Regenerate comparison graphs from saved runs without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    if args.plot_only:
        generate_comparison_graphs(output_dir)
        return
    if args.model_type == "mlp" and (
        not args.hidden_sizes or any(width < 1 for width in args.hidden_sizes)
    ):
        raise ValueError("--hidden-sizes must contain positive integers")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in (0, 1)")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.cpu_threads < 1:
        raise ValueError("epochs, batch size, patience, and CPU threads must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    print(f"Device: {device}")

    training_loader, validation_loader, test_loader = make_data_loaders(
        data_dir, args.batch_size, args.validation_fraction, args.seed,
        args.num_workers, device,
    )
    if args.model_type == "mlp":
        model = MNISTMLP(args.hidden_sizes, args.dropout)
    elif args.model_type == "cnn":
        model = MNISTCNN(args.dropout)
    elif args.model_type == "deepcnn":
        model = MNISTDeepCNN(args.dropout)
    else:
        model = MNISTResNet()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    if args.model_type == "mlp":
        architecture = "-".join(map(str, args.hidden_sizes))
    elif args.model_type == "cnn":
        architecture = "conv32-64-128_fc128"
    elif args.model_type == "deepcnn":
        architecture = "conv16-32-64-64"
    else:
        architecture = "resnet16-32-64_6blocks"
    run_name = f"{args.model_type}_{architecture}_seed{args.seed}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_model.pt"
    history_path = run_dir / "history.csv"
    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, grad_norms = train_epoch(
            model, training_loader, optimizer, criterion, device
        )
        validation = evaluate(model, validation_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            **grad_norms,
        }
        history.append(row)
        gradient_text = " ".join(
            f"{name}={value:.3e}" for name, value in grad_norms.items()
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} "
            f"acc {train_accuracy:.4%} | val loss {validation['loss']:.4f} "
            f"acc {validation['accuracy']:.4%} | gradients {gradient_text}"
        )
        if validation["loss"] < best_validation_loss:
            best_validation_loss = validation["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": args.model_type,
                    "hidden_sizes": args.hidden_sizes if args.model_type == "mlp" else None,
                    "dropout": args.dropout,
                    "epoch": epoch,
                    "validation_loss": best_validation_loss,
                    "seed": args.seed,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
                break

    fieldnames = list(dict.fromkeys(key for row in history for key in row))
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    probability_metrics = probability_summary(model, test_loader, device)
    results = {
        "architecture": {
            "model_type": args.model_type,
            "hidden_sizes": args.hidden_sizes if args.model_type == "mlp" else [],
            "convolution_channels": [32, 64, 128] if args.model_type == "cnn" else [],
            "deep_convolution_channels": [16, 32, 64, 64] if args.model_type == "deepcnn" else [],
            "residual_stage_channels": [16, 32, 64] if args.model_type == "resnet" else [],
            "residual_blocks_per_stage": [2, 2, 2] if args.model_type == "resnet" else [],
            "input_units": 784,
            "output_units": 10,
        },
        "settings": {
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "dropout": args.dropout,
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
            "cpu_threads": args.cpu_threads if device.type == "cpu" else None,
        },
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        **probability_metrics,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "history": str(history_path.relative_to(ROOT)),
    }
    results_path = run_dir / "metrics.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved training history: {history_path}")
    print(f"Saved metrics: {results_path}")
    generate_comparison_graphs(output_dir)


if __name__ == "__main__":
    main()
