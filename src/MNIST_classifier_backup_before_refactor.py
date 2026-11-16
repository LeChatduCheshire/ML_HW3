"""Train and evaluate MLP or CNN classifiers on the MNIST IDX files.

Run from the repository root with the project's ``hw3`` environment, e.g.:
    .\hw3\Scripts\python.exe src\MNIST_classifier.py

Select a convolutional network with ``--model-type cnn``.

Hidden layer sizes can be changed to compare shallow and deeper networks:
    python src/MNIST_classifier.py --hidden-sizes 256
    python src/MNIST_classifier.py --hidden-sizes 512 256 128
"""

from __future__ import annotations

import argparse
import csv
import html
import gzip
import json
import random
import struct
from pathlib import Path
from typing import BinaryIO

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
BUILD_DIR = ROOT / "build"
MNIST_BUILD_DIR = BUILD_DIR / "mnist"
RESULTS_DIR = ROOT / "results"


def _open_idx_file(dataset_dir: Path, stem: str) -> BinaryIO:
    """Find a compressed or extracted MNIST file in the repository layout."""
    candidates = (
        dataset_dir / f"{stem}.gz",
        dataset_dir / stem / f"{stem.split('-', 1)[0]}-{'images' if 'images' in stem else 'labels'}.idx3-ubyte",
        dataset_dir / stem / f"{stem.split('-', 1)[0]}-{'images' if 'images' in stem else 'labels'}.idx1-ubyte",
        dataset_dir / stem,
    )
    for path in candidates:
        if path.is_file():
            return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")
    raise FileNotFoundError(
        f"Could not find {stem}.gz or an extracted {stem} IDX file under {dataset_dir}"
    )


def load_images(dataset_dir: Path, stem: str) -> torch.Tensor:
    """Load IDX image data as float tensors in [0, 1], flattened to 784 pixels."""
    with _open_idx_file(dataset_dir, stem) as stream:
        header = stream.read(16)
        if len(header) != 16:
            raise ValueError(f"Truncated IDX image header for {stem}")
        magic, count, rows, columns = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Unexpected IDX image magic number {magic} for {stem}")
        raw = stream.read(count * rows * columns)
        if len(raw) != count * rows * columns:
            raise ValueError(f"Truncated IDX image data for {stem}")
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(count, rows * columns).float().div_(255)


def load_labels(dataset_dir: Path, stem: str) -> torch.Tensor:
    """Load IDX labels as integer class IDs."""
    with _open_idx_file(dataset_dir, stem) as stream:
        header = stream.read(8)
        if len(header) != 8:
            raise ValueError(f"Truncated IDX label header for {stem}")
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Unexpected IDX label magic number {magic} for {stem}")
        raw = stream.read(count)
        if len(raw) != count:
            raise ValueError(f"Truncated IDX label data for {stem}")
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()


class MNISTMLP(nn.Module):
    """Fully connected classifier; the final layer returns logits for CrossEntropyLoss."""

    def __init__(self, hidden_sizes: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        input_size = 28 * 28
        for width in hidden_sizes:
            layers.extend((nn.Linear(input_size, width), nn.ReLU()))
            if dropout:
                layers.append(nn.Dropout(dropout))
            input_size = width
        layers.append(nn.Linear(input_size, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class MNISTCNN(nn.Module):
    """Three convolution/pooling stages followed by a fully connected classifier."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        classifier: list[nn.Module] = [nn.Flatten(), nn.Linear(128 * 3 * 3, 128), nn.ReLU()]
        if dropout:
            classifier.append(nn.Dropout(dropout))
        classifier.append(nn.Linear(128, 10))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.reshape(-1, 1, 28, 28)
        return self.classifier(self.features(images))


def gradient_norms(model: nn.Module) -> dict[str, float]:
    """Return L2 norms of trainable layer weight gradients for diagnostics."""
    norms = {}
    layer_numbers = {nn.Linear: 0, nn.Conv2d: 0}
    for layer in model.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            layer_numbers[type(layer)] += 1
            if layer.weight.grad is not None:
                prefix = "linear" if isinstance(layer, nn.Linear) else "conv"
                norms[f"{prefix}_{layer_numbers[type(layer)]}_grad_norm"] = layer.weight.grad.detach().norm().item()
    return norms


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, dict[str, float]]:
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    gradient_sums: dict[str, float] = {}
    batches = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        for name, value in gradient_norms(model).items():
            gradient_sums[name] = gradient_sums.get(name, 0.0) + value
        optimizer.step()
        batches += 1
        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += labels.size(0)
    return (
        loss_sum / count,
        correct / count,
        {name: value / batches for name, value in gradient_sums.items()},
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += labels.size(0)
    return {"loss": loss_sum / count, "accuracy": correct / count}


@torch.inference_mode()
def probability_summary(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Measure probability quality on held-out test examples using softmax outputs."""
    model.eval()
    nll_sum = 0.0
    confidence_sum = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        logits = model(images.to(device))
        probabilities = torch.softmax(logits, dim=1)
        labels = labels.to(device)
        nll_sum += nn.functional.nll_loss(probabilities.clamp_min(1e-12).log(), labels, reduction="sum").item()
        confidence_sum += probabilities.max(dim=1).values.sum().item()
        correct += (probabilities.argmax(dim=1) == labels).sum().item()
        count += labels.size(0)
    return {
        "test_negative_log_likelihood": nll_sum / count,
        "test_mean_max_probability": confidence_sum / count,
        "test_accuracy_from_probabilities": correct / count,
    }


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


def _render_comparison_svg(
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
    colors = ("#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c", "#0891b2", "#4f46e5")

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
    tick_count = 5
    for index in range(tick_count + 1):
        value = y_min + (y_max - y_min) * index / tick_count
        y = top + plot_height - plot_height * index / tick_count
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


def generate_comparison_graphs(build_dir: Path, results_dir: Path = RESULTS_DIR) -> None:
    """Regenerate accuracy and loss SVGs from all saved MNIST run histories."""
    runs = _read_run_histories(build_dir)
    if not runs:
        raise FileNotFoundError(f"No run histories found under {build_dir}; expected */history.csv")
    results_dir.mkdir(parents=True, exist_ok=True)
    accuracy_values = [row[key] for _, rows in runs for row in rows for key in ("train_accuracy", "validation_accuracy")]
    accuracy_min = max(0.0, min(accuracy_values) - 0.02)
    accuracy_max = min(1.0, max(accuracy_values) + 0.02)
    loss_max = max(row[key] for _, rows in runs for row in rows for key in ("train_loss", "validation_loss"))
    outputs = {
        "mnist_accuracy_comparison.svg": _render_comparison_svg(
            runs, "MNIST training and validation accuracy", "train_accuracy",
            "validation_accuracy", accuracy_min, accuracy_max, "Accuracy",
        ),
        "mnist_loss_comparison.svg": _render_comparison_svg(
            runs, "MNIST training and validation loss", "train_loss",
            "validation_loss", 0.0, loss_max * 1.05, "Loss",
        ),
    }
    for filename, svg in outputs.items():
        (results_dir / filename).write_text(svg, encoding="utf-8")
        print(f"Saved comparison graph: {results_dir / filename}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=MNIST_BUILD_DIR)
    parser.add_argument("--model-type", choices=("mlp", "cnn"), default="mlp")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Regenerate comparison graphs from saved runs without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_only:
        generate_comparison_graphs(args.output_dir)
        return
    if args.model_type == "mlp" and (not args.hidden_sizes or any(width < 1 for width in args.hidden_sizes)):
        raise ValueError("--hidden-sizes must contain positive integers")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in (0, 1)")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch size, and patience must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_images = load_images(args.data_dir, "train-images-idx3-ubyte")
    train_labels = load_labels(args.data_dir, "train-labels-idx1-ubyte")
    test_images = load_images(args.data_dir, "t10k-images-idx3-ubyte")
    test_labels = load_labels(args.data_dir, "t10k-labels-idx1-ubyte")
    if len(train_images) != len(train_labels) or len(test_images) != len(test_labels):
        raise ValueError("MNIST image and label counts do not match")

    full_train = TensorDataset(train_images, train_labels)
    validation_size = max(1, round(len(full_train) * args.validation_fraction))
    training_size = len(full_train) - validation_size
    generator = torch.Generator().manual_seed(args.seed)
    training_set, validation_set = random_split(
        full_train, [training_size, validation_size], generator=generator
    )
    training_loader = DataLoader(
        training_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, generator=generator,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        TensorDataset(test_images, test_labels), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    model = (
        MNISTMLP(args.hidden_sizes, args.dropout)
        if args.model_type == "mlp"
        else MNISTCNN(args.dropout)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    architecture = "-".join(map(str, args.hidden_sizes)) if args.model_type == "mlp" else "conv32-64-128_fc128"
    run_name = f"{args.model_type}_{architecture}_seed{args.seed}"
    run_dir = args.output_dir / run_name
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
        gradient_text = " ".join(f"{key}={value:.3e}" for key, value in grad_norms.items())
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
            "hidden_sizes": args.hidden_sizes if args.model_type == "mlp" else [128],
            "convolution_channels": [32, 64, 128] if args.model_type == "cnn" else [],
            "input_units": 784, "output_units": 10,
        },
        "settings": {
            "optimizer": "Adam", "learning_rate": args.learning_rate,
            "batch_size": args.batch_size, "dropout": args.dropout,
            "epochs_requested": args.epochs, "best_epoch": best_epoch,
            "validation_fraction": args.validation_fraction, "seed": args.seed,
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
    generate_comparison_graphs(args.output_dir)


if __name__ == "__main__":
    main()
