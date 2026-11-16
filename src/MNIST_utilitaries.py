"""Data, training, evaluation, and graph helpers for the MNIST experiments."""

from __future__ import annotations

import csv
import gzip
import html
import struct
from pathlib import Path
from typing import BinaryIO

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
MNIST_BUILD_DIR = ROOT / "build" / "mnist"
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


def make_data_loaders(
    data_dir: Path,
    batch_size: int,
    validation_fraction: float,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load MNIST and return train, validation, and official test loaders."""
    train_images = load_images(data_dir, "train-images-idx3-ubyte")
    train_labels = load_labels(data_dir, "train-labels-idx1-ubyte")
    test_images = load_images(data_dir, "t10k-images-idx3-ubyte")
    test_labels = load_labels(data_dir, "t10k-labels-idx1-ubyte")
    if len(train_images) != len(train_labels) or len(test_images) != len(test_labels):
        raise ValueError("MNIST image and label counts do not match")

    full_train = TensorDataset(train_images, train_labels)
    validation_size = max(1, round(len(full_train) * validation_fraction))
    training_size = len(full_train) - validation_size
    generator = torch.Generator().manual_seed(seed)
    training_set, validation_set = random_split(
        full_train, [training_size, validation_size], generator=generator
    )
    pin_memory = device.type == "cuda"
    training_loader = DataLoader(
        training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        generator=generator, pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        TensorDataset(test_images, test_labels), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )
    return training_loader, validation_loader, test_loader


def gradient_norms(model: nn.Module) -> dict[str, float]:
    """Return L2 norms of trainable layer weight gradients for diagnostics."""
    norms = {}
    layer_numbers = {nn.Linear: 0, nn.Conv2d: 0}
    for layer in model.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            layer_numbers[type(layer)] += 1
            if layer.weight.grad is not None:
                prefix = "linear" if isinstance(layer, nn.Linear) else "conv"
                norms[f"{prefix}_{layer_numbers[type(layer)]}_grad_norm"] = (
                    layer.weight.grad.detach().norm().item()
                )
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
def probability_summary(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Measure probability quality on held-out test examples using softmax outputs."""
    model.eval()
    nll_sum = 0.0
    confidence_sum = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        probabilities = torch.softmax(model(images.to(device)), dim=1)
        labels = labels.to(device)
        nll_sum += nn.functional.nll_loss(
            probabilities.clamp_min(1e-12).log(), labels, reduction="sum"
        ).item()
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
