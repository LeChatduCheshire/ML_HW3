import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader
from torch.utils.data import random_split

# ============================================================
# CONFIG
# ============================================================

from pathlib import Path

BATCH_SIZE = 10
NUM_WORKERS = 4
IMAGE_SIZE = 128
EPOCHS = 1000
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "../build/MNIST_classifier_v1.0.pth"
VERSION = "1.0"
CONTINUE_TRAINING = True
USE_PRETREATED_DATASET = False 
# ============================================================
# MODEL
# ============================================================
class MNIST_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                128 * 1 * 1,
                128
            ),
            nn.ReLU(),
            nn.Dropout(0.07),
            nn.Linear(
                128,
                10
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def main(MODEL_SAVE_PATH):
    print("Model loaded")
    model = MNIST_Classifier().to(device)
    if CONTINUE_TRAINING:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH,weights_only=True))
        MODEL_SAVE_PATH = MODEL_SAVE_PATH[:-5]+VERSION+".pth"
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    train(model, optimizer, criterion,MODEL_SAVE_PATH)
    save_model(model,f"../build/MNIST_classifier_EOT_v1.{VERSION}.pth")

def train(model, optimizer, criterion,MODEL_SAVE_PATH):
    best_test_accuracy = 0
    for epoch in range(EPOCHS):
        # ------------------------
        # TRAIN MODE
        # ------------------------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_accuracy = 100 * correct / total

        # ------------------------
        # EVALUATION
        # ------------------------
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (
                    predicted == labels
                ).sum().item()
        test_accuracy = 100 * test_correct / test_total
        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {running_loss:.4f} "
            f"Train Acc: {train_accuracy:.2f}% "
            f"Test Acc: {test_accuracy:.2f}%"
        )
        if test_accuracy>best_test_accuracy:
            save_model(model,MODEL_SAVE_PATH)
            best_test_accuracy=test_accuracy

# ============================================================
# SAVE MODEL
# ============================================================
def save_model(model,MODEL_SAVE_PATH):
    torch.save(model.state_dict(),MODEL_SAVE_PATH)
    print(f"Model saved to : {MODEL_SAVE_PATH}")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if USE_PRETREATED_DATASET:
        DATASET_PATH = Path("../dataset_pretreated_128x128")
        base_transforms = [
            transforms.Grayscale(num_output_channels=1),
        ]
    else:
        DATASET_PATH = Path("../dataset/train-images-idx3-ubyte")
        base_transforms = [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=1)
        ]


    # ============================================================
    # DEVICE
    # ============================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device : {device}")

    # ============================================================
    # TRANSFORMS
    # ============================================================

    train_transform = transforms.Compose(
        base_transforms + [
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            )
        ]
    )

    test_transform = transforms.Compose(
        base_transforms + [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            )
        ]
    )

    # ============================================================
    # DATASET
    # ============================================================

    full_dataset = ImageFolder(root=DATASET_PATH)
    # Split train/test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_indices, test_indices = random_split(range(len(full_dataset)),[train_size, test_size])

    train_dataset = torch.utils.data.Subset(
        ImageFolder(DATASET_PATH, transform=train_transform),
        train_indices.indices
    )
    test_dataset = torch.utils.data.Subset(
        ImageFolder(DATASET_PATH, transform=test_transform),
        test_indices.indices
    )

    # ============================================================
    # DATALOADERS
    # ============================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS, 
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS, 
        pin_memory=True,
    )

    main(MODEL_SAVE_PATH)