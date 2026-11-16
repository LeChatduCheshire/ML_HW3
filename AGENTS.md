# Project Instructions

## Environment and Dependencies
- Python version: 3.12
- Virtual environment: `./hw3`
- All Python commands must use `./hw3/Scripts/python.exe`.
- This project uses PyTorch (`torch` and `torchvision`) and `scikit-learn`.
- All Python commands must be executed using the `hw3` virtual environment located in the repository root. 
- This session is running on Windows 11. Activate virtual environment using the following command in the command prompt: `hw3\Scripts\activate.bat` (or `hw3\Scripts\activate.ps1` in the powershell).
- Do not install packages or modify the environment unless explicitly requested.
- Do not use the system Python interpreter or another virtual environment.

## Project Structure
- `src/`: All source code, including data preprocessing, model definitions, and training scripts.
  - `MNIST_classifier.py`: MNIST model class definitions only (`MNISTMLP`, `MNISTCNN`, `MNISTDeepCNN`, `MNISTResNet`, and `ResidualBlock`).
  - `MNIST_utilitaries.py`: Reusable MNIST data loading, training, evaluation, metrics, and graph utilities.
  - `MNIST_main.py`: MNIST command-line options and training/evaluation orchestration. Run MNIST experiments from this entry point.
  - `spooky_classifier.py`: Spooky model class definitions only.
  - `spooky_utilitaries.py`: Reusable Spooky CSV loading, training, evaluation, submission, and graph utilities.
  - `spooky_main.py`: Spooky command-line options and training/evaluation orchestration. Run Spooky experiments from this entry point.
- `doc/`: Project documentation.
  - `hw3_2026.pdf`: Contains the project objectives and requirements.
  - Also contains a draft of the project report.
- `build/`: Generated build artifacts and training outputs, organized by task and model variant. Use paths such as `build/mnist/<variant>/` and `build/spooky/<variant>/` so outputs from different classifiers stay separate.
- `results/`: At the repository root, store graphs comparing training and validation performance across model versions. Regenerate and save comparison graphs here whenever new training runs are added.
- `temp/`: Temporary files and intermediate artifacts. This directory may be created at the repository root when necessary.
- `dataset/`: All datasets that will be used in the project. Each dataset is already divided in 'test' and 'train' data. For the MNIST dataset, reffer to their official website to see data format.
- `hw3\`: virtual environment source.

## General Guidelines
- Keep the project organized according to the directory structure above.
- Do not modify the original documentation unless explicitly requested.
- Avoid placing generated files or temporary artifacts in the repository root.
- Before making significant architectural changes, explain the proposed changes and their rationale.
