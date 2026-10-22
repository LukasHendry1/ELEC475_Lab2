# ELEC 475 Lab 2: From LeNet-5 to AlexNet

Lukas Hendry and Benjamin Syer

This repository implements and compares five CNN architectures on CIFAR-10 and Tiny ImageNet-200 at 64 × 64 resolution: LeNet-5, a modernized LeNet, AlexNet-64 with three normalization schemes, and the torchvision AlexNet. It covers a normalization ablation (none, LRN, BatchNorm2d), a data augmentation study, a two-dataset benchmark, and multi-seed evaluation.

## Directory Structure

```text
ELEC475_Lab2/
├── configs/
│   └── config.py          # Experiment configuration dataclass
├── models/
│   ├── lenet.py           # Base LeNet-5 (Tanh + AvgPool)
│   ├── lenet_modern.py    # Modernized LeNet (ReLU + overlapping MaxPool)
│   ├── alexnet.py         # AlexNet-64 with --norm none / lrn / bn
│   └── zoo.py             # torchvision AlexNet adapter
├── utils/
│   ├── dataset.py         # CIFAR-10 and Tiny ImageNet loaders
│   ├── eda.py             # Exploratory data analysis
│   ├── transforms.py      # Training augmentation pipeline
│   ├── metrics.py         # Accuracy, Macro F1, confusion matrix, early stopping
│   ├── seed.py            # Global seeding and Mean ± Std aggregation
│   ├── visualization.py   # Training curve and ablation plotters
│   └── profiler.py        # CUDA-synchronized Timer, FLOPs/MACs/params profiler
├── plots/                 # EDA figures, loss/accuracy curves, ablation summary
├── checkpoints/           # Trained weights (not tracked in git)
├── train.py               # Training engine
├── evaluate.py            # Checkpoint evaluation with classification report
├── compare_models.py      # Params, MACs, FLOPs, latency and throughput benchmark
├── demo_transforms.py     # OpenCV vs NumPy vs PyTorch transform comparison
├── prelab_verify.py       # Environment and dataset verification
├── commands.txt           # Exact commands for every experiment in the report
├── github_repo.txt        # Repository URL
├── requirements.txt       # Package dependencies
└── requirements-lock.txt  # Pinned package versions
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CIFAR-10 downloads automatically on first use. Tiny ImageNet-200 is downloaded and extracted to `data/tiny-imagenet-200` on first use.

## Running the Experiments

Every command used to produce the report's results is listed in [`commands.txt`](commands.txt). The main ones are:

```bash
# Pre-lab
python -m utils.eda --dataset both
python prelab_verify.py

# Normalization study (CIFAR-10, 10 epochs)
python train.py --model alexnet --norm {none,lrn,bn} --dataset cifar10 --epochs 10 --no-augment

# Data augmentation study (CIFAR-10, 15 epochs)
python train.py --model alexnet --norm bn --dataset cifar10 --epochs 15 [--no-augment]

# Tiny ImageNet benchmark
python train.py --model alexnet --norm bn --dataset tiny_imagenet --epochs 10

# Complexity benchmark
python compare_models.py --dataset cifar10
```

Useful `train.py` options: `--model {lenet,lenet_modern,alexnet,zoo_alexnet}`, `--norm {none,lrn,bn}`, `--dataset {cifar10,tiny_imagenet}`, `--epochs`, `--lr`, `--batch-size`, `--seed`, `--no-augment`, `--save-model`, `--plot-file`, `--early-stopping`.

All runs are seeded with `utils/seed.py` and use deterministic cuDNN, so re-running a command with the same seed reproduces its metrics exactly.

## Training Setup

| Setting | Value |
|---|---|
| Input resolution | 3 × 64 × 64 |
| Optimizer | Adam, LR = 0.001, weight decay = 1e-4 |
| Batch size | 128 |
| Seeds | 42 (main results); 42, 123 and 2026 (multi-seed study) |
| Augmentation | RandomCrop(64, padding=4, reflect), RandomHorizontalFlip, ColorJitter(0.2, 0.2, 0.2) |
| Hardware | NVIDIA GeForce RTX 3060 Laptop GPU |

## Results Summary

Best validation accuracy after 10 epochs, seed 42:

| Model | Params (M) | FLOPs (M) | CIFAR-10 Val Acc (%) | Tiny ImageNet Val Acc (%) |
|---|---|---|---|---|
| Base LeNet-5 | 0.338 | 7.17 | 55.85 | 12.31 |
| Modernized LeNet | 0.446 | 9.19 | 66.27 | 19.96 |
| AlexNet-64 (No Norm) | 6.609 | 436.70 | 73.33 | 0.50 |
| AlexNet-64 + LRN | 6.609 | 436.70 | 73.52 | 0.50 |
| AlexNet-64 + BatchNorm | 6.611 | 437.97 | 79.84 | 30.06 |
| AlexNet-64 + BN + Aug | 6.611 | 437.97 | 79.74 | 30.02 |
| torchvision AlexNet | 57.045 | 188.20 | 65.98 | 10.31 |

Full results, analysis and multi-seed statistics are in the lab report.
