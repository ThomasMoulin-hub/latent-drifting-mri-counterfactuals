# Source Tree Analysis

This document provides an annotated view of the critical directories in the project. The architecture follows a standard PyTorch Lightning + Hydra template structure.

## Core Source Code (`src/`)

```text
src/
├── train.py               # Main entry point for training pipelines
├── eval.py                # Main entry point for evaluation pipelines
│
├── data/                  # Data modules and datasets
│   ├── preprocess_oasis.py  # Script for extracting and normalizing 2D slices from 3D FreeSurfer volumes
│   ├── oasis_datamodule.py  # PyTorch Lightning DataModule for OASIS dataset
│   └── components/
│       └── oasis_dataset.py # PyTorch Dataset defining the data schema and clinical sliders
│
├── models/                # Neural network architectures (pure PyTorch)
│   ├── classifier.py      # DenseNet-121 architecture for the clinical judge
│   ├── discriminator.py   # PatchGAN discriminator for CycleGAN
│   ├── latent_diffusion.py# Diffusion models/schedulers wrappers
│   └── swin_unet.py       # Swin UNETR generator bounded by Tanh for GAN output
│
├── systems/               # PyTorch Lightning Modules (Training logic, losses, optimizers)
│   ├── classifier_system.py # Training loop for the DenseNet classifier
│   ├── cyclegan_system.py   # Complex training loop with 2 generators/discriminators and Cycle Consistency
│   └── diffusion_system.py  # Conditional denoising diffusion training loop and counterfactual trajectory sampling
│
└── utils/                 # Shared utilities, loggers, and Hydra instantiators
```

## Configuration (`configs/`)

The `configs/` directory is managed by **Hydra**. It allows compositional configuration of the entire machine learning pipeline.

```text
configs/
├── train.yaml             # Master configuration file for training
├── eval.yaml              # Master configuration file for evaluation
│
├── experiment/            # High-level experiment definitions (ties data, model, and system together)
│   ├── classifier_oasis.yaml # Train the DenseNet evaluator
│   ├── cyclegan_oasis.yaml   # Train the CycleGAN baseline
│   └── diffusion_oasis.yaml  # Train the Latent Diffusion model
│
├── model/                 # System/Model specific parameters (e.g., learning rates, losses)
│   ├── classifier.yaml
│   ├── cyclegan.yaml
│   └── diffusion.yaml
│
├── data/                  # Dataset parameters (batch size, paths)
│   └── oasis.yaml
│
└── trainer/               # PyTorch Lightning Trainer configurations (epochs, GPUs)
    ├── default.yaml
    └── gpu.yaml
```

## Other Critical Directories
- **`scripts/`**: Contains utility scripts such as `finetuning.py` (used to train the DenseNet from scratch), `export_presentation_images.py` (to generate 0-255 PNGs), and `generate_difference_maps.py` (to produce heatmaps).
- **`report/`**: Contains the LaTeX source code (`Project Report - Thomas Moulin.tex`) for the IEEE conference paper.
- **`presentation/web/`**: Contains an HTML/CSS mockup illustrating the Clinical Slider framework.