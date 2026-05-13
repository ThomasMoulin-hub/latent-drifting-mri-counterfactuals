# Project Overview: Patient-Customized Counterfactual Data Augmentation via Latent Drifting

## Executive Summary
This project aims to solve the data scarcity problem in medical deep learning, specifically the lack of longitudinal MRI data for neurodegenerative diseases. By disentangling a patient's core anatomical identity from pathological markers, the framework generates patient-specific counterfactuals. It simulates how a healthy brain would age or develop diseases like Alzheimer's (AD) by traversing a compressed latent space via "Latent Drifting."

## Tech Stack
| Category | Technology | Purpose |
|---|---|---|
| **Core Framework** | PyTorch & PyTorch Lightning | Training loops, module abstraction |
| **Configuration** | Hydra | Hierarchical experiment configuration |
| **Medical Imaging** | MONAI | Advanced medical neural network architectures (Swin UNETR) |
| **Generative Models** | Diffusers (HuggingFace) | Schedulers and components for Latent Diffusion |
| **Logging** | Weights & Biases (WandB) | Experiment tracking and visual qualitative inspection |

## Architecture Classification
**Monolith (ML Pipeline):** The project is structured as a cohesive machine learning pipeline following a standard PyTorch Lightning architecture, separating raw data loading (`data/`), model definition (`models/`), and training logic (`systems/`).

## The 3-Step Pipeline
The project is divided into three consecutive methodological steps:

1. **The Judge (Clinical Evaluator):** A 1-channel DenseNet-121 model trained from scratch on the OASIS dataset. It acts as an independent medical evaluator to compute the "Deception Rate"—the percentage of generated images that successfully mimic Alzheimer's biomarkers.
2. **The Baseline (CycleGAN):** An unpaired image-to-image translation system using Swin UNETR generators and PatchGAN discriminators. It establishes a baseline for structural preservation (Cycle Consistency) but is limited to binary (Healthy $\leftrightarrow$ Sick) translation.
3. **The Innovation (Conditional Latent Diffusion):** A diffusion-based system that allows for continuous disease trajectory modeling. By conditioning the denoising process on clinical sliders (Age, AD, ALS, FTD), it generates smooth, frame-by-frame pathology progression.