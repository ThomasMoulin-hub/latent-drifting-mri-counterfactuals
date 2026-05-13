# Development and Execution Guide

This guide details how to set up the environment and run the core pipelines of the Latent Drifting project.

## Prerequisites
- **Python:** $\ge$ 3.10
- **Hardware:** NVIDIA GPU strongly recommended for training (min 16GB VRAM for CycleGAN, 24GB+ for Diffusion).

## Setup
The project utilizes `pip` for dependency management. To install the required libraries:

```bash
pip install -r requirements.txt
```

## Running the Training Pipelines
The project uses **Hydra** for configuration management. All execution starts from `src/train.py` using predefined experiment configurations.

### 1. CycleGAN Baseline
To train the unpaired translation baseline:
```bash
python src/train.py experiment=cyclegan_oasis
```

### 2. Latent Diffusion Model
To train the conditional diffusion model (requires more VRAM):
```bash
python src/train.py experiment=diffusion_oasis
```
*Note: If you encounter CUDA Out of Memory (OOM) errors, override the batch size via command line:*
```bash
python src/train.py experiment=diffusion_oasis +data.batch_size=4
```

### Development Flags
PyTorch Lightning allows for quick debugging without running a full epoch. Append `+trainer.fast_dev_run=true` to any command to run exactly 1 batch of training, validation, and testing to ensure no code crashes occur:
```bash
python src/train.py experiment=cyclegan_oasis +trainer.fast_dev_run=true
```

## Post-Processing Scripts

### Exporting Presentation Images
The CycleGAN model normalizes images and tracks its state in `.ckpt` files. To export denormalized, high-contrast PNG images (Real $\to$ Fake $\to$ Reconstructed) for presentations:
```bash
python scripts/export_presentation_images.py --ckpt_path "path/to/your/checkpoint.ckpt" --num_patients 5
```

### Generating Difference Heatmaps
To visually highlight the precise regions modified by the generator (e.g., ventricular enlargement), run the heatmap script on the exported presentation images:
```bash
python scripts/generate_difference_maps.py
```
This generates `inferno` colormap heatmaps demonstrating the absolute pixel difference between the generated images and the ground truth.