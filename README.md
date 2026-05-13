# Patient-Customized Counterfactual Data Augmentation via Latent Drifting

This repository contains the codebase and research for generating patient-specific counterfactual MRI images. By manipulating clinical "sliders" (e.g., Age, Alzheimer's markers) within a compressed latent space, this framework simulates structural brain changes (like ventricular enlargement) while preserving the individual's core anatomical identity.

## Read the Full Report
A comprehensive IEEE-formatted conference paper detailing the methodology, architectural findings, and experimental results is available at the root of this project:

**[Read the Project Report (PDF)](./Project%20Report%20-%20Thomas%20Moulin.pdf)**

---

## Quick Start

This project is built using **PyTorch**, **PyTorch Lightning**, and orchestrated via **Hydra**.

### 1. Training the Models

**CycleGAN Baseline (Style Translation):**
```bash
python src/train.py experiment=cyclegan_oasis
```

**Conditional Latent Diffusion:**
```bash
python src/train.py experiment=diffusion_oasis
```

*Note: You can easily run a fast development test on any model by appending `+trainer.fast_dev_run=true` to your command.*

### 2. Exporting Presentation Images
To extract normalized, presentation-ready PNG sequences (Real $\rightarrow$ Fake $\rightarrow$ Reconstructed) with true black backgrounds from a trained CycleGAN checkpoint:
```bash
python scripts/export_presentation_images.py --ckpt_path "path/to/your/epoch_XXX.ckpt" --num_patients 5
```

### 3. Generating Difference Heatmaps
To visually prove that the generator isolates pathological biomarkers without distorting the skull, run the heatmap script on your exported images:
```bash
python scripts/generate_difference_maps.py
```
This will output absolute pixel-wise difference maps using the `inferno` colormap.

---

## Core Pipeline

1. **The Judge (Clinical Evaluator):** A 1-channel DenseNet-121 fine-tuned on the OASIS dataset to distinguish Cognitively Normal (CN) from Alzheimer's Disease (AD) patients. Used to calculate the *Deception Rate*.
2. **The Baseline (CycleGAN):** Unpaired image-to-image translation utilizing Swin UNETR generators bounded by $Tanh$ activations.
3. **The Innovation (Cycle Diffusion):** A `DDPMScheduler`-based conditional diffusion model enabling continuous, slider-based disease trajectory synthesis.

---

## Repository Structure
* `configs/` - Hydra configuration files (hyperparameters, experiment setups).
* `src/` - Core PyTorch Lightning modules (`systems/`), networks (`models/`), and data loaders (`data/`).
* `scripts/` - Standalone tools for preprocessing, finetuning, and exporting visualizations.
* `report/` - LaTeX source files for the IEEE conference report.
* `presentation/web/` - A visual HTML/CSS mockup illustrating the "Clinical Slider" framework.