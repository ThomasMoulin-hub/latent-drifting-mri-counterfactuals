# Data Models and Structures

This document outlines the primary data structures used in the machine learning pipelines for this project. As a PyTorch-based repository, "data models" primarily refer to Dataset classes and the structure of the tensors passed to the neural networks.

## OASISDataset

The primary dataset class for the project is `OASISDataset`, which loads 2D axial MRI slices and their corresponding metadata.

### Input Data
- **Images:** 2D `.npy` files containing MRI slices, loaded as `numpy.float32` arrays.
- **Metadata:** A CSV file mapping slices to patient labels and clinical variables.

### Output Dictionary Structure (`__getitem__`)
Each item yielded by the dataset is a dictionary containing the following keys:

| Key | Type | Shape | Description |
|---|---|---|---|
| `image` | `torch.Tensor` | `(1, H, W)` | The 1-channel grayscale MRI slice. Pixel values are typically normalized to `[-1, 1]` or `[0, 1]` depending on the transform pipeline. |
| `label` | `int` | Scalar | The binary classification label: `0` for Cognitively Normal (CN) and `1` for Alzheimer's Disease (AD). |
| `clinical_sliders` | `torch.Tensor` | `(4,)` | A 1D tensor containing clinical conditioning variables used for the Latent Diffusion model. |

### Clinical Sliders Schema
The `clinical_sliders` tensor encodes specific clinical markers to control the generative process:
1. `age_z`: Patient age normalized (e.g., `age / 100.0`).
2. `z_ad`: Alzheimer's Disease marker (matches the binary `label`).
3. `z_als`: Amyotrophic Lateral Sclerosis marker.
4. `z_ftd`: Frontotemporal Dementia marker.

## Data Preprocessing Pipeline
The `preprocess_oasis.py` script utilizes FreeSurfer outputs (`brain.mgz`) and extracts specific axial slices (typically indices 100-140) to focus on key regions of interest like the lateral ventricles and the hippocampus, which are critical for detecting AD-related atrophy.