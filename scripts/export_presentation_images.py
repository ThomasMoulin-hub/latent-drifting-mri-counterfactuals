#!/user/tmm2219/.conda/envs/DLBI/bin/python

import os
import torch
import torchvision
from PIL import Image
import numpy as np
import argparse
from pathlib import Path
import sys

# Add src to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.systems.cyclegan_system import CycleGANSystem
from src.data.oasis_datamodule import OASISDataModule

def denormalize_and_save(tensor: torch.Tensor, filepath: str):
    """
    Takes a tensor in range [-1, 1], shifts it back so the background (0 in native space) 
    becomes actual black (0 in 0-255 space), and saves it as a PNG.
    """
    # 1. Denormalize from [-1, 1] back to [0, 1] native FreeSurfer scale
    # In preprocess_oasis.py, ScaleIntensity(minv=0.0, maxv=1.0) was used.
    # But later, W&B logs used (x + 1)/2. 
    # Let's extract exactly what CycleGAN outputs [-1, 1] and map it.
    
    # If the background is exactly at 0 in the original npy, and CycleGAN scales to [-1, 1]
    # Native: [0, 1] -> CycleGAN input: (x - 0.5) / 0.5 = [-1, 1]
    # CycleGAN output: [-1, 1] -> Native: (x * 0.5) + 0.5 = [0, 1]
    
    img = (tensor * 0.5) + 0.5
    
    # Clamp just in case the generator overshot
    img = torch.clamp(img, 0.0, 1.0)
    
    # Convert to 0-255
    img_np = (img.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
    
    # Save using PIL
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    Image.fromarray(img_np, mode='L').save(filepath)

def main():
    parser = argparse.ArgumentParser(description="Export CycleGAN images for presentation")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the trained CycleGAN checkpoint (.ckpt)")
    parser.add_argument("--output_dir", type=str, default="out/presentation_images", help="Directory to save the PNGs")
    parser.add_argument("--num_patients", type=int, default=5, help="Number of patients to process from the test set")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data...")
    # We create a dummy config dict to satisfy the DataModule requirements
    from omegaconf import OmegaConf
    data_cfg = OmegaConf.create({
        "data_dir": "data/processed",
        "csv_path": "data/processed/metadata.csv",
        "train_val_test_split": [0.8, 0.1, 0.1],
        "batch_size": 1, # Process one by one for easy saving
        "num_workers": 0,
        "pin_memory": False
    })
    
    dm = OASISDataModule(
        data_dir=data_cfg.data_dir,
        csv_path=data_cfg.csv_path,
        train_val_test_split=data_cfg.train_val_test_split,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers
    )
    dm.setup()
    test_loader = dm.test_dataloader()

    # 2. Load Model
    print(f"Loading model from {args.ckpt_path}...")
    model = CycleGANSystem.load_from_checkpoint(args.ckpt_path)
    model.eval()
    model.to(device)

    # 3. Generate Images
    print(f"Generating images into {args.output_dir}...")
    out_path = Path(args.output_dir)
    
    a_count = 0
    b_count = 0
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if a_count >= args.num_patients and b_count >= args.num_patients:
                break
                
            images = batch["image"].to(device)
            labels = batch["label"]
            
            # Since batch_size=1
            label = labels[0].item()
            
            # Domain A (CN -> 0)
            if label == 0 and a_count < args.num_patients:
                real_a = images
                fake_b = model.net_g_a2b(real_a)
                rec_a = model.net_g_b2a(fake_b)
                
                # Save A -> B -> A sequence
                prefix = out_path / f"patient_CN_{a_count:02d}"
                denormalize_and_save(real_a[0], f"{prefix}_1_Real_Sain.png")
                denormalize_and_save(fake_b[0], f"{prefix}_2_Fake_Alzheimer.png")
                denormalize_and_save(rec_a[0],  f"{prefix}_3_Reconstruit_Sain.png")
                a_count += 1
                
            # Domain B (AD -> 1)
            elif label == 1 and b_count < args.num_patients:
                real_b = images
                fake_a = model.net_g_b2a(real_b)
                rec_b = model.net_g_a2b(fake_a)
                
                # Save B -> A -> B sequence
                prefix = out_path / f"patient_AD_{b_count:02d}"
                denormalize_and_save(real_b[0], f"{prefix}_1_Real_Alzheimer.png")
                denormalize_and_save(fake_a[0], f"{prefix}_2_Fake_Sain.png")
                denormalize_and_save(rec_b[0],  f"{prefix}_3_Reconstruit_Alzheimer.png")
                b_count += 1

    print(f"✅ Success! Images saved in '{args.output_dir}'.")
    print("They now have a true black background and are ready for PowerPoint/Papers.")

if __name__ == "__main__":
    main()