#!/user/tmm2219/.conda/envs/DLBI/bin/python

import os
import glob
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def process_patient_diff(patient_prefix, out_dir):
    # Find the 3 images for the sequence (1: Real, 2: Fake, 3: Rec)
    img1_path = glob.glob(f"{patient_prefix}_1_*.png")
    img2_path = glob.glob(f"{patient_prefix}_2_*.png")
    img3_path = glob.glob(f"{patient_prefix}_3_*.png")
    
    if not (img1_path and img2_path and img3_path):
        print(f"Could not find all 3 images for prefix {patient_prefix}")
        return
        
    # Load as numpy float arrays
    img1 = np.array(Image.open(img1_path[0]).convert('L')).astype(np.float32)
    img2 = np.array(Image.open(img2_path[0]).convert('L')).astype(np.float32)
    img3 = np.array(Image.open(img3_path[0]).convert('L')).astype(np.float32)
    
    # 1. Calculate absolute differences
    # Fake AD vs Real CN (What did the disease simulation modify?)
    diff_forward = np.abs(img2 - img1) 
    
    # Rec CN vs Fake AD (What did the reverse generator fix back?)
    diff_backward = np.abs(img3 - img2) 
    
    # Rec CN vs Real CN (Cycle consistency error: what was lost forever?)
    diff_cycle = np.abs(img3 - img1) 
    
    patient_name = os.path.basename(patient_prefix)
    os.makedirs(out_dir, exist_ok=True)
    
    # We use a heatmap ('inferno') to make the differences very visible for presentation.
    # The 'inferno' colormap maps 0 (no difference) to black, and higher differences to bright yellow/white.
    def save_heatmap(data, filename):
        # We set vmax to 40 (out of 255) to massively boost the contrast of small changes.
        # Any pixel difference >= 40 will appear as maximum bright yellow/white.
        # This makes even subtle cortical/ventricular modifications really pop out!
        plt.imsave(os.path.join(out_dir, filename), data, cmap='inferno', vmin=0, vmax=40)
        
    save_heatmap(diff_forward, f"{patient_name}_diff_1_Forward_Modification.png")
    save_heatmap(diff_backward, f"{patient_name}_diff_2_Backward_Correction.png")
    save_heatmap(diff_cycle, f"{patient_name}_diff_3_Cycle_Error.png")
    
    print(f"Generated difference maps for {patient_name}")

def main():
    parser = argparse.ArgumentParser(description="Generate Difference Heatmaps for Presentations")
    parser.add_argument("--img_dir", type=str, default="presentation/presentation_images", help="Directory with the exported sequence PNGs")
    args = parser.parse_args()
    
    # Get all unique patient prefixes in the folder (e.g. out/presentation_images/patient_CN_00)
    all_files = glob.glob(os.path.join(args.img_dir, "*_1_*.png"))
    
    if not all_files:
        print(f"No sequence images found in {args.img_dir}")
        return
        
    prefixes = [f.replace("_1_" + f.split("_1_")[1], "") for f in all_files]
    diff_dir = os.path.join(args.img_dir, "differences_heatmaps")
    
    print(f"Found {len(prefixes)} patient sequences. Generating heatmaps in {diff_dir}...")
    for prefix in prefixes:
        process_patient_diff(prefix, diff_dir)
        
    print("✅ All difference maps generated successfully!")

if __name__ == "__main__":
    main()