#!/user/tmm2219/.conda/envs/qt_env/bin/python
import os
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from monai.transforms import Compose, ScaleIntensity, Resize, EnsureChannelFirst, SqueezeDim
import torch
import argparse
from tqdm import tqdm

def create_dummy_oasis_data(raw_dir: Path):
    """Crée des fichiers NIfTI de test si le dossier raw est vide."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Création d'un petit volume 3D (64, 64, 64)
    for i in range(2):
        data = np.random.rand(64, 64, 64).astype(np.float32)
        affine = np.eye(4)
        image = nib.Nifti1Image(data, affine)
        
        patient_id = f"OAS1_{i:04d}"
        file_path = raw_dir / f"{patient_id}_MR1.nii.gz"
        nib.save(image, file_path)
        print(f"Dummy data created: {file_path}")

def preprocess_oasis(
    raw_dir: str,
    demographic_path: str,
    processed_dir: str,
    num_slices: int = 3,
    size: int = 128
):
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Chargement des métadonnées réelles
    print(f"Loading demographics from {demographic_path}...")
    if demographic_path.endswith('.xlsx'):
        df_demo = pd.read_excel(demographic_path)
    else:
        df_demo = pd.read_csv(demographic_path)
    
    # Nettoyage et mapping (CDR > 0 => AD, sinon CN)
    df_demo = df_demo.dropna(subset=['CDR'])
    df_demo['label'] = df_demo['CDR'].apply(lambda x: "AD" if x > 0 else "CN")
    # Création d'un dictionnaire pour un accès rapide : ID -> label, age
    demo_dict = df_demo.set_index('ID')[['label', 'Age']].to_dict('index')

    # 2. Recherche des volumes FreeSurfer (fichiers brain.mgz)
    # Structure attendue : raw_dir/OAS1_0001_MR1/mri/brain.mgz
    mgz_files = list(raw_path.glob("**/mri/brain.mgz"))
    
    if not mgz_files:
        print(f"No brain.mgz files found in {raw_dir}")
        return

    # MONAI transforms pour le preprocessing
    transforms = Compose([
        ScaleIntensity(minv=0.0, maxv=1.0), # Normalisation entre 0 et 1
        Resize((size, size, size)),
        SqueezeDim(0)
    ])

    metadata = []

    print(f"Processing {len(mgz_files)} volumes...")
    for f_path in tqdm(mgz_files):
        # f_path est .../OAS1_0001_MR1/mri/brain.mgz
        # Le patient_id est le nom du dossier parent de 'mri'
        patient_id = f_path.parent.parent.name
        
        if patient_id not in demo_dict:
            continue
            
        label = demo_dict[patient_id]['label']
        age = demo_dict[patient_id]['Age']
        
        # Chargement
        img = nib.load(f_path)
        data = img.get_fdata().astype(np.float32)
        
        # Transformation
        data_tensor = torch.from_numpy(data).unsqueeze(0) 
        processed_volume = transforms(data_tensor).numpy()
        
        # Extraction des coupes axiales centrales
        center = size // 2
        start = center - (num_slices // 2)
        
        for i in range(num_slices):
            slice_idx = start + i
            # Dans brain.mgz de FreeSurfer, l'orientation est souvent telle que l'axe 2 est axial
            slice_2d = processed_volume[:, :, slice_idx]
            
            slice_name = f"{patient_id}_slice{slice_idx:03d}.npy"
            np.save(proc_path / slice_name, slice_2d)
            
            metadata.append({
                "patient_id": patient_id,
                "slice_path": slice_name,
                "label": label,
                "age": age,
                "slice_idx": slice_idx
            })

    # Sauvegarde du nouveau metadata.csv
    df = pd.DataFrame(metadata)
    df.to_csv(proc_path / "metadata.csv", index=False)
    print(f"Finished. Saved {len(df)} slices and metadata to {proc_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/OASIS-1/FreeSurfer/oasis_cs_freesurfer_disc1/disc1")
    parser.add_argument("--demographic_path", type=str, default="data/OASIS-1/DemographicAndClinicalData/oasis_cross-sectional-5708aa0a98d82080.xlsx")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--num_slices", type=int, default=3)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    
    preprocess_oasis(args.raw_dir, args.demographic_path, args.processed_dir, args.num_slices, args.size)
