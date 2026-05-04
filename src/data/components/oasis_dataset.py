import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Callable

class OASISDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        data_dir: str,
        transform: Optional[Callable] = None,
    ):
        """
        Dataset for OASIS 2D slices.
        
        :param csv_path: Path to the metadata CSV file.
        :param data_dir: Directory where .npy slices are stored.
        :param transform: Optional transformation to be applied on a sample.
        """
        self.data_dir = Path(data_dir)
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        
        # Mapping labels to integers
        self.label_map = {"CN": 0, "AD": 1}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.data_dir / row["slice_path"]
        image = np.load(img_path).astype(np.float32) # (H, W)
        
        # Map label
        label_str = row["label"]
        label = self.label_map[label_str]
        
        # Clinical sliders (age_z, z_ad, z_als, z_ftd)
        age_z = float(row.get("age", 0.0)) / 100.0 # simple normalization
        z_ad = float(label)
        z_als = float(row.get("als", 0.0))
        z_ftd = float(row.get("ftd", 0.0))
        
        clinical_sliders = np.array([age_z, z_ad, z_als, z_ftd], dtype=np.float32)
        
        # MONAI transforms usually expect (C, H, W)
        image = image[np.newaxis, ...] # (1, H, W)
        
        if self.transform:
            image = self.transform(image)
            
        return {
            "image": image,
            "label": label,
            "clinical_sliders": torch.from_numpy(clinical_sliders)
        }
