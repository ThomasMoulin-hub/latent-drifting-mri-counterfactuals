import os
import torch
import torch.nn as nn
from torchvision import models
from torch.optim import Adam
import pandas as pd
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

class OASIS1AxialDataset(Dataset):
    def __init__(self, patient_ids, labels, data_root, transform=None, num_slices=3):
        self.data_root = data_root
        self.transform = transform
        self.samples = []

        for pid, label in zip(patient_ids, labels):
            vol_path = os.path.join(data_root, pid, "mri", "brain.mgz")
            if os.path.exists(vol_path):
                center_z = 128
                start_z = center_z - num_slices // 2
                for z in range(start_z, start_z + num_slices):
                    self.samples.append((vol_path, z, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vol_path, slice_idx, label = self.samples[idx]
        vol = nib.load(vol_path).get_fdata()
        slice_2d = vol[:, :, slice_idx].astype(np.float32)

        slice_min, slice_max = np.min(slice_2d), np.max(slice_2d)
        if slice_max - slice_min > 0:
            slice_2d = (slice_2d - slice_min) / (slice_max - slice_min)
        
        slice_2d = np.expand_dims(slice_2d, axis=-1)

        if self.transform:
            slice_2d = self.transform(slice_2d)

        return slice_2d, torch.tensor(label, dtype=torch.long)

def get_densenet_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def test_finetune_cpu():
    print("Setting up the dataset...")
    demographic_path = 'data/OASIS-1/DemographicAndClinicalData/oasis_cross-sectional-5708aa0a98d82080.xlsx'
    data_root = 'data/OASIS-1/FreeSurfer/oasis_cs_freesurfer_disc1/disc1'
    
    # Lecture des données
    df = pd.read_excel(demographic_path)
    df = df.dropna(subset=['CDR'])
    df['Label'] = (df['CDR'] > 0.0).astype(int)

    # Prendre un tout petit subset de patients pour le test rapide (ex: 10 patients)
    df_subset = df.head(10)
    patient_ids = df_subset['ID'].values
    labels = df_subset['Label'].values

    dataset = OASIS1AxialDataset(patient_ids, labels, data_root, transform=get_densenet_transforms(), num_slices=1)
    
    if len(dataset) == 0:
        print("Warning: No data loaded. Check paths and patient folders.")
        return

    # DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    print(f"Dataset length (slices): {len(dataset)}")
    
    print("Setting up DenseNet121 for linear probing...")
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    
    # Geler le backbone pour le linear probing
    for param in model.parameters():
        param.requires_grad = False
        
    # Modifier la dernière couche
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)
    
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.classifier.parameters(), lr=0.001)

    print("Starting a single epoch of training on CPU...")
    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"Batch {i+1}/{len(dataloader)} - Loss: {loss.item():.4f}")
        
        # Stop after 2 batches to keep it very fast
        if i >= 1:
            break
            
    print("Test finished successfully!")

if __name__ == '__main__':
    test_finetune_cpu()