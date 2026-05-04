import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.components.oasis_dataset import OASISDataset

def test_oasis_dataset_clinical_sliders(tmp_path: Path):
    """Test that OASISDataset returns a clinical_sliders tensor with 4 features (age_z, z_ad, z_als, z_ftd) of type float32."""
    
    # Create dummy data
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    csv_path = tmp_path / "metadata.csv"
    
    # Dummy image
    img_path = data_dir / "dummy_image.npy"
    np.save(img_path, np.zeros((128, 128)))
    
    # Dummy dataframe without als and ftd to test fallback to 0.0
    df = pd.DataFrame({
        "slice_path": ["dummy_image.npy"],
        "label": ["AD"],
        "age": [75.0]
    })
    df.to_csv(csv_path, index=False)
    
    # Instantiate dataset
    dataset = OASISDataset(csv_path=str(csv_path), data_dir=str(data_dir))
    
    assert len(dataset) == 1
    
    # Get item
    item = dataset[0]
    
    # Assertions
    assert "clinical_sliders" in item
    sliders = item["clinical_sliders"]
    
    assert isinstance(sliders, torch.Tensor)
    assert sliders.dtype == torch.float32
    assert sliders.shape == (4,)
    
    # Check values: age=75.0 -> 0.75, label="AD" -> 1.0, als=default(0.0), ftd=default(0.0)
    assert sliders[0].item() == pytest.approx(0.75)
    assert sliders[1].item() == pytest.approx(1.0)
    assert sliders[2].item() == pytest.approx(0.0)
    assert sliders[3].item() == pytest.approx(0.0)
