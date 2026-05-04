import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from lightning import Trainer
from src.models.swin_unet import SwinUNetGenerator
from src.models.discriminator import PatchGANDiscriminator
from src.models.classifier import OASISClassifier
from src.systems.cyclegan_system import CycleGANSystem
from functools import partial

def test_cyclegan_system_fast_dev_run():
    """Test the CycleGAN system with a dummy fast dev run to verify the training step logic."""
    net_g_a2b = SwinUNetGenerator(spatial_dims=2, in_channels=1, out_channels=1, feature_size=12)
    net_g_b2a = SwinUNetGenerator(spatial_dims=2, in_channels=1, out_channels=1, feature_size=12)
    net_d_a = PatchGANDiscriminator(spatial_dims=2, in_channels=1, num_layers=2, hidden_channels=16)
    net_d_b = PatchGANDiscriminator(spatial_dims=2, in_channels=1, num_layers=2, hidden_channels=16)
    evaluator = OASISClassifier(in_channels=1, num_classes=2, spatial_dims=2)
    
    optimizer_g = partial(torch.optim.Adam, lr=0.0002)
    optimizer_d = partial(torch.optim.Adam, lr=0.0002)
    
    system = CycleGANSystem(
        net_g_a2b=net_g_a2b,
        net_g_b2a=net_g_b2a,
        net_d_a=net_d_a,
        net_d_b=net_d_b,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        evaluator=evaluator
    )
    
    # Create a small dummy dataset containing both domains (labels 0 and 1)
    dummy_images = torch.randn(4, 1, 64, 64)
    dummy_labels = torch.tensor([0, 0, 1, 1])
    dataset = TensorDataset(dummy_images, dummy_labels)
    
    # Custom collate function to mimic OASISDataModule dict output
    def collate_fn(batch):
        imgs = torch.stack([item[0] for item in batch])
        lbls = torch.stack([item[1] for item in batch])
        return {"image": imgs, "label": lbls}
        
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    
    trainer = Trainer(fast_dev_run=True, logger=False, enable_checkpointing=False)
    
    # Run fit
    trainer.fit(system, train_dataloaders=dataloader, val_dataloaders=dataloader)
    
    # If we got here without errors, the training_step logic works correctly
    assert True
