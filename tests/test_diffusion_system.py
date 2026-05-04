import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from lightning import Trainer
from src.models.latent_diffusion import ConditionalDiffusionModel
from src.systems.diffusion_system import DiffusionSystem
from functools import partial

def test_diffusion_system_fast_dev_run():
    """Test the DiffusionSystem with a dummy fast dev run."""
    # Create tiny model for fast testing
    net = ConditionalDiffusionModel(
        in_channels=1,
        out_channels=1,
        sample_size=32,
        condition_dim=4,
        cross_attention_dim=32,
        block_out_channels=(32, 64),
        layers_per_block=1,
        down_block_types=(
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
        ),
    )
    
    optimizer = partial(torch.optim.AdamW, lr=1e-4)
    
    system = DiffusionSystem(
        net=net,
        optimizer=optimizer,
        num_train_timesteps=100
    )
    
    # Create dummy dataset
    dummy_images = torch.randn(4, 1, 32, 32)
    # 4 conditions: age_z, z_ad, z_als, z_ftd
    dummy_conditions = torch.rand(4, 4)
    
    dataset = TensorDataset(dummy_images, dummy_conditions)
    
    def collate_fn(batch):
        imgs = torch.stack([item[0] for item in batch])
        conds = torch.stack([item[1] for item in batch])
        return {"image": imgs, "clinical_sliders": conds}
        
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    trainer = Trainer(fast_dev_run=True, logger=False, enable_checkpointing=False)
    
    # Run fit
    trainer.fit(system, train_dataloaders=dataloader, val_dataloaders=dataloader)
    
    assert True
