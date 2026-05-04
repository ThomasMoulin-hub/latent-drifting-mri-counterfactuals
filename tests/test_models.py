import pytest
import torch

from src.models.swin_unet import SwinUNetGenerator
from src.models.discriminator import PatchGANDiscriminator


def test_swin_unet_generator():
    """Test the SwinUNetGenerator model."""
    # Create model
    model = SwinUNetGenerator(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        feature_size=24,
    )
    
    # Check that it's a torch.nn.Module
    assert isinstance(model, torch.nn.Module)
    
    # Create dummy input: Batch=2, Channels=1, Height=256, Width=256
    dummy_input = torch.randn(2, 1, 256, 256)
    
    # Forward pass
    output = model(dummy_input)
    
    # Verify dimensions are preserved
    assert output.shape == dummy_input.shape
    assert output.shape == (2, 1, 256, 256)


def test_patchgan_discriminator():
    """Test the PatchGANDiscriminator model."""
    # Create model
    model = PatchGANDiscriminator(
        spatial_dims=2,
        in_channels=1,
        num_layers=3,
    )
    
    # Check that it's a torch.nn.Module
    assert isinstance(model, torch.nn.Module)
    
    # Create dummy input: Batch=2, Channels=1, Height=256, Width=256
    dummy_input = torch.randn(2, 1, 256, 256)
    
    # Forward pass
    output = model(dummy_input)
    
    # Output should be a 2D feature map of logits/probabilities
    # e.g. for 256x256 and 3 layers, the output is typically around 30x30 or 32x32 depending on padding
    assert len(output.shape) == 4
    assert output.shape[0] == 2 # Batch size
    assert output.shape[1] == 1 # Single channel output (real/fake logit)
    # the height and width should be significantly smaller than 256 due to strided convolutions
    assert output.shape[2] < 256
    assert output.shape[3] < 256
