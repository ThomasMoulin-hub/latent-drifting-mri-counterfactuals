import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class ConditionalDiffusionModel(nn.Module):
    """
    Wrapper around diffusers UNet2DConditionModel.
    Takes an image (or latent) and a condition vector (clinical sliders),
    projects the condition vector to the expected cross_attention_dim,
    and returns the predicted noise.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        sample_size: int = 128,
        condition_dim: int = 4,
        cross_attention_dim: int = 256,
        block_out_channels: tuple = (128, 256, 512, 512),
        layers_per_block: int = 2,
        down_block_types: tuple = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: tuple = (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
    ):
        super().__init__()
        
        self.condition_dim = condition_dim
        self.cross_attention_dim = cross_attention_dim
        
        # Projection for the clinical sliders (condition_dim -> cross_attention_dim)
        # We add an extra dimension to act as a sequence of length 1 for cross-attention.
        self.cond_proj = nn.Sequential(
            nn.Linear(condition_dim, cross_attention_dim),
            nn.SiLU(),
            nn.Linear(cross_attention_dim, cross_attention_dim)
        )
        
        # UNet model
        self.unet = UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            cross_attention_dim=cross_attention_dim,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        )
        
    def forward(self, x: torch.Tensor, timestep: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The noisy input tensor. Shape: (B, C, H, W)
            timestep (torch.Tensor): The timestep tensor. Shape: (B,)
            condition (torch.Tensor): The conditional sliders. Shape: (B, condition_dim)
            
        Returns:
            torch.Tensor: The predicted noise.
        """
        # Project condition and reshape for cross attention (B, seq_len, cross_attention_dim)
        # We use seq_len = 1
        cond_emb = self.cond_proj(condition).unsqueeze(1)
        
        # Forward pass through UNet
        out = self.unet(x, timestep, encoder_hidden_states=cond_emb)
        return out.sample
