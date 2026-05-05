import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class SwinUNetGenerator(nn.Module):
    """
    Generator based on SwinUNETR from MONAI for 2D images.
    Preserves spatial dimensions of the input.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        feature_size: int = 24,
    ):
        """
        Args:
            spatial_dims: Number of spatial dimensions (default: 2)
            in_channels: Number of input channels (default: 1)
            out_channels: Number of output channels (default: 1)
            feature_size: Size of the features (default: 24)
        """
        super().__init__()

        self.model = SwinUNETR(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor. Shape: (B, C_in, H, W)
            
        Returns:
            torch.Tensor: Output tensor. Shape: (B, C_out, H, W) in range [-1, 1]
        """
        # x shape: (B, C_in, H, W)
        out = self.model(x)
        # out shape: (B, C_out, H, W)
        
        # Apply Tanh to bound output between [-1, 1]
        return torch.tanh(out)
