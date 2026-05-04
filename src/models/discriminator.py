import torch
import torch.nn as nn
from monai.networks.nets import PatchDiscriminator


class PatchGANDiscriminator(nn.Module):
    """
    Discriminator based on PatchDiscriminator from MONAI for 2D images.
    Outputs a 2D map of logits representing real/fake predictions per patch.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        num_layers: int = 3,
        hidden_channels: int = 64,
    ):
        """
        Args:
            spatial_dims: Number of spatial dimensions (default: 2)
            in_channels: Number of input channels (default: 1)
            num_layers: Number of intermediate layers (default: 3)
            hidden_channels: Number of base hidden channels (default: 64)
        """
        super().__init__()

        self.model = PatchDiscriminator(
            spatial_dims=spatial_dims,
            num_layers_d=num_layers,
            in_channels=in_channels,
            channels=hidden_channels,
            out_channels=1,
            kernel_size=4,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor. Shape: (B, C_in, H, W)
            
        Returns:
            torch.Tensor: Logits map. Shape: (B, 1, H', W')
        """
        # x shape: (B, C_in, H, W)
        out = self.model(x)
        # The PatchDiscriminator from MONAI returns a list of tensors from intermediate layers.
        # The last element is the final logits map.
        logits = out[-1]
        # logits shape: (B, 1, H', W')
        return logits
