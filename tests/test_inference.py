import pytest
import torch
import torch.nn as nn
from src.systems.diffusion_system import DiffusionSystem
from src.models.latent_diffusion import ConditionalDiffusionModel
from functools import partial

class DummyEvaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(32 * 32, 2)
        
    def forward(self, x):
        # x is (B, 1, 32, 32)
        b = x.shape[0]
        x = x.view(b, -1)
        return self.layer(x)

def test_inference_interpolation_and_sampling():
    """Test that interpolate_conditions and sample_trajectory execute correctly."""
    net = ConditionalDiffusionModel(
        in_channels=1,
        out_channels=1,
        sample_size=32,
        condition_dim=4,
        cross_attention_dim=32,
        block_out_channels=(32, 64),
        layers_per_block=1,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
    )
    
    optimizer = partial(torch.optim.AdamW, lr=1e-4)
    system = DiffusionSystem(net=net, optimizer=optimizer, num_train_timesteps=10)
    
    x_0 = torch.randn(2, 1, 32, 32)
    c_start = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]])
    c_end = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.5, 1.0, 0.0, 0.0]])
    
    evaluator = DummyEvaluator()
    
    steps = 3
    # Check interpolation
    interpolated = system.interpolate_conditions(c_start, c_end, steps=steps)
    assert interpolated.shape == (steps, 2, 4)
    # the second step should be 50% between start and end
    assert torch.allclose(interpolated[1, 0, 1], torch.tensor(0.5))
    
    # Check sampling
    trajectory_images, eval_scores = system.sample_trajectory(
        x_0=x_0,
        c_start=c_start,
        c_end=c_end,
        steps=steps,
        noise_level=0.5, # max_timestep = 5
        evaluator=evaluator
    )
    
    assert trajectory_images.shape == (steps, 2, 1, 32, 32)
    assert len(eval_scores) == steps
    assert len(eval_scores[0]) == 2 # Batch size 2
    assert isinstance(eval_scores[0][0], float)
