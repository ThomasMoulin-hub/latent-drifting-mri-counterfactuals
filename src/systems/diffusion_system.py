import torch
import torch.nn as nn
from lightning import LightningModule
from typing import Dict, Any, Tuple
from diffusers import DDPMScheduler
import torch.nn.functional as F

class DiffusionSystem(LightningModule):
    """
    LightningModule for training a Conditional Latent Diffusion Model.
    """
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        num_train_timesteps: int = 1000,
        lambda_cycle: float = 0.1,
    ):
        super().__init__()
        
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        
        # Noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        
    def forward(self, x: torch.Tensor, timestep: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(x, timestep, condition)
        
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        conditions = batch["clinical_sliders"]
        
        # Sample noise to add to the images
        noise = torch.randn_like(images)
        bsz = images.shape[0]
        
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=images.device
        ).long()
        
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)
        
        # Predict the noise residual
        noise_pred = self(noisy_images, timesteps, conditions)
        
        # Standard diffusion loss (MSE on noise)
        loss_diff = F.mse_loss(noise_pred, noise)
        
        # Optional: structural/cycle consistency approximation
        # We can predict x_0 from the noisy image and the predicted noise
        # and enforce an L1 penalty with the original image to preserve structure.
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(images.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1)
        
        pred_x0 = (noisy_images - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod
        loss_cycle = F.l1_loss(pred_x0, images)
        
        loss = loss_diff + self.hparams.lambda_cycle * loss_cycle
        
        self.log("train/loss_diff", loss_diff, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_cycle", loss_cycle, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        conditions = batch["clinical_sliders"]
        
        noise = torch.randn_like(images)
        bsz = images.shape[0]
        
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=images.device
        ).long()
        
        noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)
        noise_pred = self(noisy_images, timesteps, conditions)
        
        loss_diff = F.mse_loss(noise_pred, noise)
        
        self.log("val/loss_diff", loss_diff, on_step=False, on_epoch=True, prog_bar=True)
        return loss_diff

    @torch.no_grad()
    def interpolate_conditions(self, c_start: torch.Tensor, c_end: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Interpolate linearly between c_start and c_end.
        
        Args:
            c_start (torch.Tensor): The starting condition vector. Shape: (B, condition_dim)
            c_end (torch.Tensor): The ending condition vector. Shape: (B, condition_dim)
            steps (int): The number of intermediate steps to generate.
            
        Returns:
            torch.Tensor: Interpolated conditions. Shape: (steps, B, condition_dim)
        """
        alphas = torch.linspace(0.0, 1.0, steps, device=c_start.device)
        alphas = alphas.view(steps, 1, 1)  # Shape for broadcasting
        
        c_start_expanded = c_start.unsqueeze(0)  # Shape: (1, B, condition_dim)
        c_end_expanded = c_end.unsqueeze(0)      # Shape: (1, B, condition_dim)
        
        interpolated = (1.0 - alphas) * c_start_expanded + alphas * c_end_expanded
        return interpolated

    @torch.no_grad()
    def sample_trajectory(
        self, 
        x_0: torch.Tensor, 
        c_start: torch.Tensor, 
        c_end: torch.Tensor, 
        steps: int = 5,
        noise_level: float = 0.5,
        evaluator: nn.Module = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Generate a counterfactual trajectory by partially noising x_0 and denoising 
        with interpolated conditions.
        
        Args:
            x_0 (torch.Tensor): Original images. Shape: (B, C, H, W)
            c_start (torch.Tensor): Starting condition. Shape: (B, condition_dim)
            c_end (torch.Tensor): Ending condition. Shape: (B, condition_dim)
            steps (int): Number of steps in the trajectory.
            noise_level (float): Percentage of total timesteps to noise the image.
            evaluator (nn.Module, optional): Frozen classifier to evaluate fakes.
            
        Returns:
            Tuple[torch.Tensor, list]: 
                - Trajectory images. Shape: (steps, B, C, H, W)
                - Evaluation scores (list of lists, steps x B, containing class 1 probability)
        """
        # Set scheduler to inference mode
        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        
        conditions = self.interpolate_conditions(c_start, c_end, steps)
        
        max_timestep = int(self.noise_scheduler.config.num_train_timesteps * noise_level)
        if max_timestep <= 0:
            max_timestep = 1
            
        bsz = x_0.shape[0]
        t_max = torch.full((bsz,), max_timestep - 1, device=x_0.device, dtype=torch.long)
        
        trajectory_images = []
        eval_scores = []
        
        # We evaluate the process for each interpolated condition
        for i in range(steps):
            current_cond = conditions[i]
            
            # Start from x_0 and add noise up to max_timestep
            noise = torch.randn_like(x_0)
            x_t = self.noise_scheduler.add_noise(x_0, noise, t_max)
            
            # Denoising loop from max_timestep down to 0
            for current_t in reversed(range(max_timestep)):
                t_batch = torch.full((bsz,), current_t, device=x_0.device, dtype=torch.long)
                
                # Predict noise
                noise_pred = self(x_t, t_batch, current_cond)
                
                # Compute previous image x_{t-1}
                step_output = self.noise_scheduler.step(noise_pred, current_t, x_t)
                x_t = step_output.prev_sample
                
            trajectory_images.append(x_t)
            
            if evaluator is not None:
                evaluator.eval()
                logits = evaluator(x_t)
                probs = torch.softmax(logits, dim=1)
                # Assuming class 1 is the target condition (e.g. AD)
                ad_scores = probs[:, 1].tolist()
                eval_scores.append(ad_scores)
                
        trajectory_images = torch.stack(trajectory_images)
        return trajectory_images, eval_scores
