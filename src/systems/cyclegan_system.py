import torch
import torch.nn as nn
from lightning import LightningModule
from typing import Dict, Any, Tuple
import torchvision
import wandb


class CycleGANSystem(LightningModule):
    """
    LightningModule for CycleGAN training.
    Handles adversarial and cycle consistency losses.
    """

    def __init__(
        self,
        net_g_a2b: nn.Module,
        net_g_b2a: nn.Module,
        net_d_a: nn.Module,
        net_d_b: nn.Module,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 0.5,
        evaluator: nn.Module = None,
    ):
        """
        Args:
            net_g_a2b: Generator from domain A to B
            net_g_b2a: Generator from domain B to A
            net_d_a: Discriminator for domain A
            net_d_b: Discriminator for domain B
            optimizer_g: Optimizer for generators (partially instantiated by Hydra)
            optimizer_d: Optimizer for discriminators (partially instantiated by Hydra)
            lambda_cycle: Weight for cycle consistency loss
            lambda_identity: Weight for identity loss
            evaluator: Frozen classifier to evaluate fakes
        """
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net_g_a2b", "net_g_b2a", "net_d_a", "net_d_b", "evaluator"])

        self.net_g_a2b = net_g_a2b
        self.net_g_b2a = net_g_b2a
        self.net_d_a = net_d_a
        self.net_d_b = net_d_b
        
        self.evaluator = evaluator
        if self.evaluator is not None:
            self.evaluator.eval()
            for param in self.evaluator.parameters():
                param.requires_grad = False

        # Important for GANs where multiple optimizers are updated manually
        self.automatic_optimization = False

        # Loss functions
        self.criterion_gan = nn.MSELoss()  # Least Squares GAN loss
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

    def forward(self, x: torch.Tensor, direction: str = "a2b") -> torch.Tensor:
        """
        Args:
            x: Input tensor
            direction: 'a2b' or 'b2a'
        """
        if direction == "a2b":
            return self.net_g_a2b(x)
        elif direction == "b2a":
            return self.net_g_b2a(x)
        else:
            raise ValueError(f"Invalid direction: {direction}")

    def configure_optimizers(self) -> Tuple[list, list]:
        """Configure optimizers for generators and discriminators."""
        opt_g = self.hparams.optimizer_g(
            params=list(self.net_g_a2b.parameters()) + list(self.net_g_b2a.parameters())
        )
        opt_d = self.hparams.optimizer_d(
            params=list(self.net_d_a.parameters()) + list(self.net_d_b.parameters())
        )
        return [opt_g, opt_d], []

    def _extract_domains(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper to extract domain A and B images from the batch."""
        if "A" in batch and "B" in batch:
            return batch["A"], batch["B"]
        
        # Fallback for the OASISDataModule which returns a mixed batch
        images = batch["image"]
        labels = batch["label"]
        
        # Domain A: CN (label 0), Domain B: AD (label 1)
        real_a = images[labels == 0]
        real_b = images[labels == 1]
        
        return real_a, real_b

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        real_a, real_b = self._extract_domains(batch)
        
        # If one domain is missing in the batch, skip
        if real_a.size(0) == 0 or real_b.size(0) == 0:
            return None

        opt_g, opt_d = self.optimizers()

        # -------------------
        # Train Generators
        # -------------------
        self.toggle_optimizer(opt_g)
        
        # Identity loss
        # G_A2B should be identity if fed B
        id_b = self.net_g_a2b(real_b)
        loss_id_b = self.criterion_identity(id_b, real_b) * self.hparams.lambda_cycle * self.hparams.lambda_identity
        
        # G_B2A should be identity if fed A
        id_a = self.net_g_b2a(real_a)
        loss_id_a = self.criterion_identity(id_a, real_a) * self.hparams.lambda_cycle * self.hparams.lambda_identity
        
        # GAN loss
        fake_b = self.net_g_a2b(real_a)
        pred_fake_b = self.net_d_b(fake_b)
        loss_gan_a2b = self.criterion_gan(pred_fake_b, torch.ones_like(pred_fake_b))
        
        fake_a = self.net_g_b2a(real_b)
        pred_fake_a = self.net_d_a(fake_a)
        loss_gan_b2a = self.criterion_gan(pred_fake_a, torch.ones_like(pred_fake_a))
        
        # Cycle consistency loss
        rec_a = self.net_g_b2a(fake_b)
        loss_cycle_a = self.criterion_cycle(rec_a, real_a) * self.hparams.lambda_cycle
        
        rec_b = self.net_g_a2b(fake_a)
        loss_cycle_b = self.criterion_cycle(rec_b, real_b) * self.hparams.lambda_cycle
        
        # Total G loss
        loss_g = loss_id_a + loss_id_b + loss_gan_a2b + loss_gan_b2a + loss_cycle_a + loss_cycle_b
        
        self.manual_backward(loss_g)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # -------------------
        # Train Discriminators
        # -------------------
        self.toggle_optimizer(opt_d)
        
        # D_A loss
        pred_real_a = self.net_d_a(real_a)
        loss_d_real_a = self.criterion_gan(pred_real_a, torch.ones_like(pred_real_a))
        
        pred_fake_a_detached = self.net_d_a(fake_a.detach())
        loss_d_fake_a = self.criterion_gan(pred_fake_a_detached, torch.zeros_like(pred_fake_a_detached))
        
        loss_d_a = (loss_d_real_a + loss_d_fake_a) * 0.5
        
        # D_B loss
        pred_real_b = self.net_d_b(real_b)
        loss_d_real_b = self.criterion_gan(pred_real_b, torch.ones_like(pred_real_b))
        
        pred_fake_b_detached = self.net_d_b(fake_b.detach())
        loss_d_fake_b = self.criterion_gan(pred_fake_b_detached, torch.zeros_like(pred_fake_b_detached))
        
        loss_d_b = (loss_d_real_b + loss_d_fake_b) * 0.5
        
        # Total D loss
        loss_d = loss_d_a + loss_d_b
        
        self.manual_backward(loss_d)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)
        
        # Logging
        self.log("train/loss_g", loss_g, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss_d", loss_d, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss_cycle", loss_cycle_a + loss_cycle_b, on_step=True, on_epoch=True)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        real_a, real_b = self._extract_domains(batch)
        if real_a.size(0) == 0 or real_b.size(0) == 0:
            return None
            
        fake_b = self.net_g_a2b(real_a)
        fake_a = self.net_g_b2a(real_b)
        
        rec_a = self.net_g_b2a(fake_b)
        rec_b = self.net_g_a2b(fake_a)
        
        # Compute validation losses (cycle consistency)
        loss_cycle_a = self.criterion_cycle(rec_a, real_a)
        loss_cycle_b = self.criterion_cycle(rec_b, real_b)
        val_loss_cycle = loss_cycle_a + loss_cycle_b
        
        self.log("val/loss_cycle", val_loss_cycle, on_epoch=True, prog_bar=True)
        
        # Evaluate deception rate if evaluator is provided
        if self.evaluator is not None:
            # Domain A is CN (class 0), Domain B is AD (class 1)
            # We want fake_b (generated AD) to be classified as AD (1)
            logits_fake_b = self.evaluator(fake_b)
            preds_fake_b = torch.argmax(logits_fake_b, dim=1)
            acc_fake_b = (preds_fake_b == 1).float().mean()
            
            # We want fake_a (generated CN) to be classified as CN (0)
            logits_fake_a = self.evaluator(fake_a)
            preds_fake_a = torch.argmax(logits_fake_a, dim=1)
            acc_fake_a = (preds_fake_a == 0).float().mean()
            
            deception_rate = (acc_fake_b + acc_fake_a) / 2.0
            self.log("val/deception_rate", deception_rate, on_epoch=True, prog_bar=True)
        
        # Log images to WandB on the first batch
        if batch_idx == 0 and self.logger is not None and hasattr(self.logger.experiment, "log"):
            # Ensure images are in [0, 1] for logging
            def to_01(t):
                return (t + 1.0) / 2.0
                
            # Create a grid for Domain A -> B -> A
            grid_a = torchvision.utils.make_grid(
                torch.cat([to_01(real_a[:4]), to_01(fake_b[:4]), to_01(rec_a[:4])], dim=0),
                nrow=real_a[:4].size(0)
            )
            
            # Create a grid for Domain B -> A -> B
            grid_b = torchvision.utils.make_grid(
                torch.cat([to_01(real_b[:4]), to_01(fake_a[:4]), to_01(rec_b[:4])], dim=0),
                nrow=real_b[:4].size(0)
            )
            
            # Note: checking __class__.__name__ to be safe against other loggers
            if self.logger.__class__.__name__ == "WandbLogger":
                import wandb
                self.logger.experiment.log({
                    "val/images_A_to_B_to_A": wandb.Image(grid_a, caption="Real A, Fake B, Rec A"),
                    "val/images_B_to_A_to_B": wandb.Image(grid_b, caption="Real B, Fake A, Rec B")
                })
        
        return val_loss_cycle
