#!/user/tmm2219/.conda/envs/DLBI/bin/python

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from src.data.components.oasis_dataset import OASISDataset
from src.models.classifier import OASISClassifier
import numpy as np
import logging
import warnings
from tqdm import tqdm
import wandb
import argparse
import os

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

def build_head(in_features, num_classes=2, mode="linear"):
    """
    Builds the classification head.
    - 'linear': Standard Linear Probing (1 layer)
    - 'mlp': Richer finetuning with 2 hidden layers, BatchNorm, and Dropout
    """
    if mode == "linear":
        return nn.Linear(in_features, num_classes)
    elif mode == "mlp":
        # Architecture advice: 
        # A 2-hidden layer MLP (512 -> 128) is an excellent standard for 1024/2048 feature extractors.
        # It adds enough non-linearity to map complex combinations of visual features to Alzheimer's markers,
        # but stays small enough to prevent massive overfitting. Dropout and BatchNorm are crucial here.
        return nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    else:
        raise ValueError("Mode must be 'linear' or 'mlp'")

def train_head(model, train_loader, val_loader, test_loader, device, run_name, mode="linear", in_channels=3, epochs=15, lr=1e-3):
    # Differentiate WandB group based on mode
    group_name = "linear_probing" if mode == "linear" else "mlp_finetuning"
    wandb.init(project="lightning-hydra-template", name=f"{run_name}_{mode}", group=group_name, reinit=True)
    
    # Only optimize parameters that require gradients (the new head)
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params_to_update, lr=lr)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Training loop
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [{run_name} Train]", leave=False):
            images = batch["image"].to(device) # (B, 1, H, W)
            targets = batch["label"].to(device)
            
            # Rescale from [-1, 1] to [0, 1]
            images = (images + 1.0) / 2.0
            images = torch.clamp(images, 0.0, 1.0)
            
            # Convert to 3 channels for backbone if needed
            if in_channels == 3:
                images = images.repeat(1, 3, 1, 1)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            else:
                mean = torch.tensor([0.5]).view(1, 1, 1, 1).to(device)
                std = torch.tensor([0.5]).view(1, 1, 1, 1).to(device)
            
            # Resize and normalize
            images = torch.nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            images = (images - mean) / std
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)
            
        train_acc = train_correct / train_total
        train_loss_avg = train_loss / train_total
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [{run_name} Val]", leave=False):
                images = batch["image"].to(device)
                targets = batch["label"].to(device)
                
                images = (images + 1.0) / 2.0
                images = torch.clamp(images, 0.0, 1.0)
                
                if in_channels == 3:
                    images = images.repeat(1, 3, 1, 1)
                
                images = torch.nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
                images = (images - mean) / std
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
                
        val_acc = val_correct / val_total
        val_loss_avg = val_loss / val_total
        
        print(f"Epoch {epoch+1}/{epochs} [{run_name}] - Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save the best model
            os.makedirs("data/pretrained", exist_ok=True)
            torch.save(model.state_dict(), f"data/pretrained/best_{run_name}_{mode}.pth")
            
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss_avg,
            "train/acc": train_acc,
            "val/loss": val_loss_avg,
            "val/acc": val_acc,
            "val/acc_best": best_val_acc,
        })
            
    # --- TEST PHASE ---
    print(f"\n--- Testing Best Model for {run_name} ({mode}) ---")
    try:
        model.load_state_dict(torch.load(f"data/pretrained/best_{run_name}_{mode}.pth", weights_only=False))
    except Exception as e:
        print(f"Could not load best model weights for testing: {e}")
        
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"[{run_name} Test]", leave=False):
            images = batch["image"].to(device)
            targets = batch["label"].to(device)
            
            images = (images + 1.0) / 2.0
            images = torch.clamp(images, 0.0, 1.0)
            
            if in_channels == 3:
                images = images.repeat(1, 3, 1, 1)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            else:
                mean = torch.tensor([0.5]).view(1, 1, 1, 1).to(device)
                std = torch.tensor([0.5]).view(1, 1, 1, 1).to(device)
                
            images = torch.nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            images = (images - mean) / std
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == targets).sum().item()
            test_total += targets.size(0)
            
    test_acc = test_correct / test_total
    print(f"[{run_name}] Test Accuracy: {test_acc:.4f}\n")
    wandb.log({"test/acc": test_acc})
    
    wandb.finish()
    return best_val_acc, test_acc

def main():
    parser = argparse.ArgumentParser(description="Head Finetuning for OASIS Dataset")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--mode", type=str, default="linear", choices=["linear", "mlp"], 
                        help="Head architecture: 'linear' for simple linear probing, 'mlp' for deep fully connected layers")
    args = parser.parse_args()
    
    epochs = args.epochs
    lr = args.lr
    mode = args.mode
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    csv_path = "data/processed/metadata.csv"
    data_dir = "data/processed"
    
    # Load dataset
    dataset = OASISDataset(csv_path=csv_path, data_dir=data_dir)
    print(f"Dataset loaded with {len(dataset)} samples.")
    
    # Split dataset 80/10/10
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"Training in '{mode}' mode for {epochs} epochs with learning rate {lr}")
    
    # ---------------------------------------------------------
    # 1. DenseNet-121 (Alzheimer Pretrained)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(f"--- {mode.upper()} Finetuning DenseNet-121 (Alzheimer Pretrained) ---")
    print("="*60)
    
    # Use the project's custom wrapper which loads the weights and handles 1-channel
    alz_model = OASISClassifier(in_channels=1, pretrained=True, weights_path="data/pretrained/alzheimer_cnn_model.pth")
    
    # Freeze backbone
    for param in alz_model.model.features.parameters():
        param.requires_grad = False
        
    num_ftrs = alz_model.model.classifier.in_features
    alz_model.model.classifier = build_head(num_ftrs, 2, mode=mode)
    
    alz_model = alz_model.to(device)
    alz_best_acc, alz_test_acc = train_head(alz_model, train_loader, val_loader, test_loader, device, run_name="densenet_alzheimer", mode=mode, in_channels=1, epochs=epochs, lr=lr)

    # ---------------------------------------------------------
    # 2. DenseNet-121 (ImageNet Pretrained)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(f"--- {mode.upper()} Finetuning DenseNet-121 (ImageNet) ---")
    print("="*60)
    imgnet_model = models.densenet121(weights="IMAGENET1K_V1")
    
    # Freeze backbone
    for param in imgnet_model.parameters():
        param.requires_grad = False
        
    # Replace and unfreeze classifier
    num_ftrs = imgnet_model.classifier.in_features
    imgnet_model.classifier = build_head(num_ftrs, 2, mode=mode)
    imgnet_model = imgnet_model.to(device)
    
    imgnet_best_acc, imgnet_test_acc = train_head(imgnet_model, train_loader, val_loader, test_loader, device, run_name="densenet_imagenet", mode=mode, in_channels=3, epochs=epochs, lr=lr)
    
    # ---------------------------------------------------------
    # 3. ResNet-50 (RadImageNet)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(f"--- {mode.upper()} Finetuning ResNet-50 (RadImageNet) ---")
    print("="*60)
    resnet_rad = torch.hub.load('Warvito/radimagenet-models', 'radimagenet_resnet50', verbose=False)
    
    # Freeze backbone
    for param in resnet_rad.parameters():
        param.requires_grad = False
        
    # The RadImageNet model ends at layer4, add pooling and a trainable linear classifier
    classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        build_head(2048, 2, mode=mode)
    )
    
    resnet_rad = nn.Sequential(
        resnet_rad,
        classifier
    )
    resnet_rad = resnet_rad.to(device)
    
    rn_best_acc, rn_test_acc = train_head(resnet_rad, train_loader, val_loader, test_loader, device, run_name="resnet_radimagenet", mode=mode, in_channels=3, epochs=epochs, lr=lr)
    
    print("\n" + "="*60)
    print(f"FINAL {mode.upper()} HEAD FINETUNING RESULTS (Train: {train_size}, Val: {val_size}, Test: {test_size} samples)")
    print("="*60)
    print(f"DenseNet-121 (Alzheimer Weights) Best Val Acc : {alz_best_acc*100:.2f}%  |  Test Acc: {alz_test_acc*100:.2f}%")
    print(f"DenseNet-121 (ImageNet Weights)  Best Val Acc : {imgnet_best_acc*100:.2f}%  |  Test Acc: {imgnet_test_acc*100:.2f}%")
    print(f"ResNet-50    (RadImageNet)       Best Val Acc : {rn_best_acc*100:.2f}%  |  Test Acc: {rn_test_acc*100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()