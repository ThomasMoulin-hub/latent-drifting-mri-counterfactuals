import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from src.data.components.oasis_dataset import OASISDataset
import numpy as np
import logging
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

def train_model(model, train_loader, val_loader, device, epochs=5, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Training loop
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            images = batch["image"].to(device) # (B, 1, H, W)
            targets = batch["label"].to(device)
            
            # Rescale from [-1, 1] to [0, 1]
            images = (images + 1.0) / 2.0
            images = torch.clamp(images, 0.0, 1.0)
            
            # Convert to 3 channels for backbone
            images = images.repeat(1, 3, 1, 1)
            
            # Note: We resize and normalize here in the training loop
            # ImageNet standard
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            
            # Resize
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
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                images = batch["image"].to(device)
                targets = batch["label"].to(device)
                
                images = (images + 1.0) / 2.0
                images = torch.clamp(images, 0.0, 1.0)
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
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/train_total:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss/val_total:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
    return best_val_acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    csv_path = "data/processed/metadata.csv"
    data_dir = "data/processed"
    
    # Load dataset
    dataset = OASISDataset(csv_path=csv_path, data_dir=data_dir)
    print(f"Dataset loaded with {len(dataset)} samples.")
    
    # Split dataset 80/20
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Hyperparameters
    epochs = 10
    lr = 1e-4
    
    print("\n" + "="*50)
    print("--- Finetuning DenseNet-121 (ImageNet) ---")
    print("="*50)
    densenet = models.densenet121(weights="IMAGENET1K_V1")
    # Replace classifier
    num_ftrs = densenet.classifier.in_features
    densenet.classifier = nn.Linear(num_ftrs, 2)
    densenet = densenet.to(device)
    
    dn_best_acc = train_model(densenet, train_loader, val_loader, device, epochs=epochs, lr=lr)
    
    print("\n" + "="*50)
    print("--- Finetuning ResNet-50 (RadImageNet) ---")
    print("="*50)
    resnet_rad = torch.hub.load('Warvito/radimagenet-models', 'radimagenet_resnet50', verbose=False)
    # The RadImageNet model ends at layer4, add pooling and linear classifier
    resnet_rad = nn.Sequential(
        resnet_rad,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(2048, 2)
    )
    resnet_rad = resnet_rad.to(device)
    
    rn_best_acc = train_model(resnet_rad, train_loader, val_loader, device, epochs=epochs, lr=lr)
    
    print("\n" + "="*50)
    print(f"FINAL COMPARISON RESULTS (Fine-Tuning on {len(dataset)} samples)")
    print("="*50)
    print(f"DenseNet-121 (ImageNet) Best Val Acc : {dn_best_acc*100:.2f}%")
    print(f"ResNet-50 (RadImageNet) Best Val Acc : {rn_best_acc*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
