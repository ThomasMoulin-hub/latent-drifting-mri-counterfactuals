import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from src.data.components.oasis_dataset import OASISDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import logging
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

def extract_features(model, dataloader, device, transform):
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            images = batch["image"] # (B, 1, H, W)
            targets = batch["label"]
            
            # Rescale from [-1, 1] to [0, 1]
            images = (images + 1.0) / 2.0
            images = torch.clamp(images, 0.0, 1.0)
            
            # Convert to 3 channels
            images_3ch = images.repeat(1, 3, 1, 1) # (B, 3, H, W)
            
            # Apply ImageNet normalization and resize
            processed_images = []
            for i in range(images_3ch.shape[0]):
                # To PIL Image for resizing (or use T.Resize directly on tensor)
                img = T.ToPILImage()(images_3ch[i])
                img = transform(img)
                processed_images.append(img)
            
            batch_tensor = torch.stack(processed_images).to(device)
            
            out = model(batch_tensor)
            out = out.view(out.size(0), -1) # Flatten
            
            features.append(out.cpu().numpy())
            labels.append(targets.numpy())
            
    features = np.vstack(features)
    labels = np.concatenate(labels)
    return features, labels

def evaluate_features(features, labels):
    # Using 5-Fold Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    
    for train_idx, test_idx in cv.split(features, labels):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
        
    return np.mean(accuracies), np.std(accuracies)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration
    csv_path = "data/processed/metadata.csv"
    data_dir = "data/processed"
    
    # Load dataset
    dataset = OASISDataset(csv_path=csv_path, data_dir=data_dir)
    # Using batch_size=8 since we only have 40 samples
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    print(f"Dataset loaded with {len(dataset)} samples.")
    
    # Transformation: Resize to 224x224 and ImageNet normalize
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\n--- Setup DenseNet-121 (ImageNet) ---")
    densenet = models.densenet121(weights="IMAGENET1K_V1")
    # Remove classifier to get features
    densenet.classifier = nn.Identity()
    densenet = densenet.to(device)
    
    print("Extracting features with DenseNet-121...")
    dn_features, dn_labels = extract_features(densenet, dataloader, device, transform)
    dn_mean, dn_std = evaluate_features(dn_features, dn_labels)
    print(f"DenseNet-121 CV Accuracy: {dn_mean:.4f} ± {dn_std:.4f}")
    
    print("\n--- Setup ResNet-50 (RadImageNet) ---")
    resnet_rad = torch.hub.load('Warvito/radimagenet-models', 'radimagenet_resnet50')
    # The loaded model ends at layer4, so we need to add Global Average Pooling
    resnet_rad = nn.Sequential(
        resnet_rad,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    resnet_rad = resnet_rad.to(device)
    
    print("Extracting features with ResNet-50 (RadImageNet)...")
    rn_features, rn_labels = extract_features(resnet_rad, dataloader, device, transform)
    rn_mean, rn_std = evaluate_features(rn_features, rn_labels)
    print(f"ResNet-50 (RadImageNet) CV Accuracy: {rn_mean:.4f} ± {rn_std:.4f}")
    
    print("\n" + "="*50)
    print("FINAL COMPARISON RESULTS (5-Fold CV on 40 samples)")
    print("="*50)
    print(f"DenseNet-121 (ImageNet) : {dn_mean*100:.2f}% ± {dn_std*100:.2f}%")
    print(f"ResNet-50 (RadImageNet) : {rn_mean*100:.2f}% ± {rn_std*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
