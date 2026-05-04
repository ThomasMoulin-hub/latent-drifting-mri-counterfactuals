import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from src.models.classifier import OASISClassifier
from src.data.components.oasis_dataset import OASISDataset
import numpy as np
import logging

# Setup logging to see the model loading messages
logging.basicConfig(level=logging.INFO)

def test_classifier():
    # Configuration
    weights_path = "data/pretrained/alzheimer_model_fawazzx.pth"
    csv_path = "data/processed/metadata.csv"
    data_dir = "data/processed"
    
    # Load model with 3 channels as required by weights
    print(f"Loading model with weights from {weights_path}...")
    model = OASISClassifier(
        in_channels=3,
        num_classes=4,
        pretrained=True,
        weights_path=weights_path
    )
    model.eval()
    
    # Transforms
    # ResNet50 expects 224x224 and ImageNet normalization
    transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = OASISDataset(csv_path=csv_path, data_dir=data_dir)
    print(f"Dataset loaded with {len(dataset)} samples.")
    
    # Mapping for the 4-class model
    # Usually: 0: Mild, 1: Moderate, 2: NonDemented, 3: VeryMild
    class_names = {0: "Mild AD", 1: "Mod AD", 2: "CN", 3: "VMild AD"}
    
    indices = [0, 1, 2, 3, 4, 5, 20, 21, 22] 
    
    print("\nRunning inference...")
    print("-" * 80)
    print(f"{'Index':<7} | {'True':<5} | {'Pred (Class)':<15} | {'Confidence':<10} | {'Status'}")
    print("-" * 80)
    
    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            image_np = sample["image"] # (1, H, W)
            
            # Rescale from [-1, 1] to [0, 1]
            image_np = (image_np + 1.0) / 2.0
            image_np = np.clip(image_np, 0, 1)
            
            target_str = "AD" if sample["label"] == 1 else "CN"
            
            # Convert to 3 channels for transform
            image_3ch = np.repeat(image_np, 3, axis=0) # (3, H, W)
            # Move channels to last dim for PIL
            image_pil = (image_3ch.transpose(1, 2, 0) * 255).astype(np.uint8)
            
            input_tensor = transforms(image_pil).unsqueeze(0)
            
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()
            
            pred_name = class_names.get(pred_idx, "Unknown")
            
            # Simple binary evaluation: Class 2 is CN, others are some form of AD
            is_correct = False
            if target_str == "CN" and pred_idx == 2:
                is_correct = True
            elif target_str == "AD" and pred_idx != 2:
                is_correct = True
                
            status = "OK" if is_correct else "WRONG"
            
            print(f"{idx:<7} | {target_str:<5} | {pred_name:<15} | {confidence:.4f} | {status}")
    print("-" * 80)

if __name__ == "__main__":
    test_classifier()
