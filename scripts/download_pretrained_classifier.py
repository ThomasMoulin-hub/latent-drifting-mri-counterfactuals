import os
import urllib.request
from pathlib import Path

def download_weights():
    """
    Downloads pretrained weights for Alzheimer's Detection.
    """
    models = {
        "mrinoybanerjee": {
            "url": "https://github.com/mrinoybanerjee/Alzheimer_Detection/raw/main/Models/alzheimer_cnn_model.pth",
            "filename": "alzheimer_cnn_model.pth"
        },
        "fawazzx_adni": {
            "url": "https://huggingface.co/datasets/Duceh/datasets/resolve/main/resnet50.pth",
            "filename": "alzheimer_model_adni.pth"
        },
        "evanrsl": {
            "url": "https://huggingface.co/evanrsl/resnet-Alzheimer/resolve/main/pytorch_model.bin",
            "filename": "alzheimer_model_evanrsl.pth"
        },
        "alberto_mate": {
            "url": "https://github.com/alberto-mate/MRI-Alzheimer-Classifier/raw/master/models/ResNet50_best_model.pth",
            "filename": "alzheimer_model_alberto.pth"
        }
    }
    
    output_dir = Path("data/pretrained")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, info in models.items():
        output_path = output_dir / info["filename"]
        
        if output_path.exists():
            print(f"Weights for {name} already exist at {output_path}")
            continue
            
        print(f"Downloading {name} weights from {info['url']}...")
        try:
            # Hugging Face might require a User-Agent or handle redirects
            req = urllib.request.Request(info["url"], headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Successfully downloaded {name} weights to {output_path}")
        except Exception as e:
            print(f"Failed to download {name} weights: {e}")

if __name__ == "__main__":
    download_weights()
