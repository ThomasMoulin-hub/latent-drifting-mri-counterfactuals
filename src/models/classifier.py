import torch
import torch.nn as nn
import torchvision.models as models
import logging

log = logging.getLogger(__name__)

class OASISClassifier(nn.Module):
    """
    2D Classifier for OASIS AD vs CN slices.
    Uses ResNet50, optimized for the Alzheimer's Detection task.
    
    Pretrained Model Reference:
    This model utilizes or is inspired by the pretrained weights from:
    https://github.com/mrinoybanerjee/Alzheimer_Detection
    (Specifically the 'alzheimer_cnn_model.pth' trained on 2D Alzheimer's MRI slices)
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        spatial_dims: int = 2,
        pretrained: bool = False,
        weights_path: str = None
    ):
        super().__init__()
        
        if spatial_dims != 2:
            raise ValueError("OASISClassifier currently only supports 2D spatial dimensions.")
            
        # Using DenseNet121
        self.model = models.densenet121(weights=None)
        
        # Adapt first convolutional layer for 1 channel input instead of 3 (RGB)
        if in_channels != 3:
            original_conv0 = self.model.features.conv0
            self.model.features.conv0 = nn.Conv2d(
                in_channels, 
                original_conv0.out_channels, 
                kernel_size=original_conv0.kernel_size, 
                stride=original_conv0.stride, 
                padding=original_conv0.padding, 
                bias=original_conv0.bias
            )
            # Initialize new conv0 with the mean of the original 3 channels
            with torch.no_grad():
                self.model.features.conv0.weight[:] = original_conv0.weight.sum(dim=1, keepdim=True)
                
        # Adapt final fully connected layer
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)
        
        # Load specific pretrained weights if provided
        if pretrained and weights_path:
            try:
                state_dict = torch.load(weights_path, map_location="cpu")
                
                # Check if we need to adjust num_classes based on weights
                if "classifier.weight" in state_dict:
                    weights_classes = state_dict["classifier.weight"].shape[0]
                    if weights_classes != num_classes:
                        log.info(f"Adjusting num_classes from {num_classes} to {weights_classes} to match pretrained weights.")
                        self.model.classifier = nn.Linear(num_ftrs, weights_classes)
                
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                
                if len(pretrained_dict) == 0:
                    log.warning(f"❌ No matching weights found in {weights_path}")
                else:
                    missing_keys = set(model_dict.keys()) - set(pretrained_dict.keys())
                    log.info(f"✅ Successfully loaded {len(pretrained_dict)}/{len(model_dict)} layers from {weights_path}")
                    if missing_keys:
                        log.info(f"⚠️ Missing layers (using random init): {list(missing_keys)[:10]}...")
                    
                    model_dict.update(pretrained_dict)
                    self.model.load_state_dict(model_dict)
            except Exception as e:
                log.warning(f"Could not load weights from {weights_path}: {e}. Proceeding with random weights.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor of shape (B, C, H, W)
        :return: logits of shape (B, num_classes)
        """
        return self.model(x)

if __name__ == "__main__":
    model = OASISClassifier()
    dummy_input = torch.randn(2, 1, 128, 128)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
