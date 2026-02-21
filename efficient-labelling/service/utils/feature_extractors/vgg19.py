import torch
import torch.nn as nn
import torchvision.models as models

class VGG19FeatureExtractor(nn.Module):
    """
    Feature extractor using VGG19 (removes the classifier head).
    """

    def __init__(self, pretrained=True, remove_fc=True):
        super(VGG19FeatureExtractor, self).__init__()
        # Load VGG19 with or without pretrained weights
        self.vgg19 = models.vgg19(pretrained=pretrained)
        
        # Remove the fully connected layers (only keep convolutional features)
        if remove_fc:
            self.vgg19 = self.vgg19.features

    def forward(self, x):
        return self.vgg19(x)  # Extract feature maps

# Function to load the model from checkpoint
def vgg19_model(checkpoint_path=None):
    """
    Load the VGG19 model with optional checkpoint.
    """
    model = VGG19FeatureExtractor(pretrained=False)  # Do not download weights automatically
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
    return model

