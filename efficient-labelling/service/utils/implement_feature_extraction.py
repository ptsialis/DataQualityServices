import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from utils.feature_extractors.res12 import Res12
from utils.feature_extractors.convnet import ConvNet
from utils.feature_extractors.res18 import resnet10, resnet18, resnet34, resnet50, resnet101, resnet152
from utils.feature_extractors.WRN28 import Wide_ResNet
from utils.feature_extractors.vgg19 import vgg19_model
from torchvision import transforms
from torchvision.models import vgg19
from torch.utils.data import DataLoader
from .images_preprocessing import load_feat_data 

# Dictionary Mapping Model Names to Classes & Checkpoints
FEATURE_EXTRACTORS = {
    "resnet12": {
        "model_class": Res12,
        "checkpoint": os.path.join(BASE_DIR, "feature_extractors/Res12-pre.pth")
    },
    "resnet18": {
        "model_class": resnet18,
        "checkpoint": os.path.join(BASE_DIR, "feature_extractors/Res18-pre.pth")
    },
    "wrn": {
        "model_class": Wide_ResNet,
        "checkpoint": os.path.join(BASE_DIR, "feature_extractors/wrn_pre.pth"),
    },
    "convnet": {
        "model_class": ConvNet,
        "checkpoint": os.path.join(BASE_DIR, "feature_extractors/con-pre.pth"),
    },
    "vgg19": { 
        "model_class": vgg19,  # Use torchvision's built-in VGG19
        "checkpoint": os.path.join(BASE_DIR, "feature_extractors/VGG19-pre.pth"),
    }
}

def load_feature_extractor(model_name="resnet12"):
    """
    Loads the selected feature extractor and its pretrained weights.

    Args:
        model_name (str): Feature extractor to use. Options: 'resnet12', 'resnet18', 'wrn', 'convnet'.
    
    Returns:
        model (torch.nn.Module): Loaded feature extractor.
    """
    if model_name not in FEATURE_EXTRACTORS:
        raise ValueError(f"Unsupported feature extractor: {model_name}. Choose from {list(FEATURE_EXTRACTORS.keys())}.")

    model_class = FEATURE_EXTRACTORS[model_name]["model_class"]
    checkpoint_path = FEATURE_EXTRACTORS[model_name]["checkpoint"]

    # Initialize model
    model = model_class()
    
    # Load pretrained weights if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        if 'params' in state_dict:
            state_dict = state_dict['params']
        model.load_state_dict(state_dict, strict= False)
        print(f"[INFO] Loaded pretrained weights from {checkpoint_path}")
    else:
        print(f"[WARNING] No pretrained weights found for {model_name}. Using randomly initialized weights.")
    
    model.eval()  # Set to evaluation mode
    return model

def extract_features(data_loader, model_name="resnet12"):
    """
    Extracts features using the selected feature extractor.

    Args:
        data_loader (DataLoader): DataLoader with preprocessed images.
        model_name (str): Feature extractor to use.
    
    Returns:
        all_features (Tensor): Extracted features.
        all_labels (Tensor): Labels (if available).
        all_paths (List[str]): Image paths.
    """
    feature_extractor = load_feature_extractor(model_name)
    feature_extractor.to(torch.device('cpu')) 
    all_features, all_labels, all_paths = [], [], []

    with torch.no_grad():  # No gradients needed
        for batch_idx, (images, labels, paths) in enumerate(data_loader):
            images = images.to(torch.device('cpu'))  # Move to CPU (or GPU if available)
            features = feature_extractor(images)  # Extract features
            if isinstance(features, tuple):  # If the model outputs (features, logits), take only features
                features = features[0]

            all_features.append(features)
            all_labels.append(labels)
            all_paths.extend(paths)

    all_features = torch.cat(all_features, dim=0)  # Combine all batches
    all_labels = torch.cat(all_labels, dim=0)
    #all_labels = torch.cat([torch.tensor(label) for label in all_labels], dim=0)

    return all_features, all_labels, all_paths

# === TESTING THE SCRIPT ===
if __name__ == "__main__":
    selected_model = "resnet12"  # Change this to any supported model: 'resnet12', 'resnet18', 'wrn', 'convnet'
    print(f"[INFO] Using feature extractor: {selected_model}")

    # Load preprocessed data
    data_loader = load_feat_data('/Users/dimayasir/ai-alliance/service/utils/datasets/unlabelled_images', phase='test')
    #data_loader = load_feat_data('/Users/dimayasir/ai-alliance/service/utils/datasets/uncropped_images', phase='adaptation')

    # Extract Features
    features, labels, paths = extract_features(data_loader, model_name=selected_model)

    print(f"[INFO] Extracted Feature Shape: {features.shape}")

    #to run it: cd /Users/dimayasir/ai-alliance/service -> python -m utils.implement_feature_extraction
