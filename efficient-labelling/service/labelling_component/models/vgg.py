import logging
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VGG:
    def __init__(self, model_path=None, architecture="vgg19", dataset="imagenet"):
        self.architecture = architecture
        
        logger.info(f"Initializing VGG model with architecture: {self.architecture} and dataset:{dataset}")

        if self.architecture == "vgg19":
            self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        elif self.architecture == "vgg16":
            self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        else:
            raise ValueError("Unsupported architecture. Choose 'vgg19' or 'vgg16'.")
        
        
        if dataset == "imagenet":
            self.model.classifier[-1] = nn.Linear(4096, 1000)
        elif dataset == "miniimagenet":
            self.model.classifier[-1] = nn.Linear(4096, 100)
        else:
            raise ValueError("Unsupported dataset. Currently only 'miniimagenet' is supported.")

        logger.info(f"Classifier weights shape: {self.model.classifier[-1].weight.shape}") 

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        logger.info(f"Using device: {self.device}")
        

        if model_path:
            logger.info(f"Loading model weights from '{model_path}'...")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            # self.model.load_state_dict(torch.load(model_path, ))

        self.model.to(self.device)
        self.model.eval()

        self.transform = self._preprocess()

    def predict(self, dataloader):
        """Predict class labels for a batch of images from a DataLoader."""
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, _ = batch  # Ignore labels if present
                else:
                    images = batch

                images = images.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu())
                
                print(outputs.softmax(dim=1))  # See confidence distribution
                print(preds)

        return torch.cat(all_preds).numpy()

    def _preprocess(self):
        """Define preprocessing transform for input images."""
        return transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Standard ImageNet mean
                std=[0.229, 0.224, 0.225]    # Standard ImageNet std
            )
        ])
