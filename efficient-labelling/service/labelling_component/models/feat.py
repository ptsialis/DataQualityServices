import os
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from service.labelling_component.feat_inference.models.feat3 import FEAT 

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResNet18Encoder(nn.Module):
    def __init__(self, hidden_dim=640, dropout_rate=0.3):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.projector = nn.Linear(512, hidden_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.projector(x)


class FEATWrapper:
    def __init__(
        self,
        model_path=None,
        hidden_dim=640,
        dropout_rate=0.3,
        temperature=1.0,
        use_cosine=True,
        lr=0.01,
        fine_tune_epochs=5
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = ResNet18Encoder(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.model = FEAT(encoder=self.encoder).to(self.device)

        if model_path:
            self._load_weights(model_path)

        self.model.eval()
        self.transform = self._preprocess()
        self.fine_tune_epochs = fine_tune_epochs
        self.lr = lr

    def _load_weights(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(path, map_location=self.device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state, strict=False)

    def _preprocess(self):
        return transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def fine_tune(self, support_loader):
        """Fine-tune FEAT model on a support set provided as a DataLoader with error handling."""
        self.model.train()
        logger.info("Starting fine-tuning of FEAT model...")

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, nesterov=True)

        try:
            for epoch in range(self.fine_tune_epochs):
                epoch_loss = 0.0
                for batch_idx, (images, labels) in enumerate(support_loader):
                    try:
                        images = images.to(self.device)
                        labels = labels.to(self.device)

                        optimizer.zero_grad()
                        logits, _ = self.model(images, labels, images, labels, mode="train")

                        if logits.shape[0] != labels.shape[0]:
                            raise ValueError(f"Logits shape {logits.shape} does not match labels shape {labels.shape}")

                        loss = F.cross_entropy(logits, labels)
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.error("CUDA out-of-memory error during fine-tuning batch %d. Skipping batch.", batch_idx)
                            torch.cuda.empty_cache()
                            continue
                        else:
                            logger.exception("Runtime error during fine-tuning batch %d", batch_idx)
                            raise

                    except Exception as e:
                        logger.exception("Unexpected error in batch %d during fine-tuning", batch_idx, e)
                        raise

                logger.info(f"[Fine-tune] Epoch {epoch + 1}/{self.fine_tune_epochs}, Loss: {epoch_loss:.4f}")

        except Exception as e:
            logger.error("Fine-tuning failed due to an unrecoverable error: %s", str(e))
            self.model.eval()
            raise

        logger.info("Fine-tuning completed.")
        self.model.eval()

    def predict(self, support_loader, query_loader):
        """Predict class labels for query images given support data, all via DataLoaders"""
        self.model.eval()

        support_images, support_labels = self._gather_from_loader(support_loader)
        query_images, _ = self._gather_from_loader(query_loader)

        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)

        with torch.no_grad():
            dummy_query_labels = torch.zeros(query_images.size(0), dtype=torch.long).to(self.device)
            logits = self.model(support_images, support_labels, query_images, dummy_query_labels, mode="eval")
            preds = logits.argmax(dim=1).cpu()

        return preds

    def _gather_from_loader(self, loader, expect_labels=False):
        """Helper to concatenate all data from a DataLoader into a single batch"""
        images, labels = [], []

        for batch in loader:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                img, lbl = batch
                images.append(img)
                labels.append(lbl)
            elif isinstance(batch, torch.Tensor):
                images.append(batch)
                if expect_labels:
                    raise RuntimeError("Expected labels but got only images.")
            else:
                raise RuntimeError(f"Unexpected batch format: {type(batch)}")

        images = torch.cat(images)
        labels = torch.cat(labels) if labels else torch.zeros(images.size(0)).long()

        return images, labels