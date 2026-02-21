import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd

from sklearn import logger
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



class VGG():

    def __init__(self, num_classes=1000, architecture="vgg16", pretrained=False, finetuned=False):
        self.num_classes = num_classes
        self.architecture = architecture
        self.pretrained = pretrained
        self.finetuned = finetuned

        if self.architecture == "vgg19":
            weights = models.VGG19_Weights.IMAGENET1K_V1 if self.pretrained else None
            self.model = models.vgg19(weights=weights)
        elif self.architecture == "vgg16":
            weights = models.VGG16_Weights.IMAGENET1K_V1 if self.pretrained else None
            self.model = models.vgg16(weights=weights)
        else:
            raise ValueError("Unsupported architecture. Choose 'vgg19' or 'vgg16'.")

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        self.transform = self._preprocess()

        if self.finetuned:
            in_feats = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_feats, num_classes).to(self.device)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def fit(self, train_loader, val_loader=None, epochs=200, lr=1e-3, patience=10, device='cpu'):
        if self.finetuned:
            raise RuntimeError("Model is set up for finetuning—call `.finetune()` instead of `.fit()`")

        history_path = f"{self.architecture}_training_history.csv"
        self.model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        best_val_acc = 0.0
        epochs_wo_imp = 0

        history = {
            'epoch': [], 'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': []
        }

        for epoch in range(1, epochs+1):
            self.model.train()
            running_loss = 0.0
            correct = total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

            train_loss = running_loss / len(train_loader.dataset)
            train_acc = correct / total
            print(f"Epoch {epoch}/{epochs} — Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

            val_loss = val_acc = float('nan')
            if val_loader is not None:
                self.model.eval()
                v_loss = v_correct = v_total = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = self.model(inputs)
                        l = criterion(outputs, labels)
                        v_loss += l.item() * inputs.size(0)
                        preds = outputs.argmax(dim=1)
                        v_total += labels.size(0)
                        v_correct += (preds == labels).sum().item()

                val_loss = v_loss / len(val_loader.dataset)
                val_acc = v_correct / v_total
                print(f"           Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    epochs_wo_imp = 0
                else:
                    epochs_wo_imp += 1
                    if epochs_wo_imp >= patience:
                        print(f"Early stopping after {epoch} epochs.")
                        break

            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f"Training history saved to '{history_path}'")

    def extract_representations(self, dataloader, flatten=True, with_labels=False):
        self.model.eval()
        features = []
        labels = [] if with_labels else None

        # with torch.no_grad():
        #     for images, lbls in tqdm(dataloader, desc="Extracting representations", unit="batch"):
        #         images = images.to(self.device)
        #         out = self.model.features(images)
        #         features.append(out.cpu())
        #         if with_labels:
        #             labels.append(lbls.cpu())
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting representations", unit="batch"):
                if with_labels:
                    images, lbls = batch
                else:
                    images = batch

                # print("Images shape:", images, images[0])
                images = images.to(self.device)
                out = self.model.features(images)
                features.append(out.cpu())
                
                if with_labels and with_labels:
                    labels.append(lbls.cpu())

        # Concatenate all batches
        features = torch.cat(features, dim=0)
        if flatten:
            features = features.view(features.size(0), -1)

        if with_labels:
            labels = torch.cat(labels, dim=0)
            return features, labels
        return features


    def finetune(self, train_loader, val_loader, epochs=50):
        if not self.finetuned:
            raise RuntimeError("Model not initialised for finetuning—construct with `finetuned=True`.")

        for epoch in range(1, epochs+1):
            self.model.train()
            run_loss = 0.0
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = self.criterion(outputs, lbls)
                loss.backward()
                self.optimizer.step()
                run_loss += loss.item()
            train_loss = run_loss / len(train_loader)

            self.model.eval()
            val_loss = correct = total = 0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                    outputs = self.model(imgs)
                    val_loss += self.criterion(outputs, lbls).item()
                    preds = outputs.argmax(dim=1)
                    correct += (preds == lbls).sum().item()
                    total += lbls.size(0)

            val_loss /= len(val_loader)
            val_acc = 100 * correct / total if total else 0
            print(f"Epoch [{epoch}/{epochs}] — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    def predict(self, images):
        self.model.eval()
        imgs_t = torch.stack([self.transform(img) for img in images]).to(self.device)
        with torch.no_grad():
            outputs = self.model(imgs_t)
            preds = outputs.argmax(dim=1)
        return preds.cpu().numpy()

    def save_weights(self):
        path = f"{self.architecture}_weights.pth"
        torch.save(self.model.state_dict(), path)
        print(f"Model weights saved to '{path}'")

    def _preprocess(self):
        if self.architecture == "vgg19":
            return transforms.Compose([
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48235, 0.45882, 0.40784],
                    std=[0.00392156862745098] * 3
                ),
            ])
