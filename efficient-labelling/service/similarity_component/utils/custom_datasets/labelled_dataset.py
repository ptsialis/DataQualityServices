import logging
import zipfile
import io
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LabelledDataset(Dataset):
    def __init__(self, zip_path, transform=None):
        self.zip_path = zip_path
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.archive = zipfile.ZipFile(self.zip_path, 'r')
        all_files = self.archive.namelist()

        # Filter only valid image files
        image_exts = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in all_files if f.lower().endswith(image_exts) and len(f.split('/')) >= 2]

        # Map class folder names to integer labels
        class_names = sorted(set(os.path.dirname(f).split('/')[0] for f in image_files))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

        self.samples = []
        for path in image_files:
            class_folder = os.path.dirname(path).split('/')[0]
            label = self.class_to_idx.get(class_folder)
            if label is not None:
                self.samples.append((path, label))

        if not self.samples:
            raise RuntimeError("No valid image files found in the ZIP archive.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            with self.archive.open(img_path) as file:
                img_bytes = file.read()
                if not img_bytes:
                    raise ValueError(f"Empty file: {img_path}")
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except (UnidentifiedImageError, ValueError, OSError) as e:
            print(f"[WARNING] Failed to load image {img_path}: {e}")
            next_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(next_idx)

        if self.transform:
            img = self.transform(img)
        else:
            raise RuntimeError("No transform provided for LabelledDataset.")

        return img, label
