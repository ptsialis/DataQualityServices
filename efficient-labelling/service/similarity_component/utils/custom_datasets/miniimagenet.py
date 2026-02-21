import os
import zipfile
from PIL import Image
import torch
from torch.utils.data import Dataset

class MiniImageNet(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        self.zip_path = os.path.join(self.root, "data/archive.zip")
        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(f"{self.zip_path} not found.")
        
        # Open zip once here
        self.zip = zipfile.ZipFile(self.zip_path, 'r')
        
        # Get list of filenames
        self.filenames = sorted([
            f for f in self.zip.namelist()
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.labels = [0] * len(self.filenames)

        print(f"Found {len(self.filenames)} images in {self.zip_path}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        label = self.labels[idx]
        
        with self.zip.open(img_path) as file:
            image = Image.open(file).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return (image, label)

    def __del__(self):
        # Attempt to close the zip if it’s still open
        if hasattr(self, 'zip'):
            try:
                self.zip.close()
            except:
                pass

    # -- The critical part: custom pickling logic so the open handle isn't stored --

    def __getstate__(self):
        # Copy the object’s dict
        state = self.__dict__.copy()
        # Remove open zip references before pickling
        if "zip" in state:
            del state["zip"]
        return state

    def __setstate__(self, state):
        # Restore the rest of the state
        self.__dict__.update(state)
        # Reopen the zip in the worker process
        self.zip = zipfile.ZipFile(self.zip_path, 'r')
