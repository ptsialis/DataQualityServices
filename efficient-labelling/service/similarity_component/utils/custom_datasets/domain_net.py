import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import torch

class DomainNet(Dataset):
    def __init__(self, root, domain, transform=None, split='test', limit_ds_per_class=None, seed=42):
        super().__init__()
        self.root = root
        self.domain = domain
        self.transform = transform
        self.split = split
        self.limit_ds_per_class = limit_ds_per_class
        self.seed = seed
        self.samples = []
        self.class_to_idx = {}
        
        domain_path = os.path.join(self.root, domain)

        for i, class_name in enumerate(sorted(os.listdir(domain_path))):
            class_dir = os.path.join(domain_path, class_name)
            if not os.path.isdir(class_dir):
                continue
            self.class_to_idx[class_name] = i
            images = sorted([
                os.path.join(class_dir, fname)
                for fname in os.listdir(class_dir)
                if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            if self.limit_ds_per_class:
                torch.manual_seed(self.seed)
                images = images[:self.limit_ds_per_class]
            self.samples.extend([(img_path, i) for img_path in images])
        
        print(f"Loaded {len(self.samples)} images from DomainNet domain: {domain}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = default_loader(path)
        if self.transform:
            image = self.transform(image)
        return image, label
