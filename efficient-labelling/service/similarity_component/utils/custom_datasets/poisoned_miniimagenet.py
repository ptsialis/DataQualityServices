import os
import zipfile
import random
import numpy as np
from PIL import Image, ImageFilter
from matplotlib.colors import hsv_to_rgb
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PoisonedMiniImageNet(Dataset):
    def __init__(self, root, transform=None, seed=42, poison_ratio=0.5,
                spurious_feature='none', train_mode=True, split='train'):
        assert split in ['train', 'val', 'test']
        assert spurious_feature in ['none', 'colour', 'blur', 'artefact', 'removal']

        self.root = root
        self.transform = transform or transforms.ToTensor()
        self.seed = seed
        self.poison_ratio = poison_ratio
        self.spurious_feature = spurious_feature
        self.train_mode = train_mode
        self.split = split
        self.hues = np.linspace(0, 1, 5, endpoint=False)

        # Open ZIP archive
        self.zip_path = os.path.join(root, "data/archive.zip")
        self.zip = zipfile.ZipFile(self.zip_path, 'r')

        # Require MiniImageNet-style: folder per class
        self.class_to_files = defaultdict(list)
        for name in self.zip.namelist():
            if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                parts = name.strip('/').split('/')
                if len(parts) == 2:
                    class_name = parts[0]
                    self.class_to_files[class_name].append(name)

        # Deterministic class split
        class_names = sorted(self.class_to_files.keys())
        random.seed(seed)
        random.shuffle(class_names)

        n_total = len(class_names)
        n_train = int(n_total * 0.64)
        n_val = int(n_total * 0.16)

        if split == 'train':
            selected_classes = class_names[:n_train]
        elif split == 'val':
            selected_classes = class_names[n_train:n_train + n_val]
        else:
            selected_classes = class_names[n_train + n_val:]

        self.class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}

        self.samples = []
        for cls in selected_classes:
            label = self.class_to_idx[cls]
            for path in self.class_to_files[cls]:
                self.samples.append((path, label))

        # Poison subset
        random.seed(seed)
        indices = list(range(len(self.samples)))
        random.shuffle(indices)
        n_poisoned = int(len(indices) * poison_ratio)
        self.poisoned_indices = set(indices[:n_poisoned])

        print(f"[{split}] Total: {len(self.samples)} | Poisoned: {n_poisoned} | Classes: {len(selected_classes)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        with self.zip.open(path) as file:
            img = Image.open(file).convert('RGB')

        spurious_idx = None
        is_poisoned = idx in self.poisoned_indices

        if self.spurious_feature != 'none' and is_poisoned:
            spurious_idx = random.randint(0, 4)
            if self.spurious_feature == 'colour':
                img = self._apply_colour(img, spurious_idx)
            elif self.spurious_feature == 'blur':
                img = self._apply_blur(img, spurious_idx)
            elif self.spurious_feature == 'artefact':
                img = self._apply_artefact(img, spurious_idx)
            elif self.spurious_feature == 'removal':
                img = self._apply_removal(img, spurious_idx)

        img = self.transform(img)

        if self.train_mode:
            return img, label
        return img, label, int(is_poisoned), spurious_idx

    def _apply_colour(self, img, feature_group):
        hue = self.hues[feature_group]
        gray = transforms.ToTensor()(img)[0]
        H = torch.full_like(gray, hue)
        S = torch.ones_like(gray)
        V = gray
        hsv = torch.stack([H, S, V], dim=2).numpy()
        rgb_np = hsv_to_rgb(hsv)
        rgb = torch.from_numpy(rgb_np).permute(2, 0, 1).float()
        return transforms.ToPILImage()(rgb)

    def _apply_blur(self, img, feature_group):
        levels = [0, 0.5, 1.5, 3.0, 6.0]
        return img.filter(ImageFilter.GaussianBlur(levels[feature_group]))

    def _apply_artefact(self, img, feature_group):
        img = img.copy()
        pixels = img.load()
        w, h = img.size
        aw, ah = int(w * 0.3), int(h * 0.15)
        positions = [
            (0, 0), (w - aw, 0), (0, h - ah),
            (w - aw, h - ah), ((w - aw) // 2, (h - ah) // 2)
        ]
        x0, y0 = positions[feature_group]
        for i in range(aw):
            for j in range(ah):
                xi, yj = x0 + i, y0 + j
                if 0 <= xi < w and 0 <= yj < h:
                    pixels[xi, yj] = (255, 255, 255)
        return img

    def _apply_removal(self, img, feature_group):
        img = img.copy()
        pixels = img.load()
        w, h = img.size
        band_h = h // 5
        y_start = feature_group * band_h
        for i in range(w):
            for j in range(y_start, min(y_start + band_h, h)):
                pixels[i, j] = (0, 0, 0)
        return img

    def __del__(self):
        if hasattr(self, 'zip'):
            try:
                self.zip.close()
            except:
                pass

    def __getstate__(self):
        state = self.__dict__.copy()
        if "zip" in state:
            del state["zip"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.zip = zipfile.ZipFile(self.zip_path, 'r')
