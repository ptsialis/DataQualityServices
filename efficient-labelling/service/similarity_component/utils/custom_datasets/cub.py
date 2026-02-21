import os
import tarfile
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset

class CUB(Dataset):
    def __init__(self, root, split='train', transform=None, download=False):
        self.root = root
        self.split = split
        self.transform = transform
        self.archive_path = os.path.join(self.root, 'CUB_200_2011.tgz')
        
        if not os.path.exists(self.archive_path):
            raise FileNotFoundError(f"{self.archive_path} not found.")

        # Open archive once
        self.tar = tarfile.open(self.archive_path, 'r:gz')

        # Read metadata files
        self.image_paths = {}
        self.labels = {}
        self.split_mask = {}

        def read_text_file(name):
            member = self.tar.getmember(f'CUB_200_2011/{name}')
            return self.tar.extractfile(member).read().decode().strip().splitlines()

        for line in read_text_file('images.txt'):
            idx, path = line.strip().split()
            self.image_paths[int(idx)] = f'CUB_200_2011/images/{path}'

        for line in read_text_file('image_class_labels.txt'):
            idx, label = line.strip().split()
            self.labels[int(idx)] = int(label) - 1  # 0-based labels

        for line in read_text_file('train_test_split.txt'):
            idx, is_train = line.strip().split()
            self.split_mask[int(idx)] = (int(is_train) == 1)

        self.data = [
            (self.image_paths[idx], self.labels[idx])
            for idx in self.image_paths
            if (self.split_mask[idx] if split == 'train' else not self.split_mask[idx])
        ]

        print(f"Loaded {len(self.data)} samples from compressed CUB split: {self.split}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, label = self.data[index]
        img_file = self.tar.extractfile(path)
        image = Image.open(BytesIO(img_file.read())).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def __del__(self):
        try:
            self.tar.close()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'tar' in state:
            del state['tar']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tar = tarfile.open(self.archive_path, 'r:gz')
