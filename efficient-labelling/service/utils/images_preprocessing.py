import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils.mean_variance_calculations import compute_mean_std
from utils.boundry_boxes import detect_and_crop 


def get_dataset_mean_std(dataset_path, use_precomputed=True, precomputed_values=None):
    '''Computes or retrieves dataset mean & standard deviation for normalization.'''
    
    if use_precomputed and precomputed_values:
        return precomputed_values
    return compute_mean_std(dataset_path)

# Specify dataset path of the fine-tunning dataset and not the training dataset
DATASET_PATH = "/Users/dimayasir/Documents/FEAT/data/cub/cropped_images"

# Use dynamically computed values
MEAN, STD = get_dataset_mean_std(DATASET_PATH, use_precomputed=False)


def transform_choosen(pipeline):
    '''Chooses transformation based on task.'''
    if pipeline == 'labelling':
        transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)])

    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),  
            transforms.Resize((224, 224)),  # Resize to VGG19 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])
    return transform

class DatasetUpload(Dataset):
    '''performs dataset loading either labelled for adaptation of FEAT or unlabelled for FEAT or similarity'''
    def __init__(self, image_dir, pipeline='labelling', transform=None, labeled=True):
        self.image_dir = image_dir
        self.transform = transform
        self.labeled = labeled
        self.image_paths = []
        self.labels = []
        self.pipeline = pipeline

        if labeled:
            for label in os.listdir(image_dir):
                label_path = os.path.join(image_dir, label)
                if not os.path.isdir(label_path):
                    continue
                for fname in os.listdir(label_path):
                    if fname.lower().endswith(('jpg', 'jpeg', 'png')):
                        self.image_paths.append(os.path.join(label_path, fname))
                        self.labels.append(label)
        else:
            for fname in os.listdir(image_dir):
                if fname.lower().endswith(('jpg', 'jpeg', 'png')):
                    self.image_paths.append(os.path.join(image_dir, fname))
                    self.labels.append(-1)  # No labels for classification
        # Convert labels to numerical indices
        if labeled:
            self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
            self.labels = [self.label_to_idx[label] for label in self.labels]  # Convert labels to integers


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = None
        label = self.labels[idx]

        if self.pipeline == 'labelling':
            cropped_image = detect_and_crop(img_path)
            if cropped_image is None:
                image = Image.open(img_path).convert('RGB')
                print(f"[WARNING] No object detected in: {img_path}")
            else:
                image = cropped_image
            image = self.transform(image)  # Apply FEAT preprocessing
            return image, label, img_path
        else: 
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, label, img_path


def collate_fn(batch):
    """
    to filter out None values from batches.
    """
    batch = [x for x in batch if x is not None]  # Remove None values
    if len(batch) == 0:
        return []  # If batch is empty, return None
    return torch.utils.data.dataloader.default_collate(batch)

    
def load_feat_data(image_dir, phase='adaptation', N=5, K=5, pipeline = 'labelling'):
    '''
    loads data for any pipeline depending on the pipeline arg which is either labelling or similarity.
    '''
    transform = transform_choosen(pipeline)
    labeled = (phase == 'adaptation' and pipeline == 'labelling')
    dataset = DatasetUpload(image_dir, pipeline= pipeline, transform = transform, labeled= labeled)

    for img_path in dataset.image_paths[:5]:
        print(f'Found image path {img_path}')

    batch_size = N*K if labeled else 32

    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(phase == 'adaptation' and pipeline == 'labelling'),
        collate_fn=collate_fn  # Use custom function to remove None
    )
    for batch_idx, (images, labels, paths) in enumerate(data_loader):
        print(f"Batch {batch_idx}: {len(images)} images loaded")
        break  # Only check the first batch

    print('finalized loading the images')
    return data_loader

if __name__ == "__main__":
    # Set dataset path (Modify based on your local structure)
    feat_adaptation_path = "/Users/dimayasir/ai-alliance/service/utils/datasets/uncropped_images"
    feat_test_path = "/Users/dimayasir/ai-alliance/service/utils/datasets/unlabelled_images"
    similarity_test_path = "/Users/dimayasir/ai-alliance/service/utils/datasets/unlabelled_images"


    # print("\n===== TEST 1: LABELLING, ADAPTATION =====")
    # adaptation_loader = load_feat_data(feat_adaptation_path, phase="adaptation", N=5, K=5, pipeline="labelling")

    # print("\n===== TEST 2: LABELLING, TEST =====")
    # test_loader = load_feat_data(feat_test_path, phase="test", N=5, K=5, pipeline="labelling")

    # print("\n===== TEST 3: SIMILARITY =====")
    # similarity_loader = load_feat_data(similarity_test_path, phase="test", pipeline="similarity")