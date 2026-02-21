import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def compute_mean_std(dataset_path, img_size=(84,84), batch_size=32):
    '''
    This function computes the mean and std when called

    Args:
        dataset_path (str): Path to dataset
        img_size (tuple): Resize images to this size before computing statistics
        batch_size (int): Batch size for loading images
    Returns:
        mean (list): List of mean values for each channel
        std (list): List of standard deviation values for each channel
    '''
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root= dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle= False, num_workers = 0)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, 3, -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples

    print(f'Dataset Mean is {mean.tolist()}')
    print(f'Dataset Std is {std.tolist()}')

    return mean.tolist(), std.tolist()


## For testing purposes
# dataset_path = '/Users/dimayasir/Documents/FEAT/data/cub/cropped_images'
# compute_mean_std(dataset_path, img_size=(84,84), batch_size=32)