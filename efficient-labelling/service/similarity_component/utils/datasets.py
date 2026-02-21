import random
import logging
import numpy as np
from collections import defaultdict
import random
from torch.utils.data import Subset

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from service.similarity_component.utils.custom_datasets.cub import CUB
from service.similarity_component.utils.custom_datasets.domain_net import DomainNet
from service.similarity_component.utils.custom_datasets.miniimagenet import MiniImageNet 

logger = logging.getLogger(__name__)


default_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((480, 640)), # some default size
    ])

def limit_torchvision_dataset(dataset, limit_per_class):
    # works for miniimagenet (fast), all torchvision datasets (very slow)
    if limit_per_class is None:
        return dataset

    # try efficicent way
    try:
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            labels = np.array(dataset.labels)
        elif hasattr(dataset, 'samples'):
            labels = np.array([label for _, label in dataset.samples])
        else:
            raise AttributeError("No fast label access – falling back to slow __getitem__ loop.")
    except:
        # try slow fallback
        print("Using fallback label extraction via __getitem__, this might be slow.")
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        labels = np.array(labels)

    # select indices
    indices = []
    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        selected = np.random.choice(cls_indices, min(limit_per_class, len(cls_indices)), replace=False)
        indices.extend(selected)
    indices = sorted(indices)

    # filter dataset attributes
    try:
        if hasattr(dataset, 'data'):
            dataset.data = dataset.data[indices]
        if hasattr(dataset, 'imgs'):
            dataset.imgs = [dataset.imgs[i] for i in indices]
        if hasattr(dataset, 'samples'):
            dataset.samples = [dataset.samples[i] for i in indices]
        if hasattr(dataset, 'targets'):
            dataset.targets = labels[indices].tolist()
        elif hasattr(dataset, 'labels'):
            dataset.labels = labels[indices].tolist()
    except Exception as e:
        print(f"Warning while filtering dataset: {e}")

    return dataset


# def limit_dataset_per_class(dataset, limit_per_class, seed):
#     # works for CUB, domainnet_*
#     random.seed(seed)
#     if limit_per_class is None:
#         return dataset

#     # group sample paths by label
#     class_to_samples = defaultdict(list)
#     for path, label in dataset.samples:
#         class_to_samples[label].append((path, label))

#     # limit per class
#     limited_samples = []
#     for label, samples in class_to_samples.items():
#         if len(samples) > limit_per_class:
#             samples = random.sample(samples, limit_per_class)
#         limited_samples.extend(samples)

#     # update dataset
#     dataset.samples = limited_samples
#     dataset.targets = [label for _, label in limited_samples]

#     return dataset

def limit_dataset_per_class(dataset, max_per_class=1, seed=42):
    """
    Limits the number of samples per class in a dataset.
    Supports datasets with .samples or _samples attributes, or custom overrides.
    """
    # Try to access sample-label pairs
    if hasattr(dataset, "samples"):
        samples = dataset.samples
    elif hasattr(dataset, "_samples"):
        samples = dataset._samples
    elif hasattr(dataset, "imgs"):  # for torchvision.datasets.ImageFolder-like datasets
        samples = dataset.imgs
    elif hasattr(dataset, "data") and hasattr(dataset, "targets"):
        samples = list(zip(dataset.data, dataset.targets))
    else:
        raise AttributeError(f"Cannot find samples or labels in dataset of type {type(dataset)}")

    logger.info(f"Limiting dataset of type {type(dataset)} to {max_per_class} samples per class")

    # Build class-wise indices
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(samples):
        class_to_indices[label].append(idx)

    # Sample max_per_class indices per class
    random.seed(seed)
    selected_indices = []
    for label, indices in class_to_indices.items():
        selected = random.sample(indices, min(len(indices), max_per_class))
        selected_indices.extend(selected)

    # Return a subset
    return Subset(dataset, selected_indices)



def get_dataset(name, root_dir, transform=default_transform, download=True, split='train', limit_ds_per_class=None, seed=42):
    name = name.lower()

    ds = None

    if name == "imagenet1k":
        return datasets.ImageNet(root=root_dir+"/data/imagenet1k", transform=transform, download=download)
    
    elif name == "miniimagenet":
        ds = MiniImageNet(root=root_dir+"/data/mini_imagenet", transform=transform)
    
    elif name == "stanforddogs":
        raise NotImplementedError("Stanford Dogs is not available yet.")
    
    elif name == "stanfordcars":
        ds = datasets.StanfordCars(root=root_dir+"/data/stanford_cars", transform=transform, download=False)
    
    elif name == "aircraft":
        ds =  datasets.FGVCAircraft(root=root_dir+"/data/aircraft", transform=transform, download=download)
    
    elif name == "oxfordflowers":
        ds = datasets.Flowers102(root=root_dir+"/data/oxford_flowers", transform=transform, download=download)
    
    elif name == "food101":
        ds = datasets.Food101(root=root_dir+"/data/food101", transform=transform, download=download)
    
    elif name == "cub":
        ds = CUB(root=root_dir+"/data/cub", transform=transform)
    
    elif name == "eurosat":
        ds =  datasets.EuroSAT(root=root_dir+"/data/euro_sat", transform=transform, download=download)
    
    elif name == "domainnet_clipart":
        _, domain = name.split("_")
        ds = DomainNet(root=root_dir+"/data/domain_net", domain=domain, split=split, transform=transform)
    
    elif name == "domainnet_infograph":
        _, domain = name.split("_")
        ds = DomainNet(root=root_dir+"/data/domain_net", domain=domain, split=split, transform=transform)
    
    elif name == "domainnet_painting":
        _, domain = name.split("_")
        ds = DomainNet(root=root_dir+"/data/domain_net", domain=domain, split=split, transform=transform)
    
    elif name == "domainnet_quickdraw":
        _, domain = name.split("_")
        ds = DomainNet(root=root_dir+"/data/domain_net", domain=domain, split=split, transform=transform)
    
    elif name == "domainnet_real":
        _, domain = name.split("_")
        ds = DomainNet(root=root_dir+"/data/domain_net", domain=domain, split=split, transform=transform)
    
    elif name == "domainnet_sketch":
        _, domain = name.split("_")
        ds = DomainNet(root=root_dir+"/data/domain_net", domain=domain, split=split, transform=transform)
    
    elif name == "cropdisease":
        raise NotImplementedError("CropDisease is not available yet.")
    
    elif name == "plantae":
        raise NotImplementedError("Places is not available yet.")
        # return datasets.INaturalist(root=root_dir, transform=transform, download=download, version='2021')
    
    elif name == "places":
        raise NotImplementedError("Places is not available yet.")
    
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    if limit_ds_per_class:
        ds = limit_dataset_per_class(ds, limit_ds_per_class, seed)
    
    return ds 



def get_poisoned_dataset(name, root_dir, transform=default_transform, poison_ratio=0.5, spurious_feature='colour', train_mode=True, seed=42, split='train'):
    
    name = name.lower()
    
    ds = None

    if name == "miniimagenet":
        from service.similarity_component.utils.custom_datasets.poisoned_miniimagenet import PoisonedMiniImageNet
        ds = PoisonedMiniImageNet(
            root=root_dir+"/data/mini_imagenet",
            transform=transform,
            poison_ratio=poison_ratio,
            spurious_feature=spurious_feature,
            train_mode=train_mode,
            seed=seed,
            split=split
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return ds