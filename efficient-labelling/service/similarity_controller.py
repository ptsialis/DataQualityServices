
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from service.similarity_component.utils.datasets import get_dataset
from service.similarity_component.utils.utils import get_feature_dataset
import torch


def _load_feature_dataset(dataset_name, dataset_root="datasets/feature_datasets"):
    
    try:
        # feature_ds = get_feature_dataset(dataset_name, dataset_root, feature_root="", use_npz=True, )# max_rows=75000) # feature_root is empty b/c of path concatenation in get_feature_dataset
        feature_ds = get_feature_dataset(dataset_name, dataset_root, feature_root="", use_npz=True, max_rows=50000) # feature_root is empty b/c of path concatenation in get_feature_dataset
        feature_ds = feature_ds.features.numpy()
        
        logger.info(f"Loaded feature dataset {dataset_name} with shape {feature_ds.shape} {type(feature_ds)}.")
        
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        return None
    
    return feature_ds

def _load_target_dataset(dataset_path, transform=None):
    from service.similarity_component.utils.custom_datasets.unlabelled_dataset import UnlabelledDataset
    logger.info(f"Loading target dataset from {dataset_path}")
    dataset = UnlabelledDataset(dataset_path, transform=transform)
    return dataset


def _load_model(model_name):
    if model_name == "VGG19":
        from service.similarity_component.pretrained_models.vgg import VGG
        return VGG(architecture="vgg19", pretrained=True, finetuned=False)


def _load_similarity_measure(measure_name):
    from service.similarity_component.similarity_measures import SimilarityMeasure
    return SimilarityMeasure(measure_name)


def evaluate_dataset_similarity(config, target_dataset_name):
    
    # load the generalist model / feature extractor
    model = _load_model(config["similarity_component"]["model_name"])
    similarity_measure = _load_similarity_measure(config["similarity_component"]["similarity_measure"]) 
    
    # print(model._preprocess())
    
    # load target dataset
    
    target_dataset_path = Path(config["dataset_root"] + f"/target_datasets/{target_dataset_name}")
    target_dataset = _load_target_dataset(target_dataset_path, transform=model._preprocess())
    
    # BEGIN: due to out of memory issues with docker
    max_images = 500 
    if len(target_dataset) > max_images:
        target_dataset = torch.utils.data.Subset(target_dataset, range(max_images))
    # END -- if unlimited resources -> discard this
        
    target_dataset_loader = DataLoader(target_dataset, batch_size=32, shuffle=False)
    
    # extract the features of the target dataset
    target_featurespace = model.extract_representations(target_dataset_loader, flatten=True)
    target_featurespace = target_featurespace.cpu().numpy() # we need to convert it to numpy for the similarity measure
    logger.info(f"Extracted {target_featurespace.shape} features from the target dataset.")
    
    # retrive all available datasets/feature datasets from the config
    datasets = config["similarity_component"]["available_base_datasets"] 
    
    # load all feature datasets from the database
    similarities = []
    for dataset_name in datasets:
        
        logger.info(f"Evaluating dataset: {dataset_name}.")
        # break # TODO: HIER HÄNGENGEBLIEBEN AM 14.07. -> IMPLEMENTIEREN
        
        # load feature dataset
        base_featurespace = _load_feature_dataset(dataset_name, dataset_root=config["dataset_root"] + "/feature_datasets")
        # TODO: above -- unlimit the amount of data to pass through for production
        
        
        # evaluate the similarity of the target dataset with the feature dataset
        dist = similarity_measure.measure(base_featurespace, target_featurespace) 
        
        
        # optional similarity transformation # TODO: shift this to the similarity measure class
        gamma = config["similarity_component"]["sim_gamma"]
        exp_sim = np.exp(-gamma * dist)
        
        similarities.append({
            "dataset": dataset_name,
            "distance": dist,
            "exponential_similarity": exp_sim
        })
        
        logger.info(f"Computed similarity for dataset {dataset_name}: distance={dist}, exponential_similarity={exp_sim}")
        
    # sort the similarities by distance
    similarities.sort(key=lambda x: x['exponential_similarity'])
    
    # find the pre-trained model to the most similar dataset using the threshold, # TODO think of how to get the thresholding right use dists or sims???
    threshold = config["similarity_component"]["threshold"]
    similar_datasets = [s for s in similarities if s["exponential_similarity"] > threshold]
    logger.info(f"Found {len(similar_datasets)} similar datasets above the threshold {threshold}.")
    
    result = {}
    if similar_datasets:
        most_similar = similar_datasets[0]
        logger.info(f"Most similar dataset: {most_similar['dataset']} with similarity {most_similar['exponential_similarity']}.")
        result = {
            "sim": True,
            "message": f"The dataset is similar enough to the pre-trained model: {most_similar['dataset']}",
            "model": most_similar['dataset']
        }
    else:
        logger.info("No similar datasets found above the threshold.")
        result = {
            "sim": False,
            "message": "The dataset is not similar enough to any pre-trained model.",
            "model": "unknown"
        }
        
    return result