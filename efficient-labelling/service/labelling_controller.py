

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
import io
import zipfile
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from collections import defaultdict, Counter
from pathlib import Path


def _load_model(model_name, model_type):
    """
    Loads the specified model by name.
    Returns the model instance.
    """    
    if model_type not in ["vgg19", "feat"]:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types are 'vgg19' and 'feat'.")
    
    if not os.path.exists(model_name):
        raise FileNotFoundError(f"Model file {model_name} does not exist.")

    logger.info(f"Loading model {model_name} of type {model_type}...")

    # Load the model based on its type
    if model_type == "vgg19":
        logger.info("Loading VGG19 model...")
        from service.labelling_component.models.vgg import VGG
        model_wrapper = VGG(model_path=model_name, architecture="vgg19")
        model = model_wrapper
    elif model_type == "feat":
        logger.info("Loading FEAT model...")
        from service.labelling_component.models.feat import FEATWrapper
        model_wrapper = FEATWrapper(model_path=model_name)
        model = model_wrapper

    
    logger.info(f"Model {model_name} loaded successfully.")

    return model



def _load_target_dataset(dataset_path, transform=None): # this is copy paste from service.similarity_controller import _load_target_dataset # TODO: ?
    from service.similarity_component.utils.custom_datasets.unlabelled_dataset import UnlabelledDataset
    logger.info(f"Loading target dataset from {dataset_path}")
    dataset = UnlabelledDataset(dataset_path, transform=transform)
    return dataset


def _load_labelled_dataset(dataset_path, transform=None):
    from service.similarity_component.utils.custom_datasets.labelled_dataset import LabelledDataset
    logger.info(f"Loading labelled dataset from {dataset_path}")
    dataset = LabelledDataset(dataset_path, transform=transform)
    return dataset


def _save_labelled_dataset_to_zip(labelled_dataset, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, mode='w') as zipf:
        for idx, (img_tensor, label) in enumerate(zip(labelled_dataset["images"], labelled_dataset["labels"])):
            class_dir = f"class_{label}"
            img_name = f"image_{idx:04d}.png"
            img = to_pil_image(img_tensor)

            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            zipf.writestr(f"{class_dir}/{img_name}", img_bytes.read())

    zip_buffer.seek(0)

    with open(output_path, 'wb') as f:
        f.write(zip_buffer.read())

    logger.info(f"Labelled dataset ZIP written to {output_path}")
    return output_path


# def label_with_model(config, model_name, target_dataset_name):
    
#     logger.info(f"Labeling dataset {target_dataset_name} with model {model_name}...")
#     logger.info(f"Using config: {config}")
    
#     available_models = config["labelling_component"]["available_models"]
    
#     # check if the model is available
#     if model_name not in available_models:
#         raise ValueError(f"Model '{model_name}' is not available for labelling.")
    
#     model_entry = available_models[model_name]
#     weights_file = model_entry["weights_file_name"]
#     model_path = config["weights_root"] + f"/pretrained_models/{weights_file}"
#     model_type = model_entry["model_type"]
    
#     # load the model
#     logger.info(f"Loading model from {model_path}...")
#     model = _load_model(model_path, model_type)    
#     logger.info(f"Model {model_name} loaded successfully.")
    
#     # load the dataset
#     dataset_path = config["dataset_root"] + f"/target_datasets/{target_dataset_name}"

#     labelled_dataset = {
#         "images": [],
#         "labels": []
#     }
    
#     dataset_transformed = _load_target_dataset(dataset_path, transform=model._preprocess())
#     dataloader = DataLoader(dataset_transformed, batch_size=32, shuffle=False)
#     logger.info(f"Transformed dataset loaded for inference.")
    
#     logger.info(f"Dataset {target_dataset_name} loaded successfully.")
    
#     # load dataset again without any transforms for preview
#     dataset_original = _load_target_dataset(dataset_path, transform=None)

#     logger.info(f"Original dataset loaded for preview.")
    
#     logger.info(f"Starting labelling for dataset '{target_dataset_name}' with model '{model_name}'...")

#     with torch.no_grad():
#         for batch in dataloader:
#             batch = batch.to(model.device)
#             outputs = model.model(batch)
#             predicted_labels = torch.argmax(outputs, dim=1)
#             labelled_dataset["labels"].extend(predicted_labels.tolist())
#             labelled_dataset["images"].extend(batch.cpu())  # TODO: asses whether original images could be stored here

#     logger.info(f"Labelling completed for dataset '{target_dataset_name}'.")

#     # save as ZIP
#     clean_name = Path(target_dataset_name).stem  # remove any file extensions
#     output_zip_path = os.path.join(
#         config["dataset_root"], "labelled_datasets", f"{clean_name}_labelled.zip"
#     )
#     _save_labelled_dataset_to_zip(labelled_dataset, output_zip_path)

#     # create preview (5 samples per class)
#     class_to_images = defaultdict(list)
#     for idx, label in enumerate(labelled_dataset["labels"]):
#         if len(class_to_images[label]) < 5:
#             image = dataset_original[idx]  # get original image
#             class_to_images[label].append(image)

#     # stats
#     label_counts = Counter(labelled_dataset["labels"])
#     total_images = len(labelled_dataset["images"])
#     num_classes = len(label_counts)

#     stats = {
#         "total_images": total_images,
#         "num_classes": num_classes,
#         "class_distribution": dict(label_counts),
#         "zip_file_size_MB": round(os.path.getsize(output_zip_path) / (1024 * 1024), 2),
#         "zip_path": output_zip_path
#     }

#     return {
#         "preview": dict(class_to_images),  # label -> list of up to 5 tensors
#         "stats": stats,
#         "zip_path": output_zip_path
#     }


def label_with_model(config, model_name, target_dataset_name):
    logger.info(f"Labeling dataset '{target_dataset_name}' with pretrained model '{model_name}'...")
    logger.info(f"Using config: {config}")
    
    available_models = config["labelling_component"]["available_models"]

    # Check model availability
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' is not available in config.")
    
    model_entry = available_models[model_name]
    weights_file = model_entry["weights_file_name"]
    model_type = model_entry["model_type"]
    model_path = os.path.join(config["weights_root"], "pretrained_models", weights_file)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' does not exist.")

    # Load the model
    logger.info(f"Loading model from {model_path} (type: {model_type})...")
    model = _load_model(model_path, model_type)
    logger.info(f"Model '{model_name}' loaded successfully.")

    # Load transformed target dataset for inference
    target_dataset_path = os.path.join(config["dataset_root"], "target_datasets", target_dataset_name)
    target_dataset = _load_target_dataset(target_dataset_path, transform=model._preprocess())
    target_loader = DataLoader(target_dataset, batch_size=32, shuffle=False)
    logger.info(f"Target dataset loaded and transformed for inference.")

    # Load raw dataset for untransformed access
    raw_target_dataset = _load_target_dataset(target_dataset_path, transform=None)
    logger.info(f"Raw (untransformed) target dataset loaded for preview and final output.")

    labelled_dataset = {
        "images": [],
        "labels": []
    }

    logger.info(f"Starting inference on dataset '{target_dataset_name}' using model '{model_name}'...")
    all_preds = []

    with torch.no_grad():
        for batch in target_loader:
            batch = batch.to(model.device)
            outputs = model.model(batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())

    if len(all_preds) != len(raw_target_dataset):
        raise ValueError(f"Mismatch between predictions ({len(all_preds)}) and dataset size ({len(raw_target_dataset)}).")

    for idx, label in enumerate(all_preds):
        image_tensor = raw_target_dataset[idx]
        if isinstance(image_tensor, (tuple, list)):
            image_tensor = image_tensor[0]
        labelled_dataset["images"].append(image_tensor)
        labelled_dataset["labels"].append(int(label))

    # Save ZIP
    clean_name = Path(target_dataset_name).stem
    output_zip_path = os.path.join(config["dataset_root"], "labelled_datasets", f"{clean_name}_labelled.zip")
    _save_labelled_dataset_to_zip(labelled_dataset, output_zip_path)

    # Create preview (5 samples per class)
    class_to_images = defaultdict(list)
    for idx, label in enumerate(labelled_dataset["labels"]):
        if len(class_to_images[label]) < 5:
            image = raw_target_dataset[idx]
            if isinstance(image, (tuple, list)):
                image = image[0]
            class_to_images[label].append(image)

    # Statistics
    label_counts = Counter(labelled_dataset["labels"])
    stats = {
        "total_images": len(labelled_dataset["images"]),
        "num_classes": len(label_counts),
        "class_distribution": dict(label_counts),
        "zip_file_size_MB": round(os.path.getsize(output_zip_path) / (1024 * 1024), 2),
        "zip_path": output_zip_path
    }

    return {
        "preview": dict(class_to_images),
        "stats": stats,
        "zip_path": output_zip_path
    }


def label_with_finetuned_model(config, target_dataset_name, label_dataset_name, class_mappings):
    
    logger.info(f"Labeling dataset {target_dataset_name} with fine-tuned model from {label_dataset_name}...")
    logger.info(f"Using class mappings: {class_mappings}")  
    
    # retrieve the standard model from config
    model_name = config["labelling_component"]["default_base_model_type"]
    weights_file = config["labelling_component"]["default_weights_file_name"]
    model_path = config["weights_root"] + f"/pretrained_models/{weights_file}"  
    
    logger.info(f"Using model path: {model_path}")
    
    # check if the model is available
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    
    # load the model
    logger.info(f"Loading model from {model_path}...")
    model = _load_model(model_path, "feat")  # assuming 'feat' is the type for fine-tuning
    logger.info(f"Model {model_name} loaded successfully.")
    
    # load the label dataset
    #label_dataset_path = config["dataset_root"] + f"/labelled_datasets/{label_dataset_name}"
    label_dataset_path = label_dataset_name # this is now a full path, that could produce errors but it is like that for now
    label_dataset = _load_labelled_dataset(label_dataset_path, transform=model._preprocess())
    label_dataset_loader = DataLoader(label_dataset, batch_size=32, shuffle=False)

    logger.info(f"Label dataset loaded with {len(label_dataset)} samples.")

    # fine-tune the model
    logger.info(f"Fine-tuning model on label dataset {label_dataset_name}...")
    model.fine_tune(label_dataset_loader)
    logger.info(f"Model fine-tuned successfully.")
    
    # load the target dataset
    target_dataset_path = config["dataset_root"] + f"/target_datasets/{target_dataset_name}"
    target_dataset = _load_target_dataset(target_dataset_path, transform=model._preprocess())
    target_dataset_loader = DataLoader(target_dataset, batch_size=32, shuffle=False)
    logger.info(f"Transformed target dataset loaded for inference.")
    
    predicted_labels = model.predict(label_dataset_loader, target_dataset_loader)
    
    labelled_dataset = {
        "images": [],
        "labels": []
    }

    target_dataset_path = config["dataset_root"] + f"/target_datasets/{target_dataset_name}"
    raw_target_dataset = _load_target_dataset(target_dataset_path, transform=None)

    if len(predicted_labels) != len(raw_target_dataset):
        raise ValueError(f"Mismatch between predicted labels ({len(predicted_labels)}) and raw images ({len(raw_target_dataset)})")

    # fill final dataset
    for idx, label in enumerate(predicted_labels):
        image_tensor = raw_target_dataset[idx]
        if isinstance(image_tensor, (tuple, list)):  # In case dataset returns (img, _) tuples
            image_tensor = image_tensor[0]
        labelled_dataset["images"].append(image_tensor)
        # labelled_dataset["labels"].append(label) # error
        labelled_dataset["labels"].append(int(label))
        
    
    # save as ZIP
    clean_name = Path(target_dataset_name).stem  # remove any file extensions
    output_zip_path = os.path.join(
        config["dataset_root"], "labelled_datasets", f"{clean_name}_labelled.zip"
    )
    _save_labelled_dataset_to_zip(labelled_dataset, output_zip_path)    
    
    # # create preview (5 samples per class)
    # class_to_images = defaultdict(list)
    # for idx, label in enumerate(labelled_dataset["labels"]):
    #     if len(class_to_images[label]) < 5:
    #         image = target_dataset[idx]  # get original image
    #         class_to_images[label].append(image)
    
    # create preview (5 samples per class)
    class_to_images = defaultdict(list)
    for idx, label in enumerate(labelled_dataset["labels"]):
        if len(class_to_images[label]) < 5:
            image = raw_target_dataset[idx]  # get raw, untransformed image
            if isinstance(image, (tuple, list)):  # in case dataset returns (img, _) tuple
                image = image[0]
            class_to_images[label].append(image)

    # stats
    label_counts = Counter(labelled_dataset["labels"])
    total_images = len(labelled_dataset["images"])
    num_classes = len(label_counts) 
    
    stats = {
        "total_images": total_images,
        "num_classes": num_classes,
        "class_distribution": dict(label_counts),
        "zip_file_size_MB": round(os.path.getsize(output_zip_path) / (1024 * 1024), 2),
        "zip_path": output_zip_path 
    }
    
    return {
        "preview": dict(class_to_images),  # label -> list of up to 5 tensors
        "stats": stats,
        "zip_path": output_zip_path 
    }