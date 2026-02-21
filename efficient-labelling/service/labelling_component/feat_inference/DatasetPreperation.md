1. Dataset Preparation with DatasetCreator.py
Purpose
Creates a custom few-shot learning dataset from the CUB-200 dataset by:

    1. Randomly selecting bird species

    2. Creating support folders with few images per class

    3. Creating a mixed query folder with remaining images
   
Command Syntax: python DatasetCreator.py --source_path <path_to_cub200_images usually in data folder> --destination_root <output_directory to be decided> --n_classes N --n_support M --n_query P --seed S

source_path: Path to CUB-200 images directory
destination_root: Output directory for new dataset
n_classes N: Number of bird classes to select, example N is 5
n_support M: Support images per class, example M is 5
n_query P: Query images per class, example P is 30
seed S: Random seed for reproducibility, example S is 123

Link to download CUB200: https://data.caltech.edu/records/65de6-vp158 to be saved in labelling_component/feat_inference

The command used to run DataCreator.py: python service/labelling_component/feat_inference/Inference/DatasetCreator.py --source_path ../newProject/CUB200/images(this needs to be changed to where you store CUB200) --destination_root service/labelling_component/feat_inference/data/NewClasses --n_classes 5 --n_support 5 --n_query 35 --seed 123
