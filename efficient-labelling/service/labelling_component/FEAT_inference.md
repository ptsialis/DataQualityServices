2. Bird Classification with FEAT_inference.py
Purpose
Classifies query images using:

    1. Pre-trained FEAT model

    2. Support set with few examples per class

    3. Episodic inference with confidence aggregation
   
Command Syntax: python FEAT_inference.py --support_root <support_directory> --query_folder <query_directory> --output_root <results_directory> --checkpoint <model_path> --num_episodes N --n_way K --n_shot S --temperature T --fine_tune_epochs E

support_root: Path to support class folders
query_folder: Path to query images
output_root: Results directory, labelled folders
checkpoint: Pre-trained FEAT model path (.pth)
num_episodes N: Number of inference episodes, example N 150, or 200
n_way K: Classes per episode, example K is 5 similar to the training classes
n_shot S: Support images per class, example S is 5 similar to training samples per class per episode
temperature T: Similarity temperature scaling, example 1 or 0.1
fine_tune_epochs E:Epochs to fine-tune on support


Output is:
Folder Results:
    Subfolder: bird1
    Subfolder: bird2
    ... number of Subfolders is the same as the number of folders in the support_root
    Subfolder diagnostics:
        classification_report.csv
        class_distribution.txt
        confidence_histogram.png
        
        
command used to run FEAT_inference.py: python service/labelling_component/feat_inference/Inference/FEAT_inference.py --support_root service/labelling_component/feat_inference/data/NewClasses/support --query_folder service/labelling_component/feat_inference/data/NewClasses/query --output_root service/labelling_component/feat_inference/Labelled_Data --checkpoint service/labelling_component/feat_inference/checkpoints/best_model_feat_cub.pth --num_episodes 200 --n_way 4 --n_shot 5 --temperature 0.1 --fine_tune_epochs 10
        

