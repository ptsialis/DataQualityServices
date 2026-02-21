# Modular_FEAT_Fewshot

1. Project Title and Context
Part of the [AI Alliance: Service Labelling Component] under the path: `ai-alliance/service/labelling_component/Modular_FEAT_FewShot`

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

2. Overview
State-of-the-art few-shot image classification using Feature Embedding Adaptation with Transformers (FEAT). This implementation supports training and inference. This modular implementation supports:
- Multiple backbones (ResNet12, ResNet18, ConvNet4)
- Dataset Agnostic: Works with any image dataset in standard folder structure
- Episodic training and inference
- Flexible Training: Configurable way/shot settings for episodic training
- Fine-tuning and test-time adaptation
- Confidence-based label assignment during inference
- Configurable augmentations and normalization strategies
- Diagnostics: Detailed reports, confidence scores, and visualizations

3. Directory Structure
Modular_FEAT_FewShot/
├── train.py                  # Episodic training and validation
├── inference.py              # Label inference for unseen query images
├── configs.py                # Shared configuration across training and inference
├── models/
│   ├── backbones.py          # ResNet12, ResNet18, ConvNet4
│   └── feat.py               # Transformer-based adaptation (FEAT)
├── data/
│   └── loader.py             # Episodic task generation for meta-training
├── utils/
│   └── training_utils.py     # Training loop utilities and callbacks
├── checkpoints/              # Stores trained model weights
├── requirements.txt
└── README.md

4. Installation # To be decided if the user can clone the tool!
```bash
git clone https://github.com/TOBEDECIDED/Modular_FEAT_FewShot.git
cd ai-alliance/service/labelling_component/Modular_FEAT_FewShot

# For systems with CUDA 12.x
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# For CPU-only systems
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
5. Input Data Folder Structure
Support and query folders should look like:

support/
├── ClassA/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── ClassB/
│   └── ...

query/
├── imgX.jpg
├── imgY.jpg
└── ...

6. Config Arguments:
The following are all the config argumnets that are used wither during training or inference. Default values have been set.
[--data_path DATA_PATH] # used during training to point to the dataset used 
[--support_root SUPPORT_ROOT] # used during inference to point to the support dataset
[--query_root QUERY_ROOT] # used during inference to point to the query dataset
[--way WAY] # used during both training and inference, the number of classes per episode
[--shot SHOT] # used during both training and inference, the number of samples per class
[--query QUERY] # used during both training and inference. Number of query images per class per episode.
[--split_train SPLIT_TRAIN] # used during training to assign the percentage of the data to be used for training
[--split_val SPLIT_VAL] # used during training to assign the percentage of the data to be used for validation
[--split_test SPLIT_TEST] # used to test the trained model (unseen during training)
[--mode {train,inference}] # mode either training or inference, needs to be provided in the command
[--output_root OUTPUT_ROOT] # used during inference to assign where to save the results (labelled dataset and diagnostics)
[--image_resize IMAGE_RESIZE] # used for preprocessing data during training and inference
[--image_crop IMAGE_CROP] # used for preprocessing data during training and inference
[--use_randaugment] # optional augmentation/ Enable RandAugment during training (N, M params needed)
[--use_color_jitter] # optional augmentation to be added during training
[--use_random_erasing] # optional augmentation to be added during training
[--use_random_affine USE_RANDOM_AFFINE] # optional augmentation to be added during training
[--use_random_rotation USE_RANDOM_ROTATION] # optional augmentation to be added during training
[--use_random_sharpness USE_RANDOM_SHARPNESS] # optional augmentation to be added during training
[--use_random_vertical_flip USE_RANDOM_VERTICAL_FLIP] # optional augmentation to be added during training
[--use_gaussian_blur USE_GAUSSIAN_BLUR] # optional augmentation to be added during training
[--backbone {ResNet18,ResNet12,ConvNet4}] # ConvNet4 | ResNet12 | ResNet18
[--hidden_dim HIDDEN_DIM] # Output of ResNet12/ResNet18/ConvNet4, input to attention mechanisms (feat_attn_dim)
[--use_cosine USE_COSINE] #  If True, uses cosine similarity (direction-based). If False, uses Euclidean distance (magnitude-based).
[--proto_attn_layers PROTO_ATTN_LAYERS] # Number of Transformer layers used for prototype adaptation
[--proto_attn_heads PROTO_ATTN_HEADS] # Number of attention heads in the prototype adaptation Transformer 
[--aux_transformer_layers AUX_TRANSFORMER_LAYERS] # Number of auxiliary Transformer layers for prototype adaptation  
[--aux_transformer_heads AUX_TRANSFORMER_HEADS] # Number of attention heads in auxiliary Transformer for prototype adaptation  
[--aux_transformer_ffn_dim_factor AUX_TRANSFORMER_FFN_DIM_FACTOR] # A scaling factor used to determine the size of the feed-forward network (FFN) layer inside each auxiliary Transformer block used for prototype adaptation.
[--dropout_rate DROPOUT_RATE] # The probability that a neuron (or unit) in a neural network will be randomly deactivated during training.
[--epochs EPOCHS] # The total number of iterations of all the training data in one cycle for training.
[--episodes_per_epoch EPISODES_PER_EPOCH] # The number of tasks per epoch in training
[--val_episodes_per_epoch VAL_EPISODES_PER_EPOCH] # The number of tasks per epoch in validation
[--num_episodes NUM_EPISODES] # Number of tasks in inference
[--lr LR] # learning rate
[--optimizer_type {SGD,Adam,AdamW}] # SGD|Adam|AdamW
[--momentum MOMENTUM] # A concept from physics where an object's motion depends not only on the current force but also on its previous velocity
[--weight_decay WEIGHT_DECAY] # the optimizer adjusts the loss function by adding a penalty term based on the square of the weights
[--warmup_epochs WARMUP_EPOCHS] # Used only when backbone is ResNet18. Number of initial epochs during which all layers are frozen (not updated).
[--grad_clip GRAD_CLIP] # the maximum allowed norm of the gradients during backpropagation. If the total gradient norm exceeds this value, PyTorch will scale down all gradients proportionally so that their norm equals args.grad_clip.
[--norm_type NORM_TYPE] # 1 for L1-norm | 2 for L2-norm | 'inf' for max norm
[--min_delta MIN_DELTA] # Threshold value used in early stopping to determine whether an improvement in the monitored metric (e.g., validation loss or accuracy) is significant.
[--patience PATIENCE] # Number of epochs without improvement (based on min_delta and mode) before early stopping
[--label_smoothing LABEL_SMOOTHING] # Regularization techinque, to prevent models from becoming overly confident in their predictions during training.
[--temperature1 TEMPERATURE1] # Temperature for similarity scaling, used for validation, testing, and inference
[--temperature2 TEMPERATURE2] # Temperature for aux_similarity scaling, used for training
[--lambda_reg LAMBDA_REG] # weight of the auxiliary loss in the total episode loss
[--num_workers NUM_WORKERS] # In PyTorch's DataLoader, the num_workers parameter controls how many subprocesses are used to load the data in parallel.
[--pin_memory] # If True, preloads data into pinned (page-locked) memory to speed up GPU transfers.
[--seed SEED] # Reproducibility via seed setting
[--save_path SAVE_PATH] # used after finalizing training to save the model weights in checkpoints
[--load_checkpoint LOAD_CHECKPOINT] # used in inference to load the weights of a trained model
[--dataset_mean DATASET_MEAN [DATASET_MEAN ...]] # dataset mean used in training, specifically in loader or can be computed from dataset
[--dataset_std DATASET_STD [DATASET_STD ...]] # dataset std used in training, specifically in loader or can be computed from dataset
[--use_support_stats USE_SUPPORT_STATS] # to compute mean and std from the support data in inference
[--fine_tune_epochs FINE_TUNE_EPOCHS] # Epochs to fine-tune on support set in inference.
[--use_tta] # Test-Time Augmentation used optionally at inference
[--normalization_stats NORMALIZATION_STATS NORMALIZATION_STATS NORMALIZATION_STATS NORMALIZATION_STATS NORMALIZATION_STATS NORMALIZATION_STATS] # fed in mean and std [m1,m2,m3],[st1,st2,st3]
[--uncertain_thresh UNCERTAIN_THRESH] # confidence threshold for rejecting predictions at inference

7. Examples for training and inference trigger

    7.1 Training
    python train_feat.py --data_path ../../../../newProject/CUB200 --backbone ResNet18 --way 5 --shot 5 --query 15 --episodes_per_epoch 1000 --val_episodes_per_epoch 600 --epochs 120  --warmup_epochs 10 --dropout_rate 0.5 --weight_decay 5e-4 --lr 1e-4 --label_smoothing 0.2 --image_resize 92 --image_crop 84 --save_path checkpoints/feat_resnet18_cub_ai_alliance --lambda_reg 0.5 --patience 100
    7.2 Inference
    python Inference.py   --mode inference   --support_root Prediction/support_set   --query_root Prediction/query_set   --load_checkpoint /workdir/ModularNewProject/checkpoints/feat_resnet18_cub/best_model.pth  --use_support_stats True --backbone ResNet18 --output_root /workdir/ModularNewProject/resultsMeta --weight_decay 1e-4
    


    
8. Output
    8.1 Training Output:
        checkpoints/models_under_any_name.pth
        checkpoints/logs/
    8.2 Inference Output:
        ├── output_root                
            ├── Class1 
            ├── Class2
            ...
            ├── Uncertain
            ├── Diagnostics
                ├── Support_Samples # for visualization 
                ├── Class_distribution.txt
                ├── Classification_Report.csv
                ├── Confidence_Histogram.png

9. Citation
@inproceedings{ye2020few,
  title={Few-shot learning with adaptive feature transformation},
  author={Ye, Han-Jia and Hu, Hexiang and Zhan, De-Chuan and Sha, Fei},
  booktitle={ICLR},
  year={2020}
}

10. License    
MIT License – https://opensource.org/license/MIT