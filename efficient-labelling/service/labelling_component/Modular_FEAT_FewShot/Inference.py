#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import shutil
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import random
import pandas as pd
from tqdm import tqdm
from models.backbones import get_backbone # to call one of the 3 backbones
from models.feat import FEAT # to call the transformer part
from config import get_args # get the args from the config.py
import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter

class GeneralClassifier:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.support_images = {}
        self.query_images = []
        self.query_image_paths = []
        self.episode_predictions = []
        self.confidence_scores = []
        self.final_preds = []
        self.avg_confidences = []
        self.true_labels = []  # For evaluation if available
        self.mean = None
        self.std = None
        self.tta_transforms = []
        self.query_pil_images = []
        
        if args.mode == 'inference':
            if args.support_root is None or args.query_root is None:
                raise ValueError("Both `--support_root` and `--query_root` are required in inference mode.")
            elif args.load_checkpoint is None:
                raise ValueError('Path to checkpoint has not been shared')
            elif args.temperature1 is None:
                raise ValueError('Temperature1 parameter was not given. This is essential for computing the logits!')

        
        # Set seeds for reproducibility
        self.set_seeds()
        
        # Setup output directories
        self.create_directories()
        
        # Initialize model 
        self.load_model()
        #self.define_transforms()
    
    def set_seeds(self):
        """Set random seeds for reproducibility"""
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def create_directories(self):
        """Create output directories"""
        os.makedirs(self.args.output_root, exist_ok=True) # where to save the images after classification
        self.diagnostic_dir = os.path.join(self.args.output_root, "diagnostics") # additional info is saved in diagnostics
        os.makedirs(self.diagnostic_dir, exist_ok=True) # making sure that directory= folder diagnostics exists
        os.makedirs(os.path.join(self.diagnostic_dir, "support_samples"), exist_ok=True) # saving samples of the support images, to to check the quality of support images used
    
    def load_model(self):
        """Load FEAT model with checkpoint saved in given arg load_checkpoint"""
        # Initialize backbone
        if self.args.backbone in ['ResNet18', 'ConvNet4', 'ResNet12']:
            self.backbone = get_backbone(self.args.backbone, self.args.hidden_dim, self.args.dropout_rate).to(self.device)
        else:
            raise ValueError(f"Unsupported backbone: {self.args.backbone}")
        
        # Initialize FEAT model
        self.model = FEAT(
            encoder=self.backbone,
            hidden_dim=self.args.hidden_dim,
            temp1=self.args.temperature1,
            proto_attn_layers=self.args.proto_attn_layers,
            proto_attn_heads=self.args.proto_attn_heads,
            aux_transformer_layers=self.args.aux_transformer_layers,
            aux_transformer_heads=self.args.aux_transformer_heads,
            aux_transformer_ffn_dim_factor=self.args.aux_transformer_ffn_dim_factor
        ).to(self.device)
        
        # Load checkpoint, must be given in args
        try:
            ckpt = torch.load(self.args.load_checkpoint, map_location=self.device)
            
            # Handle different checkpoint formats
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
            else:
                self.model.load_state_dict(ckpt)
                
            print(f"Loaded model from {self.args.load_checkpoint}")
        except FileNotFoundError:
             raise FileNotFoundError(f"Checkpoint not found at {self.args.load_checkpoint}")
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load checkpoint due to architecture mismatch or corruption: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while loading checkpoint: {e}")
        
        self.model.eval() # ensuring that the model is eval, no dropout rate no BN, ...
    
    def define_transforms(self):
        # Use provided stats or load from file
        if self.args.normalization_stats:
            print("Using provided normalization stats in command prompt")
            self.mean = self.args.normalization_stats[:3]
            self.std = self.args.normalization_stats[3:]
        elif self.args.use_support_stats:
            print(f"Using support images to compute mean/std for normalization, mean is {self.mean} and std is {self.std}")
            #self.mean, self.std = compute_mean_std_from_images(support_image_paths)
        else:
            print("Using fixed normalization (from ImageNet)")
            self.mean = [0.485, 0.456, 0.406]  
            self.std = [0.229, 0.224, 0.225]  

    
        # Base transform without augmentation
        self.base_transform = transforms.Compose([
            transforms.Resize(self.args.image_resize),
            transforms.CenterCrop(self.args.image_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # Optional test-time augmentation transforms
        self.tta_transforms = []
        if self.args.use_tta:
            self.tta_transforms = [
                transforms.Compose([
                    transforms.Resize(self.args.image_resize),
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.CenterCrop(self.args.image_crop),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std)
                ]),
                transforms.Compose([
                    transforms.Resize(self.args.image_resize),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.CenterCrop(self.args.image_crop),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std)
                ])]
    def compute_mean_std(self, image_paths):
        preprocess = transforms.Compose([
        transforms.Resize(self.args.image_resize),
        transforms.CenterCrop(self.args.image_crop),
        transforms.ToTensor()
    ])
    
        images = []
        for path in tqdm(image_paths, desc= 'Computing stats'):
            try:
                img = Image.open(path).convert('RGB')
                img = preprocess(img)
                images.append(img)
            except Exception as e:
                print(f"Skipping {path}: {e}")
        if not images: # if the support images are none return ImageNet mean and std
            return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        stacked = torch.stack(images)  # Shape: [N, C, H, W]
        mean = stacked.mean(dim=[0, 2, 3]).tolist()  # per-channel mean
        std = stacked.std(dim=[0, 2, 3]).tolist()   # per-channel std
        return mean, std       
    
    def safe_copy(self, src, dest_dir):
        filename = os.path.basename(src)
        #print(f'filename is {filename}')
        dest_path = os.path.join(dest_dir, src)
        #print(f'dest_path is {dest_path}')
        if not os.path.abspath(dest_path).startswith(dest_dir):
            raise ValueError(f"Invalid destination path: {dest_path}")
        shutil.copy(src, dest_path)
        return dest_path

    def load_support_images(self):
        support_image_paths = []
        """Load support images from class folders"""
        self.class_folders = sorted([f for f in os.listdir(self.args.support_root) if os.path.isdir(os.path.join(self.args.support_root, f)) and not f.startswith('.')])
        #  self.class_folders will give ['Class_1', 'Class_2', ., ., .]
        print("\nLoading support images:")
        try:
            if len(self.class_folders) > 0:    
                print(f'Classes is {self.class_folders}')
        except:
            print('Support folder is empty!')
        for cls in self.class_folders:
            cls_path = os.path.join(self.args.support_root, cls)
            self.support_images[cls] = []
            
            # Save sample image for reference
            sample_count = 0
            
            for img_name in os.listdir(cls_path):
                if img_name.startswith('.'): 
                    continue
                img_path = os.path.join(cls_path, img_name)
                support_image_paths.append(img_path)
                    
                # Save first few samples for diagnostics
                if sample_count < 3:
                    try:
                        dest = os.path.join(self.diagnostic_dir, "support_samples", f"{cls}_{img_name}")
                        shutil.copy(img_path, os.path.join(self.diagnostic_dir, "support_samples", f"{cls}_{img_name}"))
                        sample_count += 1
                    except Exception as e:
                        print(f"Copy failed: {e}")
        # Compute stats if requested
        if self.args.use_support_stats:
            print("Computing normalization stats from support images...")
            self.mean, self.std = self.compute_mean_std(support_image_paths)
            print(f"Computed mean: {self.mean}, std: {self.std}")
        else:
            self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        
        # Now define transforms with final stats
        self.define_transforms()
        
        # Now process support images with final transform
        for cls in self.class_folders:
            cls_path = os.path.join(self.args.support_root, cls)
            for img_name in os.listdir(cls_path):
                if img_name.startswith('.'): 
                    continue
                
                img_path = os.path.join(cls_path, img_name)
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = self.base_transform(img)
                    self.support_images[cls].append(img_tensor)
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")
            
            print(f"{cls}: {len(self.support_images[cls])} images")
    
    
    def load_query_images(self):
        """Load query images from folder"""
        if not os.path.exists(self.args.query_root):
            raise FileNotFoundError(f"Query directory not found: {self.args.query_root}")
        
        print("\nLoading query images:")
        
        load_query_images =[]
    
        for img_name in sorted(os.listdir(self.args.query_root)):
            if img_name.startswith('.'):
                continue
            path = os.path.join(self.args.query_root, img_name)
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    self.query_pil_images.append(img)
                    img_tensor = self.base_transform(img)
                    self.query_images.append(img_tensor)
                    self.query_image_paths.append((img_name, path))
                    self.true_labels.append(None)  # Placeholder
            except Exception as e:
                print(f"Skipping {path}: {e}")
        
        if not self.query_images:
            print("No query images found!")
            return
        
        self.query_tensor = torch.stack(self.query_images).to(self.device)
        print(f"Loaded {len(self.query_tensor)} query images")
        
        # Initialize prediction storage
        self.episode_predictions = [[] for _ in range(len(self.query_tensor))] # initiating placeholders
        self.confidence_scores = [[] for _ in range(len(self.query_tensor))] # initiating place holders
    
    
    def fine_tune(self):
        if self.args.fine_tune_epochs <= 0:
            return

        print(f"\nFine-tuning for {self.args.fine_tune_epochs} epochs...")
        self.model.train()

        # Configure optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=0.01,
            momentum=0.9,
            nesterov=True,
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

        # Calculate iterations per epoch
        total_support = sum(len(v) for v in self.support_images.values())
        iterations_per_epoch = max(50, total_support // (self.args.way * self.args.shot))

        for epoch in range(self.args.fine_tune_epochs):
            epoch_loss = 0
            for it in range(iterations_per_epoch):
                # Create episode with fixed way/shot
                support_data, support_labels = self._create_episode()

                optimizer.zero_grad()
                logits = self.model(
                    support_data, 
                    support_labels, 
                    support_data,  # Use same as query
                    support_labels,
                    mode='train'
                )[0]

                loss = F.cross_entropy(logits, support_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / iterations_per_epoch
            scheduler.step(avg_loss)
            print(f"Epoch {epoch+1}/{self.args.fine_tune_epochs}, Loss: {avg_loss:.4f}")
            
            # ADDED LR MONITORING
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{self.args.fine_tune_epochs}, "
                  f"Loss: {avg_loss:.4f}, "
                  f"LR: {current_lr:.2e}")

        self.model.eval()

    def _create_episode(self):
        """Create an episode for fine-tuning"""
        support_data = []
        support_labels = []

        # Select classes
        sampled_classes = random.sample(self.class_folders, self.args.way)

        for new_label, cls in enumerate(sampled_classes):
            available = self.support_images[cls]
            samples = random.sample(available, min(self.args.shot, len(available)))
            support_data.extend(samples)
            support_labels.extend([new_label] * len(samples))

        return torch.stack(support_data).to(self.device), torch.tensor(support_labels).to(self.device)

    
    def apply_tta(self, image_tensor):
        augments = []
        base_img = self.base_transform(image_tensor).unsqueeze(0)
        augments.append(base_img)
        for transform in self.tta_transforms:
            try:
                aug_img = transform(image_tensor)
                aug_img = transforms.functional.normalize(aug_img, self.mean, self.std)
                augments.append(aug_img.unsqueeze(0))
            except Exception as E:
                print(f"TTA transform failed: {e}")
        return torch.cat(augments, dim=0).to(self.device)
                
    
    
    def run_inference(self):
        """Run episodic inference"""
        if not self.query_images:
            print("No query images to process!")
            return
        
        print(f"\nRunning {self.args.num_episodes} episodes...")
        class_usage = {cls: 0 for cls in self.class_folders} #initiation at the begining
        
        for ep in tqdm(range(self.args.num_episodes), desc="Episodes"):
            # Select classes with usage-based weighting
            weights = [1/(class_usage[cls]+1) for cls in self.class_folders]
            sampled_classes = random.choices(
                self.class_folders, 
                weights=weights,
                k=self.args.way
            )
            
            support_data = []
            support_labels = []
            
            for new_label, cls in enumerate(sampled_classes):
                available = self.support_images[cls]
                if not available:
                    continue
                    
                # Sample support images
                samples = random.sample(available, min(self.args.shot, len(available)))
                support_data.extend(samples)
                support_labels.extend([new_label] * len(samples))
                class_usage[cls] += 1
            
            if not support_data:
                continue
                
            support_data = torch.stack(support_data).to(self.device)
            support_labels = torch.tensor(support_labels).to(self.device)
            
            with torch.no_grad():
                # Process each query with TTA
                all_logits = []
                
                for i, query_tensor in enumerate(self.query_images):
                    pil_img = self.query_pil_images[i] 
                    # Apply TTA
                    #tta_batch = self.apply_tta(query_tensor)
                    tta_batch = self.apply_tta(pil_img)
                    
                    # Get predictions for each TTA variant
                    logits = self.model(
                        support_data, 
                        support_labels, 
                        tta_batch, 
                        mode='eval'
                    )
                    logits = logits.mean(dim=0)  # Average TTA predictions
                    all_logits.append(logits)
                
                # Stack all query logits
                logits = torch.stack(all_logits)
                logits = logits / (self.model.temperature1 + 1e-8)
                
                # Get predictions
                probs = F.softmax(logits, dim=1)
                conf, preds = torch.max(probs, dim=1)
                
                # Store results
                for i, (p, c) in enumerate(zip(preds, conf)):
                    if p < len(sampled_classes):
                        predicted_class = sampled_classes[p.item()]
                        self.episode_predictions[i].append(predicted_class)
                        self.confidence_scores[i].append(c.item())
    
    
    def aggregate_results(self):
        """Aggregate results using confidence-weighted voting"""
        print("\nAggregating results...")
        self.final_preds = []
        self.avg_confidences = []
        
        for i, votes in enumerate(self.episode_predictions):
            if not votes:
                self.final_preds.append("uncertain")
                self.avg_confidences.append(0.0)
                continue
                
            # Weight votes by confidence
            vote_weights = {}
            for vote, conf in zip(votes, self.confidence_scores[i]):
                vote_weights[vote] = vote_weights.get(vote, 0) + conf
                
            # Get top prediction
            top_pred = max(vote_weights.items(), key=lambda x: x[1])[0]
            self.final_preds.append(top_pred)
            self.avg_confidences.append(np.mean(self.confidence_scores[i]))
    
    def save_results(self):
        print("\nSaving results...")
        # Create class folders
        for cls in self.class_folders: #['BIRD1', 'BIRD2',..]
            os.makedirs(os.path.join(self.args.output_root, cls), exist_ok=True) # /workdir/ModularNewProject/results + cls
        os.makedirs(os.path.join(self.args.output_root, "uncertain"), exist_ok=True)
        class_dist = {cls:0 for cls in self.class_folders}
        class_dist['uncertain']=0
        
        # Copy images to predicted folders
        for (img_name, img_path), pred_class, avg_conf in zip(self.query_image_paths, self.final_preds, self.avg_confidences):
            print(f'prediction class is {pred_class} and average confidence is {avg_conf}')
            if pred_class in self.class_folders and avg_conf > self.args.uncertain_thresh:
                dest_dir = os.path.join(self.args.output_root, pred_class) # destination is the folder where the image is saved
                class_dist[pred_class] += 1
            else:
                dest_dir = os.path.join(self.args.output_root, "uncertain")
                class_dist["uncertain"] += 1 # the uncertainty is high, tus save the image in uncertain folder

            shutil.copy(img_path, os.path.join(dest_dir, img_name))

        # Save distribution report
        with open(os.path.join(self.diagnostic_dir, "class_distribution.txt"), "w") as f:
            for cls, count in class_dist.items():
                f.write(f"{cls}: {count} images\n")
        
        # Save CSV report
        df = pd.DataFrame({
            'Image': [name for name, _ in self.query_image_paths],
            'Prediction': self.final_preds,
            'Avg_Confidence': self.avg_confidences,
            'Num_Votes': [len(votes) for votes in self.episode_predictions]})
        report_path = os.path.join(self.diagnostic_dir, "classification_report.csv")
        df.to_csv(report_path, index=False)
        
        # Class distribution
        class_dist = {cls: 0 for cls in self.class_folders}
        for pred in self.final_preds:
            if pred in class_dist:
                class_dist[pred] += 1
        
        with open(os.path.join(self.diagnostic_dir, "class_distribution.txt"), "w") as f:
            for cls, count in class_dist.items():
                f.write(f"{cls}: {count} images\n")
        
        print("\n=== Classification Complete ===")
        print(f"Results saved to: {self.args.output_root}")
        print(f"Diagnostics saved to: {self.diagnostic_dir}")
        print("Class distribution:")
        for cls, count in class_dist.items():
            print(f"  {cls}: {count} images")
        plt.figure(figsize=(10, 6))
        plt.hist(self.avg_confidences, bins=20, alpha=0.7, color='skyblue')
        plt.title('Confidence Distribution')
        plt.xlabel('Average Confidence')
        plt.ylabel('Count')
        plt.savefig(os.path.join(self.diagnostic_dir, 'confidence_histogram.png'))
        plt.close()
    
    
#     def evaluate_performance(self):
#         """Evaluate performance if true labels are available"""
#         if not any(self.true_labels) or len(self.true_labels) != len(self.final_preds):
#             print("True labels not available for evaluation")
#             return
        
#         # Map class names to indices for confusion matrix
#         class_to_idx = {cls: i for i, cls in enumerate(self.class_folders)}
#         y_true = [class_to_idx[label] for label in self.true_labels if label in class_to_idx]
#         y_pred = [class_to_idx[pred] for pred in self.final_preds if pred in class_to_idx]
        
#         if len(y_true) != len(y_pred):
#             print("Mismatch in true/predicted labels")
#             return
        
#         # Generate classification report
#         report = classification_report(
#             y_true, y_pred, 
#             target_names=self.class_folders,
#             output_dict=True
#         )
        
#         # Save report
#         with open(os.path.join(self.diagnostic_dir, "classification_report.json"), "w") as f:
#             json.dump(report, f, indent=2)
        
#         # Generate confusion matrix
#         cm = confusion_matrix(y_true, y_pred)
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                     xticklabels=self.class_folders,
#                     yticklabels=self.class_folders)
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.title('Confusion Matrix')
#         plt.savefig(os.path.join(self.diagnostic_dir, "confusion_matrix.png"))
        
#         print("\n=== Evaluation Results ===")
#         print(f"Overall Accuracy: {report['accuracy']:.2%}")
#         for cls in self.class_folders:
#             print(f"{cls} Precision: {report[cls]['precision']:.2%}, Recall: {report[cls]['recall']:.2%}")
    
    
    
    def run(self):
        """Full classification pipeline"""
        self.load_support_images()
        self.load_query_images()
        if self.query_images:                 
            self.fine_tune()
            self.run_inference()
            self.aggregate_results()
            self.save_results()
        else:
            print("Skipping inference - no query images loaded")


def main():
    args = get_args()
    classifier = GeneralClassifier(args)
    classifier.run()


if __name__ == "__main__":
    main()

