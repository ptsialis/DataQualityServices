 #!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from models.feat3 import FEAT
from PIL import Image
import random
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import argparse
import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class BirdClassifier:
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
        
        # Set seeds for reproducibility
        self.set_seeds()
        
        # Setup output directories
        self.create_directories()
        
        # Initialize model and transforms
        self.load_model()
        self.define_transforms()
    
    def set_seeds(self):
        """Set random seeds for reproducibility"""
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def create_directories(self):
        
        if os.path.exists(self.args.output_root):
            print(f"Cleaning up existing output directory: {self.args.output_root}")
            shutil.rmtree(self.args.output_root)

        # Recreate output structure
        os.makedirs(self.args.output_root, exist_ok=True)
        self.diagnostic_dir = os.path.join(self.args.output_root, "diagnostics")
        os.makedirs(self.diagnostic_dir, exist_ok=True)
        os.makedirs(os.path.join(self.diagnostic_dir, "support_samples"), exist_ok=True)
#         """Create output directories"""
#         os.makedirs(self.args.output_root, exist_ok=True)
#         self.diagnostic_dir = os.path.join(self.args.output_root, "diagnostics")
#         os.makedirs(self.diagnostic_dir, exist_ok=True)
#         os.makedirs(os.path.join(self.diagnostic_dir, "support_samples"), exist_ok=True)
    
    def load_model(self):
        """Load FEAT model with checkpoint"""
        # Initialize backbone
        if self.args.backbone == "resnet18":
            self.backbone = ResNet18FeatEncoder(
                dropout_rate=self.args.dropout_rate,
                hidden_dim=self.args.hidden_dim
            )
        elif self.args.backbone == "convnet4":
            self.backbone = ConvNet4(
                in_channels=3,
                feat_dim=self.args.hidden_dim
            )
        else:
            raise ValueError(f"Unsupported backbone: {self.args.backbone}")
        
        # Initialize FEAT model
        self.model = FEAT(
            encoder=self.backbone,
            hidden_dim=self.args.hidden_dim,
            temp1=self.args.temperature,
            temp2=1.0,
            proto_attn_layers=self.args.proto_attn_layers,
            proto_attn_heads=self.args.proto_attn_heads,
            aux_transformer_layers=self.args.aux_transformer_layers,
            aux_transformer_heads=self.args.aux_transformer_heads,
            aux_transformer_ffn_dim_factor=self.args.aux_transformer_ffn_dim_factor
        ).to(self.device)
        
        # Load checkpoint
        if os.path.exists(self.args.checkpoint):
            ckpt = torch.load(self.args.checkpoint, map_location=self.device)
            
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
            print(f"Checkpoint parameters: {sum(p.numel() for p in ckpt.values())}")
            
            # Handle different checkpoint formats
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            else:
                self.model.load_state_dict(ckpt, strict=False)
#             ckpt_params = sum(p.numel() for p in ckpt.values())
#             model_params = sum(p.numel() for p in self.model.parameters())
#             assert ckpt_params == model_params, "Parameter count mismatch!"
            print(f"Loaded model from {self.args.checkpoint}")
        else:
            print(f"Warning: Checkpoint not found at {self.args.checkpoint}")
        
        self.model.eval()
    
    def define_transforms(self):
        """Define image transforms with normalization"""
        # Use provided stats or load from file
        if self.args.normalization_stats:
            self.mean = self.args.normalization_stats[:3]
            self.std = self.args.normalization_stats[3:]
        else:
            # Try to load dataset-specific normalization
            stats_path = os.path.join(os.path.dirname(self.args.checkpoint), "normalization_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path) as f:
                    stats = json.load(f)
                    self.mean = stats['mean']
                    self.std = stats['std']
                print(f"Loaded dataset normalization: mean={self.mean}, std={self.std}")
            else:
                self.mean = [0.4706, 0.4632, 0.3927]  # CUB training mean
                self.std = [0.1956, 0.1925, 0.1986] # CUB training std
                print("Using default CUB normalization hardcoded")
        
        # Base transform without augmentation
        self.base_transform = transforms.Compose([
            transforms.Resize(self.args.resize_size),
            transforms.CenterCrop(self.args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # Optional test-time augmentation transforms
        self.tta_transforms = [
            transforms.Compose([
                transforms.Resize(self.args.resize_size),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.CenterCrop(self.args.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ]),
            transforms.Compose([
                transforms.Resize(self.args.resize_size),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.CenterCrop(self.args.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        ] if self.args.use_tta else []
    
    def load_support_images(self):
        """Load support images from class folders"""
        self.class_folders = sorted([
            f for f in os.listdir(self.args.support_root) 
            if os.path.isdir(os.path.join(self.args.support_root, f)) and not f.startswith('.')
        ])
        
        print("\nLoading support images:")
        for cls in self.class_folders:
            cls_path = os.path.join(self.args.support_root, cls)
            self.support_images[cls] = []
            
            # Save sample image for reference
            sample_count = 0
            
            for img_name in os.listdir(cls_path):
                if img_name.startswith('.'): 
                    continue
                try:
                    img_path = os.path.join(cls_path, img_name)
                    img = Image.open(img_path).convert("RGB")
                    
                    # Save first few samples for diagnostics
                    if sample_count < 3:
                        shutil.copy(img_path, os.path.join(self.diagnostic_dir, "support_samples", f"{cls}_{img_name}"))
                        sample_count += 1
                    
                    img_tensor = self.base_transform(img)
                    self.support_images[cls].append(img_tensor)
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")
            
            print(f"  {cls}: {len(self.support_images[cls])} images")
    
    def load_query_images(self):
        """Load query images from folder"""
        print("\nLoading query images:")
        for img_name in sorted(os.listdir(self.args.query_folder)):
            if img_name.startswith('.'):
                continue
            path = os.path.join(self.args.query_folder, img_name)
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = self.base_transform(img)
                self.query_images.append(img_tensor)
                self.query_image_paths.append((img_name, path))
                
                # Extract true label if available (from filename or folder structure)
                if "true_labels" in self.args and img_name in self.args.true_labels:
                    self.true_labels.append(self.args.true_labels[img_name])
                else:
                    self.true_labels.append(None)
            except Exception as e:
                print(f"Skipping {path}: {e}")
        
        if not self.query_images:
            print("No query images found!")
            return
        
        self.query_tensor = torch.stack(self.query_images).to(self.device)
        print(f"Loaded {len(self.query_tensor)} query images")
        
        # Initialize prediction storage
        self.episode_predictions = [[] for _ in range(len(self.query_tensor))]
        self.confidence_scores = [[] for _ in range(len(self.query_tensor))]
        
    def fine_tune(self):
        """Fine-tune model on support images with proper mode handling"""
        if self.args.fine_tune_epochs <= 0:
            return

        print(f"\nFine-tuning for {self.args.fine_tune_epochs} epochs...")
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, nesterov=True)

        # Prepare full support set
        all_support = []
        all_labels = []
        for label_idx, cls in enumerate(self.class_folders):
            imgs = self.support_images[cls]
            all_support.extend(imgs)
            all_labels.extend([label_idx] * len(imgs))

        all_support_tensor = torch.stack(all_support).to(self.device)
        all_labels_tensor = torch.tensor(all_labels).to(self.device)

        for epoch in range(self.args.fine_tune_epochs):
            optimizer.zero_grad()

            # For fine-tuning we use 'train' mode but need query_y
            logits = self.model(
                all_support_tensor,
                all_labels_tensor,
                all_support_tensor,
                all_labels_tensor,  
                mode='train'
            )[0]  # Only use main logits for fine-tuning

            loss = F.cross_entropy(logits, all_labels_tensor)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{self.args.fine_tune_epochs}, Loss: {loss.item():.4f}")

        self.model.eval()

    
    def apply_tta(self, images):
        """Apply test-time augmentation to images"""
        if not self.tta_transforms:
            return [images]
            
        augmented_batches = []
        for tta_transform in self.tta_transforms:
            augmented = []
            for img_tensor in images:
                # Convert tensor to PIL for transformation
                img_pil = transforms.ToPILImage()(img_tensor.cpu())
                augmented.append(tta_transform(img_pil))
            augmented_batches.append(torch.stack(augmented).to(self.device))
        return augmented_batches
    
    def run_inference(self):
        """Run episodic inference"""
        if not self.query_images:
            print("No query images to process!")
            return
        
        print(f"\nRunning {self.args.num_episodes} episodes...")
        class_usage = {cls: 0 for cls in self.class_folders}
        
        for ep in tqdm(range(self.args.num_episodes), desc="Episodes"):
            # Select classes with usage-based weighting
            weights = [1/(class_usage[cls]+1) for cls in self.class_folders]
            sampled_classes = random.choices(
                self.class_folders, 
                weights=weights,
                k=self.args.n_way
            )
            
            support_data = []
            support_labels = []
            
            for new_label, cls in enumerate(sampled_classes):
                available = self.support_images[cls]
                if not available:
                    continue
                    
                # Sample support images
                samples = random.sample(available, min(self.args.n_shot, len(available)))
                support_data.extend(samples)
                support_labels.extend([new_label] * len(samples))
                class_usage[cls] += 1
            
            if not support_data:
                continue
                
            support_data = torch.stack(support_data).to(self.device)
            support_labels = torch.tensor(support_labels).to(self.device)
            
            with torch.no_grad():
                # Apply test-time augmentation to queries
                query_batches = [self.query_tensor] + self.apply_tta(self.query_images)
                episode_logits = []
                
            
                for query_batch in query_batches:
                    if isinstance(query_batch, list):
                        query_batch = torch.stack(query_batch).to(self.device)
                    else:
                        query_batch = query_batch.to(self.device)
                    # Get predictions
                    dummy_query_y = torch.zeros(len(self.query_tensor)).long().to(self.device)
                    logits = self.model(
                        support_data, 
                        support_labels, 
                        query_batch, 
                        dummy_query_y,
                        mode='eval')
                    episode_logits.append(logits)
                
                # Average logits across TTA variants
                logits = torch.stack(episode_logits).mean(dim=0)
                
#                 # Apply temperature scaling
#                 logits = logits / max(self.args.temperature, 1e-8)
                
                # Get confidence
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
            for vote_idx, vote in enumerate(votes):
                conf = self.confidence_scores[i][vote_idx]
                vote_weights[vote] = vote_weights.get(vote, 0) + conf
                
            # Get top prediction
            sorted_votes = sorted(vote_weights.items(), key=lambda x: x[1], reverse=True)
            top_pred = sorted_votes[0][0]
            self.final_preds.append(top_pred)
            self.avg_confidences.append(sum(self.confidence_scores[i]) / len(self.confidence_scores[i]))
    
    def evaluate_performance(self):
        """Evaluate performance if true labels are available"""
        if not any(self.true_labels) or len(self.true_labels) != len(self.final_preds):
            print("True labels not available for evaluation")
            return
        
        # Map class names to indices for confusion matrix
        class_to_idx = {cls: i for i, cls in enumerate(self.class_folders)}
        y_true = [class_to_idx[label] for label in self.true_labels if label in class_to_idx]
        y_pred = [class_to_idx[pred] for pred in self.final_preds if pred in class_to_idx]
        
        if len(y_true) != len(y_pred):
            print("Mismatch in true/predicted labels")
            return
        
        # Generate classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_folders,
            output_dict=True
        )
        
        # Save report
        with open(os.path.join(self.diagnostic_dir, "classification_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_folders,
                    yticklabels=self.class_folders)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.diagnostic_dir, "confusion_matrix.png"))
        
        print("\n=== Evaluation Results ===")
        print(f"Overall Accuracy: {report['accuracy']:.2%}")
        for cls in self.class_folders:
            print(f"{cls} Precision: {report[cls]['precision']:.2%}, Recall: {report[cls]['recall']:.2%}")
    
    def save_results(self):
        """Save predictions and diagnostics"""
        print("\nSaving results...")
        # Create class folders
        for cls in self.class_folders:
            os.makedirs(os.path.join(self.args.output_root, cls), exist_ok=True)
        
        # Copy images to predicted folders
        for (img_name, img_path), pred_class in zip(self.query_image_paths, self.final_preds):
            dest = os.path.join(self.args.output_root, pred_class, img_name)
            shutil.copy(img_path, dest)
        
        # Save CSV report
        df = pd.DataFrame({
            'Image': [name for name, _ in self.query_image_paths],
            'Prediction': self.final_preds,
            'Avg_Confidence': [f"{c:.4f}" for c in self.avg_confidences],
            'Num_Votes': [len(votes) for votes in self.episode_predictions],
            'True_Label': self.true_labels if self.true_labels else None
        })
        report_path = os.path.join(self.diagnostic_dir, "classification_report.csv")
        df.to_csv(report_path, index=False)
        
        # Class distribution report
        class_dist = {cls: 0 for cls in self.class_folders}
        for pred in self.final_preds:
            if pred in class_dist:
                class_dist[pred] += 1
        
        dist_report = []
        for cls, count in class_dist.items():
            dist_report.append(f"{cls}: {count} images")
        
        with open(os.path.join(self.diagnostic_dir, "class_distribution.txt"), "w") as f:
            f.write("\n".join(dist_report))
        
        # Evaluate performance if true labels available
        if any(self.true_labels):
            self.evaluate_performance()
        
        print("\n=== Classification Complete ===")
        print(f"Results saved to: {self.args.output_root}")
        print(f"Diagnostics saved to: {self.diagnostic_dir}")
        print("Class distribution:")
        print("\n".join(dist_report))
    
    def run(self):
        """Full classification pipeline"""
        self.load_support_images()
        self.load_query_images()
        self.fine_tune()
        self.run_inference()
        self.aggregate_results()
        self.save_results()


class ResNet18FeatEncoder(nn.Module):
    """Feature extractor based on ResNet-18"""
    def __init__(self, dropout_rate=0.3, hidden_dim=640):
        super().__init__()
        base_resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder_features = nn.Sequential(*list(base_resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(512, hidden_dim)

    def forward(self, x):
        x = self.encoder_features(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.projection(x)
        return x


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Bird Species Classifier')
    # Data paths
    parser.add_argument('--support_root', type=str, required=True,
                        help='Path to support class folders')
    parser.add_argument('--query_folder', type=str, required=True,
                        help='Path to query images folder')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Output directory for classified images')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Model architecture
    parser.add_argument('--backbone', type=str, default="resnet18",
                        choices=["resnet18", "convnet4"],
                        help='Feature extractor backbone')
    parser.add_argument('--hidden_dim', type=int, default=640,
                        help='Hidden dimension size')
    parser.add_argument('--proto_attn_layers', type=int, default=3,
                        help='Number of prototype attention layers')
    parser.add_argument('--proto_attn_heads', type=int, default=2,
                        help='Number of attention heads in prototype attention')
    parser.add_argument('--aux_transformer_layers', type=int, default=1,
                        help='Number of auxiliary transformer layers')
    parser.add_argument('--aux_transformer_heads', type=int, default=1,
                        help='Number of attention heads in auxiliary transformer')
    parser.add_argument('--aux_transformer_ffn_dim_factor', type=int, default=4,
                        help='FFN dimension factor for auxiliary transformer')
    parser.add_argument('--use_cosine', action='store_true',
                        help='determine how distance is computed either with cosine similarity or Euclidean Distance')
    
    # Training/fine-tuning
    parser.add_argument('--fine_tune_epochs', type=int, default=5,
                        help='Epochs for fine-tuning (0 to disable)')
    parser.add_argument('--optimizer', type=str, default="sgd",
                        choices=["sgd", "adam"],
                        help='Optimizer for fine-tuning')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--use_scheduler', action='store_true',
                        help='Use learning rate scheduler during fine-tuning')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout rate for feature extractor')
    
    # Inference
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of episodes for inference')
    parser.add_argument('--n_way', type=int, default=5,
                        help='Number of classes per episode')
    parser.add_argument('--n_shot', type=int, default=5,
                        help='Number of support images per class')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature scaling value')
    parser.add_argument('--use_tta', action='store_true',
                        help='Enable test-time augmentation')
    
    # Image processing
    parser.add_argument('--resize_size', type=int, default=94,
                        help='Size for resizing images')
    parser.add_argument('--crop_size', type=int, default=84,
                        help='Size for center cropping')
    parser.add_argument('--normalization_stats', nargs=6, type=float, default=None,
                        help='Normalization stats [mean_r, mean_g, mean_b, std_r, std_g, std_b]')
    
    # System
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    args = parse_args()
    classifier = BirdClassifier(args)
    classifier.run()


if __name__ == "__main__":
    main()

