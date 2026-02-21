#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import shutil
import random
import argparse
import logging
from typing import List, Optional
import shutil 

class DatasetCreator:
    def __init__(self, args):
        """
        Creates a few-shot learning dataset from CUB-200 images
        
        :param args: Command-line arguments
        """
        self.args = args
        self.source_path = args.source_path
        self.destination_root = args.destination_root
        self.n_classes = args.n_classes
        self.n_support = args.n_support
        self.n_query = args.n_query
        
        # Use default bird list if not provided
        self.bird_list = [
            '060.Glaucous_winged_Gull', '056.Pine_Grosbeak', '024.Red_faced_Cormorant',
            '008.Rhinoceros_Auklet', '009.Brewer_Blackbird', '109.American_Redstart',
            '152.Blue_headed_Vireo', '023.Brandt_Cormorant', '140.Summer_Tanager',
            '174.Palm_Warbler', '027.Shiny_Cowbird', '189.Red_bellied_Woodpecker',
            '036.Northern_Flicker', '058.Pigeon_Guillemot', '063.Ivory_Gull',
            '071.Long_tailed_Jaeger', '190.Red_cockaded_Woodpecker', '007.Parakeet_Auklet',
            '029.American_Crow', '164.Cerulean_Warbler'
        ]
        
        self.selected_classes = []
        self.support_path = os.path.join(args.destination_root, "support")
        self.query_path = os.path.join(args.destination_root, "query")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(args.destination_root, 'dataset_creation.log'))
            ]
        )
        self.logger = logging.getLogger('DatasetCreator')

    def validate_paths(self):
        """Check if source dataset exists"""
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Source path not found: {self.source_path}")
        
        # Create destination directories
        os.makedirs(self.support_path, exist_ok=True)
        os.makedirs(self.query_path, exist_ok=True)
        self.logger.info(f"Created directories: {self.support_path}, {self.query_path}")

    def clear_destination(self):
        """Remove existing support and query directories if they exist"""
        for path in [self.support_path, self.query_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
                self.logger.warning(f"Deleted existing directory: {path}")
    
    def select_classes(self):
        """Randomly select classes for the new dataset"""
        if self.n_classes > len(self.bird_list):
            raise ValueError(
                f"Requested {self.n_classes} classes, "
                f"but only {len(self.bird_list)} available in bird list"
            )
            
        self.selected_classes = random.sample(self.bird_list, self.n_classes)
        self.logger.info(f"Selected {self.n_classes} classes")
        for i, bird_class in enumerate(self.selected_classes):
            self.logger.info(f"  Class {i+1}: {bird_class}")

    def copy_images(self):
        """Copy images to support and query directories"""
        total_support = 0
        total_query = 0
        
        for class_idx, bird_class in enumerate(self.selected_classes):
            class_path = os.path.join(self.source_path, bird_class)
            
            if not os.path.exists(class_path):
                self.logger.warning(f"Skipping missing class: {bird_class}")
                continue
                
            images = sorted([
                f for f in os.listdir(class_path) 
                if os.path.isfile(os.path.join(class_path, f)) and not f.startswith('.')
            ])
            
            # Check if enough images available
            required = self.n_support + self.n_query
            if len(images) < required:
                self.logger.warning(
                    f"Skipping {bird_class}: Only {len(images)} images "
                    f"(needs {required})"
                )
                continue
                
            # Create support class directory
            support_class_dir = os.path.join(self.support_path, f"bird{class_idx+1}")
            os.makedirs(support_class_dir, exist_ok=True)
            
            # Copy support images
            for img in images[:self.n_support]:
                src = os.path.join(class_path, img)
                dst = os.path.join(support_class_dir, img)
                shutil.copy(src, dst)
                total_support += 1
                
            # Copy query images
            for img in images[self.n_support:self.n_support+self.n_query]:
                src = os.path.join(class_path, img)
                dst = os.path.join(self.query_path, img)
                shutil.copy(src, dst)
                total_query += 1
                
            self.logger.info(
                f"Copied from {bird_class}: "
                f"{self.n_support} support → bird{class_idx+1}, "
                f"{self.n_query} query → mixed"
            )

        self.logger.info(
            f"\nDataset summary:\n"
            f"  Total classes: {len(self.selected_classes)}\n"
            f"  Support images: {total_support}\n"
            f"  Query images: {total_query}"
        )

    def create_dataset(self):
        """Main method to create the dataset"""
        try:
            self.logger.info("Starting dataset creation")
            self.logger.info(f"Source: {self.source_path}")
            self.logger.info(f"Destination: {self.destination_root}")
            self.logger.info(f"Classes: {self.n_classes}, Support: {self.n_support}, Query: {self.n_query}")
            
            self.clear_destination()
            self.validate_paths()
            self.select_classes()
            self.copy_images()
            
            self.logger.info("Dataset created successfully!")
            return True
        except Exception as e:
            self.logger.error(f"Dataset creation failed: {str(e)}", exc_info=True)
            return False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create few-shot learning dataset from CUB-200')
    
    parser.add_argument('--source_path', type=str, required=True,
                        help='Path to CUB-200 images directory')
    parser.add_argument('--destination_root', type=str, required=True,
                        help='Root directory for new dataset')
    parser.add_argument('--n_classes', type=int, default=15,
                        help='Number of classes to select (default: 15)')
    parser.add_argument('--n_support', type=int, default=5,
                        help='Number of support images per class (default: 5)')
    parser.add_argument('--n_query', type=int, default=35,
                        help='Number of query images per class (default: 35)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    return parser.parse_args()


def main():
    """Main function to execute dataset creation"""
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create destination root if not exists
    os.makedirs(args.destination_root, exist_ok=True)
    
    # Create and execute dataset creator
    creator = DatasetCreator(args)
    success = creator.create_dataset()
    
    # Exit with appropriate status
    exit(0 if success else 1)


if __name__ == "__main__":
    main()

