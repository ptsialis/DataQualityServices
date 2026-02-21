import logging

import streamlit as st
import random
import time
import zipfile
from io import BytesIO
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AppController:

    def __init__(self, api=None):
        self.uploaded_file_name = None
        self.model_for_dataset = None
        self.mode = None
        
        self.api = api

    def submit_dataset(self, uploaded_file, dataset_info):
        if uploaded_file is not None:
            self.api.submit_dataset(uploaded_file, dataset_info)
        else:
            st.error("No dataset uploaded. Please upload a valid dataset.")

    def submit_labels(self, class_mapping, uploaded_file, label_dataset_info):
        if not class_mapping:
            st.warning("No class mapping provided.")
            return False

        if uploaded_file is None:
            st.warning("No uploaded file provided.")
            return False

        if label_dataset_info is None:
            st.warning("No label dataset provided.")
            return False
        
        logger.info("Submitting labels for dataset.")
        try:
            response = self.api.submit_labels(class_mapping, uploaded_file, label_dataset_info)
            logger.info(f"Labels submitted successfully: {response}")
        except Exception as e:
            logger.error(f"Error submitting labels: {e}")
            st.error(f"Error submitting labels: {e}")
            return False

        return True
    
    def process_dataset_similarity(self, dataset_name=None):
        evaluated_state = False
        
        # evaluation
        if self.mode == "development":
            logger.info("Simulating dataset evaluation...")
            time.sleep(5)
            if "imagenet" in self.uploaded_file_name:
                evaluated_state = True    
            elif "cub" in self.uploaded_file_name:
                evaluated_state = False
            else:
                logger.error("Unknown dataset for simulation!")
                raise ValueError("Unknown dataset for simulation!")
        elif self.mode == "production":
            logger.info("Processing dataset similarity evaluation...")
            try: 
                res = self.api.process_dataset_similarity(dataset_name)
                logger.info(f"Result of dataset similarity evaluation: {res}")
            except Exception as e:
                logger.error(f"Error during dataset similarity evaluation: {e}")
                st.error(f"Error processing dataset similarity: {e}")
                return False
            
            self.model_for_dataset = res["model"]  # Store the model used for evaluation
            logger.info(f"Model for dataset: {self.model_for_dataset} saved.")
            
            if res["sim"] is not None:
                evaluated_state = res["sim"] # True if similar enough, False otherwise
            else: 
                evaluated_state = None
                raise ValueError("Unknown error!")

        logger.info(f"Dataset evaluation completed with state: {evaluated_state}")

        return evaluated_state, self.model_for_dataset
    
    
    def process_dataset_labelling(self, model_name, dataset_name):
        
        if self.mode == "development":
            logger.info("Simulating dataset labelling...")
            time.sleep(5)
            return True
        
        elif self.mode == "production":
            logger.info("Processing dataset labelling...")
            try:
                logger.info(f"Dataset name: {dataset_name}, Model name: {model_name}")
                logger.info(self)
                
                res = self.api.process_dataset_labelling(dataset_name, model_name)
                
                logger.info(f"Result of dataset labelling: status_code {res["status_code"]}, zip_path {res["zip_path"]}, stats {res["stats"]}")
                return res
            except Exception as e:
                logger.error(f"Error during dataset labelling: {e}")
                st.error(f"Error processing dataset labelling: {e}")
                return False


    def process_dataset_labelling_with_labels(self, class_mappings, label_dataset_name, dataset_name):

        if self.mode == "development":
            logger.info("Simulating dataset labelling...")
            time.sleep(5)
            return True
        
        elif self.mode == "production":
            logger.info("Processing dataset labelling...")
            try:
                logger.info(f"Dataset name: {dataset_name}, Label dataset name: {label_dataset_name}, Class mappings: {class_mappings}")
                logger.info(self)

                res = self.api.process_dataset_labelling_with_labels(dataset_name, label_dataset_name, class_mappings)

                logger.info(f"Result of dataset labelling: status_code {res["status_code"]}, zip_path {res["zip_path"]}, stats {res["stats"]}")
                return res
            except Exception as e:
                logger.error(f"Error during dataset labelling: {e}")
                # st.error(f"Error processing dataset labelling: {e}")
                return False    



    def get_labelling_subset(self, num_subset=5):
        if self.uploaded_dataset is None:
            st.warning("No dataset uploaded yet.")
            return []

        subset = []
        try:
            with zipfile.ZipFile(self.uploaded_dataset, "r") as unzipped_dataset:
                image_files = [
                    f for f in unzipped_dataset.namelist()
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

                unzipped_image_files = []
                for image_file in image_files:
                    with unzipped_dataset.open(image_file) as f:
                        image_bytes = f.read()  # Read image bytes before closing
                        img = Image.open(BytesIO(image_bytes)).convert("RGB")  # Keep image in memory
                        unzipped_image_files.append(img)

                # Select a random subset of images
                if len(unzipped_image_files) > num_subset:
                    subset = random.sample(unzipped_image_files, num_subset)
                else:
                    subset = unzipped_image_files  # Use all available if less than requested

                return [{"id": idx, "file": file} for idx, file in enumerate(subset, start=1)]
        except Exception as e:
            st.error(f"Error extracting images: {e}")
            return []
        
