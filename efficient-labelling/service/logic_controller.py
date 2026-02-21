import tomllib
import logging
import traceback
import zipfile
import io
import os

from PIL import Image, UnidentifiedImageError
from service.similarity_controller import evaluate_dataset_similarity
from service.labelling_controller import label_with_model
from service.labelling_controller import label_with_finetuned_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LogicController:
    
    def __init__(self, config_path="service/config.toml"):

        #### BEGIN: These are currently not used, cause streamlit does not support stateful controllers
        self.submitted_dataset = None
        self.submitted_dataset_file_name = None

        self.submitted_labels_for_dataset = None 
        self.submitted_labels_file_name = None
        #### END: These are currently not used, cause streamlit does not support stateful controllers
        
        self.config_path = config_path
        self.config = self._load_config()
    

    def _load_config(self):
        logger.info("Loading configuration for LogicController.")
        logger.info(f"Config path: {self.config_path}")
        try:
            with open(self.config_path, "rb") as f:
                config = tomllib.load(f)
            logger.info("Configuration loaded successfully.")
            return config
        except FileNotFoundError:
            logger.error(f"Config file not found at {self.config_path}")
            raise
        except tomllib.TOMLDecodeError as e:
            logger.error(f"Failed to parse config file: {e}")
            raise
    
    
    def _preprocess_zip(self, zip_path):
        """ Preprocess the ZIP file to ensure it contains valid image files."""
        logger.info(f"Preprocessing zip file: {zip_path}")
        temp_zip_path = zip_path + ".tmp"
        valid_image_files = []

        try:
            with zipfile.ZipFile(zip_path, 'r') as original_zip:
                with zipfile.ZipFile(temp_zip_path, 'w') as cleaned_zip:
                    for item in original_zip.infolist():
                        with original_zip.open(item) as file:
                            try:
                                # Try to open the file as an image
                                img_data = file.read()
                                Image.open(io.BytesIO(img_data)).verify()
                                # Write the valid image back into the new archive
                                cleaned_zip.writestr(item.filename, img_data)
                                valid_image_files.append(item.filename)
                            except (UnidentifiedImageError, OSError):
                                logger.warning(f"Removed non-image file: {item.filename}")

            # Replace original zip with cleaned version
            os.replace(temp_zip_path, zip_path)

            if not valid_image_files:
                logger.error("No valid images found in zip file.")
                raise ValueError("The dataset is not suitable for evaluation.")
            
            logger.info(f"Zip file {zip_path} preprocessed successfully with {len(valid_image_files)} image(s).")
            return valid_image_files

        except zipfile.BadZipFile:
            logger.error(f"Invalid zip file: {zip_path}")
            raise ValueError("The uploaded file is not a valid ZIP archive.")
    
    def post_submitted_dataset(self, dataset, file_name):
        """
        dataset: The dataset to be submitted.
        file_name: The name of the file containing the dataset.
        
        This method is used to submit a dataset for processing. It stores the dataset
        and its file name in the controller.
        """
        logger.info("Received request to submit dataset with file name: %s", file_name)
        
        if dataset is None or file_name is None:
            logger.error("Dataset or file name is None.")
            raise ValueError("Dataset and file name must not be None.")
        
        # preprocess the dataset if it is a zip file
        if file_name.endswith('.zip'):
            logger.info("Preprocessing ZIP file for dataset submission.")
            try:
                valid_files = self._preprocess_zip(dataset)
                logger.info(f"Valid files after preprocessing: {len(valid_files)}")
            except ValueError as e:
                logger.error(f"Error during preprocessing: {e}")
                raise
        
        self.submitted_dataset = dataset # currently it does not store the dataset on the server, which might be changed in the future
        self.submitted_dataset_file_name = file_name
        print(f"Dataset {file_name} submitted successfully."
            f"\n Dataset size: {len(dataset)} samples.")
        logger.info("Dataset submitted and stored successfully.")
        
        return 200
    
    def post_submitted_labels_for_dataset(self, class_mappings, label_dataset, label_file_name):
        """
        labels: The labels for the submitted dataset.
        
        This method is used to submit labels for the dataset that has been previously
        submitted. It stores the labels in the controller.
        """
        logger.info("Received request to submit labels for dataset.")
        
        # if self.submitted_dataset is None:
        #     logger.error("No dataset has been submitted yet.")
        #     return {"status_code": 400, "message": "No dataset has been submitted yet."}
        
        # if label_file_name is None:
        #     logger.error("Labels cannot be None.")
        #     return {"status_code": 400, "message": "Labels cannot be None."}
        
        
        # preprocess the dataset if it is a zip file
        if label_file_name.endswith('.zip'):
            logger.info("Preprocessing ZIP file for dataset submission.")
            try:
                valid_files = self._preprocess_zip(label_dataset)
                logger.info(f"Valid files after preprocessing: {len(valid_files)}")
            except ValueError as e:
                logger.error(f"Error during preprocessing: {e}")
                raise

        self.class_mappings = class_mappings
        self.submitted_labels_for_dataset = label_dataset
        self.submitted_labels_file_name = label_file_name
        logger.info("Labels received for dataset.")
        
        return {"status_code": 200, "message": "Labels submitted successfully."}


    def get_processed_dataset_similarity(self, dataset_name):
        """
        dataset: The dataset to be evaluated for similarity.
        
        Returns the evaluation of the similarity of the handed over dataset. Assessing
        whether the dataset is similar enough one of the pre-trained models.
        
        returns: JSON object with the evaluation of the similarity of the dataset.
        The evaluation is in the form of a dictionary with the following keys:
        - "sim": boolean indicating whether the dataset is similar enough to a pre-trained model.
        - "message": string containing a message about the evaluation.
        - "model": string indicating the model used for the evaluation (e.g., "imagenet", "cub").
        Example:
        {
            "sim": True,
            "message": "The dataset is similar enough to the pre-trained model.",
            "model": "imagenet"
        }
        If the dataset is not similar enough, the "sim" key will be False and the
        "message" key will contain a message indicating that the dataset is not similar enough.
        Example:
        {
            "sim": False,
            "message": "The dataset is not similar enough to the pre-trained model.",
            "model": "cub"
        }
        If there is an error during the evaluation, the "sim" key will be False and
        the "message" key will contain an error message.
        Example:
        {            
            "sim": False,
            "message": "An error occurred during the evaluation: <error_message>",
            "model": "unknown"
        }
        If the dataset is not suitable for evaluation, the function will raise a ValueError.
        Example:
        {   
            "sim": False,
            "message": "The dataset is not suitable for evaluation.",
            "model": "unknown"
        }
        """
        if self.submitted_dataset is None:
            logger.error("Received None dataset for similarity evaluation.")
            raise ValueError("The dataset is not suitable for evaluation.")
        
        
        result = None
        logger.info("Processing dataset similarity evaluation.")
        
        try:
            if self.config["mode"] == "development":
                # assemble the result
                result = { # test success result
                    "sim": True,
                    "message": "The dataset is similar enough to the pre-trained model.",
                    "model": "imagenet"  # Example model name, can be "imagenet", "cub", etc.
                }
                
            else:
                # evaluate the similarity of the dataset with the pre-trained models
                result = evaluate_dataset_similarity(self.config, dataset_name)
                result["status_code"] = 200  # Success
            
            logger.info("Similarity evaluation completed: %s", result["message"])
            
        except Exception as e:
            logger.error("Error during dataset similarity evaluation: %s", str(e))
            logger.error("Traceback:\n%s", traceback.format_exc())
            
            # default error result
            result = {
                "sim": None,
                "message": f"An error occurred during the evaluation: {str(e)}",
                "model": "unknown",
                "status_code": 500,
            }
            
        return result


    def get_processed_dataset_labelling(self, dataset_name, model_name):
        """
        Returns the labels of the unlabelled dataset.
        
        returns: JSON object with the labels of the dataset.
        The labels are in the form of a dictionary with the image file names as keys
        and the labels as values.
        Example:
        {            
            "image1.jpg": "label1",
            "image2.jpg": "label2",
            "image3.jpg": "label3",
            ...
        }    
        """        
        logger.info("Processing dataset labelling.")
        logger.info(f"Dataset name: {dataset_name}, Model name: {model_name}")
        
        result = None

        try:
            # label the dataset using a pre-trained model or a fine-tuned model
            if self.config["mode"] == "development":
                # Simulate processing time
                result = {
                    "image1.jpg": "label1",
                    "image2.jpg": "label2",
                    "image3.jpg": "label3"
                }
                logger.info("Labelling processing completed successfully.")
                
                
            else:
                result = label_with_model(self.config, model_name, dataset_name)
                result["status_code"] = 200  # Success

            logger.info("Dataset labelling completed successfully.")
        
        
        except Exception as e:
            
            logger.error("Error during dataset labelling processing: %s", str(e))
            
            result = {
                "message": f"An error occurred during the evaluation: {str(e)}",
                "error": f"An error occurred during the labelling processing: {str(e)}",
                "status_code": 500,
            }
            

        return result


    def get_processed_dataset_labelling_with_labels(self, dataset_name, label_dataset_name, class_mappings):
        """
        Returns the labels of the unlabelled dataset with provided class mappings and labels

        returns: JSON object with the labels of the dataset.
        The labels are in the form of a dictionary with the image file names as keys
        and the labels as values.
        Example:
        {            
            "image1.jpg": "label1",
            "image2.jpg": "label2",
            "image3.jpg": "label3",
            ...
        }    
        """        
        logger.info("Processing dataset labelling.")
        logger.info(f"Dataset name: {dataset_name}, Label dataset name: {label_dataset_name}, Class mappings: {class_mappings}")
        
        result = None

        try:
            # label the dataset using a pre-trained model or a fine-tuned model
            if self.config["mode"] == "development":
                # Simulate processing time
                result = {
                    "image1.jpg": "label1",
                    "image2.jpg": "label2",
                    "image3.jpg": "label3"
                }
                logger.info("Labelling processing completed successfully.")
                
                
            else:
                result = label_with_finetuned_model(self.config, dataset_name, label_dataset_name, class_mappings)
                result["status_code"] = 200  # Success

            logger.info("Dataset labelling completed successfully.")
        
        
        except Exception as e:
            
            logger.error("Error during dataset labelling processing: %s", str(e))
            
            result = {
                "message": f"An error occurred during the evaluation: {str(e)}",
                "error": f"An error occurred during the labelling processing: {str(e)}",
                "status_code": 500,
            }
            

        return result


if __name__ == "__main__":
    # Example usage
    # logic_controller = LogicController()
    
    
    # dataset = "path/to/dataset.zip"  # Replace with actual dataset path
    # similarity_result = logic_controller.get_processed_dataset_similarity(dataset)
    # print(f"Dataset similarity result: {similarity_result}")
    
    # labelling_result = logic_controller.get_processed_dataset_labelling(dataset)
    # print(f"Dataset labelling result: {labelling_result}")
    
    pass
