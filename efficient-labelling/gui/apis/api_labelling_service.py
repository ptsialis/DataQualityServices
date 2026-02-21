import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LabellingServiceAPI:
    
    def __init__(self, api_client):
        self.api_client = api_client

    def submit_dataset(self, dataset, file_name):
        """
        Submits a dataset for processing.
        """
        logger.info(f"Submitting dataset with file name: dataset {dataset}, file_name {file_name}")
        
        if dataset is None or file_name is None:
            logger.error("Failed to submit dataset: dataset or file_name is None.")
            raise ValueError("Dataset and file name must not be None.")
        
        response = self.api_client.post_submitted_dataset(dataset, file_name)
        logger.info("Dataset submitted successfully.")
        return response
    
    
    def submit_labels(self, class_mappings, label_dataset, file_name):
        """
        Submits labels for a dataset.
        """
        logger.info("Submitting labels.")
        
        if not label_dataset:
            logger.warning("No labels provided.")
            return False
        
        response = self.api_client.post_submitted_labels_for_dataset(class_mappings, label_dataset, file_name)
        logger.info("Labels submitted successfully.")
        return response
    

    def process_dataset_similarity(self, dataset_name):
        """
        Retrieves the evaluation of the similarity of two datasets.
        """
        logger.info("Processing dataset similarity.")
        response = self.api_client.get_processed_dataset_similarity(dataset_name)
        logger.info("Similarity processing completed.")
        return response


    def process_dataset_labelling(self, dataset, model_name):
        """
        Retrieves the labels for an unlabelled dataset.
        """
        logger.info(f"Processing dataset labelling for dataset: {dataset}, model: {model_name}")
        response = self.api_client.get_processed_dataset_labelling(dataset, model_name)
        logger.info("Labelling processing completed.")
        return response

    def process_dataset_labelling_with_labels(self, dataset, label_dataset, class_mappings):
        """
        Retrieves the labels for an unlabelled dataset.
        """
        logger.info(f"Processing dataset labelling for dataset: {dataset}, label dataset: {label_dataset}, class mappings: {class_mappings}")
        response = self.api_client.get_processed_dataset_labelling_with_labels(dataset, label_dataset, class_mappings)
        logger.info("Labelling processing completed.")
        return response
