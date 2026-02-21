import pandas as pd
import json
import os
from datetime import datetime

def generate_metadata_json_imputation(df, dataset_name, save_dir="data/meta_data_piveau/imputation"):
    """
    Build DCAT-like metadata for a pandas DataFrame, including per-column stats
    (total values, missing counts/percent, unique values), write it to disk,
    and RETURN the metadata dict instead of the save path.
    """
    variables = []
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    for column in df.columns:
        total_values = df[column].count()
        missing_values_total = df[column].isnull().sum()
        missing_values_percent = (missing_values_total / len(df)) * 100 if len(df) else 0.0
        unique_values = df[column].nunique()
        variables.append({
            "column_name": column,
            "totalValues": int(total_values),
            "missingValues_total": int(missing_values_total),
            "missingValues_percent": float(missing_values_percent),
            "uniqueValues": int(unique_values)
        })
    
    metadata = {
        "@context": "https://www.w3.org/ns/dcat",
        "@type": "dcat:DataService",
        "@id": "https://example.com/imputation-service",
        "name": dataset_name,
        "description": f"Metadata for the dataset {dataset_name}, including missing values analysis.",
        "keywords": ["data imputation", "machine learning", "statistics"],
        "creator": {"@type": "foaf:Person","name": "Petros Tsialis","email": "petros.tsialis@hs-aalen.de"},
        "publisher": {"@type": "foaf:Hochschule Aalen","name": "Data Science Team","homepage": "https://www.hs-aalen.de/"},
        "contactPoint": {"@type": "vcard:Contact","fn": "Support Team","email": "petros.tsialis@hs-aalen.de"},
        "dateCreated": current_datetime.split('_')[0],
        "dateModified": current_datetime.split('_')[0],
        "conformsTo": "https://www.w3.org/TR/dcat/",
        "hasVersion": "https://example.com/imputation-service/v1.1",
        "serviceType": "Imputation Service",
        "endpointDescription": "https://example.com/api-docs",
        "endpointURL": "https://api.example.com/imputation",
        "distribution": [{"@type": "dcat:Distribution","format": "JSON","accessURL": "https://api.example.com/imputation"}],
        "variables": variables
    }
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"metadata_imputation_{current_datetime}_{dataset_name}.json")
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    
    return metadata  # <-- return JSON dict


def generate_metadata_json_anomalies(df, dataset_name, save_dir="data/meta_data_piveau/anomalies"):
    """
    Build DCAT-like metadata for anomaly detection results, write to disk,
    and RETURN the metadata dict instead of the save path.
    """
    variables = []
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if "Anomaly" not in df.columns:
        raise ValueError("The DataFrame must contain an 'Anomaly' column indicating anomalies.")

    total_rows = len(df)
    total_anomalies = int(df["Anomaly"].sum())
    anomaly_percent = (total_anomalies / total_rows) * 100 if total_rows else 0.0
    anomaly_indices = df.index[df["Anomaly"] == 1].tolist()

    variables.append({
        "column_name": "Anomaly",
        "totalRows": int(total_rows),
        "anomalies_total": total_anomalies,
        "anomalies_percent": float(anomaly_percent),
        "anomaly_indices": anomaly_indices
    })

    metadata = {
        "@context": "https://www.w3.org/ns/dcat",
        "@type": "dcat:DataService",
        "@id": "https://example.com/anomaly-detection-service",
        "name": dataset_name,
        "description": f"Metadata for the dataset {dataset_name}, including anomaly detection results.",
        "keywords": ["anomaly detection", "data quality", "machine learning"],
        "creator": {"@type": "foaf:Person","name": "Petros Tsialis","email": "petros.tsialis@hs-aalen.de"},
        "publisher": {"@type": "foaf:Hochschule Aalen","name": "Data Science Team","homepage": "https://www.hs-aalen.de/"},
        "contactPoint": {"@type": "vcard:Contact","fn": "Support Team","email": "petros.tsialis@hs-aalen.de"},
        "dateCreated": current_datetime.split('_')[0],
        "dateModified": current_datetime.split('_')[0],
        "conformsTo": "https://www.w3.org/TR/dcat/",
        "hasVersion": "https://example.com/anomaly-detection-service/v1.0",
        "serviceType": "Anomaly Detection Service",
        "endpointDescription": "https://example.com/api-docs",
        "endpointURL": "https://api.example.com/anomaly-detection",
        "distribution": [{"@type": "dcat:Distribution","format": "JSON","accessURL": "https://api.example.com/anomaly-detection"}],
        "variables": variables
    }

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"metadata_anomalies_{current_datetime}_{dataset_name}.json")
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return metadata  # <-- return JSON dict


def generate_metadata_json_feature_types(df, dataset_name, save_dir="data/meta_data_piveau/inference"):
    """
    Build DCAT-like metadata capturing per-feature type predictions, write to disk,
    and RETURN the metadata dict instead of the save path.
    """
    variables = []
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if not {"Attribute_name", "prediction"}.issubset(df.columns):
        raise ValueError("The DataFrame must contain both 'Attribute_name' and 'prediction' columns.")

    attribute_predictions = {row["Attribute_name"]: row["prediction"] for _, row in df.iterrows()}
    variables.append(attribute_predictions)

    metadata = {
        "@context": "https://www.w3.org/ns/dcat",
        "@type": "dcat:DataService",
        "@id": "https://example.com/feature-type-inference",
        "name": dataset_name,
        "description": f"Metadata for the dataset {dataset_name}, including inferred feature types.",
        "keywords": ["feature type", "schema inference", "data profiling"],
        "creator": {"@type": "foaf:Person","name": "Petros Tsialis","email": "petros.tsialis@hs-aalen.de"},
        "publisher": {"@type": "foaf:Hochschule Aalen","name": "Data Science Team","homepage": "https://www.hs-aalen.de/"},
        "contactPoint": {"@type": "vcard:Contact","fn": "Support Team","email": "petros.tsialis@hs-aalen.de"},
        "dateCreated": current_datetime.split('_')[0],
        "dateModified": current_datetime.split('_')[0],
        "conformsTo": "https://www.w3.org/TR/dcat/",
        "hasVersion": "https://example.com/feature-type-inference/v1.0",
        "serviceType": "Feature Type Inference",
        "endpointDescription": "https://example.com/api-docs",
        "endpointURL": "https://api.example.com/feature-type-inference",
        "distribution": [{"@type": "dcat:Distribution","format": "JSON","accessURL": "https://api.example.com/feature-type-inference"}],
        "variables": variables
    }

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"metadata_feature_type_{current_datetime}_{dataset_name}.json")
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return metadata  # <-- return JSON dict


def generate_metadata_json_personal_data(df, dataset_name, save_dir="data/meta_data_piveau/personal"):
    """
    Build DCAT-like metadata for personal data classification results, write to disk,
    and RETURN the metadata dict instead of the save path.
    """
    variables = []
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if not {"Column", "Prediction"}.issubset(df.columns):
        raise ValueError("The DataFrame must contain both 'Column' and 'Prediction' columns.")

    personal_data_predictions = {row["Column"]: row["Prediction"] for _, row in df.iterrows()}
    variables.append(personal_data_predictions)

    metadata = {
        "@context": "https://www.w3.org/ns/dcat",
        "@type": "dcat:DataService",
        "@id": "https://example.com/personal-data-detection",
        "name": dataset_name,
        "description": f"Metadata for the dataset {dataset_name}, including personal data detection results.",
        "keywords": ["personal data", "privacy", "GDPR", "data classification"],
        "creator": {"@type": "foaf:Person","name": "Petros Tsialis","email": "petros.tsialis@hs-aalen.de"},
        "publisher": {"@type": "foaf:Hochschule Aalen","name": "Data Science Team","homepage": "https://www.hs-aalen.de/"},
        "contactPoint": {"@type": "vcard:Contact","fn": "Support Team","email": "petros.tsialis@hs-aalen.de"},
        "dateCreated": current_datetime.split('_')[0],
        "dateModified": current_datetime.split('_')[0],
        "conformsTo": "https://www.w3.org/TR/dcat/",
        "hasVersion": "https://example.com/personal-data-detection/v1.0",
        "serviceType": "Personal Data Detection",
        "endpointDescription": "https://example.com/api-docs",
        "endpointURL": "https://api.example.com/personal-data-detection",
        "distribution": [{"@type": "dcat:Distribution","format": "JSON","accessURL": "https://api.example.com/personal-data-detection"}],
        "variables": variables
    }

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"metadata_personal_data_{current_datetime}_{dataset_name}.json")
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return metadata  # <-- return JSON dict
