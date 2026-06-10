import pandas as pd
import json
import os
from datetime import datetime


def _base_metadata_template(
    dataset_name,
    service_id,
    description,
    keywords,
    version,
    service_type,
    endpoint_url
):
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    current_date = current_datetime.split('_')[0]

    return {
        "@context": "https://www.w3.org/ns/dcat",
        "@type": "dcat:DataService",
        "@id": service_id,
        "name": dataset_name,
        "description": description,
        "keywords": keywords,
        "creator": {
            "@type": "foaf:Person",
            "name": "Petros Tsialis",
            "email": "petros.tsialis@hs-aalen.de"
        },
        "publisher": {
            "@type": "foaf:Hochschule Aalen",
            "name": "Data Science Team",
            "homepage": "https://www.hs-aalen.de/"
        },
        "contactPoint": {
            "@type": "vcard:Contact",
            "fn": "Support Team",
            "email": "petros.tsialis@hs-aalen.de"
        },
        "dateCreated": current_date,
        "dateModified": current_date,
        "conformsTo": "https://www.w3.org/TR/dcat/",
        "hasVersion": version,
        "serviceType": service_type,
        "endpointDescription": "https://example.com/api-docs",
        "endpointURL": endpoint_url,
        "distribution": [
            {
                "@type": "dcat:Distribution",
                "format": "JSON",
                "accessURL": endpoint_url
            }
        ],
        "variables": []
    }


def _make_imputation_metadata_dict(df, dataset_name):
    variables = []

    if df is None or getattr(df, "empty", True):
        cols = []
        df_len = 0
    else:
        cols = df.columns
        df_len = len(df)

    for column in cols:
        total_values = df[column].count()
        missing_values_total = df[column].isnull().sum()
        missing_values_percent = (missing_values_total / df_len) * 100 if df_len else 0
        unique_values = df[column].nunique()

        variables.append({
            "column_name": column,
            "totalValues": int(total_values),
            "missingValues_total": int(missing_values_total),
            "missingValues_percent": float(missing_values_percent),
            "uniqueValues": int(unique_values)
        })

    metadata = _base_metadata_template(
        dataset_name=dataset_name,
        service_id="https://example.com/imputation-service",
        description=f"Metadata for the dataset {dataset_name}, including missing values analysis.",
        keywords=["data imputation", "machine learning", "statistics"],
        version="https://example.com/imputation-service/v1.1",
        service_type="Imputation Service",
        endpoint_url="https://api.example.com/imputation"
    )

    metadata["variables"] = variables
    return metadata


def _make_feature_types_metadata_dict(data, dataset_name):
    """
    Erwartet ein Dictionary wie:
    {
        "Pregnancies": "numeric",
        "Outcome": "categorical"
    }
    """
    metadata = _base_metadata_template(
        dataset_name=dataset_name,
        service_id="https://example.com/feature-type-inference",
        description=f"Metadata for the dataset {dataset_name}, including inferred feature types.",
        keywords=["feature type", "schema inference", "data profiling"],
        version="https://example.com/feature-type-inference/v1.0",
        service_type="Feature Type Inference",
        endpoint_url="https://api.example.com/feature-type-inference"
    )

    if isinstance(data, dict):
        metadata["variables"] = [data]
    else:
        metadata["variables"] = []

    return metadata


def _make_personal_data_metadata_dict(data, dataset_name):
    """
    Erwartet ein Dictionary wie:
    {
        "Glucose": "personal",
        "BMI": "non-personal"
    }
    """
    metadata = _base_metadata_template(
        dataset_name=dataset_name,
        service_id="https://example.com/personal-data-detection",
        description=f"Metadata for the dataset {dataset_name}, including personal data detection results.",
        keywords=["personal data", "privacy", "GDPR", "data classification"],
        version="https://example.com/personal-data-detection/v1.0",
        service_type="Personal Data Detection",
        endpoint_url="https://api.example.com/personal-data-detection"
    )

    if isinstance(data, dict):
        metadata["variables"] = [data]
    else:
        metadata["variables"] = []

    return metadata


def _make_anomalies_metadata_dict(data, dataset_name):
    """
    Erwartet ein Dictionary wie:
    {
        "column_name": "Anomaly",
        "totalRows": 768,
        "anomalies_total": 39,
        "anomalies_percent": 5.08,
        "anomaly_indices": [...]
    }
    """
    metadata = _base_metadata_template(
        dataset_name=dataset_name,
        service_id="https://example.com/anomaly-detection-service",
        description=f"Metadata for the dataset {dataset_name}, including anomaly detection results.",
        keywords=["anomaly detection", "data quality", "machine learning"],
        version="https://example.com/anomaly-detection-service/v1.0",
        service_type="Anomaly Detection Service",
        endpoint_url="https://api.example.com/anomaly-detection"
    )

    if isinstance(data, dict):
        metadata["variables"] = [data]
    else:
        metadata["variables"] = []

    return metadata


def generate_metadata_json(df, dataset_name, save_dir="data/meta_data_piveau"):
    """Schreibt Imputation-Metadaten wie bisher auf Disk."""
    metadata = _make_imputation_metadata_dict(df, dataset_name)

    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"metadata_{current_datetime}_{dataset_name}.json")

    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    return save_path


def build_metadata_json(df, dataset_name):
    return _make_imputation_metadata_dict(df, dataset_name)


def generate_metadata_json_imputation(df, base_name):
    return _make_imputation_metadata_dict(df, f"{base_name}_imputation")


def generate_metadata_json_anomalies(data, base_name):
    return _make_anomalies_metadata_dict(data, f"{base_name}_anomalies")


def generate_metadata_json_feature_types(data, base_name):
    return _make_feature_types_metadata_dict(data, f"{base_name}_feature_types")


def generate_metadata_json_personal_data(data, base_name):
    return _make_personal_data_metadata_dict(data, f"{base_name}_personal_data")