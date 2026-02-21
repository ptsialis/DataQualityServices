# AI Services Production Platform (AI_Services_Production_BaseVersion)

A unified AI platform offering **Data Quality Analysis**, **Outlier Detection**, **Efficient Lebelling** and **Image Deblurring** services and deployment been done in **HLRS infrastructure** environment, developed by Aalen University and funded by KI-Allianz.

![alt text](image-1.png)

---

## Features

- **Four AI services integrated**:
  - **Data Quality AI:** Feature type detection, imputation, anomaly detection, personalized summary
  - **Outlier Detection (XGBOD):** ML-based outlier detection for tabular data
  - **Efficient Lebelling:** Large image datasets are often unlabeled or partially labeled, making them unusable for supervised ML workflows.
  - **Image Deblurring:** Restores blurred images while preserving resolution, format, and EXIF metadata.
- **Wizard-based UI with "Back / Next / Reset" navigation**
- **REST API routes available** 
- Supports **CSV/XLSX uploads**, runs inference, and exports results
- Integration with **Piveau Hub** for dataset publishing.
- Fully containerized using **Docker and docker-compose**
- Easy to extend with more AI services

---

---

## Local Docker Development Setup(Powershell) for Window & Linux
# 1. Clone repo
git clone <repo-url>
cd AI_Services_Production_BaseVersion
# 2. Check Python version (should be 3.11+)
python --version      # For Window
python3 --version     # For Linux
# 3. Create & activate virtualenv
# For Window
py -3.11 -m venv venv   
.\venv\bin\activate     
# For Linux
python3 -m venv venv     
source venv/bin/activate 
# 4. Install dependencies (Window & Linux)
pip install --upgrade pip
pip install -r requirements.txt

### Build & run docker stack (Window & Linux)
docker compose down         # stop/remove old containers if any
docker compose build        # build ai-services + efficient-labelling
docker compose up -d        # start in background
docker compose logs -f      # follow logs from both services

View the Main UI at:

```
http://localhost:8000
```

---

## AI Services Overview

### Data Quality AI Service
Descriptions:
- Automated Feature Type Inference
The automated feature type inference service analyzes each column in a dataset and assigns it a semantic type (such as numeric, categorical, sentence, URL, list, embedded-number, context-specific, not-generalizable or datetime). This enables downstream preprocessing and machine learning components to handle each feature appropriately. It currently distinguishes between nine different feature classes and allows users to review and manually adjust the inferred types where necessary. Link: …

- Detection of Personal Data
The personal data detection service scans structured datasets to identify columns and fields that likely contain personal or sensitive information (such as names, contact details, or identifiers). It analyses column names, descriptions, and values to flag potentially privacy-relevant attributes so they can be handled appropriately in privacy-preserving and compliant data preparation workflows. Link: https://arxiv.org/abs/2506.22305 

-Automated Imputation of Tabular Data
The imputation service automatically handles missing values in tabular datasets, for both single- and multi-column missing data, using mean and mode imputation. Extensive evaluation showed that simple mean/mode imputation offers competitive accuracy (within roughly ±3% of advanced methods such as autoencoders or random forests) at a fraction of the computational cost. Consequently, mean/mode imputation is used as a robust default to provide complete, consistently imputed data for downstream analyses and machine learning models. Link: https://dl.acm.org/doi/full/10.1145/3643643 

- Anomaly Detection
The anomaly detection service automatically finds unusual or inconsistent data points in tabular and time-series datasets. It highlights values and patterns that deviate from expected behaviour, helping to reveal potential errors, sensor faults, or rare events as part of the overall data-quality process.  Link: https://github.com/yzhao062/pyod  / https://arxiv.org/abs/2201.07284 

Steps:
1. Upload CSV/XLSX
2. Select target column (optional)
3. Run pipeline (Feature Type → Imputation → Anomaly Detection →Personalized_detection→ Summary)
4. Download outputs or publish to Piveau

Key files:
- `feature_type_inference.py`
- `data_imputation.py`
- `anomaly_detection.py`
- `personalized_detection.py`

---

### Outlier Detection (XGBOD)
- Upload a CSV/XLSX
- Uses pretrained XGBOD model (`artifacts/`)
- Outputs:
  - `results.csv`
  - `inliers_no_outliers.csv`
  - `only_outliers.csv`

Key module: `xgbod_runtime.py`

---

## Environment Variables

Create `.env` file:
```bash
OUTPUT_DIR=./output
XGBOD_ARTIFACTS_DIR=./artifacts
PIVEAU_TOKEN=<optional>
HUB_STORE_URL=<optional>
HUB_STORE_BUCKET=ai-results
```

---

## API
```
For DataQuality API:http://localhost:8503
```
```
For Outlier Detection API:http://localhost:8000/services/outlier
```
```
For Image Deblurring API:http://localhost:8502
```
```
For Efficient Labelling API:http://localhost:8501
```

---

## Publishing to Piveau Hub 

Handled by:
```
src/services/piveau_publish.py
```

Only enabled if:
- Token provided in `.env`
- Proper MinIO / Piveau variables configured

---


## License

```
Apache License 2.0
```

---

## Maintainers

| Name | Affiliation |
|------|-------------|
| **Bhuvneshwar Bajpeyee** | AI Services Development & Integration Ownership | Aalen University |
| **Miroslav** | Hosting & infrastructure support |
| **Petros Tsialis & Albert Agisha** | DataQuality Services Owner | Aalen University |
| **Dima Al-Obaidi & Felix Gerschner** | Efficient Labelling Service Owner | Aalen University |
| **Niloofar Kalashtari** | Outlier Detection Service Owner | Aalen University |
| **Patrick Krawczyk** | Image deblurringg Service Owner | Aalen University |

---

## How to Contribute

1. Fork this repo
2. Create your branch: `git checkout -b feature/new-service`
3. Commit: `git commit -m 'Add new service'`
4. Push: `git push origin feature/new-service`
5. Create a Pull Request

---
