# AI Service Metadata Checklist Template


---

## 1. Basic Service Information

- **Service Name**:  Efficient Labelling
- **Service Version**:  1.0.0-rc.1
- **Short Description**:  Automatically labels unlabelled image datasets to ensure domain-faithful annotations. Ideal for scenarios with limited labelled data, this service delivers high-quality, task-relevant labelled datasets.
- **Keywords / Tags**: labelling, image data, 
- **License Type**: CC BY-NC-SA 4.0
- **Institution / Team Name**:  Hochschule Aalen - Technik, Wirtschaft und Gesundheit
- **Contact Person Name**:  Andreas Theissler, Dima Al-Obaidi, Felix Gerschner
- **Contact Email**:  {Andreas.Theissler, Dima.Al-Obaidi, Felix.Gerschner}@hs-aalen.de

---

## 2. API Access Information

- **API Endpoint URL**:  
- **Supported Methods**: (GET, POST, etc.)  
- **Requires Authentication?**: Yes / No  
- **Authentication Method (if any)**: (e.g., API key, OAuth2)  
- **Input Format**: (e.g., JSON, text, image)  
- **Output Format**: (e.g., JSON, XML, image)  
- **Content-Type Header**: (e.g., application/json)  

---

## 3. Input/Output Examples

- **Sample Input** (attach or paste JSON/text below):  
\`\`\`json
{
  "input_zip": "dataset.zip",
  "description": "The ZIP file contains unlabelled images and optionally a small set of labelled examples for few-shot adaptation."
}
\`\`\`

- **Sample Output** (attach or paste JSON/text below):  
\`\`\`json
{
  "output_zip": "labelled_dataset.zip",
  "description": "The ZIP file contains the same images and a metadata file with predicted labels for each image."
}
\`\`\`

---

## 4. Documentation & Links

- **OpenAPI/Swagger Spec File** (if available):  N/A (not available)
- **Demo UI or Frontend**:  Streamlit app, URL not available yet 
- **Git Repository URL** (if available): URL not available yet
- **Documentation Link** (if available): N/A (not available)

---

## 5. Model / System Info 

- **Model Type**: CNN-based feature extractors
- **Framework Used**: PyTorch
- **Accuracy or F1-score**:  Varies by domain; few-shot accuracy between 70–75\% on benchmark dataset (CUB)
- **Training Dataset / Source**:  Pretrained on diverse meta-learning datasets (CUB) with custom domain adaptation
- **Inference Hardware Requirements**: GPU recommended for optimal performance; CPU fallback supported

---
 
