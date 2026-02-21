# AI Alliance – Labelling Service (Core)

This folder contains the **core functionality** of the Efficient Labelling Service.  
It provides two modular pipelines:

- **Similarity Pipeline**: Calculates similarity between datasets and guides transfer decisions.
- **Labelling Pipeline**: Performs few-shot adaptation or training to label new datasets.

Pipelines can be:
- Used independently
- Combined into an end-to-end service
- Connected to the [Streamlit GUI](../gui/README.md)

---

## Structure

```
service/
├── similarity_component/   # Similarity pipeline
├── labelling_component/    # Labelling pipeline
└── main.py                 # Entry point for combined workflow
```

---

## Usage

### 1. Run the full service
```bash
python main.py
```

This runs the integrated workflow:  
1. Similarity analysis  
2. Labelling pipeline (adaptation / training / inference)  

### 2. Run individual pipelines
- [Similarity Pipeline](similarity_component/README.md)  
- [Labelling Pipeline](labelling_component/README.md)  

---

## Requirements
- Python 3.10+
- PyTorch
- Additional dependencies listed in `requirements.txt`

---

## Notes
- The **datasets** and **model_weights** folders are excluded from this repo. Request them from the AI Alliance team.
- For reproducibility, we recommend using a virtual environment.
