# AI Alliance – Labelling Service GUI

This is the **Streamlit-based GUI** for the Efficient Labelling Service.  
It demonstrates the workflow developed in the project and was originally showcased at **Hannover Messe**.

![Workflow of the labelling service](../assets/labelling-service_workflow.png)

---

## Requirements
- Python 3.10+
- Streamlit
- Dependencies in `requirements.txt`

Install dependencies:
```bash
pip install -r ../requirements.txt
```

---

## Usage

Run the GUI from the `gui` folder:
```bash
streamlit run app.py
```

You will be guided through the workflow:

1. Start the service.  
2. Upload an **unlabelled dataset** (`*_unlabelled.zip` from `/gui/assets/hannover_messe_data/`).  
3. Progress through the workflow steps.  
4. When asked for a small labelled dataset, upload the corresponding `*_small_labelled.zip`.  

---

## Showcase Datasets
The folder `./assets/hannover_messe_data/` contains example datasets for demonstration.  
In real use cases, you can replace them with your own datasets.

---

## Notes
The GUI is optional – you can also use the [service pipelines directly](../service/README.md).
