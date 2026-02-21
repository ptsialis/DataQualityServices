# AI Alliance – Similarity Pipeline

The **Similarity Pipeline** estimates how similar an incoming dataset is to existing reference datasets.  
This is used to decide whether to:
- Reuse an existing model directly
- Perform few-shot adaptation
- Retrain from scratch

---

## Features
- Multiple similarity metrics (e.g., KL divergence, EMD, cosine similarity, …)
- Flexible integration into the service or standalone usage
- Outputs interpretable similarity scores

---

## Usage

### Run as a module
```bash
python similarity_pipeline.py --dataset path/to/unlabelled.zip --reference path/to/reference/
```

### Example output
```
Dataset similarity scores:
- Reference A: 0.82
- Reference B: 0.45
Recommended action: few-shot adaptation
```

---

## Integration
- Can be run standalone (e.g., pre-screening new datasets).
- Automatically used when running `python service/main.py`.

---

## Notes
Default thresholds and metrics can be modified in the config file (`config.yaml`).
