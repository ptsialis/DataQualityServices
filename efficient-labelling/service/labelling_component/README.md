# AI Alliance – Labelling Pipeline

The **Labelling Pipeline** applies few-shot learning to efficiently label new datasets.  
It can train from scratch, adapt to new classes, or perform inference with confidence scoring.

---

## Features
- Built on FEAT (Few-Shot Embedding Adaptation with Transformer)
- Supports few-shot adaptation and full retraining
- Confidence-based auto-labelling with human-in-the-loop fallback
- Modular backbone and dataset support

---

## Usage

### Run as a module
```bash
python labelling_pipeline.py --train path/to/small_labelled.zip --test path/to/unlabelled.zip
```

### Example output
```
Accuracy: 79.2%
Average confidence: 0.84
Images auto-labelled: 73%
Images sent to review: 27%
```

---

## Integration
- Can be run standalone for dataset labelling.
- Used as part of the full service (`python service/main.py`).
- Optional connection to the [Streamlit GUI](../../gui/README.md).

---

## Notes
- Model weights are not included in this repository. Contact the AI Alliance team for access.
- Parameters (e.g., backbone, confidence thresholds) can be configured in `config.yaml`.
