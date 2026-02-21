## Quick orientation for AI coding agents

This repository is a Streamlit demonstrator that detects feature types, anomalies, imputes values and flags personal data using classical code and locally-loaded LLMs/models.
Keep guidance short and actionable — follow examples below when editing or adding features.

Key entry points
- Frontend (user-facing): `src/streamlit_app.py` (preferred entry) and `streamlit_auto.py` (larger demo UI). Use `cd src && streamlit run streamlit_app.py` to run the app (see Run section).
- Orchestration / utilities: `src/global_controller.py` — central metadata store (via `gc.add_metadata`), anomaly detection, imputation, and helpers used across pages.
- LLM & feature logic: `services_featuretype_personal/` — contains `feature_generator.py` (featurizer), `gpt.py` (OpenAI helpers), `huggingface_model.py` (HF model loader), and wrappers used by the Streamlit UI.

What matters (big picture)
- The Streamlit UI stores most runtime state in `st.session_state` and calls functions from `src/global_controller` and `services_featuretype_personal/*` to do heavy work.
- Feature-type inference: featurize with `feature_generator.FeaturizeFile()` → call LLM wrapper (`execute_featuretype_LLM`) → return a DataFrame with a `prediction` column (labels like `numeric`, `categorical`, `datetime`, etc.). See `services_featuretype_personal/gpt.py` and `services_featuretype_personal/feature_generator.py` for prompting and label extraction.
- Personal-data detection: `execute_personal_llm(df, model)` builds a per-column prompt and returns (metadata_df, results_df) where `results_df` has columns `Column` and `Prediction` (values `Personal`/`Non-personal`).
- Models: Flair `TextClassifier` loads `src/best-model.pt` for the personal-data classifier. Hugging Face models (Phi-4) are loaded via the huggingface wrapper. Some code expects quantized pipelines and GPU if available.

Project-specific conventions & warnings
- Python path: modules import as `import src.<module>`. Run code from repo root or `src/` so imports resolve. Many devs use `cd src && streamlit run streamlit_app.py` (documented in `ReadMe.txt`).
- Requirements & Python version: `ReadMe.txt` mentions Python 3.10.13 and pip install of `requirements.txt`. Use a virtualenv/conda environment to isolate installs.
- Local/vendor code: a local `sortinghat` dependency is referenced (e.g., `import sortinghat.pylib as pl`). The ReadMe suggests copying vendor modules into site-packages if not pip-installable. Check for `dictionaryName.pkl` / `dictionarySample.pkl` and other model artifacts in the repo root — they are required by `feature_generator.py`.
- Secrets: API keys may be read from `OpenAIAPIKey.txt` (helper in `services_featuretype_personal/gpt.py`). Do NOT hardcode keys in new code and do not commit secrets. If a file with a key exists, treat it as sensitive.

Developer workflows (how to run & test quickly)
- Setup (recommended):
  1. Create conda/env with Python 3.10.13 (or compatible 3.10.x).
  2. Install dependencies: `pip install -r requirements.txt` (there are multiple requirements files—use `requirements.txt` unless instructed otherwise).
  3. From repo root: `cd src && streamlit run streamlit_app.py` — this mirrors the documented workflow in `ReadMe.txt`.
- Running the demo page in `streamlit_auto.py`: this file is standalone at repo root and imports `services_featuretype_personal.services_LLM`. It can be run directly with `streamlit run streamlit_auto.py`, but verify Python path and current working directory for asset paths used by the UI.

Patterns and small examples
- Add metadata: `import src.global_controller as gc` then `gc.add_metadata("Key", value)`; metadata is stored in `st.session_state.metadata_store`.
- Feature-type call path (example):
  - UI collects DataFrame `df` → calls `model = get_phi4_model()` from `services_featuretype_personal` → `execute_featuretype_LLM(df, model)` → returns `df_pred` (DataFrame with `prediction`).
- Personal-data call path (example):
  - UI: `st.session_state.personal_0, st.session_state.personal_1 = execute_personal_llm(df, model)`
  - `personal_1` is a DataFrame with columns `Column` and `Prediction`.

Integration & external deps to be aware of
- Hugging Face: model loading happens in `services_featuretype_personal/huggingface_model.py` (model id `microsoft/Phi-4` is referenced). Loading/quantization is time and memory intensive — prefer caching and check `@st.cache_resource` decorators where used.
- OpenAI: `services_featuretype_personal/gpt.py` and `services_featuretype_personal/gpt` helpers use OpenAI client and may expect `OpenAIAPIKey.txt` or environment variables.
- Flair/TextClassifier: loads `src/best-model.pt`. Ensure correct Flair version and PyTorch compatibility when updating models.
- Pickles / joblib: `dictionaryName.pkl` and `dictionarySample.pkl` are loaded by `feature_generator.py` — they must be present.

Editing guidance for agents
- When modifying LLM prompts, edit the system/user prompts in `services_featuretype_personal/gpt.py` or `services_featuretype_personal/*` and keep changes small and well-tested. Prompts are spread across `gpt.py`, `huggingface_model.py` and `feature_generator.py`.
- Preserve `st.session_state` keys: UI expects specific keys like `original`, `inference`, `impute`, `personal_0`, `personal_1`, `metadata_store` — adding/removing keys may break navigation and stateful flows.
- Avoid heavy synchronous work on the Streamlit main thread. Long LLM/model loads should be cached (`@st.cache_resource`) and backgrounded if possible.

Where to look for more context
- Quick: `ReadMe.txt` (top-level) — environment notes.
- Core orchestrator: `src/global_controller.py` (metadata, imputation, anomaly, helper utilities).
- LLM pipeline: `services_featuretype_personal/huggingface_model.py`, `services_featuretype_personal/gpt.py`, `services_featuretype_personal/feature_generator.py`.

If anything above is unclear or you need more detailed examples (unit tests, small harness to run a single pipeline step), tell me which area to expand and I will add runnable examples or tests. 
