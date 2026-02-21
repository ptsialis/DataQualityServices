# streamlit_auto.py
# -----------------
# Interactive Streamlit demo UI for dataset preprocessing, feature-type
# inference, personal-data detection, imputation and anomaly detection.
#
# High-level architecture:
# - Frontend/UI: this file (many Streamlit dialogs/containers).
# - Orchestration: `src/global_controller.py` (metadata store, imputation,
#   anomaly detection helpers and utility functions).
# - LLM & model wrappers: `services_featuretype_personal/*` (feature featurizer,
#   prompt builders, HuggingFace model loader, Flair classifier wrapper).
#
# Important: this file relies heavily on `st.session_state` keys created and
# consumed across dialogs. Do not remove keys unless you update all callers.
import os
cwd = os.getcwd()
print(cwd)
# --- Standard library imports
import os
import io
import re
import gc as gcl  # renamed to avoid confusion with your `src.global_controller as gc`
import sys
import json
import time
import uuid
import random
import zipfile
import pathlib
from datetime import datetime
from io import StringIO
import base64
import webbrowser
import io

# --- Third‑party imports
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components  # used to provide st.html shim
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.grid import grid
from PIL import Image
import torch
import nltk
st.set_page_config(layout="wide")
# --- Local/Project imports
import src.global_controller as gc  # orchestrator utilities
import src.meta_data_export as md
import src.train_autogluon as trau
import src.plots as plt  # NOTE: this name can be confused with matplotlib.pyplot
import src.summary as sm
from services_featuretype_personal.services_LLM import get_phi4_model, execute_featuretype_LLM,execute_personal_llm
from services_featuretype_personal.services_LLM import FeaturizeFile,Load_RF,FeatureExtraction
# Reduce noisy warnings in the UI
import warnings
warnings.filterwarnings("ignore")

# Ensure NLTK stopwords are available (download once)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")




def load_css(file_path): # Define helper to load CSS file
    # """Load a CSS file and inject it into the Streamlit page.

    # This helper reads a CSS file from disk and uses `st.markdown` to apply
    # styling globally. Called early during module initialization to style the
    # app UI.
    # """
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)  # Inject CSS into page

 # Path object to CSS file
css_path = pathlib.Path("assets//style.css")
 # Apply CSS theme
load_css(css_path)


# Initialize session state variables
if "language" not in st.session_state:
    st.session_state.language = "en"  # Default language is Englisch

# Function to toggle language
def toggle_language():
    """Cycle the app language stored in `st.session_state.language`.

    The UI supports a small set of languages ('en' -> 'de' -> 'sw' -> 'en').
    Toggling triggers `st.rerun()` so the UI re-renders immediately.
    """
    current = st.session_state.language
    if current == "en":
        st.session_state.language = "de"
    # elif current == "de":
    #     st.session_state.language = "sw"
    else:
        st.session_state.language = "en"
    st.rerun()


# Button to toggle language
st.markdown(
    """
        <style>
                .stAppHeader {
                    background-color: rgba(255, 255, 255, 0.0);  /* Transparent background */
                    visibility: hidden;  /* Ensure the header is visible */
                }

            
        </style>
        """,
    unsafe_allow_html=True,
)

#Mean/Mode Imports
if 'starting_process' not in st.session_state:
     st.session_state.starting_process = True
if 'intermediate_process' not in st.session_state:
     st.session_state.intermediate_process = False
if 'ending_process' not in st.session_state:
     st.session_state.ending_process = False



#Session states  # Initialize various dataframes and UI flags in session
if "original" not in st.session_state:  # Original dataset holder
    st.session_state.original = pd.DataFrame()  # Empty DataFrame default
if "anomaly" not in st.session_state:  # Anomalies results holder
    st.session_state.anomaly = pd.DataFrame()  # Empty DataFrame default
if "impute" not in st.session_state:  # Imputed dataset holder
    st.session_state.impute = pd.DataFrame()  # Empty DataFrame default
if "inference" not in st.session_state:  # Feature-type inference holder
    st.session_state.inference = pd.DataFrame()  # Empty DataFrame default
if "personal_0" not in st.session_state:  # Personal data summary holder
    st.session_state.personal_0 = pd.DataFrame()  # Empty DataFrame default
if "personal_1" not in st.session_state:  # Personal data detailed holder
    st.session_state.personal_1 = pd.DataFrame()  # Empty DataFrame default
if "metadata_store" not in st.session_state:  # Metadata collection holder
    st.session_state.metadata_store = pd.DataFrame()  # Empty DataFrame default
if "show_upload" not in st.session_state:  # Flag to reveal upload section
    st.session_state.show_upload = False  # Default hidden
if "dataset_name" not in st.session_state:  # Name of uploaded dataset
    st.session_state.dataset_name = ""  # Default empty
if "image_team_bern" not in st.session_state:  # Flag for image team feature
    st.session_state.image_team_bern = False  # Default false
if "image" not in st.session_state:  # UI mode for image preprocessing page
    st.session_state.image = False  # Default false
if "uploaded_zip" not in st.session_state:  # Store uploaded ZIP for images
    st.session_state.uploaded_zip = None  # Default none
if "anomaly" not in st.session_state:  # Store uploaded ZIP for images
     st.session_state.anomaly = False  # Default none

if "anomalies_delete_or_not" not in st.session_state:
    st.session_state.anomalies_delete_or_not = False  # Default false
if "personal_delete_or_not" not in st.session_state:
    st.session_state.personal_delete_or_not = False  # Default false  
 

# Check if button state exists in session state
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False


# --- Small UI utilities ----------------------------------------------------
def show_meta_data():
    
  
   
    st.html("<span class='realy-big-dialog'></span>") 




@st.dialog("Finished Data Processing")
def show_finished(): 

    findal_df = show_clean_data()
    
    # (normalize_to_mapping unchanged / still unused)
    def normalize_to_mapping(items):
        mapping = {}
        if isinstance(items, dict):
            mapping = dict(items)
        elif isinstance(items, list):
            # try list of tuples
            if items and isinstance(items[0], tuple) and len(items[0]) == 2:
                for name, data in items:
                    mapping[str(name)] = data
            # try list of {"name","data"}
            elif items and isinstance(items[0], dict) and "name" in items[0] and "data" in items[0]:
                for d in items:
                    mapping[str(d["name"])] = d["data"]
            else:
                # fallback: auto-name
                for idx, obj in enumerate(items, start=1):
                    mapping[f"item_{idx:03d}.json"] = obj
        else:
            # single object fallback
            mapping["item_001.json"] = items
        return mapping

    name = st.session_state.data["name"]
    name = name.name.split(".")[0]

    try:
        st.session_state["run_jsons"] = {
            "imputation.json": md.generate_metadata_json_imputation(st.session_state.original,  name),
            "anomalies.json": md.generate_metadata_json_anomalies(st.session_state.anomaly,     name),
            "feature_types.json": md.generate_metadata_json_feature_types(st.session_state.inference, name),
            "personal_data.json": md.generate_metadata_json_personal_data(st.session_state.personal_1, name),
        }
    except NameError as e:
        st.error("Missing one of the required DataFrames (df1, df2, df3, df4). Define them before this block.")
        st.stop()

    run_map = st.session_state.get("run_jsons", {})
    if not run_map:
        st.info("No JSONs were generated.")
        st.stop()

    # --- Selection + filename inputs ---
    all_names = sorted(run_map.keys())
    selected = st.multiselect("Select Services to include", options=all_names, default=all_names)

    # NEW: create cleaned data based on the selected services
    # clean_data_exctraction should return a DataFrame (or something with .to_csv)
    cleaned_df = clean_data_exctraction(selected)
    cleaned_csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = (
        st.text_input("ZIP filename", f"metadata_bundle_{ts}.zip").strip()
        or f"metadata_bundle_{ts}.zip"
    )

    st.write(f"Selected: **{len(selected)}** / {len(all_names)}")

    # --- Zip helper (extended to also include cleaned_data.csv) ---
    def make_zip_bytes(mapping, names, cleaned_bytes, folder=""):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # 1) write JSON metadata files
            for name in names:
                payload = json.dumps(mapping[name], indent=2, ensure_ascii=False).encode("utf-8")
                arcname = f"{folder}/{name}" if folder else name
                zf.writestr(arcname, payload)

            # 2) write cleaned_data.csv
            if cleaned_bytes is not None:
                csv_arcname = f"{folder}/cleaned_data.csv" if folder else "cleaned_data.csv"
                zf.writestr(csv_arcname, cleaned_bytes)

        buf.seek(0)
        return buf.getvalue()

    # --- Single ZIP download button ---
    zip_col, upload_col = st.columns(2)
    with zip_col:  
        st.download_button(
            label="Download",
            data=make_zip_bytes(run_map, selected, cleaned_csv_bytes, folder="metadata") if selected else b"",
            file_name=zip_name,
            mime="application/zip",
            disabled=len(selected) == 0,
        )

    with upload_col:
        if st.button("Upload Data"):
            st.write("")

    st.html("<span class='realy-big-dialog'></span>")


def clean_data_exctraction(selected):

    #st.write(selected)
    # 0:"anomalies.json"
    # 1:"feature_types.json"
    # 2:"imputation.json"
    # 3:"personal_data.json"


    if "anomalies.json" in selected and "personal_data.json" not in selected:
        clean_df =gc.one_hot_encode_columns(st.session_state.impute, st.session_state.inference,  df_anomalies =st.session_state.anomaly,target_col=st.session_state.data["target_variable"])  
    if "anomalies.json" not in selected and "personal_data.json" in selected:
        clean_df =gc.one_hot_encode_columns(st.session_state.impute, st.session_state.inference,  df_personal= st.session_state.personal_1,target_col=st.session_state.data["target_variable"])  
    if "anomalies.json" in selected and "personal_data.json" in selected:
        clean_df =gc.one_hot_encode_columns(st.session_state.impute, st.session_state.inference,df_anomalies= st.session_state.anomaly, df_personal=st.session_state.personal_1,target_col=st.session_state.data["target_variable"])
    else:
        clean_df =gc.one_hot_encode_columns(st.session_state.impute, st.session_state.inference,target_col=st.session_state.data["target_variable"])
    
    return clean_df
            

# Detect button click and run function
if st.session_state.button_clicked:
    show_finished()
    st.session_state.button_clicked = False  # Reset after execution

def toggle_columns():
    """Simple flag used elsewhere to reveal a columns panel."""
    st.session_state.show_columns = True
    
def render_svg(svg):
    """Render an SVG string via base64 <img> embedding (handy for inline icons)."""
    # Convert SVG string to base64 and render as an inline <img> element.
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)



def load_streamlit_csv(
    uploaded_file,
    max_mb: int = 50,
    max_rows: int = 1_000_000,     # hard limit
    max_cols: int = 1_000,         # hard limit
    styler_max_cells: int = 262_144,  # Styler/Streamlit rendering limit
    truncate_rows: int = 1000,   # rows to keep if over Styler limit
):
    """
    Load a CSV uploaded via Streamlit's file_uploader with safety checks.

    Returns
    -------
    (df, error, truncated)
        df : pd.DataFrame or None
        error : str or None (fatal problems only)
        truncated : bool - True if rows were cut down to avoid Styler issues
    """
    if uploaded_file is None:
        return None, "No file uploaded.", False

    # 1) File size check
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > max_mb:
        return None, (
            f"File is too large ({file_size_mb:.1f} MB). "
            f"Limit is {max_mb} MB."
        ), False

    # 2) Try separators: comma first, then semicolon
    df = None
    last_error = None
    for sep in [",", ";"]:
        try:
            uploaded_file.seek(0)  # reset pointer each try
            df = pd.read_csv(uploaded_file, sep=sep)
            break
        except Exception as e:
            last_error = e
            df = None

    if df is None:
        return None, (
            "Could not read CSV with ',' or ';' as separator. "
            f"Last error: {last_error}"
        ), False

    # 3) Drop index-like first column if it exists
    if len(df.columns) > 0:
        first_col = df.columns[0]
        col = df[first_col]

        drop_index_col = False

        # Name suggests it's an index
        if str(first_col).lower() in ["unnamed: 0", "index"]:
            drop_index_col = True
        else:
            # Looks like a default RangeIndex 0..n-1
            try:
                if (
                    pd.api.types.is_integer_dtype(col)
                    and col.is_monotonic_increasing
                    and col.iloc[0] == 0
                    and col.iloc[-1] == len(df) - 1
                ):
                    drop_index_col = True
            except Exception:
                pass

        if drop_index_col:
            df = df.drop(columns=[first_col])

    # 4) Hard DataFrame size check
    n_rows, n_cols = df.shape
    if n_rows > max_rows or n_cols > max_cols:
        return None, (
            f"DataFrame too large: {n_rows} rows, {n_cols} columns. "
            f"Limits: {max_rows} rows, {max_cols} columns."
        ), False

    # 5) Styler cells limit: trim rows to at most `truncate_rows` if needed
    original_rows = n_rows
    truncated = False
    total_cells = n_rows * n_cols

    if total_cells > styler_max_cells:
        truncated = True
        df = df.head(min(truncate_rows, n_rows))
        n_rows = len(df)  # update n_rows if needed

    return df, None, truncated





# Dialog width presets for different modal classes.
st.markdown(
    """
<style>
div[data-testid="stDialog"] div[role="dialog"]:has(.realy-big-dialog) {
    width: 90%;
    height: 90%;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
div[data-testid="stDialog"] div[role="dialog"]:has(.big-dialog) {
    width: 70vw;
    height: 90vh;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
div[data-testid="stDialog"] div[role="dialog"]:has(.semi-big-dialog) {
    width: 60vw;
    height: 80vh;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
div[data-testid="stDialog"] div[role="dialog"]:has(.small-big-dialog) {
    width: 60vw;
    height: 80vh;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
div[data-testid="stDialog"] div[role="dialog"]:has(.personal-small-dialog) {
    width: 35vw;
    height: 70vh;
}
</style>
""",
    unsafe_allow_html=True,
)





# Define your custom CSS


def update_container_state(key, color, visible):
    """Maintain per-container style/visibility state in session."""
    st.session_state.container_states[key] = {"color": color, "visible": visible}



def get_base64(bin_file):
    """Read a binary file and return its base64 representation (used for bg images)."""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Change Page background image via base64 CSS.
st.markdown(
    """
    <style>
        /* Targeting the specific column container for col2 */
        [data-testid="stBlock"] > div:nth-child(2) {
            background-color: transparent !important;
        }

        /* Ensuring the main container also allows the background to be visible */
        .stApp {
            background-color: rgba(0,0,0,0); /* Fully transparent */
            background-image: url("data:image/png;base64,%s"); /* Your background image */
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed; /* Fixed background */
        }
    </style>
    """ % get_base64('./data/logos/back.png'), unsafe_allow_html=True
)



# --- Legacy RandomForest label mapping (used in make_predictions) For Not LLM ----------
RandomForest_classes = ['numeric', 'categorical', 'datetime',
                        'sentence', 'url', 'embedded-number',
                        'list', 'not-generalizable', 'context-specific'
                        ]



def make_predictions(df, model):
    # """Run a (legacy) RandomForest feature-type prediction using `sortinghat.pylib`<- (Extracted now in services_featuretype_personal.services_LLM).

    # The application originally used a local RandomForest pipeline in the
    # `sortinghat` helper. This wrapper converts Prediction of column Types and builds
    # a small result DataFrame with a 'prediction' column (mapped to
    # `RandomForest_classes`). Currently this path is only used when
    # LLM-based feature inference is disable.
    # """
    df = df*1 # turn true and false entries to 0 and 1 (required for RF)
    dataFeaturized = FeaturizeFile(df)
    columns_of_interest = dataFeaturized.columns[:15]
    result = dataFeaturized.filter(columns_of_interest)
    result.insert(loc=0, column='prediction', value="") # predictions as first column


    if model == 'RandomForest':
        dataFeaturized1 = FeatureExtraction(dataFeaturized)
        # new columns are int, make all strings to escape error from model.fit
        dataFeaturized1.columns = dataFeaturized1.columns.astype(str)
        y_RF = Load_RF(dataFeaturized1.values)
        result['prediction'] = y_RF
        # map numbers back to classes
        for i in range(len(result['prediction'])):
            result.loc[i, 'prediction'] = RandomForest_classes[int(result['prediction'][i])]
    
    return result

@st.dialog("Upload your Data",width="large")
def upload_data_beginning():
                                        
                    # """End-to-end orchestration for a single CSV upload and processing run.

                    # Steps:
                    # 1) Upload CSV ➜ load DataFrame and basic metadata.
                    # 2) Choose target column and whether to use local LLM.
                    # 3) Run feature-type inference (LLM or RF fallback) and check if time-series.
                    # 4) Detect personal data (LLM or classic fallback).
                    # 5) Impute (mean/mode) and detect anomalies (tabular vs. TranAD for series).
                    # 6) Persist results in session_state and re-render the UI.
                    # """
                    df=None
                    column_list = []
                    
                    uploaded_file = st.file_uploader("", type=["csv"])
                    if uploaded_file is not None:
                        df, err, truncated = load_streamlit_csv(uploaded_file)

                        if err is not None:
                            st.error(err)
                        else:
                            st.success("File loaded successfully!")

                            if truncated:
                                st.info(
                                    "The dataframe is very large, so only the first 1,000 rows "
                                    "are shown to avoid rendering limits."
                                )

                          


                        st.info("Data uploaded",icon="🙌")
                        on = st.toggle("Use local LLM model. Brief initial load, major performance boost.")
                        df_original= df.copy()
                        column_list += df.columns.tolist()
                        target_variable = st.selectbox("Select the target variable",
                                                    column_list,)
                        


                        if st.button("Submit"):
                            if on:
                                with st.spinner("Loading LLM... This may take a few minutes the first time.", show_time=True):
                                    model = get_phi4_model()

                            gc.add_metadata("Target variable", target_variable)
                            problem_type = gc.infer_problem_type(df[target_variable])

                            gc.add_metadata("User has chosen", problem_type)
                            if on:
                                with st.spinner("Detecting Feature Types", show_time=True):
                                    feature_typ_inference = execute_featuretype_LLM(df,model)
                            else:
                                feature_typ_inference = gc.make_predictions(df,"RandomForest")
                                    
                            
                            time_or_tab = gc.check_time_series(df,feature_typ_inference)
                            gc.add_metadata("Time or Tab", time_or_tab)

                            
                            if on : 
                                with st.spinner("Detecting Pesonal Data.", show_time=True):
                                    st.session_state.personal_0,st.session_state.personal_1 = execute_personal_llm(df,model)
                            else:
                                st.session_state.personal_0,st.session_state.personal_1 = gc.make_prediciton_personal(df_original)
                            
            
                            

                            imputed_df, imputation_info = gc.impute_mean_mode(df)
                            if time_or_tab == "Tabular":

                                if target_variable is not None:
                                    df_anomalies,threshold = gc.detect_anomalies(df)
                                    st.session_state.anomaly = df_anomalies
                                    
                            else:
                                    original_data, normalized_data, feats = gc.load_and_preprocess_data(df,feature_typ_inference)
                                    threshold_percentile = 95
                                    anomalies, scores, preds, threshold = gc.detect_anomalies_tranad(normalized_data, feats, threshold_percentile)
                                    anomalies_final_pred= df.copy()
                                    anomalies_final_pred["Anomaly"] = anomalies
                                    st.session_state.anomaly = anomalies_final_pred
                            

                                 # clear Python garbage
                            torch.cuda.empty_cache() 
                            st.session_state.data = {"State":True,"problem Type": problem_type, "threshold" : threshold,"time_or_tab":time_or_tab,"original_df":df,"df_imputed":imputed_df,"feature_typ_inference":feature_typ_inference,"name":uploaded_file, "target_variable":target_variable}
                            st.session_state.inference = feature_typ_inference
                            st.session_state.original = df_original
                            st.session_state.impute = imputed_df
                            
                            

                            
                            # with st.spinner("Wait for it...", show_time=True):
                            #     time.sleep(3)
                            st.session_state.starting_process = False
                            st.session_state.intermediate_process = True
                            st.rerun()


    
@st.dialog("Dataset",width="large")
def show_orig_data(): 
    """Raw data preview, basic statistics, and quick plots of the dataset."""
    metadata,metadata_featuerwise =gc.extract_metadata(st.session_state.original,st.session_state.inference)   
    with st.expander("See Raw Data"):
        original_df= st.session_state.original
        
        df_styled = original_df.style.set_properties(**{'font-size': '50pt'})
        st.dataframe(df_styled)
        

    with st.expander("See Statistics of Original Data"):
            
            desc= sm.describe_dataframe(st.session_state.original)
            
            st.dataframe(desc)
        
    with st.expander("See Graphs of Data"):
        #st.dataframe(metadata_featuerwise)  
        fig1_box,fig2_bar = plt.plot_boxplots(st.session_state.original,st.session_state.inference)
        fig_1_orig,fig_2_orig = st.columns(2)
       

        with fig_1_orig:
            if fig1_box is None:
                 st.write("")
            else:
                 
                st.plotly_chart(fig1_box) 

        with fig_2_orig:

            if fig2_bar is None:
                 st.write("")
            else:
                 
                st.plotly_chart(fig2_bar) 
         
          
    st.html("<span class='big-dialog'></span>")           


@st.dialog("Feature Type Inference",width="large")
def show_inference(): 
    #"""Show feature-type predictions and a brief explanation figure."""
    with st.expander("See Feature Typ Inference Results"):
        inference_df_highlighted = gc.highlight_columns(st.session_state.inference,["prediction"])
        st.dataframe(inference_df_highlighted,width=4000, height = 300)
    with st.expander("See Explanation Feature Type Inference"):
         fig = sm.explanation_feature_type_inference(st.session_state.original,st.session_state.data["target_variable"])
         st.plotly_chart(fig)
    st.html("<span class='big-dialog'></span>") 
               
@st.dialog("Imputation",width="large")
def show_imputation(): 
    #"""Highlight imputed cells and provide a textual explanation."""
    imputed_df_highlighted=gc.highlight_imputed_cells(st.session_state.original,st.session_state.impute)
    st.dataframe(imputed_df_highlighted,width=1300, height = 400) 
    with st.expander("See explanation"):
         st.write(sm.explanation_imputation(st.session_state.original,st.session_state.inference))
    st.html("<span class='big-dialog'></span>")    


@st.dialog("Anomaly Detection",width="large")
def show_anomalies(): 
    #"""Display anomalies table and explain the thresholding approach."""
    anomaly_df_highlighted = gc.highlight_anomaly_cells(st.session_state.anomaly)
    st.dataframe(anomaly_df_highlighted,width=1300, height = 400)
    with st.expander("See explanation anomaly"):
         st.write(sm.explanation_anomaly(st.session_state.data["threshold"],st.session_state.data["time_or_tab"]))
    
    
    st.html("<span class='semi-big-dialog'></span>") 


@st.dialog("Personalized Data Detection")
def show_personal(): 
    #"""Show personal-data detection results with highlighted predictions."""
    personal_df_highlighted = gc.highlight_personal_cells(st.session_state.personal_1)
    st.dataframe(personal_df_highlighted)
    # personal_delete_or_not = st.toggle("Delete personal data from dataset?")
    # if personal_delete_or_not:
    #     st.session_state.personal_delete_or_not = True
   
    st.html("<span class='personal-small-dialog'></span>") 


def get_newest_file_path(directory_path):
    # List all files in the directory with full path
    files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    
    if not files:
        return None  # or raise an error if preferred

    # Find the file with the latest modification time
    newest_file = max(files, key=os.path.getmtime)
    
    return newest_file


# --- Image deblurring demo (ZIP of PNGs ➜ side-by-side display) -----------

@st.dialog("Image Deblurring")
def show_deblurring():
    #if "image_team_bern" not in st.session_state:
    # """Collect a .zip of .png files and switch the UI into image mode."""
        uploaded_zip  = st.file_uploader("Drag and drop file here", type=[".zip"], accept_multiple_files=False)
        
        st.session_state.uploaded_zip = uploaded_zip
        st.session_state.image = True
        
        if st.button("Submit"):
            if uploaded_zip == None:
                st.error("Upload a File")
            else:
                with st.spinner("Please, wait ..."):
                    time.sleep(5)
                    st.rerun()
         



def show_results_page(zip_file):
    # """Parse the ZIP in-memory and display (blurred, deblurred) image pairs."""
    st.title("AI Alliance - Image deblurring")
    #st.subheader("Image deblurring result")

    zip_bytes = io.BytesIO(zip_file.read())

    # Open the ZIP file from memory
    with zipfile.ZipFile(zip_bytes, 'r') as zip_ref:
        # Collect the list of PNG images
        png_images = []
        invalid_files = []
        
        # Iterate over all files inside the ZIP
        for file_name in zip_ref.namelist():
            if file_name.lower().endswith('.png'):
                # Extract image bytes and wrap them in a BytesIO object
                with zip_ref.open(file_name) as file:
                    img_data = io.BytesIO(file.read())
                    png_images.append((file_name, img_data))
            else:
                  # Track non-PNG files to show an error message later
                invalid_files.append(file_name)
    
     # If invalid files were found, display an error message
    if invalid_files:
        st.error(f"Fehler: Es wurden ungültige Dateien gefunden: {', '.join(invalid_files)}. Nur .png-Dateien sind erlaubt.")

    # If no PNG images were found, inform the user and exit
    if not png_images:
        st.write("Es wurden keine PNG-Bilder in der ZIP-Datei gefunden.")
        return

    # Initialize the index for the currently displayed image pair (if missing)
    if 'current_image_idx' not in st.session_state:
        st.session_state.current_image_idx = 0

    st.session_state.process_result = []
    st.session_state.process_result.append({"name": png_images[0][0],
                    "blurred": png_images[0][1],
                    "deblurred": png_images[1][1]})
    st.session_state.process_result.append({"name": png_images[0][0],
                    "blurred": png_images[2][1],
                    "deblurred": png_images[3][1]})

    if st.session_state.process_result:       
        img1 = st.session_state.process_result[st.session_state.current_image_idx]["blurred"]
        img2 = st.session_state.process_result[st.session_state.current_image_idx]["deblurred"]

        #  Navigation layout for "Next" and "Back" (buttons handled elsewhere)
        col1,col2, col3 = st.columns(3)
                
        with col1:            
            st.image(img1, caption="Original",width=300)
            
        with col3:
            st.image(img2, caption="Deblurred", width=300)
    
        
        #st.write(f"Image {st.session_state.current_image_idx+1} of {len(st.session_state.process_result)}")

      
    
# Needs to be integrated into show_results_page()
# # Funktion zum Extrahieren von Bildern aus einer ZIP-Datei im Speicher und Überprüfen der Dateiendung
# def extract_and_preview_zip(zip_file):
#     # BytesIO-Objekt für die ZIP-Datei im Speicher
#     zip_bytes = io.BytesIO(zip_file.read())
    
#     # ZIP-Datei im Speicher öffnen
#     with zipfile.ZipFile(zip_bytes, 'r') as zip_ref:
#         # Liste der PNG-Bilder
#         png_images = []
#         invalid_files = []
        
#         # Durch alle Dateien in der ZIP-Datei iterieren
#         for file_name in zip_ref.namelist():
#             if file_name.lower().endswith('.png'):
#                 # Bilddaten extrahieren und in BytesIO-Objekt umwandeln
#                 with zip_ref.open(file_name) as file:
#                     img_data = io.BytesIO(file.read())
#                     png_images.append((file_name, img_data))
#             else:
#                 # Nicht-PNG-Dateien sammeln, um eine Fehlermeldung anzuzeigen
#                 invalid_files.append(file_name)
        
#     # Wenn ungültige Dateien gefunden wurden, eine Fehlermeldung anzeigen
#     if invalid_files:
#         st.error(f"Fehler: Es wurden ungültige Dateien gefunden: {', '.join(invalid_files)}. Nur .png-Dateien sind erlaubt.")

#     # Wenn keine PNG-Bilder gefunden wurden
#     if not png_images:
#         st.write("Es wurden keine PNG-Bilder in der ZIP-Datei gefunden.")
#         return

#     # Initialisiere den Zustand für den Bildindex, wenn er noch nicht existiert
#     if 'current_image_idx' not in st.session_state:
#         st.session_state.current_image_idx = 0
#     st.write(st.session_state.current_image_idx)
    

#     # Zeige das aktuelle Bild an
#     img_name, img_data = png_images[st.session_state.current_image_idx]
#     img = Image.open(img_data)

#     st.session_state.current_image_idx += 1
#     st.write(st.session_state.current_image_idx)
#     st.write(st.session_state.current_image_idx)

#     img_2_name, img_2_data =  png_images[st.session_state.current_image_idx ]
#     img_2 = Image.open(img_2_name)
    
#     # Navigation zwischen den Bildern mit "Next" und "Back"
#     col1, col2, col3 = st.columns([1, 4, 1])
    
#     # Button für "Back"
#     with col1:
#         if st.button("Back") and st.session_state.current_image_idx > 0:
#             st.session_state.current_image_idx -= 1

#     with col2:
#         st.image(img, use_column_width=True)
#         st.image(img_2)

        
#     # Button für "Next"
#     with col3:
#         if st.button("Next") and st.session_state.current_image_idx < len(png_images) - 1:
#             st.session_state.current_image_idx += 1
    
#     #st.write(f"Image {st.session_state.current_image_idx+1} of {len(png_images)}")

#     # Daten verarbeiten
#     if st.button("Process"):
#         with st.spinner("Processing data..."):
#             process_data(png_images)
#             st.session_state["page"] = "results"


@st.dialog("Show Meta Data")
def show_meta_data():
    #"""Show computed metadata and let the user download a ZIP of JSON files."""
    metadata,metadata_featuerwise =gc.extract_metadata(st.session_state.original,st.session_state.inference)      
    general_meta, demo_df= st.columns(2)
    with general_meta:
         st.dataframe(metadata)
    with demo_df:
          st.dataframe(sm.meta_data_from_demos(st.session_state.inference,st.session_state.original,st.session_state.anomaly,st.session_state.personal_1))
         
    # ---------- 2) Normalize to {filename: python_obj} ----------
    def normalize_to_mapping(items):
        mapping = {}
        if isinstance(items, dict):
            mapping = dict(items)
        elif isinstance(items, list):
            # try list of tuples
            if items and isinstance(items[0], tuple) and len(items[0]) == 2:
                for name, data in items:
                    mapping[str(name)] = data
            # try list of {"name","data"}
            elif items and isinstance(items[0], dict) and "name" in items[0] and "data" in items[0]:
                for d in items:
                    mapping[str(d["name"])] = d["data"]
            else:
                # fallback: auto-name
                for idx, obj in enumerate(items, start=1):
                    mapping[f"item_{idx:03d}.json"] = obj
        else:
            # single object fallback
            mapping["item_001.json"] = items
        return mapping

    name = st.session_state.data["name"]
    name = name.name.split(".")[0]
    # EXPECTED: a mapping of {filename.json: python_obj}
    #You can set this earlier in your app, e.g.:
    try:
        st.session_state["run_jsons"] = {
        "imputation.json": md.generate_metadata_json_imputation(st.session_state.original,name),
        "anomalies.json": md.generate_metadata_json_anomalies(st.session_state.anomaly,name),
        "feature_types.json": md.generate_metadata_json_feature_types(st.session_state.inference,name),
        "personal_data.json": md.generate_metadata_json_personal_data(st.session_state.personal_1,name),
    }

    except NameError as e:
        st.error("Missing one of the required DataFrames (df1, df2, df3, df4). Define them before this block.")
        st.stop()

 
    run_map = st.session_state.get("run_jsons", {})
    if not run_map:
        st.info("No JSONs were generated.")
        st.stop()

    # --- Selection + filename inputs ---
    all_names = sorted(run_map.keys())
    selected = st.multiselect("Select JSONs to include", options=all_names, default=all_names)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = st.text_input("ZIP filename", f"metadata_bundle_{ts}.zip").strip() or f"metadata_bundle_{ts}.zip"
    #folder_in_zip = st.text_input("Folder name inside ZIP (optional)", "metadata").strip()

    st.write(f"Selected: **{len(selected)}** / {len(all_names)}")

    # --- Zip helper ---
    def make_zip_bytes(mapping, names, folder=""):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name in names:
                payload = json.dumps(mapping[name], indent=2, ensure_ascii=False).encode("utf-8")
                arcname = f"{folder}/{name}" if folder else name
                zf.writestr(arcname, payload)
        buf.seek(0)
        return buf.getvalue()

    # --- Single ZIP download button ---
    st.download_button(
        label="Download ZIP",
        data=make_zip_bytes(run_map, selected, folder='metadata') if selected else b"",
        file_name=zip_name,
        mime="application/zip",
        disabled=len(selected) == 0,
    )
   
    st.html("<span class='realy-big-dialog'></span>") 

#@st.dialog("Trained Model",width="medium")
# Simple decision-tree training demo on the imputed/encoded dataset. Not in Project yet, might change in future
# def show_trained_model_on_data():
#     final_df =gc.one_hot_encode_df(st.session_state.impute)
#     best_model_hyper,figure_best_model=trau.train_and_visualize_model_decisiontree(final_df,st.session_state.data["target_variable"],sm.summary_problemtype(st.session_state.data["time_or_tab"],st.session_state.data["target_variable"],st.session_state.inference))
#     st.pyplot(figure_best_model)
#     st.write(best_model_hyper)
#     st.html("<span class='realy-big-dialog'></span>")   


# --- Convenience panel to compare original vs final (encoded) data --------
def show_clean_data():
    # Two-column preview + CSV download for original and final (encoded) data.
    
    old_df,cleaned_df = st.columns(2)
    with old_df:
        st.dataframe(st.session_state.original)
    with cleaned_df:
         
        if st.session_state.anomalies_delete_or_not == True and st.session_state.personal_delete_or_not == False:
            final_df =gc.one_hot_encode_columns(st.session_state.impute, st.session_state.inference,  df_anomalies =st.session_state.anomaly,target_col=st.session_state.data["target_variable"])  
        if st.session_state.anomalies_delete_or_not == False and st.session_state.personal_delete_or_not == True:
            final_df =gc.one_hot_encode_columns(st.session_state.impute, st.session_state.inference,  df_personal= st.session_state.personal_1,target_col=st.session_state.data["target_variable"])  
        if st.session_state.anomalies_delete_or_not == True and st.session_state.personal_delete_or_not == True:
            final_df =gc.one_hot_encode_columns(st.session_state.impute, st.session_state.inference,df_anomalies= st.session_state.anomaly, df_personal=st.session_state.personal_1,target_col=st.session_state.data["target_variable"])
        else:
            final_df =gc.one_hot_encode_columns(st.session_state.impute, st.session_state.inference,target_col=st.session_state.data["target_variable"])
        st.dataframe(final_df)
        return final_df
        # if st.button("download final"):
        #     end_download_df=final_df
        #     end_download_df.to_csv("data/final_df.csv")


#expander for show summary
@st.dialog("Show Summary",width="large")
def show_summary():
    # Dashboard-style summary: inference, anomalies, imputation, personal-data.
    with st.container():

        plot_inference,plot_anomaly = st.columns(2)
       
            
        with plot_inference:
            st.plotly_chart(plt.plot_inference_bar(st.session_state.inference))

        with plot_anomaly:
            st.plotly_chart(plt.plot_anomaly_pie(st.session_state.anomaly))

        plot_imputation, plot_personal  = st.columns(2)
        with plot_imputation:
            
            st.plotly_chart(plt.plot_imputation_bar(st.session_state.original,st.session_state.inference))

        with plot_personal:
           
            st.plotly_chart(plt.plot_personal_pie(st.session_state.personal_0))
        

        # with st.expander("See Final Data"):
        #      show_clean_data()
    st.html("<span class='realy-big-dialog'></span>") 



def main():
 torch.cuda.empty_cache() 

 if st.session_state.language == "en" and st.session_state.image == False: 
    # Container CSS with dynamic visibility and color
    #Top-level UI flow switcher between dataset mode and image mode.

    # * English/tabular mode: header with actions (Start, Image Preprocessing,
    #   Upload Data, Language toggle) and a large results panel showing each step
    #   (Dataset, Feature Type Inference, Imputation, Anomaly Detection, Personal
    #   Data, Summary, Metadata). Each action opens its corresponding dialog.
    # * Image mode: simplified layout to upload a ZIP and preview blurred vs.
    #   deblurred image pairs.

         # Header bar with logo + action buttons.
    
    
        # Top large container
        # Css for container, contains Buttons to start, upload data, language toggle
        with stylable_container(
                    key="container_with_border",
                    css_styles="""
                    
                    {
                            box-sizing: border-box;                  /* Include padding + border inside the element's total width/height */

                            border: 1px solid rgba(170, 10, 10, 0.2);   /* Light red border around the container */
                            border-radius: 0.5rem;                      /* Slightly rounded corners */
                            background-color: rgba(248, 245, 245, 0.91);/* Light grey background with a bit of transparency */

                            font-family: "Nunito-Sans", sans-serif;     /* Use Nunito Sans if available, else any sans-serif font */

                            margin-top: -15%;                              /* No extra space above the container */
                            margin-right: auto;                         /* Auto right margin to help center the container */
                            margin-bottom: 10%;                        /* Space below the container to separate from the next block */
                            margin-left: auto;                          /* Auto left margin to help center the container */

                            width: 100%;                                /* Take up the full available width of the parent */
                            max-width: 1200px;                          /* But don’t be wider than 1200px on large screens (better readability) */
                            min-width: 900px;
                            padding-top: 1rem;
                            padding-right: 0px;
                            padding-bottom: 0px;
                            padding-left: 0px;                    /* Inner spacing so content doesn’t touch the edges */

                            display: flex;                              /* Use flexbox for layout of direct children */
                            flex-direction: column;                     /* Stack children vertically (each child behaves like a row) */
                            row-gap: 0.75rem;                           /* Vertical gap between children so rows don’t touch each other */

                            pointer-events: auto;                       /* Allow mouse interactions (clicks, hovers) on this container */
                            position: block;                         /* Establish a positioning context for absolutely positioned children */
                            z-index: 1;  
                            }
                        """,
                ):
                    pic, bt_container = st.columns(2)  # Adjust ratio to shift buttons further right

                    with pic:
                        
                        # image=Image.open("data/logos/logo_ki.png")
                        # buffered = io.BytesIO()
                        # image.save(buffered, format="PNG")
                        # img_str = base64.b64encode(buffered.getvalue()).decode()

                    
                        # st.markdown(
                        #     f"""
                        #     <div class='uploaded-image'>
                        #         <img src="data:image/png;base64,{img_str}" alt="Uploaded Image">
                        #     </div>
                        #     """,
                        #     unsafe_allow_html=True,
                        # )
                        st.image("data/logos/logo.svg",width=300)

                    with bt_container:
                        piveau,upload_data= st.columns(2,gap="large")
                        
                        # with image_preprocess:
                        #     st.write("")
                        # with language:
                        #     st.write("")
                        
                        with piveau:
                            if st.button("Dataset Catalogue", key="nicebt"):
                                #
                                webbrowser.open("https://ask.ki-allianz.de/catalogues/dataservices?locale=en")
                                
                                #show_dawid()
                        # with image_preprocess:
                        #     if st.button("Image Preprocessing", key="image_preprocess"):
                                
                        #         show_deblurring()
                                

                        with upload_data:
                            if st.button("Load & Prepare Data", key="upload"):
                                upload_data_beginning()

                        # with language:
                        #     if st.button("DE/EN", key = "language_bt"):
                        #         toggle_language() 
    
        col1, col2= st.columns(2,gap="small")
        with col1:
            # Main Contianer that contains all results
                st.markdown(
                        """
                        <style>
                            @import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@200;400;600;700;800&display=swap');

                            .custom-text {
                                font-family: 'Nunito Sans', sans-serif;
                                /* Responsive font size:
                                - minimum: 16px (1rem)
                                - preferred: 2vw + 0.5rem (scales with width)
                                - maximum: 24px (1.5rem)
                                */
                                font-size: clamp(0.5rem, 2vw + 0.5rem, 1rem);
                                
                                color: #333333;  /* Adjust color if needed */
                                line-height: 1.5;  /* Improves readability */
                                font-weight: 700;
                            }

                           
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                # If data is not uploaded, show empty container
                if "data" not in st.session_state:
                    st.write("")
                else:

                    with stylable_container(
                        key="container_with_cols",
                        css_styles="""
                            {   box-sizing: border-box;                  /* Include padding and border inside the element's total width/height */

                                border: 1px solid rgba(170, 10, 10, 0.2); /* Light red border around the container */
                                border-radius: 0.5rem;                    /* Slightly rounded corners */
                                background-color: rgba(248, 245, 245, 0.91); /* Light grey / off-white background with a bit of transparency */

                                font-family: "Nunito-Sans", sans-serif;  /* Use Nunito Sans if available, otherwise a generic sans-serif font */


                                margin-top: -15%;                              /* No extra space above the container */
                                margin-right: auto;                         /* Auto right margin to help center the container */
                                margin-bottom: 10%;                        /* Space below the container to separate from the next block */
                                margin-left: 11%;                          /* Auto left margin to help center the container */

                                                  

                                padding-top: 0.75rem;                    /* Inner space at the top (so content doesn’t touch the border) */
                                padding-right: 1.5rem;                   /* Inner space on the right side */
                                padding-bottom: 2rem;                    /* Extra inner space at the bottom for visual breathing room */
                                padding-left: 1.5rem;                    /* Inner space on the left side */

                                width: 280%;                             /* Take all available width of the parent container */
                                max-width: 1200px;                       /* But never be wider than 1200px (better readability on big screens) */
                                min-width: 900px;
                                display: flex;                           /* Use flexbox layout for direct child elements inside this container */
                                position: block; 
                                flex-direction: column;                  /* Stack direct children vertically (one “row” under another) */
                                row-gap: 0.75rem;                        /* Add vertical space between each direct child (so rows don’t touch) */
                            }
                        """,
                    ):
                        
                        
                    # Create columns in the main container with three subcontainers for each displayed result:
                    # first the name, then a button to open the dialog, then a short text summary of the results.
                    # --- Row 1: Dataset (label • open dialog button • short summary) ---
                        col_orig, col_dialogue_orig, orig_text = st.columns(3)

                        with col_orig:
                            # Left: section label
                            st.markdown('<div class="custom-text">Dataset</div>', unsafe_allow_html=True)

                        with col_dialogue_orig:
                            # Middle: button named after the uploaded dataset; opens the dataset dialog
                            name = st.session_state.data["name"]
                            name = name.name.split(".")[0]
                            if st.button(name):
                                show_orig_data()

                        with orig_text:
                            # Right: short textual summary about problem type/time series status
                            orig_text_summary_display = sm.summary_problemtype(
                                st.session_state.data["time_or_tab"],
                                st.session_state.data["target_variable"],
                                st.session_state.inference
                            )
                            st.markdown(f'<div class="custom-text">{orig_text_summary_display}</div>', unsafe_allow_html=True)


                        # --- Row 2: Feature Type Inference (label • open dialog • summary) ---
                        col_name, col_dialogue, col_text_inference = st.columns(3)

                        with col_name:
                            st.markdown('<div class="custom-text">Feature Type Inference</div>', unsafe_allow_html=True)

                        with col_dialogue:
                            # Button opens dialog showing the inference table and explanation
                            if st.button("Feature Type Inference Results"):
                                show_inference()

                        with col_text_inference:
                            # Short summary of inferred feature types
                            inference_text_summary_display = sm.summary_inference(st.session_state.inference)
                            st.markdown(f'<div class="custom-text">{inference_text_summary_display}</div>', unsafe_allow_html=True)


                        # --- Row 3: Imputation (label • open dialog • summary) ---
                        col_imputation, col_dialogue_imputation, col_text_imputation = st.columns(3)

                        with col_imputation:
                            st.markdown('<div class="custom-text">Imputation of Data</div>', unsafe_allow_html=True)

                        with col_dialogue_imputation:
                            # Button opens dialog highlighting imputed cells and explanation
                            if st.button("Imputation Results"):
                                show_imputation()

                        with col_text_imputation:
                            # Short summary of imputation coverage/columns
                            imputation_text_summary_display = sm.summary_imputation(st.session_state.original)
                            st.markdown(f'<div class="custom-text">{imputation_text_summary_display}</div>', unsafe_allow_html=True)


                        # --- Row 4: Anomaly Detection (label • open dialog • summary) ---
                        col_anomaly, col_dialogue_anomaly, col_anomaly_text = st.columns(3)

                        with col_anomaly:
                            st.markdown('<div class="custom-text">Anomaly Detection</div>', unsafe_allow_html=True)

                        with col_dialogue_anomaly:
                            # Button opens dialog showing anomalies and threshold explanation
                            if st.button("Anomaly Detection Results"):
                                show_anomalies()

                        with col_anomaly_text:
                            # Short summary of detected anomalies
                            anomaly_text_summary_display = sm.summary_anomaly(st.session_state.anomaly)
                            st.markdown(f'<div class="custom-text">{anomaly_text_summary_display}</div>', unsafe_allow_html=True)


                        # --- Row 5: Personal Data Detection (label • open dialog • summary) ---
                        col_personal, col_dialogue_personal, col_personal_text = st.columns(3)

                        with col_personal:
                            st.markdown('<div class="custom-text">Personalize Data Detection</div>', unsafe_allow_html=True)

                        with col_dialogue_personal:
                            # Button opens dialog showing PII detection results
                            if st.button("Personalize Data Detection Results"):
                                show_personal()

                        with col_personal_text:
                            # Short summary (counts/categories) of detected personal data
                            personal_text_summary_display = sm.summary_personal(st.session_state.personal_0)
                            st.markdown(f'<div class="custom-text">{personal_text_summary_display}</div>', unsafe_allow_html=True)


                        # --- Row 6: Summary (label • open dialog • static hint) ---
                        col_summary, col_dialogue_summary, col_text_summary = st.columns(3)

                        with col_summary:
                            st.markdown('<div class="custom-text">Summary</div>', unsafe_allow_html=True)

                        with col_dialogue_summary:
                            # Button opens a dashboard-like summary (charts for all steps)
                            if st.button("Summary"):
                                show_summary()

                        with col_text_summary:
                            # Static hint to indicate what the button shows
                            st.markdown(f'<div class="custom-text">Show Dataset Summary</div>', unsafe_allow_html=True)


                        # --- Row 7: Metadata (label • open dialog + set flags • static hint) ---
                        # col_meta, col_dialogue_meta, col_text_meta = st.columns(3)

                        # with col_meta:
                        #     st.markdown('<div class="custom-text">Create Meta-Data</div>', unsafe_allow_html=True)

                        # with col_dialogue_meta:
                        #     # Button opens metadata dialog and flips flags to enable upload step
                        #     if st.button("Metadata"):
                        #         show_meta_data()
                        #         st.session_state.starting_process = False
                        #         st.session_state.intermediate_process = False
                        #         st.session_state.ending_process = True
                        #         st.session_state.show_upload = True
                        #         # (Optional rerun logic was intentionally left commented)

                        # with col_text_meta:
                        #     # Static hint about metadata dialog
                        #     st.markdown(f'<div class="custom-text">Show Meta Data</div>', unsafe_allow_html=True)


                        # --- Row 8: Upload to Piveau (conditional row; appears after metadata step) ---
                        col_f, col_buttons, col_text = st.columns(3)

                        with col_f:
                            # Left label shown only when the upload step is enabled
                            #if st.session_state.show_upload:
                                st.markdown('<div class="custom-text">Upload</div>', unsafe_allow_html=True)

                        with col_buttons:
                            # Upload button triggers the final "Finished" dialog
                            #if st.session_state.show_upload:
                                if st.button("Upload"):
                                    show_finished()

                        with col_text:
                            # Instructional text for the upload step
                            #if st.session_state.show_upload:
                                st.markdown('<div class="custom-text">Upload your Data into Piveau</div>', unsafe_allow_html=True)

                                                #    
                                
                    
    
        with col2:
            
            
                if "data" not in st.session_state:
                    
                        st.write("")

                else:
                    st.write("")
                    
          
        # download font and set styles for the fixed bottom container                   
        st.markdown(
            """
            <style>
                .fixed-container {
                    @import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@200;400&display=swap');
                    background: rgba(0, 0, 0, 0); /* Fully transparent */
                    padding: 0px;
                    margin-top: 5%;
                    z-index: 1000; /* Ensures it's always above expanding content */
                }
                .big-white-text {
                    color: black;
                    font-family: 'Nunito Sans', sans-serif;
                    font-weight: 400;
                    text-align: left; /* Change to center if needed */

                    /* Responsive font size:
                    - minimum: 16px (1rem)
                    - preferred: 2vw + 0.5rem (scales with width)
                    - maximum: 24px (1.5rem)
                    */
                    font-size: clamp(1rem, 2vw + 0.5rem, 1.5rem);
                                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        
        with stylable_container(
            key="mlrdy-container",
            css_styles="""
                {
                
                position: fixed;                           /* Keep the bar fixed on screen even when the page scrolls */
                bottom: 1.5rem;                            /* Distance from the bottom of the viewport (responsive unit) */
                left: 50%;                                 /* Position the bar starting from the horizontal center */
                transform: translateX(-50%);               /* Shift it left by 50% of its own width to truly center it */

                width: 98%;                               /* Let the bar span the full available width */
                max-width: 1800px;                         /* But on large screens, don’t exceed 1200px for readability */
                min-width: 280px;                          /* Prevent the bar from collapsing on very small screens */

                /* Let the height adapt to content instead of being fixed */
                min-height: 4.5rem;                        /* Ensure a minimum height so it looks like a proper bar */

                background: rgba(248, 245, 245, 0.91);     /* Light background with a bit of transparency */
                border-radius: 0.625rem;                   /* Nicely rounded corners for a softer, modern look */
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); /* Soft shadow so it appears elevated above the page */

                padding: 0.5rem 1.5rem;                    /* Inner space so text/logos don’t touch the bar edges */

                box-sizing: border-box;                    /* Include padding and border inside width/height calculations */
                z-index: 9999;                             /* Keep the bar on top of other elements */

                display: block;                            /* Treat this as a normal block element; Streamlit manages layout inside */
            
                }
            """,
            ):
            
                # Load in text, Aalen Image and BW logo
                text_mlrdy,aalen,pic_bw= st.columns(3,gap="large")
                with text_mlrdy:
                    st.markdown(
                        '<div class="fixed-container"><p class="big-white-text">Make Your Dataset Machine-Learning Ready!</p></div>',
                        unsafe_allow_html=True,
                    ) 
  
              
    
                with aalen:
                        st.image("data/logos/Hochschule-aalen.png")

                with pic_bw:
                                
                        st.image("data/logos/logo-m-wm_de.svg",width=400)
                    

 
                
if __name__ == "__main__":
    main()

