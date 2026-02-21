from enum import Enum
import numpy as np
from pandas import Series
import logging
import streamlit as st
from openai import OpenAI
import nltk 
nltk.download('stopwords') 
nltk.download('punkt_tab')
import pandas as pd
from pyod.models.iforest import IForest
import torch
import torch.nn as nn
import pandas.io.formats.style
from sklearn.preprocessing import MinMaxScaler
import os
from flair.data import Sentence
from flair.models import TextClassifier
from collections import Counter
import random
import json
from datetime import datetime
import string
from services_featuretype_personal.services_LLM import FeaturizeFile,Load_RF,FeatureExtraction
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words

#     from src.models import TranAD
# else:
from src.demonstrators.ANOMALYDETECTIONHMTEAM.ANOMALYDETECTIONHMTEAM.src.models import TranAD

from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import torch

from services_featuretype_personal.services_LLM import FeaturizeFile,Load_RF
from IPython.display import HTML
from src.autogluon.constants import (
    BINARY,
    LARGE_DATA_THRESHOLD,
    MULTICLASS,
    MULTICLASS_UPPER_LIMIT,
    QUANTILE,
    REGRESS_THRESHOLD_LARGE_DATA,
    REGRESS_THRESHOLD_SMALL_DATA,
    REGRESSION,
)

logger = logging.getLogger(__name__)

#method for feature type inference

RandomForest_classes = ['numeric', 'categorical', 'datetime',
                        'sentence', 'url', 'embedded-number',
                        'list', 'not-generalizable', 'context-specific'
                        ]







def make_predictions(df_inference, model):
    df_inference = df_inference*1 # turn true and false entries to 0 and 1 (required for RF)
    dataFeaturized = FeaturizeFile(df_inference)
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



#Personal Data Prediciton
def make_prediciton_personal(df_personal): 

    classifier_personal = load_model()
    results = process_file(classifier_personal, df_personal)
    
    num_features = len(results)
    num_personal = results[results["Prediction"] == "personal"].shape[0]
    num_non_personal = num_features - num_personal

    metadata_personal = {
                    "Total Features": num_features,
                    "Personal Data Exists": num_personal > 0,
                    "Personal Features": num_personal,
                    "Non-Personal Features": num_non_personal,
                    "Personal Columns": results[results["Prediction"] == "personal"]["Column"].tolist(),
                    "Non-Personal Columns": results[results["Prediction"] == "non-personal"]["Column"].tolist()
                }
    metadata_df = pd.DataFrame([metadata_personal])

    return metadata_df,results


@st.cache_resource
def load_model():
    MODEL_PATH_PERSONAL = "data/models/best-model.pt"
    """Load the Flair model with error handling."""
    try:
        model = TextClassifier.load(MODEL_PATH_PERSONAL)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def predict_column(classifier, column_values):
    """Predict whether a column contains personal data based on the most common values."""
    value_counts = Counter(column_values)
    most_common_values = [value for value, _ in value_counts.most_common(10)]
    sample_values = most_common_values[:min(10, len(most_common_values))]

    sentences = [Sentence(str(value)) for value in sample_values]
    classifier.predict(sentences)
    
    predictions = [sentence.labels[0].value for sentence in sentences]
    return "personal" if "personal" in predictions else "non-personal"

def process_file(classifier, df):
    """Process the uploaded dataset and classify columns."""
    column_predictions = {}
    total_columns = len(df.columns)

    
    for i, column in enumerate(df.columns):
        column_values = df[column].dropna().astype(str)
        prediction_class = predict_column(classifier, column_values)
        column_predictions[column] = prediction_class
        

    result_df = pd.DataFrame(list(column_predictions.items()), columns=["Column", "Prediction"])
    return result_df

# Funktion: Metadaten hinzufügen
def add_metadata(key, metadata):
    if "metadata_store" not in st.session_state:
        st.session_state.metadata_store = {}
    st.session_state.metadata_store[key] = metadata

# Funktion: Einzelne Metadaten abrufen
def get_metadata(key):
    if "metadata_store" in st.session_state and key in st.session_state.metadata_store:
        return st.session_state.metadata_store[key]
    return None

# Funktion: Alle Metadaten ausgeben
def get_all_metadata():
    if "metadata_store" in st.session_state:
        return st.session_state.metadata_store
    return {}


def load_and_preprocess_data(data_anomalie_df,feature_type_inference):

    if feature_type_inference['prediction'].str.contains('datetime').any():
    
        row = feature_type_inference[feature_type_inference['prediction'] == "datetime"]
        
        data_anomalie_df = data_anomalie_df.drop([row["Attribute_name"].iloc[0]], axis=1) 
    
   
        # extract a list of ages
        #col_to_encode = feature_type_inference['Attribute_name'].tolist()
    onehot_encoder = ce.OneHotEncoder( use_cat_names=True)
    data_anomalie_df = onehot_encoder.fit_transform(data_anomalie_df)
    

    original_data = data_anomalie_df.copy()
    
    if not pd.api.types.is_numeric_dtype(data_anomalie_df.iloc[:, 0]):
        data_anomalie_df = data_anomalie_df.iloc[:, 1:]
    
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(data_anomalie_df), columns=data_anomalie_df.columns)
    
    return original_data, normalized_data, data_anomalie_df.shape[1]

def convert_to_windows(data, model):
    windows = []
    w_size = model.n_window
    for i in range(len(data)):
        if i >= w_size:
            w = data[i-w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
        windows.append(w)
    return torch.stack(windows)

def set_random_seed(seed=10000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def detect_anomalies_tranad(data, feats, percentile):
    set_random_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TranAD(feats).to(device)
    data_tensor = torch.tensor(data.values, dtype=torch.float).to(device)
    windows = convert_to_windows(data_tensor, model)
    
    model.eval()
    with torch.no_grad():
        losses, preds = [], []
        for window in windows:
            window = window.unsqueeze(0)
            elem = window[-1, :, :].view(1, window.shape[1], feats).to(device)
            z = model(window, elem)[1]
            loss = nn.MSELoss(reduction='none')(z, elem).mean(dim=[1, 2]).cpu().numpy()
            losses.append(loss)
            preds.append(z.squeeze(0).cpu().numpy())
    
    losses = np.array(losses).flatten()
    preds = np.array(preds)
    threshold = np.percentile(losses, percentile)
    anomalies = (losses > threshold).astype(int)
    return anomalies, losses, preds, threshold


def one_hot_encode_columns(
    df: pd.DataFrame,
    feature_inference_df: pd.DataFrame,
    df_anomalies: pd.DataFrame | None = None,
    df_personal: pd.DataFrame | None = None,
    target_col: str | None = None,
) -> pd.DataFrame:
    """
    One-hot-encodes categorical columns inferred from `feature_inference_df`, with optional
    row/column filtering based on `df_anomalies` and `df_personal`.

    The `target_col` (if provided) is NEVER modified:
      - never dropped (even if marked Personal),
      - never encoded or treated as datetime,
      - values & dtype are restored after all transforms.

    Parameters
    ----------
    df : pd.DataFrame
    feature_inference_df : pd.DataFrame
        Needs columns ["Attribute_name", "prediction"] with values like "datetime", "categorical".
    df_anomalies : pd.DataFrame, optional
        Last column is "Anomaly" (1 marks anomalous rows). Those rows are removed before encoding.
    df_personal : pd.DataFrame, optional
        Columns ["Column", "Prediction"]. Rows with Prediction == "Personal" drop that Column.
    target_col : str, optional
        Name of the target column in `df` that must not be altered.

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame.
    """
    df_out = df.copy()

    # --- Validate / stash the target for protection ---
    target_series = None
    if target_col is not None:
        if target_col not in df_out.columns:
            raise ValueError(f"`target_col`='{target_col}' not found in df.")
        # Stash original target (values + dtype) for restore at the end
        target_series = df_out[target_col].copy()

    # --- Remove personal columns first (if provided), but NEVER the target ---
    if df_personal is not None:
        required_cols = {"Column", "Prediction"}
        missing_req = required_cols - set(df_personal.columns)
        if missing_req:
            raise ValueError(f"`df_personal` is missing required columns: {sorted(missing_req)}")

        personal_cols = (
            df_personal.loc[df_personal["Prediction"].astype(str).str.lower() == "personal", "Column"]
            .dropna().astype(str).tolist()
        )
        if personal_cols:
            # Exclude target from being dropped
            cols_to_drop = [c for c in personal_cols if c in df_out.columns and c != target_col]
            if cols_to_drop:
                df_out = df_out.drop(columns=cols_to_drop, errors="ignore")

    # --- Remove anomalous rows next (if provided) ---
    if df_anomalies is not None and len(df_anomalies) > 0:
        # Use the last column (expected "Anomaly")
        anomaly_series = df_anomalies.iloc[:, -1]
        # Align by position if lengths match, else by index (missing -> non-anomaly)
        if len(anomaly_series) == len(df_out):
            mask = pd.Series(anomaly_series.values, index=df_out.index)
        else:
            mask = anomaly_series.reindex(df_out.index).fillna(0)
        mask_bool = mask.astype(int) == 1
        df_out = df_out.loc[~mask_bool]

        # Re-align target stash to current index if present
        if target_series is not None:
            target_series = target_series.reindex(df_out.index)

    # --- Expand datetime features (first inferred datetime only), but NEVER on target ---
    dt_rows = feature_inference_df[feature_inference_df["prediction"] == "datetime"]
    if len(dt_rows) >= 1:
        dt_col = dt_rows["Attribute_name"].iloc[0]
        if dt_col != target_col and dt_col in df_out.columns:
            # Assumes `encode_datetime_features` is defined elsewhere
            df_out = encode_datetime_features(df_out, dt_col)

    # --- Determine categorical columns to encode (exclude target and missing) ---
    row_cat = feature_inference_df[feature_inference_df["prediction"] == "categorical"]
    columns_to_encode = (
        row_cat["Attribute_name"].astype(str).tolist()
        if "Attribute_name" in row_cat
        else []
    )
    columns_to_encode = [c for c in columns_to_encode if c in df_out.columns and c != target_col]

    if not columns_to_encode:
        # Final safeguard: restore target exactly as-original
        if target_series is not None:
            df_out[target_col] = target_series
        return df_out

    # --- One-hot encode (non-target only) ---
    encoder = ce.OneHotEncoder(cols=columns_to_encode, use_cat_names=True)
    encoded_df = encoder.fit_transform(df_out)

    # --- FINAL: restore target exactly (values + dtype), ensuring no accidental change ---
    if target_series is not None:
        encoded_df[target_col] = target_series

    return encoded_df


def encode_datetime_features(df, column_name):
    # Ensure the column is in datetime format
    df[column_name] = pd.to_datetime(df[column_name])

    # Extract basic datetime features
    df[f'{column_name}_hour'] = df[column_name].dt.hour
    df[f'{column_name}_day'] = df[column_name].dt.day
    df[f'{column_name}_dayofweek'] = df[column_name].dt.dayofweek
    df[f'{column_name}_month'] = df[column_name].dt.month
    df[f'{column_name}_is_weekend'] = df[column_name].dt.dayofweek.isin([5, 6]).astype(int)

    # Cyclical encoding
    df[f'{column_name}_hour_sin'] = np.sin(2 * np.pi * df[f'{column_name}_hour'] / 24)
    df[f'{column_name}_hour_cos'] = np.cos(2 * np.pi * df[f'{column_name}_hour'] / 24)

    df[f'{column_name}_dayofweek_sin'] = np.sin(2 * np.pi * df[f'{column_name}_dayofweek'] / 7)
    df[f'{column_name}_dayofweek_cos'] = np.cos(2 * np.pi * df[f'{column_name}_dayofweek'] / 7)

    #df[f'{column_name}_month_sin'] = np.sin(2 * np.pi * df[f'{column_name}_month'] / 12)
    #df[f'{column_name}_month_cos'] = np.cos(2 * np.pi * df[f'{column_name}_month'] / 12)
    df= df.drop(columns=[column_name])
    

    return df

def one_hot_encode_df(df):
    columnames = list(df.columns)
    for c in df.columns:
        type = infer_problem_type(df[c])
        if type not in ["multiclass","binary"]:
            columnames.remove(c)
    #columnames.remove(get_metadata("Target variable"))
    #print(columnames) #debug
    return one_hot_encode_columns(df, columnames)

def detect_pictures(format):
    image_formats = [
    "png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif", "svg", "webp",
    "ico", "heif", "heic", "raw", "eps", "pdf", "psd", "ai", "xcf",
    "jp2", "j2k", "jxr", "hdp", "wdp", "dds", "exr", "pbm", "pgm",
    "ppm", "pnm", "pfm"
    ]
    name = name.name.split(".")[1]

    if name in image_formats:
        return "picture"


    return "no_picture"

def detect_anomalies(df, model=IForest(), contamination=0.1, scale_data=True):
    """
    Detect anomalies in a given DataFrame using PyOD.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - model (pyod model, default=IForest()): The anomaly detection model to use.
    - contamination (float, default=0.1): Expected proportion of anomalies.
    - scale_data (bool, default=True): Whether to standardize the data before detection.
    
    Returns:
    - pd.DataFrame: Original DataFrame with an added 'Anomaly' column (0=normal, 1=anomaly).
    """
    df_clean = df.select_dtypes(include=['number']).dropna()  # Keep only numerical columns
    if df_clean.empty:
        raise ValueError("No valid numerical data found for anomaly detection.")

    # Scale the data if required
    if scale_data:
        scaler = StandardScaler()
        data = scaler.fit_transform(df_clean)
   
    # Fit the model and predict anomalies
    model.set_params(contamination=contamination)
    model.fit(data)
    df_clean["Anomaly"] = model.predict(data)  # 0 = normal, 1 = anomaly
    treshhold = model.threshold_
    

    return df_clean,treshhold


def highlight_columns(df: pd.DataFrame, columns: list) -> pd.io.formats.style.Styler:
    if isinstance(columns, str):
        columns = [columns]  # Convert single column name to list

    def highlight(val, column):
        if column in columns:
            return "background-color: lightgreen"
        return "background-color: white"

    styled_df = df.style.apply(lambda x: [highlight(v, x.name) for v in x], axis=0)
    
    # Set font size for all cells
    styled_df = styled_df.set_properties(**{'font-size': '30px'})

    return styled_df



def highlight_imputed_cells(original_df: pd.DataFrame, imputed_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    
    
    missing_mask = original_df.isna()
    
    def highlight(val, mask):
        return "background-color: lightgreen" if mask else "background-color: white"
    
    styled_df = imputed_df.style.apply(lambda x: [highlight(v, missing_mask.loc[x.name, col]) for col, v in x.items()], axis=1)
    
    return styled_df


def highlight_personal_cells(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    
    def highlight_personal(val):
        if val == "personal":
            return "background-color: lightgreen" 
        elif val == "non-personal":
            return "background-color: white" 
        else:
            return ""

    styled_df = df.style.applymap(highlight_personal, subset=['Prediction'])
    
    return styled_df

def highlight_anomaly_cells(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    
    def highlight_anomaly(val):
        if val == 0:
            return "background-color: white" 
        elif val == 1:
            return "background-color: lightgreen" 
        else:
            return ""

    styled_df = df.style.applymap(highlight_anomaly, subset=['Anomaly'])
    
    return styled_df



def extract_metadata(df: pd.DataFrame,feature_type_inference) -> dict:
   
    #feature_type_inference = pd.DataFrame(feature_type_inference)
    metadata = {
        "Number of Rows": df.shape[0],
        "Number of Columns": df.shape[1],
        "Column Names": list(df.columns),
        "Missing Values": df.isnull().sum().to_dict(),
        "Duplicate Rows": df.duplicated().sum(),
        "Unique Values Per Column": df.nunique().to_dict(),
    }

    # Initialize DataFrame to store feature-level statistics
    df_features = pd.DataFrame()

    for i, col in enumerate(df.columns):
        current_column = df[col]
        
        # General statistics
        total_vals = len(current_column)
        num_nans = current_column.isna().sum()
        perc_nans = (num_nans / total_vals) * 100 if total_vals > 0 else 0
        num_of_dist_val = current_column.nunique()
        perc_dist_val = (num_of_dist_val / total_vals) * 100 if total_vals > 0 else 0
        

        row = feature_type_inference[feature_type_inference['Attribute_name'] == current_column.name]
        
        if  row["prediction"].iloc[0]=="numeric":
             mean = current_column.mean()
             
        else :
            np.nan
       
        if row["prediction"].iloc[0]=="numeric":
             std_dev = current_column.std()     
        else :
            np.nan

        if row["prediction"].iloc[0]=="numeric":
            min_val = current_column.min() 
        else:
            np.nan

        if row["prediction"].iloc[0]=="numeric":
            max_val = current_column.max()
        else:
            np.nan
        
        # Random samples
        non_nan_values = current_column.dropna().tolist()
        random_samples = random.sample(non_nan_values, min(5, len(non_nan_values))) if non_nan_values else [np.nan] * 5

        # String-based metrics (for text columns)
        if row["prediction"].iloc[0]=="sentence" or row["prediction"].iloc[0]=="categorical":
            word_counts = current_column.dropna().apply(lambda x: len(str(x).split()))
            stopword_counts = current_column.dropna().apply(lambda x: sum(1 for word in str(x).split() if word.lower() in stop_words))
            char_counts = current_column.dropna().apply(lambda x: len(str(x)))
            whitespace_counts = current_column.dropna().apply(lambda x: str(x).count(" "))
            delim_counts = current_column.dropna().apply(lambda x: sum(str(x).count(d) for d in string.punctuation))
            is_long_sentence = (word_counts > 15).sum() #we want to retrieve the cars with prices less than 20,000 you might try the followin

            mean_word_count = word_counts.mean()
            std_dev_word_count = word_counts.std()
            mean_stopword_total = stopword_counts.mean()
            stdev_stopword_total = stopword_counts.std()
            mean_char_count = char_counts.mean()
            stdev_char_count = char_counts.std()
            mean_whitespace_count = whitespace_counts.mean()
            stdev_whitespace_count = whitespace_counts.std()
            mean_delim_count = delim_counts.mean()
            stdev_delim_count = delim_counts.std()
        else:
            mean_word_count = std_dev_word_count = mean_stopword_total = stdev_stopword_total = np.nan
            mean_char_count = stdev_char_count = mean_whitespace_count = stdev_whitespace_count = np.nan
            mean_delim_count = stdev_delim_count = is_long_sentence = np.nan
        
        df_features = FeaturizeFile(df)
      
    #metadata["Per Column Statistics"] = df_features

    return metadata,df_features


def metadata_to_dataframe(metadata):
    # Convert the dictionary (excluding the nested DataFrame) into a DataFrame
    meta_df = pd.DataFrame.from_dict({k: [v] for k, v in metadata.items() if k != "Per Column Statistics"})
    
    # Extract the 'Per Column Statistics' DataFrame if it exists
    df_features = metadata.get("Per Column Statistics")
    
    # Concatenate both DataFrames if df_features exists
    if isinstance(df_features, pd.DataFrame):
        result_df = pd.concat([meta_df.T, df_features], axis=1)
    else:
        result_df = meta_df.T
    
    return result_df



def impute_mean_mode(df_to_impute):
    """Function to impute missing values with mean for numeric columns and mode for categorical columns."""
    imputation_info = []
    for col in df_to_impute.columns:
        if df_to_impute[col].dtype in [np.float64, np.int64]:
            mean_value = df_to_impute[col].mean()
            df_to_impute[col].fillna(mean_value, inplace=True)
            imputation_info.append((col, 'Mean', mean_value))
        else:
            mode_value = df_to_impute[col].mode()[0]
            df_to_impute[col].fillna(mode_value, inplace=True)
            imputation_info.append((col, 'Mode', mode_value))
    return df_to_impute, imputation_info




# Metadaten eines DataFrame berechnen in einem dict
def calculate_dataframe_metadata(df):
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Die Eingabe muss ein gültiger Pandas-DataFrame sein.")

    metadata = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
    }

    return metadata


# enum for the different dataset types
class Dataset_Type(Enum): #enum to manually set the type of the dataset
    DEFAULT = "default"
    TIME_SERIES = "time series dataset"
    TABULAR = "tabular dataset"


#method to get all possible problem types of a tabular dataset
def get_problem_type_list():
    problem_type_list = []
    problem_type_list.append(BINARY)
    problem_type_list.append(MULTICLASS)
    problem_type_list.append(REGRESSION)
    return problem_type_list

def check_time_series(df: pd.DataFrame,feature_df:pd.DataFrame) -> str:
    """
    Iterates through a DataFrame, extracts each column and its values,
    checks if any column is of datetime type, and returns 'Time series'
    if a datetime column is found.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        str: 'Time series' if a datetime column is found, else 'No time series'.
    """

    if feature_df['prediction'].str.contains('datetime').any():
            
            return "Time series"
    else:
            return "Tabular"

#method to slect which problem type is possibly
def infer_problem_type(y: Series, silent=False) -> str: 
    """Identifies which type of prediction problem we are interested in (if user has not specified).
    Ie. binary classification, multi-class classification, or regression.
    """
    # treat None, NaN, INF, NINF as NA
    y = y.replace([np.inf, -np.inf], np.nan, inplace=False)
    y = y.dropna()
    num_rows = len(y)

    if num_rows == 0:
        raise ValueError("Label column cannot have 0 valid values")

    unique_values = y.unique()

    if num_rows > LARGE_DATA_THRESHOLD:
        regression_threshold = (
            REGRESS_THRESHOLD_LARGE_DATA  # if the unique-ratio is less than this, we assume multiclass classification, even when labels are integers
        )
    else:
        regression_threshold = REGRESS_THRESHOLD_SMALL_DATA

    unique_count = len(unique_values)
    if unique_count == 2:
        problem_type = BINARY
        reason = "only two unique label-values observed"
    elif y.dtype.name in ["object", "category", "string"]:
        problem_type = MULTICLASS
        reason = f"dtype of label-column == {y.dtype.name}"
    elif np.issubdtype(y.dtype, np.floating):
        unique_ratio = unique_count / float(num_rows)
        if (unique_ratio <= regression_threshold) and (unique_count <= MULTICLASS_UPPER_LIMIT):
            try:
                can_convert_to_int = np.array_equal(y, y.astype(int))
                if can_convert_to_int:
                    problem_type = MULTICLASS
                    reason = "dtype of label-column == float, but few unique label-values observed and label-values can be converted to int"
                else:
                    problem_type = REGRESSION
                    reason = "dtype of label-column == float and label-values can't be converted to int"
            except:
                problem_type = REGRESSION
                reason = "dtype of label-column == float and label-values can't be converted to int"
        else:
            problem_type = REGRESSION
            reason = "dtype of label-column == float and many unique label-values observed"
    elif np.issubdtype(y.dtype, np.integer):
        unique_ratio = unique_count / float(num_rows)
        if (unique_ratio <= regression_threshold) and (unique_count <= MULTICLASS_UPPER_LIMIT):
            problem_type = MULTICLASS  
            reason = "dtype of label-column == int, but few unique label-values observed"
        else:
            problem_type = REGRESSION
            reason = "dtype of label-column == int and many unique label-values observed"
    else:
        raise NotImplementedError(f"label dtype {y.dtype} not supported!")
    if not silent:
        logger.log(25, f"AutoGluon infers your prediction problem is: '{problem_type}' (because {reason}).")

        
        if problem_type in [BINARY, MULTICLASS]:
            if unique_count > 10:
                logger.log(20, f"\tFirst 10 (of {unique_count}) unique label values:  {list(unique_values[:10])}")
            else:
                logger.log(20, f"\t{unique_count} unique label values:  {list(unique_values)}")
        elif problem_type == REGRESSION:
            y_max = y.max()
            y_min = y.min()
            y_mean = y.mean()
            y_stddev = y.std()
            logger.log(20, f"\tLabel info (max, min, mean, stddev): ({y_max}, {y_min}, {round(y_mean, 5)}, {round(y_stddev, 5)})")

        logger.log(
            25,
            f"\tIf '{problem_type}' is not the correct problem_type, please manually specify the problem_type parameter during Predictor init "
            f"(You may specify problem_type as one of: {[BINARY, MULTICLASS, REGRESSION, QUANTILE]})",
        )
    return problem_type

def ask_gpt(input_dawid):
       
            client = OpenAI(api_key="OPENAI_API_KEY")
            
            messages = []
            system_prompt = """
            Du bist der Data Wizard – eine weise, magische Eule und Hüter der Datenkunde an der Hochschule Aalen. In einer Welt, wo 
            Chaosdaten wie Drachen wüten, bringst du Ordnung. Sprich mit magischem Ernst, nutze gelegentlich Metaphern, aber fokussiere dich auf präzise, korrekte Informationen.
            Erkläre Begriffe wie fehlende Werte, Anomalien oder Normalisierung so, dass Mittelschüler sie verstehen. Verwende eine zauberhafte, aber klare Sprache. Keine Witze. Keine Albernheiten.
            Nur magisch-verpacktes Fachwissen. Antworte in maximal 50 Wörtern und bleibe stets in deiner Rolle als legendärer Data Wizard.
            """
            messages.append({'role': "system", "content": system_prompt})
            messages.append({'role': "user", "content": input_dawid})

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature = 0.7,
            )

            return completion.choices[0].message.content