import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from services_featuretype_personal.services_LLM import FeaturizeFile,FeatureExtraction
#from treeinterpreter import treeinterpreter as ti
import pickle
#import shap


# summary_problemtype
# --------------------
# Purpose: Determine the high-level ML problem type (Regression, Classification,
# Time-Series Forecasting, or Time-Series Classification) based on
# the overall data modality (problem_type) and the inferred data type of
# the target variable.
# Inputs:
# - problem_type (str): "Tabular" or "Time series".
# - target_variable (str): Column name of the target in the dataset.
# - problem_type_inference (pd.DataFrame): Must contain columns
# ["Attribute_name", "prediction"], where "prediction" is one of
# {"numeric", "categorical", "datetime"} for each attribute.
# Returns:
# - str: Human-readable description of the ML problem type. Empty string if
# no decision could be made.

def summary_problemtype(problem_type,target_variable,problem_type_inference):
    row_pt = problem_type_inference[problem_type_inference['Attribute_name'] == target_variable]

    
    if problem_type=="Tabular" :
        if row_pt["prediction"].iloc[0]=="numeric":
            return "Machine Learning Problem: \n\n Regression"
        if row_pt["prediction"].iloc[0]=="categorical":
            return "Machine Learning Problem: \n\n Classification"
        
        
    elif problem_type=="Time series" :
        if row_pt["prediction"].iloc[0]=="numeric":
            return "Machine Learning Problem: \n\n Time-Series Forecasting"
        if row_pt["prediction"].iloc[0]=="categorical":
            return "Machine Learning Problem: \n\n Time series classification"
        if row_pt["prediction"].iloc[0]=="datetime":
            return "Machine Learning Problem: \n\n Time-Series Forecasting"

    return ""

# summary_problemtype_de
# ----------------------
# Purpose: German-language variant of summary_problemtype.
# Inputs/Returns: Same as summary_problemtype, but strings are in German.
def summary_problemtype_de(problem_type,target_variable,problem_type_inference):
    row_pt = problem_type_inference[problem_type_inference['Attribute_name'] == target_variable]
    
    if problem_type=="Tabular" :
        if row_pt["prediction"].iloc[0]=="numeric":
            return "Machine Learning Problem: \n\n Regression"
        if row_pt["prediction"].iloc[0]=="categorical":
            return "Machine Learning Problem: \n\n Klassifizierung"
        
        
    elif problem_type=="Time series" :
        if row_pt["prediction"].iloc[0]=="numeric":
            return "Machine Learning Problem: \n\n Zeitreihenprognose"
        if row_pt["prediction"].iloc[0]=="categorical":
            return "Machine Learning Problem: \n\n Zeitreihen Klassifizierung"
        if row_pt["prediction"].iloc[0]=="datetime":
            return "Machine Learning Problem: \n\n Zeitreihenprognose"

    return ""

# summary_imputation
# ------------------
# Purpose: Summarize how many missing values are present in a DataFrame.
# Inputs:
# - df_check_summary (pd.DataFrame): Any DataFrame to be checked for NaNs.
# Returns:
# - str: English message with total count of missing values.
def summary_imputation(df_check_summary: pd.DataFrame):
    
    num_nans = df_check_summary.isna().sum().sum()
    string_imp_nans ='{} Missing Values Detected'.format(num_nans)



    return string_imp_nans

# summary_imputation_de
# ---------------------
# Purpose: German-language variant of summary_imputation.
# Returns a German message with the total count of missing values.
def summary_imputation_de(df_check_summary: pd.DataFrame):
    
    num_nans = df_check_summary.isna().sum().sum()
    string_imp_nans ='{} fehlende Werte gefunden'.format(num_nans)


    return string_imp_nans

# summary_inference
# -----------------
# Purpose: Count the number of unique inferred data types in a DataFrame that
# contains a column named "prediction" describing inferred types per
# attribute.
# Inputs:
# - df_check_summary (pd.DataFrame): Must include a "prediction" column.
# Returns:
# - str: English message with number of unique types detected.
def summary_inference(df_check_summary: pd.DataFrame):
    
    num_classes = df_check_summary["prediction"].nunique()
    string_inference_class ='{} Different Types Detected'.format(num_classes)


    return string_inference_class

# summary_inference_de
# --------------------
# Purpose: German-language variant of summary_inference.
# Returns a German message with number of unique types detected.
def summary_inference_de(df_check_summary: pd.DataFrame):
    
    num_classes = df_check_summary["prediction"].nunique()
    string_inference_class ='{} Datentypen erkannt'.format(num_classes)


    return string_inference_class



# summary_anomaly
# ---------------
# Purpose: Count the number of rows flagged as anomalies.
# Inputs:
# - df_check_summary (pd.DataFrame): Must include an "Anomaly" column with
# binary indicators (1=anomaly, 0=normal).
# Returns:
# - str: English message with number of anomalies detected
def summary_anomaly(df_check_summary: pd.DataFrame):

    num_anomalies=(df_check_summary["Anomaly"] == 1).sum()
    string_anomalies= "{} Anomalies Detected".format(num_anomalies)

    return string_anomalies

# summary_anomaly_de
# ------------------
# Purpose: German-language variant of summary_anomaly.
# Returns a German message with number of anomalies detected.
def summary_anomaly_de(df_check_summary: pd.DataFrame):

    num_anomalies=(df_check_summary["Anomaly"] == 1).sum()
    string_anomalies= "{} Anomalien identifiziert".format(num_anomalies)

    return string_anomalies




# summary_personal
# ----------------
# Purpose: Report the number of personal features detected, assuming the
# DataFrame carries an aggregated count at row index 0, column
# 'Personal Features'.
# Returns an English message with this count.

def summary_personal(df_check_summary: pd.DataFrame):

    num_personal=df_check_summary.loc[0, 'Personal Features']
    string_personal= "{} Personal Features Detected".format(num_personal)

    return string_personal

# summary_personal_de
# -------------------
# Purpose: German-language variant of summary_personal.
# Returns a German message with the detected count.
def summary_personal_de(df_check_summary: pd.DataFrame):

    num_personal=df_check_summary.loc[0, 'Personal Features']
    string_personal= "{} persönliche Merkmale erkannt".format(num_personal)

    return string_personal


# explanation_imputation
# ----------------------
# Purpose: Explain missing-value imputation strategy per column, referencing
# inferred feature types to decide mean (numeric) vs mode (categorical).
# Inputs:
# - df_explain_imp (pd.DataFrame): Original data used for imputation checks.
# - df_inference (pd.DataFrame): Must include columns ["Attribute_name",
# "prediction"] where prediction is the inferred type for the attribute.
# Returns:
# - str: Detailed German explanation summarizing missing values per affected
# column and the chosen imputation method.
def explanation_imputation(df_explain_imp,df_inference):
    stat_missing_values_relative= []
    stat_missing_values_absolut= []
    col_name= []

    columns_missing_values =df_explain_imp.columns[df_explain_imp.isnull().any()].tolist()
    missing_df_mask= df_explain_imp[columns_missing_values]
    for i, col in enumerate(missing_df_mask.columns):
        
        current_column = df_explain_imp[col]
        col_name.append(col)
        
        total_vals = len(current_column)
        num_nans = current_column.isna().sum()
        stat_missing_values_absolut.append(num_nans)
        perc_nans = (num_nans / total_vals) * 100 if total_vals > 0 else 0
        stat_missing_values_relative.append(np.round(perc_nans,2))

    list_with_explanation = []
    for i in range(len(col_name)):
        
        row = df_inference[df_inference['Attribute_name'] == col_name[i]]
        if row["prediction"].iloc[0]=="numeric":
            column_type= ["mean",row["prediction"].iloc[0]]
        else:

            column_type= ["mode",row["prediction"].iloc[0]]
        stat_for_colummns= f" • Spalte {col_name[i]} enthält insgesamt {stat_missing_values_absolut[i]} fehlende Werte, was {stat_missing_values_relative[i]} % aller Werte entspricht. Da diese Spalte {column_type[1]} ist, wurde sie mit dem {column_type[0]} imputiert. "
        list_with_explanation.append(stat_for_colummns)

    num_missing_values=df_explain_imp.isnull().sum().sum()
    
    nested_string= " \n\n ".join(list_with_explanation)
    return f"In deinem Datensatz wurden insgesamt {num_missing_values} fehlende Werte erkannt, die sich in den Spalten {columns_missing_values} befanden. \n\n "+nested_string


# explanation_feature_type_inference
# ----------------------------------
# Purpose: Visualize the top-10 most important engineered features, using a
# pre-trained RandomForest model and a featurization pipeline.
# Inputs:
# - df (pd.DataFrame): Raw dataframe to featurize.
# - target_column (str): Column name of the target (not directly used here but
# kept for interface symmetry/extension).
# Side Effects:
# - Loads a model from "data/models/RandomForest.pkl".
# Returns:
# - plotly.graph_objects.Figure: Horizontal bar chart of feature importances
# (ascending order) for the top 10 features.
def explanation_feature_type_inference(df, target_column):
    model_rf = pickle.load(open("data/models/RandomForest.pkl", 'rb'))
    
    # Feature importance
    dataFeaturized = FeaturizeFile(df)
    dataFeaturized1 = FeatureExtraction(dataFeaturized)
    dataFeaturized1.columns = dataFeaturized1.columns.astype(str)
    
    feature_importance = pd.Series(model_rf.feature_importances_, index=dataFeaturized1.columns)
    
    # Select top 10 most important features
    top_features = feature_importance.nlargest(10)
    
    # Sort the top features in ascending order
    top_features = top_features.sort_values(ascending=True)
    
    # Visualize feature importance using Plotly
    fig = px.bar(top_features, 
                 x=top_features.values, 
                 y=top_features.index, 
                 orientation='h', 
                 labels={'index': 'Features', 'value': 'Importance'},
                 title='Top 10 Most Important Features (Ascending Order)')
    
    return fig


# explanation_anomaly
# -------------------
# Purpose: Produce a short English explanation of the anomaly detection method
# and report the threshold, differing by data modality.
# Inputs:
# - treshold (float): Model threshold value (scientific notation shown).
# - data_type (str): "Tabular" -> PYOD iForest; "Time series" -> TransAD
# Returns:
# - str or np.nan: Explanation text, or NaN if data_type unknown.
def explanation_anomaly(treshold,data_type):

    if data_type == "Tabular":
        return f"For anomaly detection, PYOD was used, and it detected anomalies with the iForest model. The treshhold is { treshold:.2e}"
    elif data_type == "Time series":
        return f"For anomaly detection, TransAD was used, and it detected anomalies with a Transformer model. The treshhold is {treshold:.2e}"
    else:
        return np.nan

    return np.nan

# describe_dataframe
# ------------------
# Purpose: Provide a quick descriptive summary for all columns of a DataFrame.
# Inputs:
# - df (pd.DataFrame): The DataFrame to summarize.
# Returns:
# - pd.DataFrame: The result of df.describe(include="all").

def describe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    
    summary_df = df.describe(include="all")
    return summary_df

# meta_data_from_demos
# --------------------
# Purpose: Build a compact metadata table summarizing (1) inferred feature types
# per column, (2) indexes of rows with missing values, (3) indexes of
# rows flagged as anomalies, and (4) columns predicted to contain
# personal data.
# Inputs:
# - feature_inference_df (pd.DataFrame): Columns ["Attribute_name",
# "prediction"].
# - imputation_df (pd.DataFrame): Dataset used to locate rows with NaNs.
# - anomaly_df (pd.DataFrame): Must include an "Anomaly" column (1/0).
# - personal_df (pd.DataFrame): Columns ["Column", "Prediction"].
# Returns:
# - pd.DataFrame: Indexed by the categories, with a single column "value"
# storing dicts/lists for each category.

def meta_data_from_demos(feature_inference_df,imputation_df,anomaly_df,personal_df):
    
    attribute_dict = dict(zip(feature_inference_df["Attribute_name"], feature_inference_df["prediction"]))
    missing_indexes = imputation_df[imputation_df.isnull().any(axis=1)].index.tolist()
    anomaly_indexes = anomaly_df[anomaly_df["Anomaly"] == 1].index.tolist()
    personal_dict = dict(zip(personal_df["Column"], personal_df["Prediction"]))

    # Creating the DataFrame with separate rows
    df_missing_info = pd.DataFrame({
        "Category": [
            "Columns with their Feature Type",
            "Rows with Missing Values",
            "Rows with Anomalies",
            "Columns with Personal Data"
        ],
        "value": [attribute_dict, missing_indexes, anomaly_indexes, personal_dict]
    })

    # Set "Category" column as the index
    df_missing_info.set_index("Category", inplace=True)

    # Return the DataFrame
    return df_missing_info