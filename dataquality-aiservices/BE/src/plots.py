import plotly.express as px
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn import tree
import pickle
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def plot_inference_bar(plot_inference_df):
    # Count occurrences of each prediction type
    prediction_counts = plot_inference_df['prediction'].value_counts().reset_index()
    prediction_counts.columns = ['prediction', 'count']

    # Create the stacked bar chart
    fig = px.bar(
        prediction_counts, 
        x='prediction', 
        y='count', 
        title='Distribution of Feature Types',
        labels={'prediction': 'Feature Type', 'count': 'Count'},
        color='prediction'
    )

    return fig

def plot_imputation_bar(df, feature_df):
    # Identify columns with NaN values
    nan_columns = [col for col in df.columns if df[col].isna().any()]

    # Sicherstellen, dass mindestens eine NaN-Spalte existiert
    if len(nan_columns) >= 2:
        column_to_plot = nan_columns[1]
    elif len(nan_columns) == 1:
        column_to_plot = nan_columns[0]
    else:
        # Notfallauswahl: erste nicht-leere Spalte
        fallback_cols = [col for col in df.columns if not df[col].dropna().empty]
        if not fallback_cols:
            raise ValueError("Keine geeignete Spalte zum Plotten gefunden.")
        column_to_plot = fallback_cols[0]

    # Drop NaN values for plotting
    data = df[column_to_plot].dropna()

    if data.empty:
        raise ValueError(f"No valid data to plot for column: {column_to_plot}")

    row_pt = feature_df[feature_df['Attribute_name'] == column_to_plot]

    if row_pt.empty:
        raise ValueError(f"Keine Feature-Information für Spalte '{column_to_plot}' gefunden.")

    # Determine if the column is categorical or numerical
    if row_pt["prediction"].iloc[0] == 'categorical':
        ref_value = data.mode()[0]
        is_categorical = True
    else:
        ref_value = data.median()
        is_categorical = False

    fig = px.histogram(data, x=column_to_plot, title=f'Imputation of {column_to_plot}')

    if not is_categorical:
        fig.add_vline(x=ref_value, line_dash="dash", line_color="red", annotation_text=f"Median: {ref_value}")
    else:
        fig.add_annotation(
            text=f"Mode: {ref_value}",
            x=ref_value,
            y=data.value_counts().max(),
            showarrow=True,
            arrowhead=2,
            arrowcolor="red"
        )

    return fig


def plot_personal_pie(plot_personal_df):
    
    personal_features = plot_personal_df["Personal Features"].iloc[0]
    non_personal_features = plot_personal_df['Non-Personal Features'].iloc[0]
    
    fig = px.pie(
        names=["Personal Features", "Non-Personal Features"],
        values=[personal_features, non_personal_features],
        title='Distribution of Personal Data',
        labels={"Personal Features": "Personal Features", "Non-Personal Features": "Non-Personal Features"}
    )
    
    return fig

def plot_anomaly_pie(plot_anomaly_df):
    anomaly_counts = plot_anomaly_df['Anomaly'].value_counts()

    # Define custom labels
    labels = {1: "Anomaly", 0: "No Anomaly"}
    labeled_index = [labels[i] for i in anomaly_counts.index]

    # Create the pie chart
    fig = px.pie(
        values=anomaly_counts, 
        names=labeled_index, 
        title="Anomaly Distribution",
        color_discrete_sequence=['lightcoral', 'lightskyblue']
    )
    return fig


def plot_randomforest_trees(df_randomforest_plot,target_variable):
    model_rf =  pickle.load(open("data/models/RandomForest.pkl", 'rb'))
    fn=df_randomforest_plot.columns
    cn=model_rf.classes_
    fig_rf, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
    tree.plot_tree(model_rf.estimators_[0],
                feature_names = fn, 
                #class_names=cn,
                filled = True)
    return fig_rf

def plot_boxplots(df):

    fig = px.box(df.melt(var_name="Feature", value_name="Value"), x="Feature", y="Value")

    return fig

def plot_histograms(df: pd.DataFrame, nbins: int = 30, max_cols: int = 12):
    figs = []
    if df is None or df.empty:
        return figs

    num_cols = df.select_dtypes(include="number").columns.tolist()[:max_cols]
    for col in num_cols:
        fig = px.histogram(df, x=col, nbins=nbins, title=f"Distribution: {col}")
        fig.update_layout(width=900, height=450, title_x=0.5)
        figs.append(fig)

    return figs

def plot_correlation_matrix(df: pd.DataFrame):
    if df is None or df.empty:
        return None

    num_df = df.select_dtypes(include="number")
    if num_df.empty or num_df.shape[1] < 2:
        return None

    corr = num_df.corr(numeric_only=True)

    corr = corr.astype(float)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix (numeric features)"
    )
    fig.update_layout(width=900, height=700, title_x=0.5)
    return fig

def plot_feature_importance(df_encoded: pd.DataFrame, target: str, top_n: int = 20):
    if df_encoded is None or df_encoded.empty:
        return None
    if not target or target not in df_encoded.columns:
        return None

    X = df_encoded.drop(columns=[target], errors="ignore")
    y = df_encoded[target]

    if X.empty or y is None or y.dropna().empty:
        return None

    y_is_numeric = pd.api.types.is_numeric_dtype(y)
    unique_count = y.nunique(dropna=True)

    if not y_is_numeric:
        y_model = LabelEncoder().fit_transform(y.astype(str))
        model = RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=-1)
    else:
        if unique_count <= 15:
            y_model = y.astype(int, errors="ignore")
            model = RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=-1)
        else:
            y_model = y.astype(float, errors="ignore")
            model = RandomForestRegressor(n_estimators=150, random_state=0, n_jobs=-1)

    model.fit(X, y_model)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    imp_df = (
        importances.sort_values(ascending=True)
        .tail(top_n)
        .reset_index()
        .rename(columns={"index": "Feature", 0: "Importance"})
    )
    imp_df.columns = ["Feature", "Importance"]

    fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top {top_n} Feature Importances for '{target}'",
        text="Importance",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(width=900, height=650, title_x=0.5, yaxis={"categoryorder": "total ascending"})
    return fig



