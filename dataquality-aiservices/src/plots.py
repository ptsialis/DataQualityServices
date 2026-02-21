import plotly.express as px
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn import tree
import pickle
import streamlit as st
import plotly.graph_objects as go
from pandas.api.types import is_numeric_dtype

# Define a custom color palette
custom_colors = ['#1f77b4', '#aec7e8','#ff7f0e' ,'#2ca02c', '#98df8a' ] 


def plot_inference_bar(plot_inference_df):
    prediction_counts = plot_inference_df['prediction'].value_counts().reset_index()
    prediction_counts.columns = ['prediction', 'count']

    fig = px.bar(
        prediction_counts,
        x='prediction',
        y='count',
        title='Distribution of Feature Types',
        labels={'prediction': 'Feature Type', 'count': 'Count'},
        color='prediction',
        color_discrete_sequence=custom_colors
    )

    fig.update_layout(
        title_font_size=24,
        width=250,
        height=300
    )

    return fig


# def plot_imputation_bar(df, feature_df):
#     nan_columns = [col for col in df.columns if df[col].isna().any()]
#     column_to_plot = nan_columns[0] if nan_columns else df.columns[0]
#     data = df[column_to_plot].dropna()

#     if data.empty:
#         raise ValueError(f"No valid data to plot for column: {column_to_plot}")

#     row_pt = feature_df[feature_df['Attribute_name'] == column_to_plot]
#     is_categorical = row_pt["prediction"].iloc[0] == 'categorical'
#     ref_value = data.mode()[0] if is_categorical else data.median()

#     fig = px.histogram(data, x=column_to_plot, title=f'Imputation of {column_to_plot}', color_discrete_sequence=custom_colors)

#     if not is_categorical:
#         fig.add_vline(
#             x=ref_value,
#             line_dash="dash",
#             line_color="red",
#             annotation_text=f"Median: {ref_value}"
#         )
#     else:
#         fig.add_annotation(
#             text=f"Mode: {ref_value}",
#             x=ref_value,
#             y=data.value_counts().max(),
#             showarrow=True,
#             arrowhead=2,
#             arrowcolor="red"
#         )

#     fig.update_layout(
#         title_font_size=24,
#         width=250,
#         height=300
#     )

#     return fig
def plot_imputation_bar(df, feature_df, custom_colors=None):
    # 1) If there are NO missing values anywhere → return empty figure with message
    nan_columns = [col for col in df.columns if df[col].isna().any()]
    if not nan_columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No missing Data",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=18)
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title="No Missing Values",title_font_size=24, width=250, height=300)
        return fig

    # Otherwise, pick the first column that has NaNs and use its non-NaN values
    column_to_plot = nan_columns[0]
    series = df[column_to_plot].dropna()

    if series.empty:
        raise ValueError(f"No valid data to plot for column: {column_to_plot}")

    # 2) Safely determine categorical/numeric
    is_categorical = None
    # Try to use feature_df['prediction'] if available for this column
    try:
        row_pt = feature_df[feature_df.get('Attribute_name') == column_to_plot]
        if not row_pt.empty and 'prediction' in row_pt.columns:
            pred = str(row_pt['prediction'].iloc[0]).lower()
            if pred in {'categorical', 'numeric'}:
                is_categorical = (pred == 'categorical')
    except Exception:
        # Any lookup error falls through to dtype inference
        pass

    # Fallback: infer from dtype if feature_df info is missing/invalid
    if is_categorical is None:
        is_categorical = not is_numeric_dtype(series)

    # 3) Compute reference value
    ref_value = series.mode()[0] if is_categorical else series.median()

    # 4) Build histogram; default to Plotly palette if custom_colors is None
    hist_kwargs = dict(title=f'Imputation of {column_to_plot}')
    if custom_colors is not None:
        hist_kwargs['color_discrete_sequence'] = custom_colors
    fig = px.histogram(x=series, **hist_kwargs)

    # 5) Overlay reference marker
    if not is_categorical:
        fig.add_vline(
            x=ref_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median: {ref_value}"
        )
    else:
        counts = series.value_counts(dropna=True)
        max_y = counts.max() if not counts.empty else 0
        fig.add_annotation(
            text=f"Mode: {ref_value}",
            x=ref_value,
            y=max_y,
            showarrow=True,
            arrowhead=2,
            arrowcolor="red"
        )

    # 6) Styling (keep original size)
    fig.update_layout(
        title_font_size=24,
        width=250,
        height=300,
        xaxis_title=column_to_plot,
        yaxis_title="Count"
    )

    return fig


# def plot_personal_pie(plot_personal_df):
#     personal_features = plot_personal_df["Personal Features"].iloc[0]
#     non_personal_features = plot_personal_df['Non-Personal Features'].iloc[0]

#     fig = px.pie(
#         names=["Personal Features", "Non-Personal Features"],
#         values=[personal_features, non_personal_features],
#         title='Distribution of Personal Data',
#         color_discrete_sequence=custom_colors
#     )

#     fig.update_layout(
#         title_font_size=24,
#         width=250,
#         height=300
#     )

#     return fig
def plot_personal_pie(plot_personal_df, custom_colors=None):
    """
    Plot a 2-slice pie for Personal vs Non-Personal features.
    If known data issues occur, return a full-circle donut with an in-center error message.

    Handled error cases → donut with message:
      - Missing columns ("Personal Features", "Non-Personal Features")
      - Empty DataFrame (no rows)
      - Non-numeric types (after coercion)
      - NaN values
      - Negative values
      - Both values are zero
    """

    def _error_pie(msg, title="Distribution of Personal Data"):
        """Return a full-circle donut with a centered error message."""
        # Use a single-slice donut to render a full circle
        pie = go.Pie(
            values=[1],
            hole=0.7,
            textinfo="none",
            sort=False,
            direction="clockwise",
            showlegend=False
        )
        # Try to use the first custom color if available (optional)
        if custom_colors and len(custom_colors) >= 1:
            pie.marker = dict(colors=[custom_colors[0]])

        fig = go.Figure(pie)
        fig.update_layout(
            title=title,
            title_font_size=24,
            width=250,
            height=300,
            annotations=[dict(
                text=msg,
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=14)
            )],
            margin=dict(l=10, r=10, t=40, b=10)
        )
        return fig

    # 1) Validate columns
    required_cols = ["Personal Features", "Non-Personal Features"]
    missing = [c for c in required_cols if c not in plot_personal_df.columns]
    if missing:
        return _error_pie(f"Missing columns: {', '.join(missing)}")

    # 2) Validate non-empty DataFrame
    if len(plot_personal_df) == 0:
        return _error_pie("No data rows found")

    # 3) Extract first-row values
    raw_personal = plot_personal_df["Personal Features"].iloc[0]
    raw_non_personal = plot_personal_df["Non-Personal Features"].iloc[0]

    # 4) Coerce to numeric to catch non-numeric types
    personal = pd.to_numeric(pd.Series([raw_personal]), errors="coerce").iloc[0]
    non_personal = pd.to_numeric(pd.Series([raw_non_personal]), errors="coerce").iloc[0]

    # 5) Check for NaNs (includes cases where coercion failed)
    if pd.isna(personal) or pd.isna(non_personal):
        return _error_pie("Values must be numeric and non-NaN")

    # 6) Non-negative check
    if personal < 0 or non_personal < 0:
        return _error_pie("Values must be non-negative")

    # 7) Both zeros check
    if personal == 0 and non_personal == 0:
        return _error_pie("Both values are zero")

    # 8) Build the normal pie chart (keep figure size unchanged)
    pie_kwargs = {
        "names": ["Personal Features", "Non-Personal Features"],
        "values": [personal, non_personal],
        "title": "Distribution of Personal Data"
    }
    # Only pass custom colors if provided and long enough
    if custom_colors and len(custom_colors) >= 2:
        pie_kwargs["color_discrete_sequence"] = custom_colors

    fig = px.pie(**pie_kwargs)
    fig.update_layout(
        title_font_size=24,
        width=250,
        height=300
    )
    return fig


def plot_anomaly_pie(plot_anomaly_df, custom_colors=None):
    """
    Plot a pie chart of anomaly vs. non-anomaly counts.
    Robust to missing column, empty data, NaNs, and unexpected labels.
    Keeps original figure size (250x300).
    """

    def _error_pie(msg, title="Anomaly Distribution"):
        pie = go.Pie(values=[1], hole=0.7, textinfo="none", showlegend=False, sort=False)
        if custom_colors and len(custom_colors) >= 1:
            pie.marker = dict(colors=[custom_colors[0]])
        fig = go.Figure(pie)
        fig.update_layout(
            title=title,
            title_font_size=24,
            width=250, height=300,
            annotations=[dict(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False, font=dict(size=14))]
        )
        return fig

    # 1) Validate column
    if 'Anomaly' not in plot_anomaly_df.columns:
        return _error_pie("Missing column: 'Anomaly'")

    # 2) Handle empty frame
    if len(plot_anomaly_df) == 0:
        return _error_pie("No data rows found")

    # 3) Count values (keep NaN as a category if present)
    counts = plot_anomaly_df['Anomaly'].value_counts(dropna=False)
    if counts.empty:
        return _error_pie("No values in 'Anomaly'")

    # 4) Map labels robustly: 1->Anomaly, 0->No Anomaly, NaN->Unknown, others->Other:<val>
    label_map = {1: "Anomaly", 0: "No Anomaly"}
    labeled_names = []
    for idx in counts.index:
        if pd.isna(idx):
            labeled_names.append("Unknown")
        elif idx in label_map:
            labeled_names.append(label_map[idx])
        else:
            labeled_names.append(f"Other:{idx}")

    # 5) Build pie (respect custom_colors if provided)
    pie_kwargs = dict(
        values=counts.values,
        names=labeled_names,
        title="Anomaly Distribution"
    )
    if custom_colors and len(custom_colors) >= len(counts):
        pie_kwargs["color_discrete_sequence"] = custom_colors

    fig = px.pie(**pie_kwargs)

    # 6) Layout (keep size)
    fig.update_layout(
        title_font_size=24,
        width=250,
        height=300
    )
    return fig


def plot_randomforest_trees(df_randomforest_plot, target_variable):
    model_rf = pickle.load(open("data/models/RandomForest.pkl", 'rb'))
    fn = df_randomforest_plot.columns
    cn = model_rf.classes_
    fig_rf, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
    tree.plot_tree(model_rf.estimators_[0],
                   feature_names=fn,
                   filled=True)
    return fig_rf




def plot_boxplots(df, feature_inference_df): 
    # Get list of categorical columns from feature inference
    row_cat = feature_inference_df[feature_inference_df['prediction'] == "categorical"]
    categorical_cols = row_cat["Attribute_name"].to_list()
    
    # Identify non-categorical columns
    non_categorical_cols = [col for col in df.columns if col not in categorical_cols]

    # --- Boxplot for non-categorical columns ---
    fig_box = None
    if non_categorical_cols:
        df_numeric = df[non_categorical_cols]
        fig_box = px.box(df_numeric.melt(var_name="Feature", value_name="Value"),
                         x="Feature",
                         y="Value",
                         color_discrete_sequence=custom_colors)
        fig_box.update_layout(
            title='Boxplot of Numerical Features',  # <-- Title added here
            xaxis_title='Features',
            yaxis_title='Values',
            width=400,
            height=350
        )

    # --- Stacked barplot for actual categorical dtype columns ---
    fig_cat = None
    if categorical_cols:
        # Select only categorical dtype columns
        categorical_df = df[categorical_cols].select_dtypes(include=["object", "category"])

        # Calculate frequency of each category per column
        frequency_df = categorical_df.apply(lambda col: col.value_counts(dropna=False)).fillna(0).astype(int)

        # Create stacked bar plot
        fig_cat = go.Figure()

        for i, category in enumerate(frequency_df.index):
            fig_cat.add_trace(go.Bar(
                name=str(category),
                x=frequency_df.columns,
                y=frequency_df.loc[category],
                marker_color=custom_colors[i % len(custom_colors)]
            ))

        fig_cat.update_layout(
            barmode='stack',
            title='Stacked Bar Plot of Categorical Frequencies',
            xaxis_title='Columns',
            yaxis_title='Frequency',
            showlegend=False,
            width=400,
            height=400
        )

    return fig_box, fig_cat