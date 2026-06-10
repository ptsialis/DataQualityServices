import os
import json
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from services.session_state import load_dataframe_from_session, save_dataframe_to_session
import src.global_controller as gc

custom_colors = ["#5B5B5B", "#D24740", "#1F4E79", "#E4B600", "#000000", "#BFA74A"]

def load_metadata():
    meta_path = os.path.join("session_data", "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_metadata(data: dict):
    os.makedirs("session_data", exist_ok=True)
    meta_path = os.path.join("session_data", "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def fig_to_base64(fig):
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def safe_load_df(key):
    df = load_dataframe_from_session(key)
    return df if df is not None and not df.empty else pd.DataFrame()

def generate_html(output_html):
    imputed_df = load_dataframe_from_session("impute")
    metadata = load_metadata()
    target_variable = metadata.get("target_variable", None)
    file_name = metadata.get("name", "Unbekannt")

    df = gc.one_hot_encode_df(imputed_df)
    if df.empty:
        with open(output_html, "w", encoding="utf-8") as f:
            f.write("<h2>Keine Daten geladen</h2>")
        return

    numeric_cols = df.select_dtypes(include="number").columns

    html_parts = []
    html_parts.append(f"<h1>Overall Report</h1>")
    html_parts.append(f"<p><b>Datei:</b> {file_name}</p>")
    html_parts.append(f"<p><b>Zielvariable:</b> {target_variable}</p>")

    #  Histogramme für numerische Features
    for col in numeric_cols:
        fig = px.histogram(
            df, x=col, nbins=10, color_discrete_sequence=custom_colors,
            title=f"Verteilung von {col}"
        )
        fig.update_traces(texttemplate="<b>%{y}</b>", textposition="outside",
                          marker_line_color="black", marker_line_width=1.2)
        fig.update_layout(title_x=0.5, bargap=0.15, width=700, height=400)
        html_parts.append(pio.to_html(fig, full_html=False, include_plotlyjs="cdn"))

    # Pairplot (Seaborn)
    if len(numeric_cols) >= 2:
        pairplot = sns.pairplot(df[numeric_cols[:5]])
        img = BytesIO()
        pairplot.savefig(img, format="png")
        img.seek(0)
        img_base64 = base64.b64encode(img.read()).decode("utf-8")
        html_parts.append("<h2>Pairplot</h2>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}" width="800"/>')

    #  Korrelationsmatrix
    if not df[numeric_cols].empty:
        fig = px.imshow(df[numeric_cols].corr(), text_auto=True,
                        aspect="auto", color_continuous_scale="RdBu_r")
        fig.update_layout(title="Korrelationsmatrix", width=700, height=500)
        html_parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    #  Korrelation mit Zielvariable
    if target_variable and target_variable in df.columns:
        corr_df = pd.DataFrame()
        if target_variable in numeric_cols:
            corr_series = df.corr(numeric_only=True)[target_variable].drop(target_variable)
            corr_df = corr_series.sort_values().to_frame("Korrelationskoeffizient").reset_index()
            corr_df.rename(columns={"index": "Feature"}, inplace=True)
        else:
            df_encoded = df.copy()
            df_encoded[target_variable] = LabelEncoder().fit_transform(df[target_variable])
            corr_series = df_encoded.corr(numeric_only=True)[target_variable].drop(target_variable)
            corr_df = corr_series.sort_values().to_frame("Korrelationskoeffizient").reset_index()
            corr_df.rename(columns={"index": "Feature"}, inplace=True)

        if not corr_df.empty:
            fig = px.bar(
                corr_df, x="Korrelationskoeffizient", y="Feature",
                orientation="h", color="Feature",
                color_discrete_sequence=px.colors.qualitative.Set3,
                text="Korrelationskoeffizient",
                title=f"Korrelation mit Zielvariable: {target_variable}"
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, width=700, height=500)
            html_parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    #  Feature Importance
    if target_variable and target_variable in df.columns:
        X = df.drop(columns=[target_variable], errors="ignore")
        y = df[target_variable]
        if not X.empty and y.nunique() > 1:
            if not pd.api.types.is_numeric_dtype(y):
                y = LabelEncoder().fit_transform(y)

            model = RandomForestClassifier(n_estimators=100, random_state=0)
            model.fit(X, y)

            importances = pd.Series(model.feature_importances_, index=X.columns)
            imp_df = importances.sort_values(ascending=True).tail(20).reset_index()
            imp_df.columns = ["Feature", "Importance"]

            fig = px.bar(
                imp_df, x="Importance", y="Feature", orientation="h",
                color="Feature", color_discrete_sequence=px.colors.qualitative.Set3,
                text="Importance", title=f"Top Merkmale für Vorhersage von '{target_variable}'"
            )
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig.update_layout(yaxis_title="Feature", xaxis_title="Bedeutung",
                              legend_title="Feature", bargap=0.2, width=700, height=500)
            html_parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    with open(output_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
