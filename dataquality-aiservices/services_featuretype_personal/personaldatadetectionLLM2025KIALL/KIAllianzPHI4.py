'''
Author : Albert Agisha
Department of Computer Science, University of Applied Sciences Aalen, Germany
Email  : albert.agishantwali@hs-aalen.de
Date   : 2025-10-27
Description : Streamlit app to classify columns in a CSV file as containing personal data or not using the Phi-4 model.
Ki-allianz Project 2024-2025
'''

import streamlit as st
import pandas as pd
import json
import time
from typing import Dict, Any
import huggingface_model
import plotly.graph_objects as go

# Prompts
initial_prompt_kaggle = """
As a classifier of personal related data in tabular datasets, your task is to analyze the provided columns (each containing up to ten distinct values)
and determine whether they contain information that originates from or relates to a person, even if it is not directly identifiable. 
Detecting personal related information helps ensure compliance with data protection regulations and safeguards individuals' privacy and security.
Output your results in a dictionary format with a boolean indicating if the column contains personal related data or not.
"""

example_prompt_kaggle = """
You can use the following example as a guideline:
Classify the following column with careful consideration of the dataset description:
Dataset: Title: "Test Dataset"
Description: "This dataset was used for a linear regression."
Features: ['first_name_en_10', 'last_name_en_10', 'email_en_10', 'phone_number', 'address_en_10', 'city_en_10', 'country_en_10', 'date', 'target']
Column of the dataset to classify: 'first_name_en_10': ['Tom', 'Walter', 'Mia', 'Lena', 'John', 'Jack', 'Felice', 'Anna', 'Lukas', 'Will']
Does this column, in the context of the dataset, contain information relating to a natural person?
"""

example_answer_kaggle = "Example Answer: {first_name_en_10: true}"
classification_prompt_kaggle = "Does this column, in the context of the dataset, contain information relating to a natural person?"

def extract_context_from_data(df: pd.DataFrame) -> Dict[str, Any]:
    features = list(df.columns)
    sample_rows = df.head(3).to_dict(orient='records')
    context = {
        "title": "Uploaded Dataset",
        "description": f"Dataset with {len(df)} rows and {len(df.columns)} columns.",
        "features": features,
        "sample_rows": sample_rows
    }
    return context


@st.cache_resource
def get_phi4_model():
    return huggingface_model.load_model(model="microsoft/Phi-4", quantization='4bit')

st.title("Personal Data Detection")

with st.spinner("Loading Phi-4 model (may take a while on first run)..."):
    model = get_phi4_model()
st.success("Phi-4 model loaded!")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    use_context = st.radio("Does your dataset have a specific context?", ("No", "Yes"))
    if use_context == "Yes":
        user_context = st.text_area("Please provide the dataset context (title, description, etc.):")
        if user_context.strip():
            try:
                context = json.loads(user_context)
            except Exception:
                context = {"title": "User Provided Context", "description": user_context, "features": list(df.columns)}
        else:
            st.warning("Please provide some context or switch to 'No'.")
            st.stop()
    else:
        context = extract_context_from_data(df)

    # Use session_state to avoid re-running detection after download
    if (
        "results" not in st.session_state
        or st.session_state.get("last_file") != uploaded_file.name
        or st.session_state.get("last_context") != str(context)
    ):
        results = []
        personal_count = 0
        nonpersonal_count = 0
        results_placeholder = st.empty()
        plot_placeholder = st.empty()

        # Start total timer
        total_start_time = time.time()

        for i, col in enumerate(df.columns):
            val_list = df[col].dropna().unique().tolist()[:10]
            data_prompt = (
                f"Classify the following column with careful consideration of the dataset description. "
                f"Dataset: Title: {context.get('title', '')}\n"
                f"Description: {context.get('description', '')}\n"
                f"Features: {context.get('features', list(df.columns))}\n"
                f"Column of the dataset to classify: '{col}': {val_list}\n"
                f"{classification_prompt_kaggle}"
            )
            conversation = [
                {"role": "system", "content": initial_prompt_kaggle},
                {"role": "user", "content": example_prompt_kaggle},
                {"role": "assistant", "content": example_answer_kaggle},
                {"role": "user", "content": data_prompt}
            ]
            start_time = time.time()
            with st.spinner(f"Classifying column '{col}'..."):
                response = huggingface_model.ask_model_mes(messages=conversation, pipe=model)# , max_new_tokens=512, 1024
            elapsed = time.time() - start_time
            try:
                if isinstance(response, str):
                    response_dict = json.loads(response.replace("Example Answer:", "").strip())
                elif isinstance(response, dict):
                    response_dict = response
                else:
                    response_dict = {}
                is_personal = response_dict.get(col, None)
                if is_personal is None:
                    is_personal = "true" in str(response).lower()
            except Exception:
                is_personal = "true" in str(response).lower()
            results.append({
                "Column": col,
                "Classification": "Personal" if is_personal else "Non-personal",
                #"Time (s)": f"{elapsed:.2f}"
            })

            if is_personal:
                personal_count += 1
            else:
                nonpersonal_count += 1

            # Live Plotly pie chart
            fig = go.Figure(data=[go.Pie(
                labels=["Personal", "Non-personal"],
                values=[personal_count, nonpersonal_count],
                hole=0.4,
                marker=dict(colors=["#EF553B", "#636EFA"])
            )])
            fig.update_layout(title_text="Live Classification Distribution")
            plot_placeholder.plotly_chart(fig, use_container_width=True)
            results_placeholder.table(pd.DataFrame(results))

        # End total timer
        total_elapsed = time.time() - total_start_time

        st.session_state["results"] = results
        st.session_state["personal_count"] = personal_count
        st.session_state["nonpersonal_count"] = nonpersonal_count
        st.session_state["last_file"] = uploaded_file.name
        st.session_state["last_context"] = str(context)
        st.session_state["total_elapsed"] = total_elapsed
        st.success("Classification complete!")
        st.info(f"Total detection time: {total_elapsed:.2f} seconds")
    else:
        # Display cached results and plot
        results = st.session_state["results"]
        personal_count = st.session_state["personal_count"]
        nonpersonal_count = st.session_state["nonpersonal_count"]
        total_elapsed = st.session_state.get("total_elapsed", None)
        st.success("Classification complete!")
        st.table(pd.DataFrame(results))
        fig = go.Figure(data=[go.Pie(
            labels=["Personal", "Non-personal"],
            values=[personal_count, nonpersonal_count],
            hole=0.4,
            marker=dict(colors=["#EF553B", "#636EFA"])
        )])
        fig.update_layout(title_text="Live Classification Distribution")
        st.plotly_chart(fig, use_container_width=True)
        if total_elapsed is not None:
            st.info(f"Total detection time: {total_elapsed:.2f} seconds")

    # Download button for results (does NOT trigger re-processing)
    results_df = pd.DataFrame(results)
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="classification_results823.csv",
        mime="text/csv"
    )
