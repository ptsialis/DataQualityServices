import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import global_controller as gc

def display_meta_data(df):
    """Function to display metadata about the dataset."""
    st.write("### Metadata")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")

    missing_values = df.isnull().sum()
    missing_info = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Values': missing_values.values,
        'Percentage': (missing_values.values / len(df)) * 100
    })
    st.write("### Columns with Missing Values")
    st.dataframe(missing_info)


def impute_mean_mode(df):
    """Function to impute missing values with mean for numeric columns and mode for categorical columns."""
    imputation_info = []
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            mean_value = df[col].mean()
            df[col].fillna(mean_value, inplace=True)
            imputation_info.append((col, 'Mean', mean_value))
        else:
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
            imputation_info.append((col, 'Mode', mode_value))
    return df, imputation_info

# Streamlit app
st.header("CSS Dataset Viewer and Imputation", divider="rainbow")

file =st.session_state.file_uploder_obj
file.seek(0)
uploaded_file = file #st.file_uploader("Upload a CSV Dataset", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    try:
        df = pd.read_csv(uploaded_file)
        st.info (f"Your uploaded dataset:", icon="ℹ️")
        st.dataframe(df)

        # Display metadata
        #display_meta_data(df)

        # Dropdown menu for methods
        st.subheader("Data Processing Methods", divider="rainbow")
        method = st.selectbox("Which imputation method do you want to use?",
                              ["Mean/Mode"],
                              placeholder="select a method",
                              index=None
                              )

        if method == "Mean/Mode":
            df, imputation_info = impute_mean_mode(df)
            #st.write("### Dataset After Mean/Mode Imputation")
            st.info (f"The dataset after the imputation:", icon="ℹ️")
            st.dataframe(df)
            gc.add_metadata("Imputation method", "Mean/Mode")

            # Show imputation details
            st.info (f"See below the details of the imputation:", icon="ℹ️")
            imputation_df = pd.DataFrame(imputation_info, columns=['Column', 'Imputation Type', 'Value'])
            st.dataframe(imputation_df)
            gc.add_metadata("Imputation details", imputation_df.to_dict())

            # Show updated metadata
            #display_meta_data(df)

            st.info (f"To have an overview about the collected metadata switch to the summary section.", icon="ℹ️")
            st.success ('Successfully run trough this demonstrator!', icon="✅")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a dataset to begin.")
