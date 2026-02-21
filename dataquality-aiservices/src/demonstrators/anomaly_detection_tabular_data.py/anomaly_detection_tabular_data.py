import streamlit as st
import pandas as pd
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.pca import PCA
from pyod.models.lof import LOF
from pyod.models.cof import COF
from pyod.models.mcd import MCD
from pyod.models.sod import SOD
from pyod.models.cblof import CBLOF
from pyod.models.ocsvm import OCSVM
from pyod.models.hbos import HBOS
from pyod.models.abod import ABOD
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import global_controller as gc

# Streamlit app
st.info (f"If you want to skip this part of the demonstrator please click on the button below:", icon="ℹ️")
_,right=st.columns([5,1]) #Hack to be on the right side
if right.button("Skip Page"):
    st.session_state.data_type = "t3_page"
    st.rerun()

st.header("Anomaly Detection with PyOD", divider="rainbow")
#st.write("Upload a dataset and select an algorithm to detect anomalies.")



# Sidebar for dataset upload
data_file = st.session_state.file_uploder_obj#.file_uploader("Upload a CSV file", type="csv")
data_file.seek(0)
if data_file != None:
    data = pd.read_csv(data_file, index_col=0)
    st.info (f"Your uploaded dataset:", icon="ℹ️")
    st.dataframe(data)

    # Ensure numeric columns only
    numeric_data = data.select_dtypes(include=["number"])
    st.info (f"The numeric columns of your uploaded dataset:", icon="ℹ️")
    st.dataframe(numeric_data)

    # Select PyOD algorithm
    algorithms = {
        "KNN": KNN,
        "Isolation Forest": IForest,
        "AutoEncoder": AutoEncoder,
        "PCA": PCA,
        "Local Outlier Factor (LOF)": LOF,
        "Connectivity-Based LOF (COF)": COF,
        "Minimum Covariance Determinant (MCD)": MCD,
        "Subspace Outlier Detection (SOD)": SOD,
        "Cluster-Based LOF (CBLOF)": CBLOF,
        "One-Class SVM (OCSVM)": OCSVM,
        "Histogram-Based Outlier Detection (HBOS)": HBOS,
        "Angle-Based Outlier Detection (ABOD)": ABOD,
    }

    st.subheader("Detect anomalies in your dataset", divider="rainbow")
    algorithm_name = st.selectbox("Which algorithm do you want to use?",
                                  options=(algorithms.keys()),
                                  placeholder="Select an algorithm",
                                  index=None
                                  )

    # Apply the selected algorithm
    if algorithm_name != None:
        button = False
        if st.button("Detect Anomalies"):
            button = True
        if button: #ausgabe bleibt nicht TODO warum?

            gc.add_metadata("Anomaly detection algorithm", algorithm_name)
            Algorithm = algorithms[algorithm_name]
            clf = Algorithm()

            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(numeric_data)

            # Fit the model and predict
            clf.fit(data_scaled)
            anomaly_scores = clf.decision_function(data_scaled)
            predictions = clf.predict(data_scaled)

            # Create a new DataFrame for results
            results = pd.DataFrame(
                {
                    "Anomaly Score": anomaly_scores,
                    "Prediction": ["anomaly" if p == 1 else "not anomaly" for p in predictions],
                },
                index=data.index
            )

            st.info(f"The results of the Anomaly detection:", icon="ℹ️")
            st.dataframe(results)
            gc.add_metadata("Anomaly Detection Results", results.to_dict())
            st.success ('Successfully run trough this demonstrator!', icon="✅")


            # Evaluate performance (if labels are available)
            if "Label" in data.columns:
                st.write("Evaluation:")
                evaluate_print(algorithm_name, data["Label"], anomaly_scores)

    if "metadata_store" in st.session_state and "Anomaly Detection Results" in st.session_state.metadata_store:

        if st.button("NextPage"):
            #st.write( st.session_state.data_type)
            st.session_state.data_type = "t3_page" #TODO warum geht es hier nicht weiter
            st.rerun ()