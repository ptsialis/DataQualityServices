
# Temporary
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import logging 
import streamlit as st
from views.landing_view import landing_view
from views.upload_view import upload_view
from views.processing_view import processing_view
from views.labelling_view import labelling_view
from views.download_view import download_view
from gui.controllers.app_controller import AppController
from apis.api_labelling_service import LabellingServiceAPI

# backend dependencies
from service.logic_controller import LogicController 



APP_MODE = st.secrets["mode"]

def main():
    st.set_page_config(page_title="AI Alliance - Data Labelling Service")
    st.title("AI Alliance - Efficient Labelling GUI")
    st.divider()
    
    if "mode" not in st.session_state: 
        st.session_state.mode = APP_MODE 
    if "page" not in st.session_state:
        st.session_state.page = "landing"
    if "dataset_labelled" not in st.session_state:
        st.session_state.dataset_labelled = False
    if "dataset_controller" not in st.session_state:
        api = LabellingServiceAPI(LogicController())  # Inject backend dependency
        controller = AppController(api) 
        controller.mode = st.session_state.mode
        st.session_state.controller = controller
    
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = None
    if "dataset_labelling_done" not in st.session_state:
        st.session_state.dataset_labelling_done = False
    
    # Show landing page if user has not landed
    if st.session_state.page == "landing":
        landing_view()
    # Route to the correct view
    elif st.session_state.page == "upload":
        upload_view()
    elif st.session_state.page == "processing":
        processing_view()
    elif st.session_state.page == "labelling":
        labelling_view()
    elif st.session_state.page == "download":
        download_view()

if __name__ == "__main__":
    main()