import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps

from views.processing_views.unsuccessful_labelling_view import unsuccessful_labelling_view
from views.processing_views.successful_labelling_view import successful_labelling_view

def processing_view():
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = "pending"
        
    if 'button_pressed' not in st.session_state:
        st.session_state.button_pressed = False
        
    if 'subset_labelled' not in st.session_state:
        st.session_state.subset_labelled = False
    
    with st.expander("🔎 View the workflow of the service!", expanded=True):
        image_path = Path(__file__).parent.parent / "assets" / "workflow" / "data_processing-1.png"
        
        if st.session_state.processing_status == "failure":
            # print("failure")
            image_path = Path(__file__).parent.parent / "assets" / "workflow" / "data_processing-1.png"
        elif st.session_state.processing_status == "success":
            # print("success")
            image_path = Path(__file__).parent.parent / "assets" / "workflow" / "data_processing-2.png"
            
        image = Image.open(image_path)
        
        # White Background 
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
        
        # Padding
        padding = 20
        image = ImageOps.expand(image, border=padding, fill=(255, 255, 255))
        
        st.image(image, caption="Overall workflow -- Data Processing")
    st.divider()
    
    st.subheader("Data Processing")
    
    process_dataset_button = st.button(
        "Process Dataset", 
        on_click=lambda: setattr(st.session_state, "button_pressed", True), 
        disabled=st.session_state.button_pressed
    )
    
    if st.session_state.subset_labelled: # TODO: Remove
            pseudo_labels = ["A", "B", "C"] # for test purposes
            st.session_state.controller.submit_labels(pseudo_labels)
            st.session_state.processing_status = "success"    

    if process_dataset_button:
        with st.spinner("Processing dataset..."):
            
            # print("Uplodaded File test: ", st.session_state.uploaded_file)
            try:
                st.session_state.controller.submit_dataset(st.session_state.uploaded_file, st.session_state.dataset_info)
            except Exception as e:
                st.error(f"Error submitting dataset: {e}")
                return
            
            try: 
                can_label, model_name = st.session_state.controller.process_dataset_similarity(st.session_state.dataset_info)

                if can_label:
                    st.session_state.processing_status = "success"
                else:
                    st.session_state.processing_status = "failure"
                
                if model_name:
                    st.session_state.model_for_dataset = model_name
                    st.success(f"The dataset can be labelled using model for {model_name}.")
                    # TODO: Integrate some user feedback that is better.
                else:
                    st.error("No model found for the dataset.")
                    st.session_state.model_for_dataset = None
                
            except Exception as e:
                # st.error(f"Error processing dataset similarity: {e}")
                st.session_state.processing_status = "error"
            
            st.rerun()

    
    if st.session_state.processing_status == "failure":
        st.warning("The dataset you provided is too specific, so manual labelling is required.")
        unsuccessful_labelling_view()
        
    elif st.session_state.processing_status == "success":
        successful_labelling_view()
        
    if st.session_state.processing_status == "error":
        st.session_state.page = "upload"
        st.rerun()
        
    st.divider() # TODO: Actually also reset the saved thing in st.session_state
    if st.button("Back"):
        st.session_state.dataset_labelling_done = False
        st.session_state.page = "upload"
        st.rerun()
        # st.session_state.clear()   # wipes all stored state
        # st.rerun()
