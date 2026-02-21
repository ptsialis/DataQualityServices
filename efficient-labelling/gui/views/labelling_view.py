import streamlit as st
from PIL import Image, ImageOps
from pathlib import Path

from views.labelling_views.interactive_labelling_view import interactive_labelling_view
from views.labelling_views.upload_labelling_view import upload_labelling_view


def labelling_view():
    with st.expander("🔎 View the workflow of the service!", expanded=True):
        image_path = Path(__file__).parent.parent / "assets" / "workflow" / "data_labelling_yellow.png"
        image = Image.open(image_path)
        
        # White Background 
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
        
        # Padding
        padding = 20
        image = ImageOps.expand(image, border=padding, fill=(255, 255, 255))
        
        st.image(image, caption="Overall workflow -- Data Labelling")
    st.divider()
    
    st.title("Labelling Options (Choose your option)")
    
    labelling_options = (
        "None", 
        "Upload subset for labelling", 
        # "Interactive labelling" # TODO: Implement interactive labelling
        )
    labelling_method = st.radio("Select labelling method", labelling_options, index=1)
    
    
    st.divider()
    
    if labelling_method == labelling_options[1]:
        upload_labelling_view()
    # elif labelling_method == labelling_options[2]: # TODO: Implement interactive labelling
    #     interactive_labelling_view()
    else:
        st.warning("Please select a labelling method.")


    st.divider() 
    if st.button("Back"): # TODO: Actually also reset the saved thing in st.session_state
        st.session_state.page = "upload"
        st.rerun()
        # st.session_state.clear()   # wipes all stored state
        # st.rerun()
