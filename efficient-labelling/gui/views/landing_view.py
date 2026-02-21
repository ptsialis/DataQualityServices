import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps


def landing_view():
    
    st.subheader("Welcome to the AI Alliance Efficient Data Labelling Service GUI")
    
    st.write("This GUI is designed to help you label your datasets efficiently.")
    
    with st.expander("🔎 View the Overall workflow ", expanded=True):
        image_path = Path(__file__).parent.parent / "assets" / "workflow" / "overall.png"
        image = Image.open(image_path)
        
        # White Background 
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
        
        # Padding
        padding = 20
        image = ImageOps.expand(image, border=padding, fill=(255, 255, 255))
        
        st.image(image, caption="Overall workflow")
    
    
    st.write("**The GUI is divided into four main sections:**")
    st.write("")
    st.write("*Data Upload*: Upload your dataset and provide a brief description.")
    st.write("*Data Processing*: Process your dataset to prepare it for labelling and if possible label it.")
    st.write("*Data Checking*: Check the processed data if it is suitable for fast labelling.")
    st.write("*Data Labelling*: Manually label a very small dataset.")
    st.write("*Data Download*: Download your large fully labelled dataset.")
    
    
    if st.button("Start the service"):
        st.session_state.page = "upload"
        # Trigger a rerun to load the next view
        st.rerun()
