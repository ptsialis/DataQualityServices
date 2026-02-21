import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps

def unsuccessful_labelling_view():
    
    with st.expander("🔎 View the workflow  of the service!", expanded=True):
        image_path = Path(__file__).parent.parent.parent / "assets" / "workflow" / "data_check_yellow.png"
        image = Image.open(image_path)
        
        # White Background 
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
        
        # Padding
        padding = 20
        image = ImageOps.expand(image, border=padding, fill=(255, 255, 255))
        
        st.image(image, caption="Overall workflow -- Data Check")
        
    st.subheader("Manual Labelling Required")
    
    st.write("The dataset's domain similarity is too low, so automatic labelling is not possible.")
    st.write("Please proceed to the labelling view to manually label the dataset.")
    
    if st.button("Proceed to \"Labelling\""):
        st.session_state.page = "labelling"
        st.rerun()
    
    # st.divider()
    # if st.button("Back"):
    #     st.session_state.page = "upload"
    #     st.rerun()
