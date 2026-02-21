import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps

def download_view():
    with st.expander("🔎 View the workflow of the service!", expanded=True):
        image_path = Path(__file__).parent.parent / "assets" / "workflow" / "data_download.png"
        image = Image.open(image_path)

        # White Background
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background

        # Padding
        padding = 20
        image = ImageOps.expand(image, border=padding, fill=(255, 255, 255))

        st.image(image, caption="Overall workflow")

    st.divider()

    st.subheader("📦 Final Dataset Ready")
    st.success("Your dataset has been successfully processed and labelled.")

    # Load actual labelled ZIP file from previous step
    zip_path = st.session_state.get("output_zip_path")

    if zip_path and Path(zip_path).exists():
        with open(zip_path, "rb") as f:
            zip_data = f.read()

        st.download_button(
            label="⬇️ Download Labelled Dataset",
            data=zip_data,
            file_name=Path(zip_path).name,
            mime="application/zip"
        )
    else:
        st.warning("No labelled dataset found to download.")

    st.divider()
    if st.button("Back"):
        # st.session_state.page = "upload"
        # st.session_state.dataset_labelling_done = False  # Optional: reset
        # st.rerun()
        st.session_state.clear()   # wipes all stored state
        st.rerun()
