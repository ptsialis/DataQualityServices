import streamlit as st
import zipfile
import random
from io import BytesIO
from PIL import Image, ImageOps
from pathlib import Path

def get_available_datasets():
    dataset_dir = Path(st.secrets["datasets"]["dataset_root"])
    # dataset_dir = Path(__file__).parent.parent 
    # print(f"Looking for datasets in: {dataset_dir}")
    return sorted([f.name for f in dataset_dir.glob("*.zip")])

def upload_view():
    if st.session_state.processing_status == "error":
        st.toast("An error occurred during processing. Please redo the data upload.", icon="❌")
    
    with st.expander("🔎 View the workflow of the service!", expanded=True):
        image_path = Path(__file__).parent.parent / "assets" / "workflow" / "data_upload.png"
        image = Image.open(image_path)
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = ImageOps.expand(background, border=20, fill=(255, 255, 255))
        st.image(image, caption="Overall workflow -- Data Upload")
    st.divider()

    st.title("Data Upload")

    st.text("Expected Folder Structure:")
    st.code("""
    dataset_name.zip
    │── image1.jpg
    │── image2.jpg
    │── image3.jpg
    │── ...
    """, language="plaintext")
    st.html("<div>Supported file types: .jpg, .png</div>")
    st.divider()

    st.subheader("📂 Select a dataset from our saved collection")
    datasets = get_available_datasets()
    selected_dataset = st.selectbox("Choose a dataset", options=["-- Select --"] + datasets)
    

    if selected_dataset != "-- Select --":
        # dataset_path = Path(__file__).parent.parent / "assets" / "unlabelled_datasets" / selected_dataset
        dataset_path = Path(st.secrets["datasets"]["dataset_root"] + "/" + selected_dataset)
        
        dataset_info = selected_dataset

        if (
            "preview_images" not in st.session_state or 
            "uploaded_file" not in st.session_state or 
            st.session_state.uploaded_file != str(dataset_path)
        ):
            st.session_state.uploaded_file = str(dataset_path)
            st.session_state.preview_images = []
            st.session_state.preview_index = 0

            try:
                with zipfile.ZipFile(dataset_path, "r") as zf:
                    image_files = [f for f in zf.namelist() if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                    if len(image_files) > 5:
                        image_files = random.sample(image_files, 5)
                    else:
                        random.shuffle(image_files)

                    for img_file in image_files:
                        with zf.open(img_file) as f:
                            st.session_state.preview_images.append(f.read())

            except Exception as e:
                st.error(f"Failed to load dataset: {e}")
                return

        with st.expander("🔎 Preview Dataset", expanded=True):
            if st.session_state.preview_images:
                image_data = st.session_state.preview_images[st.session_state.preview_index]
                image = Image.open(BytesIO(image_data))
                st.image(image)

                col1, _, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button("Previous"):
                        st.session_state.preview_index = (
                            st.session_state.preview_index - 1
                        ) % len(st.session_state.preview_images)
                        st.rerun()
                with col3:
                    if st.button("Next"):
                        st.session_state.preview_index = (
                            st.session_state.preview_index + 1
                        ) % len(st.session_state.preview_images)
                        st.rerun()
            else:
                st.write("No images found in this dataset.")

        if st.button("Proceed to \"Processing\""):
            st.session_state.page = "processing"
            st.session_state.dataset_info = dataset_info
            
            # print("Uplodaded File test: ", st.session_state.uploaded_file
            
            st.rerun()

        st.divider() 
        if st.button("Back"): # TODO: Actually also reset the saved thing in st.session_state
            st.session_state.page = "landing"
            st.rerun()
            # st.session_state.clear()   # wipes all stored state
            # st.rerun()
