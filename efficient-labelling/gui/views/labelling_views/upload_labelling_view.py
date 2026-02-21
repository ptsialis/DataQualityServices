import logging
import streamlit as st
import zipfile
from PIL import Image
import os
import io
import shutil
import tempfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_zip_file_in_place(original_zip_path):
    """Clean unwanted files in-place inside the original zip."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
        temp_cleaned_path = tmp_file.name

    # Create cleaned zip
    with zipfile.ZipFile(original_zip_path, 'r') as zin, \
        zipfile.ZipFile(temp_cleaned_path, 'w', zipfile.ZIP_DEFLATED) as zout:

        for item in zin.infolist():
            filename = item.filename

            if (
                filename.startswith("__MACOSX/") or
                "/__MACOSX/" in filename or
                os.path.basename(filename).startswith("._") or
                os.path.basename(filename).startswith(".DS_Store") or
                filename.endswith('/')
            ):
                continue

            if filename.lower().endswith((".jpg", ".jpeg", ".png")) and len(filename.split("/")) >= 2:
                buffer = zin.read(filename)
                zout.writestr(filename, buffer)

    # Replace original zip
    shutil.move(temp_cleaned_path, original_zip_path)
    logger.info(f"Replaced original ZIP with cleaned version: {original_zip_path}")
    return original_zip_path


def upload_labelling_view():
    st.header("Labelling via Dataset Selection")

    st.subheader("Expected Folder Structure")
    st.code("""
    data.zip
    │── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   ├── ...
    │
    │── class2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   ├── ...
    │
    │── class3/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   ├── ...
    │             """, language="plaintext")

    controller = st.session_state.controller
    dataset_path = st.secrets["datasets"]["small_labelled_path"]

    logger.info(f"Dataset path: {dataset_path}")

    available_datasets = [
        name for name in os.listdir(dataset_path)
        if name.endswith(".zip") and os.path.isfile(os.path.join(dataset_path, name))
    ]

    logger.info(f"Available datasets: {available_datasets}")

    selected_dataset = st.selectbox("Select a labelled dataset", available_datasets)

    if selected_dataset:
        selected_zip_path = os.path.join(dataset_path, selected_dataset)

        # Clean in-place
        cleaned_zip_path = clean_zip_file_in_place(selected_zip_path)

        with zipfile.ZipFile(cleaned_zip_path, 'r') as z:
            file_list = z.namelist()

            # Build tree: {folder: [file1.jpg, ...]}
            directory_tree = {}
            for file in file_list:
                parts = file.split('/')
                if len(parts) >= 2 and parts[0] and not file.endswith('/'):
                    folder = parts[0]
                    directory_tree.setdefault(folder, []).append(file)

            with st.expander("📁 Directory Tree", expanded=False):
                for folder, files in directory_tree.items():
                    st.write(f"**{folder}/**")
                    for f in files:
                        st.write(f" - {f}")

            with st.expander("🏷️ Label Classes", expanded=False):
                class_names = {}

                for folder, files in directory_tree.items():
                    st.markdown(f"**Folder: {folder}**")
                    preview_file = next((f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))), None)

                    if preview_file:
                        try:
                            with z.open(preview_file) as image_file:
                                image_bytes = image_file.read()
                                image = Image.open(io.BytesIO(image_bytes))
                                st.image(image, caption=f"Preview of {folder}")
                        except Exception as e:
                            st.error(f"Error loading image: {e}")
                    else:
                        placeholder_path = "assets/placeholder.svg"
                        if os.path.exists(placeholder_path):
                            st.image(placeholder_path, caption="No image available")
                        else:
                            st.error("Placeholder image not found!")

                    new_name = st.text_input(f"Input name for '{folder}'", value=folder, key=f"class_{folder}")
                    class_names[folder] = new_name

                st.write("### **Class Mappings:**")
                st.write(class_names)

            submitted = st.session_state.get("labelling_submitted", False)

            if not submitted:
                if st.button("Submit Labels", key="submit_labels_button"):
                    st.session_state.class_names = class_names
                    st.session_state.zip_file = cleaned_zip_path
                    # st.session_state.dataset_info = selected_dataset  

                    st.session_state.dataset_labelling_done = False  

                    with st.spinner("⏳ Processing and applying labels..."):
                        # Actually apply the labels to the dataset
                        st.session_state.controller.submit_labels(
                            class_names, 
                            cleaned_zip_path, 
                            st.session_state.dataset_info
                        )


                        # Do the actual processing
                        result = st.session_state.controller.process_dataset_labelling_with_labels(
                            class_names, 
                            cleaned_zip_path,
                            st.session_state.dataset_info
                        )
                        if result:
                            st.session_state.stats = result["stats"]
                            st.session_state.preview = result["preview"]
                            st.session_state.output_zip_path = result["zip_path"]
                            st.session_state.labelling_submitted = True
                            st.session_state.processing_status = "success"
                            st.session_state.page = "processing"
                            st.session_state.dataset_labelling_done = False 
                            st.rerun()
                        else:
                            st.session_state.processing_status = "failure"
                            st.session_state.page = "processing"
                            st.rerun()
                                
            else:
                st.info("✅ Labels have already been submitted. Reload the page to start over.")









if False:

    st.code("""
    data.zip
    │── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   ├── ...
    │
    │── class2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   ├── ...
    │
    │── class3/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   ├── ...
    │             """, language="plaintext")
