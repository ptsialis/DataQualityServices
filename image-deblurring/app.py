import streamlit as st
from PIL import Image
import zipfile
import io
import torch
import torchvision.transforms as transforms
import onnxruntime as ort
import numpy as np

# --- Globale CSS & Logos ---
def apply_css():
    st.markdown("""
    <style>
    .block-container { max-width: 1000px; padding: 2rem 1rem; }
    .custom-text { font-family: 'Nunito Sans', sans-serif; font-size: 18px; }
    .custom-title { font-family: 'Nunito Sans', sans-serif; font-size: 48px; font-weight: 700; }
    .custom-subheader { font-family: 'Nunito Sans', sans-serif; font-size: 40px; font-weight: 600; }
    .stButton>button { background-color: #000AFA !important; color: white !important; border-radius: 0px !important; padding: 10px 20px !important;}
    .stButton>button:hover { filter: brightness(1.1) !important; background: #E7008A !important;}
    div.stDownloadButton>button { background-color: #000AFA !important; color: white !important; border-radius: 0px !important; padding: 10px 20px !important; }
    div.stDownloadButton>button:hover { filter: brightness(1.1); background: #E7008A !important; }
    </style>
    """, unsafe_allow_html=True)
    

def load_logo():
    return Image.open("Logo.png")

# --- ONNX Session einmal laden ---
def get_onnx_session():
    if 'onnx_session' not in st.session_state:
        providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        st.session_state.onnx_session = ort.InferenceSession("Deblurring.onnx", providers=providers)
    return st.session_state.onnx_session

# --- Landing Page ---
def show_landing_page():
    st.image(load_logo(), width=800)
    st.markdown('<h2 class="custom-subheader">Welcome to the KI-Allianz image deblurring service</h2>', unsafe_allow_html=True)
    st.markdown('<p class="custom-text">The quality of digital images can vary significantly depending on acquisition conditions. Factors such as distortion and a small depth of field often result in blurred images, which can directly compromise the accuracy of image analysis. Ensuring and verifying data quality is therefore essential. Our service provides a solution to correct image blur in light-optical microscopy images, enabling users to work with clearer, more accurate data.</p>', unsafe_allow_html=True)
    
    image = Image.open("Service_overview.png")
    st.image(image, width=800)
    
    st.markdown('<p class="custom-text"><b>The service is divided into three main sections.</b></p>', unsafe_allow_html=True)
    st.markdown('<p class="custom-text">1. Data upload: Upload your images which should be deblurred.</p>', unsafe_allow_html=True)
    st.markdown('<p class="custom-text">2. Image deblurring: Run the image deblurring method.</p>', unsafe_allow_html=True)
    st.markdown('<p class="custom-text">3. Data download: Download your deblurred images.</p>', unsafe_allow_html=True)

    if st.button("Start the service"):
        st.session_state.page = "upload"

# --- ZIP-Verarbeitung & Bildvorschau ---
def extract_zip(uploaded_zip):
    zip_bytes = io.BytesIO(uploaded_zip.read())
    png_images = []
    with zipfile.ZipFile(zip_bytes, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.lower().endswith('.png'):
                with zip_ref.open(file_name) as f:
                    png_images.append((file_name, io.BytesIO(f.read())))
    return png_images

def show_image_navigation(images, key_prefix):
    if 'current_image_idx' not in st.session_state:
        st.session_state.current_image_idx = 0

    col1, col2, col3 = st.columns([1, 4, 1])
    if col1.button("⬅", key=f"{key_prefix}_back") and st.session_state.current_image_idx > 0:
        st.session_state.current_image_idx -= 1
    col2.image(Image.open(images[st.session_state.current_image_idx][1]), caption=images[st.session_state.current_image_idx][0])
    if col3.button("➡", key=f"{key_prefix}_next") and st.session_state.current_image_idx < len(images)-1:
        st.session_state.current_image_idx += 1

# --- Bildverarbeitung ---
def process_images(images):
    session = get_onnx_session()
    results = []
    for name, img_data in images:
        img = Image.open(img_data).convert("RGB")
        tensor = torch.unsqueeze(transforms.ToTensor()(img), 0)
        tensor = torch.permute(tensor, (0,2,3,1))
        tensor = tensor.cuda() if torch.cuda.is_available() else tensor
        out = session.run(None, {"input": tensor.cpu().numpy()})[0]
        out = (np.squeeze(out) * 255).clip(0, 255).astype(np.uint8)
        results.append({"name": name, "blurred": img_data, "deblurred": Image.fromarray(out)})
    st.session_state.process_result = results

# --- ZIP-Erstellung zum Download ---
def create_zip(process_result):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for d in process_result:
            img_buf = io.BytesIO()
            d["deblurred"].save(img_buf, format="PNG")
            zf.writestr(d["name"][:-4]+"_processed.png", img_buf.getvalue())
    buf.seek(0)
    return buf

# --- Upload Page ---
def show_upload_page():
    st.image(load_logo(), width=800)
    st.markdown('<h2 class="custom-subheader">KI-Allianz image deblurring service</h2>', unsafe_allow_html=True)
    st.markdown('<h2 class="custom-subheader">Data upload</h2>', unsafe_allow_html=True)
    
    st.markdown('<p class="custom-text"><b>Expected folder structure:</b></p>', unsafe_allow_html=True)
    st.markdown("""
                ```plaintext
                   data.zip
                    |-- image1.png
                    |-- image2.png
                    |-- image3.png
                    |-- ...""")
    st.markdown('<p class="custom-text">Zip files with .png or .tif extensions are supported.</p>', unsafe_allow_html=True)
    
    uploaded_zip = st.file_uploader("Drag and drop file here", type=["zip"], accept_multiple_files=False)
    if uploaded_zip:
        images = extract_zip(uploaded_zip)
        if images:
            show_image_navigation(images, "upload")
            if st.button("Process Images"):
                with st.spinner("Processing..."):
                    process_images(images)
                    st.session_state.page = "results"

# --- Results Page ---
def show_results_page():
    st.image(load_logo(), width=800)
    st.markdown('<h2 class="custom-subheader">KI-Allianz image deblurring service</h2>', unsafe_allow_html=True)
    st.markdown('<p class="custom-text"><b>Image deblurring result</b></p>', unsafe_allow_html=True)
    st.markdown('<p class="custom-text">You can browse through the individual images to view the deblurring results, and you also have the option to download them.</p>', unsafe_allow_html=True)

    if st.session_state.process_result:
        img1 = st.session_state.process_result[st.session_state.current_image_idx]["blurred"]
        img2 = st.session_state.process_result[st.session_state.current_image_idx]["deblurred"]

        # Navigation zwischen den Bildern mit "Next" und "Back"
        col1, col2, col3, col4 = st.columns([1, 4, 4, 1])

        with col1:
            if st.button("⬅") and st.session_state.current_image_idx > 0:
                st.session_state.current_image_idx -= 1
                
        with col2:            
            st.image(img1, caption="Original", use_column_width=True)
            
        with col3:
            st.image(img2, caption="Deblurred", use_column_width=True)
        
        with col4:
            if st.button("➡") and st.session_state.current_image_idx < len(st.session_state.process_result) - 1:
                st.session_state.current_image_idx += 1
                
        # Download-Button für Bildpaare
        zip_file = create_zip(st.session_state.process_result)
        st.download_button("Download deblurred images", data=zip_file, file_name="processed_data.zip", mime="application/zip")
        
    if st.button("Back to Start"):
        st.session_state.page = "landing"

# --- Main ---
def main():
    st.set_page_config(page_title="Landing Page", layout="centered")
    if "page" not in st.session_state:
        st.session_state.page = "landing"
    apply_css()

    if st.session_state.page == "landing":
        show_landing_page()
    elif st.session_state.page == "upload":
        show_upload_page()
    elif st.session_state.page == "results":
        show_results_page()

if __name__ == "__main__":
    main()
