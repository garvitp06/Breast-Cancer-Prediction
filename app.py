# app.py

import streamlit as st
import torch
import pydicom
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from model import CrossViewTransformer
import time

# --- UI Configuration ---
st.set_page_config(page_title="Mammogram Screening AI", page_icon="🩺", layout="wide")


# --- Model Loading ---
@st.cache_resource
def load_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CrossViewTransformer().to(device)

    try:
        model.load_state_dict(torch.load('165gb training/best_dicom_mps_model.pth', map_location=device))
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("⚠️ Could not find 'best_mammo_model_Latest.pth'. Make sure it's in the same folder!")
        return None, device


model, device = load_model()


# --- Universal File Processor ---
def process_uploaded_file(uploaded_file):
    """Detects file type and converts it to a standard [1, H, W] PyTorch Tensor"""

    file_name = uploaded_file.name.lower()

    # 1. Handle PyTorch Tensors (.pt)
    if file_name.endswith('.pt'):
        tensor_img = torch.load(uploaded_file, weights_only=True).float()
        # Ensure it has the [1, H, W] channel dimension
        if len(tensor_img.shape) == 2:
            tensor_img = tensor_img.unsqueeze(0)
        return tensor_img

    # 2. Handle Standard Images (.png, .jpg, .jpeg)
    elif file_name.endswith(('.png', '.jpg', '.jpeg')):
        # Open image and convert to grayscale ('L')
        img = Image.open(uploaded_file).convert('L')
        img_array = np.array(img).astype(np.float32)

        # Normalize 8-bit (0-255) to float (0.0-1.0)
        img_array = img_array / 255.0
        return torch.from_numpy(img_array).unsqueeze(0).float()

    # 3. Handle Clinical DICOMs (.dcm)
    elif file_name.endswith('.dcm'):
        dcm = pydicom.dcmread(uploaded_file)
        img_array = dcm.pixel_array.astype(np.float32)

        # Fix negative X-rays
        if hasattr(dcm, 'PhotometricInterpretation') and dcm.PhotometricInterpretation == "MONOCHROME1":
            img_array = np.max(img_array) - img_array

        # Normalize 16-bit to float (0.0-1.0)
        min_val = np.min(img_array)
        max_val = np.max(img_array)
        img_array = (img_array - min_val) / (max_val - min_val + 1e-8)

        return torch.from_numpy(img_array).unsqueeze(0).float()

    else:
        st.error(f"Unsupported file type: {file_name}")
        return None


# --- Patch Extraction (Unchanged) ---
def extract_patches(tensor_img, patch_size=256, num_patches=60):
    _, h, w = tensor_img.shape
    patches = []
    stride_h = (h - patch_size) // int(np.sqrt(num_patches))
    stride_w = (w - patch_size) // int(np.sqrt(num_patches))

    if stride_h <= 0 or stride_w <= 0:
        resized = TF.resize(tensor_img, (patch_size * 8, patch_size * 8))
        return extract_patches(resized, patch_size, num_patches)

    for i in range(0, h - patch_size + 1, stride_h):
        for j in range(0, w - patch_size + 1, stride_w):
            patch = tensor_img[:, i:i + patch_size, j:j + patch_size]
            patches.append(patch)
            if len(patches) == num_patches: break
        if len(patches) == num_patches: break

    while len(patches) < num_patches:
        patches.append(torch.zeros((1, patch_size, patch_size)))

    return torch.stack(patches).unsqueeze(0)


# --- The Frontend Dashboard ---
st.title("🩺 Clinical Breast Cancer Screening AI")
st.markdown("### Powered by Cross-View Transformer Architecture (v2.0)")
st.markdown(
    "Upload a patient's **CC** and **MLO** views. Supports **DICOM (.dcm)**, **Standard Images (.png/.jpg)**, or **Pre-computed Tensors (.pt)**.")

# Added multi-format support to the uploaders
ALLOWED_TYPES = ["dcm", "pt", "png", "jpg", "jpeg"]

col1, col2 = st.columns(2)

with col1:
    st.header("1. CC View Input")
    cc_file = st.file_uploader("Upload CC View", type=ALLOWED_TYPES, key="cc")

with col2:
    st.header("2. MLO View Input")
    mlo_file = st.file_uploader("Upload MLO View", type=ALLOWED_TYPES, key="mlo")

st.divider()

if cc_file and mlo_file:
    if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
        with st.spinner(f"Processing inputs and running Cross-View Attention on {device.type.upper()}..."):
            start_time = time.time()

            # Universal Processing
            cc_tensor = process_uploaded_file(cc_file)
            mlo_tensor = process_uploaded_file(mlo_file)

            if cc_tensor is not None and mlo_tensor is not None:
                # Patch Extraction
                cc_patches = extract_patches(cc_tensor).to(device)
                mlo_patches = extract_patches(mlo_tensor).to(device)

                # Model Inference
                with torch.no_grad():
                    output = model(cc_patches, mlo_patches)
                    probability = torch.sigmoid(output).item()

                inference_time = time.time() - start_time

                # Display Results
                st.success(f"Analysis Complete in {inference_time:.2f} seconds!")

                res_col1, res_col2 = st.columns(2)

                with res_col1:
                    st.metric(label="Malignancy Probability", value=f"{probability * 100:.2f}%")
                    st.progress(probability)

                with res_col2:
                    if probability > 0.5:
                        st.error("🚨 **Verdict: High Risk (Malignant Signature Detected)**")
                        st.write(
                            "The Cross-View Transformer detected geometric correlations between the CC and MLO views indicative of a solid mass or micro-calcifications.")
                    else:
                        st.success("✅ **Verdict: Low Risk (Benign)**")
                        st.write("No correlating malignant structures were detected across the spatial views.")
else:
    st.info("Waiting for both files to be uploaded...")