# 🩺 Cross-View Mammogram Transformer
**A Multi-View Spatial Transformer for Clinical Breast Cancer Screening**

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MPS-blue?style=for-the-badge&logo=apple)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

## 📌 Project Overview
The **Cross-View Mammogram Transformer** is a deep learning architecture designed to replicate the spatial reasoning of a human radiologist. Instead of using standard "late-fusion" techniques on downsampled images, this model leverages high-resolution patch extraction and a Multi-Head Self-Attention Transformer to mathematically map tissue geometries across two distinct camera angles: the **CC (Top-Down)** and **MLO (Side-Angle)** views.

This repository tracks the evolution of the project from a Kaggle-based proof-of-concept (Version 1.0) to a highly optimized, hardware-aware clinical pipeline (Version 2.0).

---

## 🧬 Core Architectural Novelty
1. **Zero-Information-Loss Pipeline:** Bypasses standard `224x224` resizing. Raw images are sliced into 60 individual `256x256` patches per view to preserve microscopic biological anomalies (e.g., early-stage calcifications).
2. **Spatial Feature Extraction:** A modified ResNet-18 backbone mathematically compresses each patch into a 512-dimensional feature vector.
3. **Sequence-Level Fusion:** CC and MLO patches are concatenated into a single 120-item sequence.
4. **Cross-View Attention:** A Transformer Encoder uses Query-Key-Value (QKV) matrices to mathematically link a suspicious shadow in the top-down view to its corresponding physical depth in the side-angle view, confirming 3D malignancies.

---

## 🚀 Version History

### 🔹 Version 1.0: The Proof of Concept
* **Data Source:** Kaggle CBIS-DDSM Subset.
* **Format:** 8-bit JPEG (256 shades of gray).
* **Scope:** Mass detection only.
* **Objective:** Prove that sequence-level Transformer fusion outperforms standard CNN averaging.

### 🔹 Version 2.0: The Clinical-Grade Pipeline (Current)
* **Data Source:** 165GB Official TCIA Database (Full CBIS-DDSM).
* **Format:** Raw 16-bit DICOM (65,536 shades of gray, MONOCHROME1 inverted).
* **Scope:** Unified Mass + Micro-calcification detection across 1,000+ patients.
* **MLOps / Hardware Optimization:** Engineered specifically for **Apple Silicon (M3 Pro)**. 
  * Implemented an offline pre-processing engine to convert massive DICOMs into lightweight `float16 .pt` tensors.
  * Utilizes `num_workers=4` and the `MPS` backend to flood the Mac's Unified Memory, eliminating the CPU-decoding bottleneck and maintaining ~100% GPU utilization.

---

## 📊 Benchmarks

| Metric | Version 1.0 (Prototype) | Version 2.0 (Clinical Pipeline) |
| :--- | :--- | :--- |
| **Pathology Scope** | Masses Only | Masses **+** Micro-calcifications |
| **Data Fidelity** | Lossy 8-bit JPEGs | Lossless 16-bit DICOM Tensors |
| **Data Engine** | On-the-fly decoding | Offline `.pt` pre-computation |
| **Processing Speed** | ~6.6s / patient | **~5.7s / patient** |
| **Validation AUC** | 0.8100 | **0.7439** *(at Epoch 6, harder dataset)* |

---

## 💻 Installation & Setup

### 1. Environment
Ensure you have Python 3.10+ installed. Install the required dependencies:
```bash
pip install torch torchvision pydicom pandas numpy streamlit pillow
2. Pre-Processing the DICOMs (V2.0 Pipeline)
To prevent I/O bottlenecks during training, run the offline preprocessor to convert your raw DICOM files into PyTorch .pt tensors. (Note: Adjust paths in the script to point to your external SSD/Storage).

Bash
python preprocess.py
3. Training the Model
Launch the training engine. This script utilizes Pandas to merge the clinical CSV answer keys, drops missing views, and initializes the MPS dataloader.

Bash
python main.py
🖥️ Interactive Clinical Dashboard
The project includes a production-ready Streamlit Frontend that allows users to test the model using .dcm, .pt, or standard .png/.jpg files. The UI handles all complex photometric inversions and patch-extraction natively.

To launch the dashboard:

Bash
streamlit run app.py
The dashboard will automatically open in your browser at http://localhost:8501.

⚙️ Project Structure
Plaintext
├── preprocess.py       # V2.0 Offline DICOM-to-Tensor Engine
├── dataset.py          # Custom PyTorch Dataset & Patch Extraction
├── model.py            # ResNet-18 + Cross-View Transformer Architecture
├── main.py             # Unified Training Loop & Evaluation
├── app.py              # Streamlit Clinical Dashboard
├── mass_case.csv       # Clinical Labels (Masses)
├── calc_case.csv       # Clinical Labels (Calcifications)
└── best_model.pth      # Saved Model Weights (Generated post-training)
🤝 Author
Built as a demonstration of applying MLOps, hardware-aware data engineering, and spatial Deep Learning to the medical domain.
