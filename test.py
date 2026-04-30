# test.py

import pandas as pd
import os
import glob
import torch
from torch.utils.data import DataLoader
from dataset import PairedMammoDataset
from model import CrossViewTransformer
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# --- 1. Configuration ---
DATASET_DIR = './Dataset'
# Point this to the test set CSV provided by the Kaggle dataset
TEST_CSV_PATH = os.path.join(DATASET_DIR, 'csv', 'mass_case_description_test_set.csv')
MODEL_WEIGHTS = 'best_mammo_model_latest.pth'
BATCH_SIZE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Testing on: {device}")

# --- 2. Data Wrangling (Test Set) ---
print(f"\nLoading Test CSV from: {TEST_CSV_PATH}")
df = pd.read_csv(TEST_CSV_PATH)

print("🔍 Mapping Test CSV UIDs to your folders...")
all_folders = {}
jpeg_root = os.path.join(DATASET_DIR, 'jpeg')

for root, dirs, files in os.walk(jpeg_root):
    for d in dirs:
        if d.startswith("1.3.6"):
            all_folders[d] = root


def find_real_image_path(csv_image_path):
    parts = str(csv_image_path).split('/')
    for part in parts:
        if part in all_folders:
            folder_path = os.path.join(all_folders[part], part)
            images = glob.glob(os.path.join(folder_path, "**", "*.jpg"), recursive=True)
            if images:
                return images[0]
    return None


df['real_image_path'] = df['image file path'].apply(find_real_image_path)

df_paired = df.pivot_table(
    index=['patient_id', 'left or right breast', 'pathology'],
    columns='image view',
    values='real_image_path',
    aggfunc='first'
).reset_index()

df_paired = df_paired.rename(columns={'CC': 'cc_image_path', 'MLO': 'mlo_image_path'})
df_paired = df_paired.dropna(subset=['cc_image_path', 'mlo_image_path'])

df_paired['pathology'] = df_paired['pathology'].astype(str).apply(
    lambda x: 1 if 'MALIGNANT' in x.upper() else 0
)

print(f"✅ Total Test sequences ready: {len(df_paired)}")

# --- 3. Model Initialization & Loading ---
print(f"\nLoading saved weights from {MODEL_WEIGHTS}...")
model = CrossViewTransformer().to(device)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True))
model.eval()  # CRITICAL: Turn off dropout and batchnorm for testing

test_dataset = PairedMammoDataset(dataframe=df_paired, image_dir='')
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- 4. Inference Loop ---
all_preds = []
all_labels = []

print("\n🚀 Running Inference on Unseen Test Data...")
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        cc = batch['cc_patches'].to(device)
        mlo = batch['mlo_patches'].to(device)
        labels = batch['label'].numpy()  # Keep labels on CPU

        outputs = model(cc, mlo)
        probs = torch.sigmoid(outputs).cpu().numpy()  # Convert logits to probabilities (0 to 1)

        all_preds.extend(probs.flatten())
        all_labels.extend(labels)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# --- 5. Clinical Metrics & Visualization ---
print("\n📊 Calculating Clinical Metrics...")

# Calculate AUC
test_auc = roc_auc_score(all_labels, all_preds)
print(f"🔥 FINAL TEST AUC: {test_auc:.4f}")

# Calculate ROC Curve to find the optimal threshold (Youden's J statistic)
fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal Prediction Threshold: {optimal_threshold:.4f}")

# Convert probabilities to strict binary predictions using the optimal threshold
binary_preds = (all_preds >= optimal_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds).ravel()

sensitivity = tp / (tp + fn)  # Recall: How many cancers did we actually catch?
specificity = tn / (tn + fp)  # True Negative Rate: How many healthy patients were correctly cleared?

print("-" * 30)
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity:          {specificity:.4f}")
print("-" * 30)

# --- Generate Plots ---
plt.figure(figsize=(12, 5))

# Plot 1: ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red',
            label=f'Optimal Threshold ({optimal_threshold:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Plot 2: Confusion Matrix
plt.subplot(1, 2, 2)
cm = confusion_matrix(all_labels, binary_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.ylabel('Actual Pathology')
plt.xlabel('Predicted Pathology')
plt.title(f'Confusion Matrix (Threshold = {optimal_threshold:.2f})')

plt.tight_layout()
plt.savefig('clinical_results.png', dpi=300)
print("\n📸 Saved charts to 'clinical_results.png'!")
plt.show()