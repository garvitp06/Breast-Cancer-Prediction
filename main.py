# main.py

import pandas as pd
import os
import glob

# Import the training function we wrote in train.py
from train import train_model

# --- 1. Configuration ---
DATASET_DIR = './Dataset'
CSV_PATH = os.path.join(DATASET_DIR, 'csv', 'mass_case_description_train_set.csv')

# M3 Pro is powerful. We will stick to 4 for safety on the first run.
BATCH_SIZE = 6
EPOCHS = 10 # Let's do 10 epochs for the first real run# main.py

import pandas as pd
import os
import glob

# Import the training function we wrote in train.py
from train import train_model

# --- 1. Configuration ---
DATASET_DIR = './Dataset'
CSV_PATH = os.path.join(DATASET_DIR, 'csv', 'mass_case_description_train_set.csv')

# M3 Pro is powerful. We will stick to 4 for safety on the first run.
BATCH_SIZE = 6
EPOCHS = 10 # Let's do 10 epochs for the first real run

# --- 2. Data Wrangling (Pandas) ---
print(f"Loading CSV from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

print("🔍 Mapping CSV UIDs to your folders...")
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

# Binarize labels
df_paired['pathology'] = df_paired['pathology'].astype(str).apply(
    lambda x: 1 if 'MALIGNANT' in x.upper() else 0
)

print(f"✅ Total paired sequences ready: {len(df_paired)}")

# --- 3. Kick off Training ---
print("\n🚀 Starting the Training Engine...")

# We pass '' for image_dir because our dataframe already has the absolute paths
train_model(df_paired, image_dir='', epochs=EPOCHS, batch_size=BATCH_SIZE)
 
# --- 2. Data Wrangling (Pandas) ---
print(f"Loading CSV from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

print("🔍 Mapping CSV UIDs to your folders...")
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

# Binarize labels
df_paired['pathology'] = df_paired['pathology'].astype(str).apply(
    lambda x: 1 if 'MALIGNANT' in x.upper() else 0
)

print(f"✅ Total paired sequences ready: {len(df_paired)}")

# --- 3. Kick off Training ---
print("\n🚀 Starting the Training Engine...")

# We pass '' for image_dir because our dataframe already has the absolute paths
train_model(df_paired, image_dir='', epochs=EPOCHS, batch_size=BATCH_SIZE)