# main.py

import pandas as pd
import os
from train import train_model

# --- MAC PATHS ---
# Update these paths to where you saved your CSVs
MASS_CSV_PATH = "./mass_case_description_train_set.csv"
CALC_CSV_PATH = "./calc_case_description_train_set.csv"

# The folder on your external drive where the pre-processor saved the .pt files
FAST_TENSOR_DIR = "./Fast_Mammograms_PT"

BATCH_SIZE = 6
EPOCHS = 10

if __name__ == '__main__':
    print("📚 Loading Answer Keys...")

    # 1. Load both VIP Guest Lists
    try:
        df_mass = pd.read_csv(MASS_CSV_PATH)
        print(f"   -> Found Mass CSV: {len(df_mass)} rows")
    except FileNotFoundError:
        print(f"❌ Could not find {MASS_CSV_PATH}. Check the path!")
        exit()

    try:
        df_calc = pd.read_csv(CALC_CSV_PATH)
        print(f"   -> Found Calcification CSV: {len(df_calc)} rows")
    except FileNotFoundError:
        print(f"❌ Could not find {CALC_CSV_PATH}. Download it from TCIA!")
        exit()

    # 2. Merge them into one massive dataset
    print("🧬 Merging Datasets...")
    df = pd.concat([df_mass, df_calc], ignore_index=True)
    print(f"   -> Total combined raw patients: {len(df)}")


    # 3. Build the file paths to your fast .pt tensors
    def build_tensor_path(row):
        filename = f"{row['patient_id']}_{row['left or right breast']}_{row['image view']}.pt"
        return os.path.join(FAST_TENSOR_DIR, filename)


    print("\n🔗 Linking CSV to fast .pt tensors on external drive...")
    df['real_image_path'] = df.apply(build_tensor_path, axis=1)

    # 4. Pivot to pair CC and MLO views
    df_paired = df.pivot_table(
        index=['patient_id', 'left or right breast', 'pathology'],
        columns='image view',
        values='real_image_path',
        aggfunc='first'
    ).reset_index()

    df_paired = df_paired.rename(columns={'CC': 'cc_image_path', 'MLO': 'mlo_image_path'})

    # 5. Clean missing data (Drops anyone who doesn't have BOTH views)
    df_paired = df_paired.dropna(subset=['cc_image_path', 'mlo_image_path'])

    # 6. Verify the files actually exist (Drops anyone whose .pt file is missing)
    df_paired = df_paired[
        df_paired['cc_image_path'].apply(os.path.exists) &
        df_paired['mlo_image_path'].apply(os.path.exists)
        ]

    # 7. Standardize the labels (Both Mass and Calc use the word "MALIGNANT")
    df_paired['pathology'] = df_paired['pathology'].astype(str).apply(
        lambda x: 1 if 'MALIGNANT' in x.upper() else 0
    )

    print(f"✅ Total perfect, paired 16-bit sequences ready: {len(df_paired)}")
    print("🚀 Starting the Unified M3 Pro Training Engine...")

    # Pass it to the exact same training loop!
    train_model(df_paired, image_dir='', epochs=EPOCHS, batch_size=BATCH_SIZE)