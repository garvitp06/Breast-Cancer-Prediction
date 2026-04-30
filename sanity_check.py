import torch
from torch.utils.data import DataLoader

# --- 1. Configuration ---
# Point this to where you unzipped the Kaggle images
# Example: './kaggle_dataset/images/'
IMAGE_DIR = './dataset/csv/mass_case_description_train_set.csv'
BATCH_SIZE = 4  # Process 4 patients at a time

# --- 2. Initialize the Dataset and DataLoader ---
print("Initializing Dataset...")
dataset = PairedMammoDataset(dataframe=df_paired, image_dir=IMAGE_DIR)

print("Initializing DataLoader...")
# num_workers speeds up loading by using multiple CPU cores. Set to 0 if you get Windows errors.
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# --- 3. Run a Single Batch ---
print("Fetching a single batch...")
for batch_idx, batch in enumerate(dataloader):
    # Extract the tensors from the dictionary
    cc_tensors = batch['cc_patches']
    mlo_tensors = batch['mlo_patches']
    labels = batch['label']

    # Print the shapes to verify
    print("\n✅ SUCCESS! Batch loaded.")
    print("-" * 30)
    print(f"CC Patches Shape:  {cc_tensors.shape}  -> Expected: [{BATCH_SIZE}, 60, 1, 256, 256]")
    print(f"MLO Patches Shape: {mlo_tensors.shape}  -> Expected: [{BATCH_SIZE}, 60, 1, 256, 256]")
    print(f"Labels Shape:      {labels.shape}           -> Expected: [{BATCH_SIZE}]")
    print(f"Sample Labels:     {labels.tolist()}")
    print("-" * 30)

    # Break immediately so we ONLY run one batch!
    break