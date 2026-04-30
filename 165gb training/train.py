# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import PairedMammoDataset
from model import CrossViewTransformer
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training Engine Initialized on: {device} (Apple Silicon)")


def train_model(df_paired, image_dir, epochs=10, batch_size=4):
    full_dataset = PairedMammoDataset(df_paired, image_dir)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # M3 PRO ADVANTAGE: We can use num_workers=4 safely on macOS!
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = CrossViewTransformer().to(device)

    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.feature_projection.parameters(), 'lr': 1e-4},
        {'params': model.transformer.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))

    best_val_auc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch in train_pbar:
            cc = batch['cc_patches'].to(device)
            mlo = batch['mlo_patches'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)

            optimizer.zero_grad()

            # Pure MPS Execution
            outputs = model(cc, mlo)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_labels = []

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")
        with torch.no_grad():
            for batch in val_pbar:
                cc = batch['cc_patches'].to(device)
                mlo = batch['mlo_patches'].to(device)
                labels = batch['label'].to(device).unsqueeze(1)

                outputs = model(cc, mlo)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu().numpy()
                all_val_preds.extend(probs)
                all_val_labels.extend(labels.cpu().numpy())

                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        try:
            val_auc = roc_auc_score(all_val_labels, all_val_preds)
        except ValueError:
            val_auc = 0.5

        print(
            f"\n✅ Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val AUC: {val_auc:.4f}\n")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_dicom_mps_model.pth')
            print(f"⭐ New Best MPS DICOM Model Saved with AUC: {best_val_auc:.4f}!")


if __name__ == "__main__":
    pass