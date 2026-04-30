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

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on: {device}")


def train_model(df_paired, image_dir, epochs=20, batch_size=4):
    full_dataset = PairedMammoDataset(df_paired, image_dir)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = CrossViewTransformer().to(device)

    # 1. DIFFERENTIAL LEARNING RATES
    # The ResNet is already smart; it needs a tiny learning rate (1e-5).
    # The Transformer is randomly initialized; it needs a larger one (1e-4).
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.feature_projection.parameters(), 'lr': 1e-4},
        {'params': model.transformer.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)

    # 2. LEARNING RATE SCHEDULER
    # Gradually reduces the learning rate to help the model settle into the minimum
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. CLASS IMBALANCE HANDLING
    # Malignant cases are rare. We force the loss function to penalize the model
    # 5x harder if it misses a malignant case (pos_weight).
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
            outputs = model(cc, mlo)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Step the scheduler at the end of the epoch
        scheduler.step()

        # --- VALIDATION AND AUC TRACKING ---
        model.eval()
        val_loss = 0

        # We need to collect all predictions and true labels to calculate AUC
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

                # Apply sigmoid to get probabilities for the AUC calculation
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_val_preds.extend(probs)
                all_val_labels.extend(labels.cpu().numpy())

                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        # 4. CALCULATE ROC-AUC
        try:
            val_auc = roc_auc_score(all_val_labels, all_val_preds)
        except ValueError:
            # Occurs if a small validation batch randomly gets only 1 class (all benign)
            val_auc = 0.5

        print(
            f"\n✅ Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val AUC: {val_auc:.4f}\n")

        # Save model based on highest AUC, not lowest loss!
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_mammo_model_Latest.pth')
            print(f"⭐ New Best Model Saved with AUC: {best_val_auc:.4f}!")


if __name__ == "__main__":
    pass