# dataset1.py

import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms.functional as TF


class PairedMammoDataset(Dataset):
    def __init__(self, dataframe, image_dir='', patch_size=256, num_patches=60):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.num_patches = num_patches

    def __len__(self):
        return len(self.dataframe)

    def extract_patches(self, tensor_img):
        _, h, w = tensor_img.shape
        patches = []

        stride_h = (h - self.patch_size) // int(np.sqrt(self.num_patches))
        stride_w = (w - self.patch_size) // int(np.sqrt(self.num_patches))

        if stride_h <= 0 or stride_w <= 0:
            resized = TF.resize(tensor_img, (self.patch_size * 8, self.patch_size * 8))
            return self.extract_patches(resized)

        for i in range(0, h - self.patch_size + 1, stride_h):
            for j in range(0, w - self.patch_size + 1, stride_w):
                patch = tensor_img[:, i:i + self.patch_size, j:j + self.patch_size]
                patches.append(patch)
                if len(patches) == self.num_patches:
                    break
            if len(patches) == self.num_patches:
                break

        while len(patches) < self.num_patches:
            patches.append(torch.zeros((1, self.patch_size, self.patch_size)))

        return torch.stack(patches)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Load the pre-computed .pt file directly into RAM!
        # We explicitly cast back to .float() (float32) because the MPS GPU prefers it
        cc_tensor = torch.load(row['cc_image_path'], weights_only=True).float()
        mlo_tensor = torch.load(row['mlo_image_path'], weights_only=True).float()

        cc_patches = self.extract_patches(cc_tensor)
        mlo_patches = self.extract_patches(mlo_tensor)

        label = torch.tensor(row['pathology'], dtype=torch.float32)

        return {
            'cc_patches': cc_patches,
            'mlo_patches': mlo_patches,
            'label': label
        }