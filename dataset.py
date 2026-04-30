# dataset.py

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os


class PatchExtractor(nn.Module):
    def __init__(self, patch_size=256):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        c, h, w = x.size()
        assert h % self.patch_size == 0, f"Height {h} not divisible by {self.patch_size}"
        assert w % self.patch_size == 0, f"Width {w} not divisible by {self.patch_size}"

        patches = x.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(c, -1, self.patch_size, self.patch_size)
        patches = patches.permute(1, 0, 2, 3)
        return patches


class PairedMammoDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe
        self.image_dir = image_dir

        self.base_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((2560, 1536)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.custom_transform = transform
        self.patch_extractor = PatchExtractor(patch_size=256)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        cc_path = os.path.join(self.image_dir, row['cc_image_path'])
        mlo_path = os.path.join(self.image_dir, row['mlo_image_path'])

        img_cc = Image.open(cc_path)
        img_mlo = Image.open(mlo_path)

        if self.custom_transform:
            img_cc = self.custom_transform(img_cc)
            img_mlo = self.custom_transform(img_mlo)

        tensor_cc = self.base_transform(img_cc)
        tensor_mlo = self.base_transform(img_mlo)

        patches_cc = self.patch_extractor(tensor_cc)
        patches_mlo = self.patch_extractor(tensor_mlo)

        label = torch.tensor(row['pathology'], dtype=torch.float32)
        age = torch.tensor([row.get('patient_age', 50) / 100.0], dtype=torch.float32)
        density = torch.tensor([row.get('breast_density', 1)], dtype=torch.long)

        return {
            'cc_patches': patches_cc,
            'mlo_patches': patches_mlo,
            'age': age,
            'density': density,
            'label': label
        }