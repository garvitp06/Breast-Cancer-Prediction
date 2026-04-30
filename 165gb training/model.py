# model.py

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class CrossViewTransformer(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, num_layers=4):
        super(CrossViewTransformer, self).__init__()

        # 1. Feature Extractor (Backbone)
        # Fixed the warning by using the modern weights parameter
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.feature_projection = nn.Linear(512, feature_dim)

        # 2. Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 60, feature_dim))

        # 3. Cross-Attention Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )

        # 4. Final Classification Head
        # FIX: Changed feature_dim * 2 to just feature_dim (512)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, cc_patches, mlo_patches):
        b, s, c, h, w = cc_patches.shape

        cc_flat = cc_patches.view(-1, c, h, w)
        mlo_flat = mlo_patches.view(-1, c, h, w)

        cc_feats = self.backbone(cc_flat).view(b, s, -1)
        mlo_feats = self.backbone(mlo_flat).view(b, s, -1)

        cc_feats = self.feature_projection(cc_feats) + self.pos_embedding
        mlo_feats = self.feature_projection(mlo_feats) + self.pos_embedding

        # Concatenate along the sequence dimension
        # Output shape: (Batch, 120, 512)
        combined_feats = torch.cat([cc_feats, mlo_feats], dim=1)

        transformed = self.transformer(combined_feats)

        # Global Average Pooling across the 120 sequence items
        # Output shape: (Batch, 512)
        pooled = torch.mean(transformed, dim=1)

        # Final prediction
        logits = self.classifier(pooled)

        # Return raw logits so BCEWithLogitsLoss handles it safely in train.py!
        return logits