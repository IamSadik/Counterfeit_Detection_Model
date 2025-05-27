# models/capsule_verification_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class CapsuleVerificationModel(nn.Module):
    def __init__(self, embedding_size=512):
        super(CapsuleVerificationModel, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.fc = nn.Linear(self.backbone[-1].in_features, embedding_size)
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)  # Normalize embeddings
        return x
