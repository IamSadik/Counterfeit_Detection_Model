import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from pathlib import Path
import os
import gc  # for memory cleanup

# -------------------------------
# Dataset
# -------------------------------
class VerificationDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img1 = Image.open(row["image1"]).convert("RGB")
        img2 = Image.open(row["image2"]).convert("RGB")
        label = torch.tensor([float(row["label"])])

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

    def __len__(self):
        return len(self.df)

# -------------------------------
# Siamese Network
# -------------------------------
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=True)
        base.fc = nn.Linear(base.fc.in_features, 256)
        self.feature_extractor = base
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        return self.feature_extractor(x)

    def forward(self, x1, x2):
        f1 = self.forward_once(x1)
        f2 = self.forward_once(x2)
        distance = torch.abs(f1 - f2)
        return self.head(distance)

# -------------------------------
# Training Loop
# -------------------------------
def train_siamese():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🖥️ Using device:", device)

    # Path to verification dataset CSV
    csv_path = Path("verification_dataset/verification_pairs.csv")
    
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])

    dataset = VerificationDataset(csv_path, transform=transform)

    # Efficient batch size for most PCs (adjust if needed)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

    model = SiameseNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    EPOCHS = 30
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for img1, img2, labels in dataloader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Free memory
            del img1, img2, labels, outputs, loss
            torch.cuda.empty_cache()
            gc.collect()

        print(f"Epoch [{epoch}/{EPOCHS}] - Loss: {total_loss / len(dataloader):.4f}")

    # Save the model
    model_dir = Path("models/verification_model")
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "siamese.pth")
    print(f"✅ Model saved to {model_dir / 'siamese.pth'}")

if __name__ == "__main__":
    train_siamese()
# This script trains a Siamese network for image verification using a dataset of image pairs.
# It uses a ResNet18 backbone to extract features and a simple head to predict similarity.