import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os
from pathlib import Path
import numpy as np


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
        label = float(row["label"])

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

    def __len__(self):
        return len(self.df)

# -------------------------------
# Siamese Model
# -------------------------------
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=False)
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
# Evaluate on Dataset
# -------------------------------
def evaluate_from_dataset(model, device, transform):
    csv_path = "verification_dataset_test/verification_pairs_test.csv"
    dataset = VerificationDataset(csv_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_preds, all_labels = [], []
    results = []

    with torch.no_grad():
        for i, (img1, img2, labels) in enumerate(tqdm(dataloader, desc="🔎 Evaluating", unit="batch")):
            img1, img2 = img1.to(device), img2.to(device)
            outputs = model(img1, img2).squeeze().cpu().numpy()
            preds = (outputs > 0.5).astype(int)

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

            # Save per-pair results
            for j in range(len(preds)):
                row_idx = i * dataloader.batch_size + j
                img1_path = dataset.df.iloc[row_idx]["image1"]
                img2_path = dataset.df.iloc[row_idx]["image2"]
                true_label = int(labels[j].item())
                pred_label = int(preds[j])
                score = float(outputs[j]) if isinstance(outputs, (list, np.ndarray)) else float(outputs)
                results.append([img1_path, img2_path, true_label, pred_label, score])

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\n📊 Evaluation Results:")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # -------------------------------
    # Save Results
    # -------------------------------
    os.makedirs("results/verification_eval", exist_ok=True)

    # Save metrics
    with open("results/verification_eval/metrics.txt", "w") as f:
        f.write("Verification Evaluation Results\n")
        f.write("===============================\n")
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"Precision : {precision:.4f}\n")
        f.write(f"Recall    : {recall:.4f}\n")
        f.write(f"F1 Score  : {f1:.4f}\n")

    # Save prediction per pair
    result_df = pd.DataFrame(results, columns=["Image1", "Image2", "TrueLabel", "PredictedLabel", "SimilarityScore"])
    result_df.to_csv("results/verification_eval/predictions.csv", index=False)

    print("📁 Results saved to: results/verification_eval/")

# -------------------------------
# Custom Pair Evaluation
# -------------------------------
def evaluate_custom_pair(model, device, transform):
    print("\n📷 Enter two image paths to compare:")
    image1_path = input("Path to Image 1: ").strip()
    image2_path = input("Path to Image 2: ").strip()

    if not Path(image1_path).exists() or not Path(image2_path).exists():
        print("❌ One or both image paths are invalid.")
        return

    img1 = transform(Image.open(image1_path).convert("RGB")).unsqueeze(0).to(device)
    img2 = transform(Image.open(image2_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img1, img2).item()

    print(f"\n🔍 Similarity Score (0 = different, 1 = same): {output:.4f}")
    if output > 0.5:
        print("✅ Prediction: SAME capsule (likely authentic)")
    else:
        print("❌ Prediction: DIFFERENT capsules (possibly counterfeit)")

# -------------------------------
# Runner
# -------------------------------
def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🖥️ Using device:", device)

    model_path = "models/verification_model/siamese.pth"
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])

    print("\nChoose Evaluation Mode:")
    print("1️⃣  Custom Test (Input two image paths)")
    print("2️⃣  Evaluate full dataset from CSV")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        evaluate_custom_pair(model, device, transform)
    elif choice == "2":
        evaluate_from_dataset(model, device, transform)
    else:
        print("❌ Invalid option selected.")

if __name__ == "__main__":
    evaluate_model()
