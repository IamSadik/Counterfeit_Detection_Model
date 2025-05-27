import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.capsule_verification_model import CapsuleVerificationModel
from utils.triplet_loss import triplet_loss_with_mining
from src.dataset import CapsuleDataset

def train(model, train_dataset, valid_dataset, device, epochs=10, batch_size=8, learning_rate=1e-4):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            loss = triplet_loss_with_mining(model, images, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
        
        if epoch % 2 == 0:
            model.eval()
            validate(model, valid_loader, device)

def validate(model, valid_loader, device):
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            # Optional: add accuracy evaluation
            pass

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CapsuleDataset(split="train")
    valid_dataset = CapsuleDataset(split="valid")

    model = CapsuleVerificationModel()
    model.to(device)

    train(model, train_dataset, valid_dataset, device)
