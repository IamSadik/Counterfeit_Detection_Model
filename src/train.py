# src/train.py

import os
from ultralytics import YOLO

# Configuration
DATA_YAML = os.path.abspath("datasets/counterfeit_med_detection/data.yaml")
MODEL_NAME = "yolov8n.pt"  # You can switch to yolov8s.pt or yolov8m.pt for better accuracy
SAVE_DIR = os.path.abspath("models")  # Directory to save the trained model

# Training hyperparameters
EPOCHS = 10
BATCH_SIZE = 8
IMG_SIZE = 1280

def train_model():
    print("🚀 Training started on counterfeit_med_detection dataset...")

    model = YOLO(MODEL_NAME)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        project=SAVE_DIR,
        name="counterfeit_capsule_model",
        pretrained=True
    )

    print(f"\n✅ Training complete! Model saved in: {SAVE_DIR}/counterfeit_capsule_model")

if __name__ == "__main__":
    train_model()
