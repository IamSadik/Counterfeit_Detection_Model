# src/evaluate.py

import os
from ultralytics import YOLO

# Paths
MODEL_PATH = os.path.abspath("models/counterfeit_capsule_model_v2/weights/best.pt")
DATA_YAML = os.path.abspath("datasets/counterfeit_med_detection/data.yaml")
IMG_SIZE = 1440  # Use high resolution consistent with training

def evaluate_model():
    print("📊 Evaluating the trained model on test data...")

    model = YOLO(MODEL_PATH)

    # Perform validation with the same image size used during training
    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,  # Ensure image size matches training
        split='test'
    )

    print("✅ Evaluation Complete")
    print(metrics)

if __name__ == "__main__":
    evaluate_model()
