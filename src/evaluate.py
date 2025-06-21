# src/evaluate.py

import os
from ultralytics import YOLO

# Paths
MODEL_PATH = os.path.abspath("models/counterfeit_capsule_model_v2/weights/best.pt")
DATA_YAML = os.path.abspath("datasets/counterfeit_med_detection/data.yaml")
IMG_SIZE = 1440  # high resolution consistent with training

def evaluate_model():
    print("📊 Evaluating the trained model on test data...\n")

    model = YOLO(MODEL_PATH)

    # Perform validation
    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        split='test'
    )

    # Extract important results
    results = metrics.results_dict
    fitness = results.get('fitness', 0.0)

    print("✅ Evaluation Complete\n")
    print("📈 Evaluation Metrics Summary:")
    print("=" * 35)
    print(f"🔹 Precision (B):        {results['metrics/precision(B)']:.4f}")
    print(f"🔹 Recall (B):           {results['metrics/recall(B)']:.4f}")
    print(f"🔹 mAP@0.5 (B):          {results['metrics/mAP50(B)']:.4f}")
    print(f"🔹 mAP@0.5:0.95 (B):     {results['metrics/mAP50-95(B)']:.4f}")
    print(f"⭐ Fitness Score:        {fitness:.4f}")
    print("=" * 35)

    print(f"\n📂 Results saved to: {metrics.save_dir}")

if __name__ == "__main__":
    evaluate_model()
