import cv2
from pathlib import Path

img_path = Path("datasets/counterfeit_med_detection/train/images")
label_path = Path("datasets/counterfeit_med_detection/train/labels")

sample = list(label_path.glob("*.txt"))[0]  # pick first label file
img_file = img_path / f"{sample.stem}.jpg"

print("Sample Label File:", sample)
print("Image Exists:", img_file.exists())

with open(sample, "r") as f:
    print("Label File Content:")
    print(f.read())

img = cv2.imread(str(img_file))
print("Image shape:", img.shape if img is not None else "Image not loaded")
