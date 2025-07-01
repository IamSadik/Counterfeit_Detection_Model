import os
import cv2
from pathlib import Path

# Paths for TEST data
images_path = Path("datasets/counterfeit_med_detection/test/images")
labels_path = Path("datasets/counterfeit_med_detection/test/labels")
output_path = Path("verification_dataset_test")

# Class folders
class_names = [
    'authentic_BrandW', 'authentic_BrandX', 'authentic_BrandY', 'authentic_BrandZ',
    'counterfeit_BrandW', 'counterfeit_BrandX', 'counterfeit_BrandY', 'counterfeit_BrandZ'
]

for class_name in class_names:
    os.makedirs(output_path / class_name, exist_ok=True)

# Class mapping
class_map = {i: name for i, name in enumerate(class_names)}

# Process labels
for label_file in labels_path.glob("*.txt"):
    img_file_jpg = images_path / f"{label_file.stem}.jpg"
    img_file_png = images_path / f"{label_file.stem}.png"
    img_file = img_file_jpg if img_file_jpg.exists() else img_file_png

    if not img_file.exists():
        print(f"❌ Image not found for: {label_file.stem}")
        continue

    image = cv2.imread(str(img_file))
    if image is None:
        print(f"❌ Failed to load image: {img_file}")
        continue

    h, w = image.shape[:2]

    with open(label_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            parts = list(map(float, line.strip().split()))
            if len(parts) < 6:
                print(f"⚠️ Skipping line with too few points: {line}")
                continue

            class_id = int(parts[0])
            coords = parts[1:]
            xs = coords[::2]
            ys = coords[1::2]

            x_min = int(min(xs) * w)
            x_max = int(max(xs) * w)
            y_min = int(min(ys) * h)
            y_max = int(max(ys) * h)

            # Safe boundary clipping
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            # Skip if bbox too small or invalid
            if x_max - x_min < 5 or y_max - y_min < 5:
                print(f"⚠️ Skipping tiny or invalid crop for {img_file.name} index {i}")
                continue

            cropped = image[y_min:y_max, x_min:x_max]
            save_path = output_path / class_map[class_id] / f"{label_file.stem}_{i}.jpg"
            cv2.imwrite(str(save_path), cropped)
            print(f"✅ Saved: {save_path}")
