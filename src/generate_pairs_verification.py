import os
import csv
import random
from pathlib import Path

# Constants
BASE_DIR = Path("verification_dataset")
OUTPUT_CSV = Path("verification_dataset/verification_pairs.csv")
TOTAL_PAIRS = 10_000
POSITIVE_COUNT = TOTAL_PAIRS // 2
NEGATIVE_COUNT = TOTAL_PAIRS // 2

# Gather all images by class
class_folders = [f for f in BASE_DIR.iterdir() if f.is_dir()]
class_to_images = {cls.name: list(cls.glob("*.jpg")) for cls in class_folders}

# Function to create positive pairs
def generate_positive_pairs(count):
    pairs = []
    for class_name, images in class_to_images.items():
        if len(images) < 2:
            continue
        for _ in range(count // len(class_to_images)):
            img1, img2 = random.sample(images, 2)
            pairs.append((img1.as_posix(), img2.as_posix(), 1))
    return pairs

# Function to create negative pairs
def generate_negative_pairs(count):
    pairs = []
    class_names = list(class_to_images.keys())
    while len(pairs) < count:
        cls1, cls2 = random.sample(class_names, 2)
        imgs1, imgs2 = class_to_images[cls1], class_to_images[cls2]
        if imgs1 and imgs2:
            img1 = random.choice(imgs1)
            img2 = random.choice(imgs2)
            pairs.append((img1.as_posix(), img2.as_posix(), 0))
    return pairs

# Generate pairs
print("🔄 Generating image pairs...")
positive_pairs = generate_positive_pairs(POSITIVE_COUNT)
negative_pairs = generate_negative_pairs(NEGATIVE_COUNT)
all_pairs = positive_pairs + negative_pairs
random.shuffle(all_pairs)

# Save to CSV
os.makedirs(OUTPUT_CSV.parent, exist_ok=True)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image1", "image2", "label"])
    writer.writerows(all_pairs)

print(f"✅ Done! {len(all_pairs)} pairs saved to {OUTPUT_CSV}")
