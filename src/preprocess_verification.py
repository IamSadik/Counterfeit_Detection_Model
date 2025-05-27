import os
import random
from glob import glob
from PIL import Image
from tqdm import tqdm

# === CONFIG ===
SOURCE_DIR = "datasets/counterfeit_med_detection/train/images"
TARGET_DIR = "datasets/capsule_verification_dataset_triplet"
SPLITS = ["train", "valid", "test"]
TRIPLETS_PER_SPLIT = {"train": 3000, "valid": 800, "test": 800}

def create_dir(path):
    os.makedirs(path, exist_ok=True)

def get_class_images(source_dir):
    class_map = {}
    for img_path in glob(os.path.join(source_dir, "*.jpg")):
        filename = os.path.basename(img_path)
        cls = "_".join(filename.split("_")[:2])  # e.g., authentic_BrandX
        class_map.setdefault(cls, []).append(img_path)
    return class_map

def create_triplets(class_map, split, n_triplets):
    split_dir = os.path.join(TARGET_DIR, split)
    anchor_dir = os.path.join(split_dir, "anchor")
    positive_dir = os.path.join(split_dir, "positive")
    negative_dir = os.path.join(split_dir, "negative")
    label_file = os.path.join(split_dir, "triplets.txt")

    create_dir(anchor_dir)
    create_dir(positive_dir)
    create_dir(negative_dir)

    valid_classes = [cls for cls in class_map if len(class_map[cls]) >= 2]

    with open(label_file, "w") as f:
        for i in tqdm(range(n_triplets), desc=f"Generating {split} triplets"):
            pos_class = random.choice(valid_classes)
            anchor_img, positive_img = random.sample(class_map[pos_class], 2)

            neg_class = random.choice([cls for cls in class_map if cls != pos_class])
            negative_img = random.choice(class_map[neg_class])

            try:
                Image.open(anchor_img).save(os.path.join(anchor_dir, f"triplet_{i}.jpg"))
                Image.open(positive_img).save(os.path.join(positive_dir, f"triplet_{i}.jpg"))
                Image.open(negative_img).save(os.path.join(negative_dir, f"triplet_{i}.jpg"))
                f.write(f"triplet_{i}.jpg\n")
            except Exception as e:
                print(f"❌ Failed triplet {i}: {e}")
                continue

def preprocess_triplet_verification_dataset():
    print("🔍 Preparing triplet verification dataset...")
    class_map = get_class_images(SOURCE_DIR)
    print(f"✅ Found {len(class_map)} distinct classes.")

    for split in SPLITS:
        create_triplets(class_map, split, TRIPLETS_PER_SPLIT[split])

    print("✅ Triplet preprocessing complete at:", TARGET_DIR)

if __name__ == "__main__":
    preprocess_triplet_verification_dataset()
