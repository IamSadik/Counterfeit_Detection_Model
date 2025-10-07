# Capsule Counterfeit Detection using YOLOv8

This project aims to detect and verify counterfeit medicinal capsules using a deep learning approach. Built on the YOLOv8 object detection framework, the system identifies whether a capsule is **genuine** or **counterfeit** based on visual characteristics.

---

## 📁 Project Structure
```
capsule-counterfeit-detection/
├── .git/
├── datasets/
├── models/
│   ├── counterfeit_capsule_model/
│   └── verification_model/
├── results/
├── runs/
├── src/
│   ├── train.py
│   ├── train_verification.py
│   ├── evaluate.py
│   ├── evaluate_verification.py
│   ├── preprocess.py
│   └── preprocess_verification.py
├── test/
├── utils/
├── verification_dataset/
│   ├── authentic_BrandW/
│   ├── authentic_BrandX/
│   ├── authentic_BrandY/
│   ├── authentic_BrandZ/
│   ├── counterfeit_BrandW/
│   ├── counterfeit_BrandX/
│   ├── counterfeit_BrandY/
│   ├── counterfeit_BrandZ/
│   └── verification_pairs.csv
│
├── verification_dataset_test/
│   ├── authentic_BrandW/
│   ├── authentic_BrandX/
│   ├── authentic_BrandY/
│   ├── authentic_BrandZ/
│   ├── counterfeit_BrandW/
│   ├── counterfeit_BrandX/
│   ├── counterfeit_BrandY/
│   ├── counterfeit_BrandZ/
│   └── verification_pairs.csv
│
├── .gitignore
├── main.py
├── README.md
├── requirements.txt
├── yolov8m.pt
└── yolov10n.pt


```

---

## 🧠 Model Overview

- **Type**: Object Detection / Verification
- **Framework**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Input**: RGB images of medicinal capsules
- **Output**: Bounding boxes with class labels:
  - `genuine_capsule`
  - `counterfeit_capsule`

## 🧠 Model Overview

- **Model Type**: Object Detection & Multi-class Classification
- **Framework**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Input**: Images of medicinal capsules
- **Output**: Bounding boxes with one of 8 class labels

---

## 🧾 Dataset Overview

- **Format**: YOLOv8 format (images + `.txt` labels)
- **Source**: [Roboflow](https://universe.roboflow.com/medetect/medetect-9kphx/dataset/11)
- **License**: CC BY 4.0

**Classes (Total: 8):**
0: authentic_BrandW
1: authentic_BrandX
2: authentic_BrandY
3: authentic_BrandZ
4: counterfeit_BrandW
5: counterfeit_BrandX
6: counterfeit_BrandY
7: counterfeit_BrandZ

**Sample `data.yaml`:**
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 8
names: ['authentic_BrandW', 'authentic_BrandX', 'authentic_BrandY', 'authentic_BrandZ',
        'counterfeit_BrandW', 'counterfeit_BrandX', 'counterfeit_BrandY', 'counterfeit_BrandZ']

roboflow:
  workspace: medetect
  project: medetect-9kphx
  version: 11
  license: CC BY 4.0
  url: https://universe.roboflow.com/medetect/medetect-9kphx/dataset/11
```


🚀 How to Run

©️ 0. Clone Project
```
git clone https://github.com/IamSadik/Counterfeit_Detection_Model.git

```
✅ 1. Install Dependencies
```
pip install -r requirements.txt

```
⚙️ 2. Train the Model
```
python src/train.py

```
📊 3. Evaluate the Model
```
python src/evaluate.py

```

---
🔍 Features

✅ Multi-Class Detection – Detects 8 distinct capsule classes (authentic + counterfeit across 4 brands)

⚡ Lightweight Architecture – Powered by YOLOv8n/s for fast inference and low-latency use cases

📦 Production-Ready – Trained on a clean, curated Roboflow dataset

🚀 Real-Time Capable – Optimized for live verification scenarios

🔁 Easily Extendable – Fine-tune or scale with minimal setup

---
🛠️ Future Enhancements

🔁 Integrate Triplet-Loss Verification Models (e.g., Siamese Network) for one-shot/few-shot learning

🧠 Add Grad-CAM / Heatmaps for model interpretability and capsule focus visualization

🌐 Deploy as a Flask/Streamlit Web App for interactive demos and real-world testing

📈 Expand the dataset to include more capsule brands and variations

---
🤝 Contributing

We welcome contributions!
If you're planning major changes, please open an issue first to discuss the scope and approach.
Pull requests for improvements, bug fixes, or new features are always appreciated!

