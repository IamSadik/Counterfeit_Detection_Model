# Capsule Counterfeit Detection using YOLOv8

This project aims to detect and verify counterfeit medicinal capsules using a deep learning approach. Built on the YOLOv8 object detection framework, the system identifies whether a capsule is **genuine** or **counterfeit** based on visual characteristics.

---

## 📁 Project Structure
**```**
capsule-counterfeit-detection/
├── datasets/
│ └── capsule_counterfeit_dataset/
│ ├── train/
│ │ ├── images/
│ │ └── labels/
│ ├── valid/
│ │ ├── images/
│ │ └── labels/
│ ├── test/
│ │ ├── images/
│ │ └── labels/
│ └── capsule_data.yaml
│
├── models/
│ ├── verification_model/ # Siamese or triplet-based model (future)
│ └── counterfeit_capsule_model/ # Trained YOLOv8 model weights
│
├── src/
│ ├── train.py # YOLOv8 training script
│ ├── train_verification.py # Triplet/Siamese model training (future)
│ ├── evaluate.py # YOLOv8 evaluation script
│ ├── evaluate_verification.py # Verification model evaluation (future)
│ ├── preprocess.py # Image/data prep for YOLO
│ └── preprocess_verification.py # Image/data prep for verification model
│
├── requirements.txt
├── README.md
└── .gitignore
**```**

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
✅ 1. Install Dependencies
git clone https://github.com/IamSadik/Counterfeit_Detection_Model.git
cd capsule-counterfeit-detection
pip install -r requirements.txt

⚙️ 2. Train the Model
python src/train.py

📊 3. Evaluate the Model
python src/evaluate.py



🔍 Features
Multi-class detection (8 capsule types)

Lightweight and fast model (YOLOv8n or YOLOv8s)

Optimized for real-time verification scenarios

Trained on clean Roboflow-curated dataset

Ready-to-use for deployment or further fine-tuning

🛠️ Future Enhancements
 Incorporate triplet-loss based verification network

 Add Grad-CAM or heatmaps for model explainability

 Convert into a Flask/Streamlit web app for demo

 Expand dataset to cover more capsule brands

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change or add.

