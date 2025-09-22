# 🧠 Alzheimer’s Disease Detection using Deep Learning

This project implements deep learning models for **MRI-based classification of Alzheimer’s Disease stages**, including:

- **VGG16** (baseline CNN with fine-tuning)
- **EfficientNetB4 with Attention** (advanced model for better feature extraction)
- **Dual-Input Fusion Model** (combines VGG16 and EfficientNetB4 features)

The models classify MRI scans into four classes:

- **AD** → Alzheimer’s Disease
- **CN** → Cognitively Normal
- **EMCI** → Early Mild Cognitive Impairment
- **LMCI** → Late Mild Cognitive Impairment

---

## 📂 Project Structure

Alzheimers-Detection-DeepLearning/
│── README.md
│── requirements.txt
│── VGG16_train.py
│── EfficientNetB4_train.py
│── DualInput_train.py

- `VGG16_train.py` → Training script for VGG16
- `EfficientNetB4_train.py` → Training script with EfficientNetB4 + Attention
- `DualInput_train.py` → Training script combining VGG16 + EfficientNetB4
- `requirements.txt` → List of dependencies

---

## 📊 Dataset

We use the [**Augmented Alzheimer’s Dataset**](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset) from Kaggle, which contains four classes of brain MRI images:

- **AD, CN, EMCI, LMCI**

### Preprocessing steps include:

- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Resizing images (224×224 for VGG16, 380×380 for EfficientNetB4)
- Normalization & augmentation (rotation, flips, shifts, zoom, brightness jitter)
- Handling class imbalance (class weights / oversampling)

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Satish1222231/Alzheimers-Classification.git
cd Alzheimers-Classification
pip install -r requirements.txt
```
