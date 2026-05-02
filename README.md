# 🌙 Deep Low-Light Image Enhancement System

A production-grade neural network pipeline designed to restore visibility, color fidelity, and sharp details in severely underexposed photography. This project implements a custom U-Net architecture built in PyTorch, enhanced with mathematically robust native tensor post-processing to handle noise amplification and color distortion.

---

## 🎯 Project Overview

Standard low-light enhancement models often suffer from "halo" artifacts, amplified camera noise, and washed-out colors when deployed in real-world scenarios. This project bridges the gap between a standard academic U-Net and a production-ready application by embedding dynamic contrast masking, saturation boosting, and full-resolution handling natively into the model's inference pipeline.

## 🌟 Core Technical Features

* **100% Resolution Preservation:** Instead of downsampling images to 256x256 (which destroys high-frequency details), the inference pipeline mathematically pads images to multiples of 16, allowing the fully-convolutional U-Net to process 4K images natively without losing sharpness.
* **Embedded Tensor Post-Processing:** The `forward()` pass has been mathematically engineered to handle edge cases natively on the GPU/CPU:
  * *Deep Shadow Masking:* Dynamically detects pitch-black pixels and forces them to stay dark, preventing the artificial "gray/grainy" shadow effect common in enhanced low-light images.
  * *Adaptive Gamma & Saturation:* Automatically balances mid-tones and restores washed-out colors lost during the aggressive artificial brightening process.
* **Interactive Streamlit Dashboard:** A sleek, minimal web interface allowing users to drag, drop, and process images in real time with side-by-side comparisons.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running the Application (Inference)
The repository comes pre-loaded with optimized trained weights (`working/model.pth`). You do **not** need the training dataset to run the application!

Simply launch the Streamlit dashboard:
```bash
python3 -m streamlit run app.py
```
This will automatically open the dashboard in your default web browser (typically at `http://localhost:8501`).

---

## 🧠 Training the Model from Scratch (Optional)

Because the training dataset consists of high-resolution images, it exceeds GitHub's file size limits and is not included in this repository. If you wish to retrain the model or experiment with the architecture, follow these steps:

**1. Download the Dataset**
* This project utilizes the standard **LOL (Low-Light) Dataset**.
* You can easily download it publicly from [Kaggle's LOL Dataset Page](https://www.kaggle.com/datasets/bguberfain/lol-dataset) or the official academic source.

**2. Organize the Directory**
Extract the dataset and place it in the root directory of this project so the folder structure looks exactly like this:
```text
image_enhancer_project/
├── lol_dataset/
│   ├── eval15/
│   │   ├── high/
│   │   └── low/
│   └── our485/
│       ├── high/
│       └── low/
├── src/
├── app.py
└── ...
```

**3. Run the Training Pipeline**
Once the dataset is accurately placed, execute the training script. The script automatically handles batching, data augmentation, and saves the best model weights:
```bash
python3 src/train.py
```

---

## 📊 Model Generalization & Performance
While the model was trained exclusively on the LOL dataset, it has been rigorously evaluated against out-of-distribution (OOD) random un-paired low-light images from external sources. The embedded dynamic masking guarantees that the model correctly outputs balanced highlights and lifted shadows without catastrophic overfitting, proving it learned generalized lighting feature-maps rather than simply memorizing the training data.