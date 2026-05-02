# Deep Low-Light Image Enhancement System

A production-grade neural network pipeline for restoring visibility, color, and sharp details in extremely underexposed photography. This project features a custom U-Net architecture built in PyTorch with advanced embedded tensor-based post-processing.

## 🌟 Key Features

* **Guided Illumination Strategy:** Extracts a perfectly smooth illumination map to scale brightness without introducing neural-network "halos", "checkerboard artifacts", or blurry textures.
* **100% Resolution Preservation:** Unlike typical models that resize inputs to 256x256 and destroy sharpness, this pipeline pads and processes images at full high-resolution.
* **Native Tensor Post-Processing:** The `forward()` pass has been mathematically engineered to handle edge cases natively:
  * *Bilateral Noise Filtering:* Kills static camera grain in pitch-black shadows before enhancement.
  * *Deep Shadow Masking:* Dynamically forces pure noise to stay dark, preventing the "gray/grainy" shadow effect.
  * *Adaptive Gamma & Saturation:* Restores washed-out colors typically lost during aggressive artificial brightening.
* **Streamlit Dashboard:** A sleek, minimal web interface allowing users to drag, drop, and process images in real time.

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed. You will need `torch`, `torchvision`, `opencv-python`, `Pillow`, `numpy`, and `streamlit`.

```bash
pip install -r requirements.txt
```

### Running the Application
The repository comes pre-loaded with the trained weights (`working/model.pth`). You do not need to retrain the model to use the app!

Simply launch the Streamlit dashboard:
```bash
python3 -m streamlit run app.py
```
This will open the dashboard in your default web browser (usually at `http://localhost:8501`).

## 🧠 Architecture Details

* **Base Model:** Fully Convolutional U-Net.
* **Backend:** PyTorch (with MPS fallback handling for Apple Silicon).
* **Pipeline:** 
  1. Input image undergoes Bilateral Denoising.
  2. Image is downsampled strictly for generating a stable light map.
  3. U-Net infers the structural light changes.
  4. The Illumination Map is extracted, smoothed, and upscaled.
  5. The Map is multiplied against the perfectly sharp, original high-resolution image.

## 📊 Proof of Generalization
While the model was trained on the LOL (Low-Light) dataset, it has been robustly tested against out-of-distribution (OOD) random un-paired low-light images from the internet. The model correctly outputs balanced, decent highlights and lifted shadows without catastrophic overfitting, proving it learned generalized lighting feature-maps rather than memorizing training data.