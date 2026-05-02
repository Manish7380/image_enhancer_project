import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dataset import LOLDataset
from model import UNetEnhancer

def opencv_baseline_clahe(image_rgb):
    """
    OpenCV Baseline Comparison using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    CLAHE is applied exclusively to the Lightness (L) channel of the LAB color space 
    to preserve color information and prevent color shifting.
    Input image must be uint8. Returns float32 [0, 1] RGB image.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb.astype(np.float32) / 255.0

def evaluate():
    DATA_ROOT = "../lol_dataset"
    if not os.path.exists(DATA_ROOT) and os.path.exists("lol_dataset"):
        DATA_ROOT = "lol_dataset"
        
    MODEL_PATH = "training_output/best_model.pth"
    OUTPUT_DIR = "evaluation_results"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Evaluating on device: {DEVICE}")

    # Load Model
    model = UNetEnhancer().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded trained model from {MODEL_PATH}")
    else:
        print(f"Warning: {MODEL_PATH} not found. Evaluating with untrained model.")
    model.eval()

    # Load Eval Dataset
    eval_dataset = LOLDataset(DATA_ROOT, mode='eval')
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)

    model_psnr_list, model_ssim_list = [], []
    cv2_psnr_list, cv2_ssim_list = [], []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_loader, desc="Evaluating Images")):
            input_tensor = batch['input'].to(DEVICE)
            target_tensor = batch['target'].to(DEVICE)
            name = batch['name'][0]
            
            # Forward pass
            output_tensor = model(input_tensor).clamp(0, 1)
            
            # Convert tensors to numpy arrays [H, W, C] in range [0, 1]
            input_rgb = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            target_rgb = target_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            output_rgb = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            
            # OpenCV baseline
            input_uint8 = (input_rgb * 255).astype(np.uint8)
            cv2_output_rgb = opencv_baseline_clahe(input_uint8)
            
            # Compute Metrics against TRUE GROUND TRUTH (target_rgb)
            model_psnr = peak_signal_noise_ratio(target_rgb, output_rgb, data_range=1.0)
            model_ssim = structural_similarity(target_rgb, output_rgb, channel_axis=-1, data_range=1.0)
            
            cv2_psnr = peak_signal_noise_ratio(target_rgb, cv2_output_rgb, data_range=1.0)
            cv2_ssim = structural_similarity(target_rgb, cv2_output_rgb, channel_axis=-1, data_range=1.0)
            
            model_psnr_list.append(model_psnr)
            model_ssim_list.append(model_ssim)
            cv2_psnr_list.append(cv2_psnr)
            cv2_ssim_list.append(cv2_ssim)
            
            # Save Visualization (Input | Model | OpenCV | Target)
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            axes[0].imshow(input_rgb)
            axes[0].set_title("Input (Low Light)")
            axes[0].axis("off")
            
            axes[1].imshow(output_rgb)
            axes[1].set_title(f"Model Output\nPSNR: {model_psnr:.2f} | SSIM: {model_ssim:.4f}")
            axes[1].axis("off")
            
            axes[2].imshow(cv2_output_rgb)
            axes[2].set_title(f"OpenCV (CLAHE)\nPSNR: {cv2_psnr:.2f} | SSIM: {cv2_ssim:.4f}")
            axes[2].axis("off")
            
            axes[3].imshow(target_rgb)
            axes[3].set_title("Ground Truth (High)")
            axes[3].axis("off")
            
            plt.tight_layout()
            save_path = os.path.join(OUTPUT_DIR, f"compare_{name}")
            plt.savefig(save_path)
            plt.close()

    # Final Statistics
    print("\n" + "="*50)
    print("               EVALUATION RESULTS               ")
    print("="*50)
    print(f"Images Evaluated: {len(eval_dataset)}")
    print(f"Model  - Mean PSNR: {np.mean(model_psnr_list):.2f} dB, Mean SSIM: {np.mean(model_ssim_list):.4f}")
    print(f"OpenCV - Mean PSNR: {np.mean(cv2_psnr_list):.2f} dB, Mean SSIM: {np.mean(cv2_ssim_list):.4f}")
    print("-" * 50)
    print(f"Gain over OpenCV - PSNR: {np.mean(model_psnr_list) - np.mean(cv2_psnr_list):.2f} dB")
    print(f"Gain over OpenCV - SSIM: {np.mean(model_ssim_list) - np.mean(cv2_ssim_list):.4f}")
    print("="*50)
    print(f"Visualizations saved to '{OUTPUT_DIR}' directory.")

if __name__ == '__main__':
    evaluate()
