import cv2
import torch
import numpy as np
from PIL import Image
import os
import sys

# Import model architecture securely
from model import UNetEnhancer

def enhance_new_image(image_path, model_path="training_output/best_model.pth", output_path="enhanced_output.png"):
    """
    Complete corrected inference pipeline for PyTorch low-light image enhancement.
    This fixes overexposed highlights, glowing edges, and washed-out colors.
    """
    # Limit CPU threads to reduce system load
    torch.set_num_threads(4)
    
    # FORCE CPU INSTEAD OF MPS to prevent PyTorch crashes on Mac
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 1. INPUT PREPROCESSING ---
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    # Read image using OpenCV (BGR format)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to load image at {image_path}")
        
    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = img_rgb.shape[:2]
    
    # Detect if image is extreme dark (mean intensity < 0.05)
    mean_intensity = np.mean(img_rgb) / 255.0
    is_extreme_dark = mean_intensity < 0.05
    
    # Resize to (256, 256) using high-quality interpolation
    img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    
    # Normalize to [0,1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convert to tensor + permute (HWC -> CHW) + unsqueeze (add batch dim)
    input_tensor = torch.tensor(img_normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # --- 2. MODEL INFERENCE ---
    model = UNetEnhancer().to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
        
    # Load model weights safely
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # VERY IMPORTANT: Use eval() mode and disable gradients
    model.eval() 
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    # --- 3. OUTPUT POSTPROCESSING (CRITICAL FIX) ---
    # Remove batch dimension and move to CPU
    output_tensor = output_tensor.squeeze(0).cpu()
    
    # Convert tensor -> numpy (CHW -> HWC)
    output_np = output_tensor.permute(1, 2, 0).numpy()
    
    # Clip values to [0,1] strictly
    output_clipped = np.clip(output_np, 0.0, 1.0)
    
    # Resize back to original size
    output_resized = cv2.resize(output_clipped, (original_w, original_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Gamma correction to balance midtones (less aggressive for extreme dark)
    gamma = 0.9 if is_extreme_dark else 0.8
    output_gamma = output_resized ** gamma
    
    # --- 4. ADD POST-PROCESSING IMPROVEMENTS ---
    
    # Fix Washed-Out Colors: Boost saturation
    # Convert float32 RGB [0,1] to HSV [0-360, 0-1, 0-1]
    hsv = cv2.cvtColor(output_gamma.astype(np.float32), cv2.COLOR_RGB2HSV)
    # Boost saturation (less aggressively for dark images to avoid color noise)
    sat_boost = 1.1 if is_extreme_dark else 1.25
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_boost, 0.0, 1.0)
    enhanced_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Prevent Blown Highlights & Glowing Edges:
    # Blend with original image in bright areas to prevent overexposure
    orig_norm = img_rgb.astype(np.float32) / 255.0
    orig_lum = cv2.cvtColor(orig_norm, cv2.COLOR_RGB2GRAY)
    orig_lum = np.expand_dims(orig_lum, axis=2)
    
    # Soft mask: 0 in dark areas, up to 1 in bright areas
    highlight_mask = np.clip((orig_lum - 0.5) * 2.5, 0.0, 1.0)
    
    # Combine: Keep enhancement in shadows, use original for highlights
    final_output = enhanced_rgb * (1.0 - highlight_mask) + orig_norm * highlight_mask
    
    # --- 5. HANDLE EXTREME DARK CASES ---
    if is_extreme_dark:
        # Prevent unrealistic flat outputs/noise by reducing unnatural brightness
        final_output = np.clip(final_output * 0.85, 0.0, 1.0)
        
    # --- 6. FINAL IMAGE CONVERSION ---
    # Convert back to uint8 safely
    final_img_uint8 = (final_output * 255.0).astype(np.uint8)
    
    # Convert RGB -> BGR for OpenCV saving
    final_img_bgr = cv2.cvtColor(final_img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, final_img_bgr)
    
    print(f"Enhanced image successfully saved to {output_path}")
    return final_img_uint8

if __name__ == "__main__":
    # You can call this file directly to test the pipeline
    # Example usage:
    # enhance_new_image("../lol_dataset/eval15/low/1.png")
    pass
