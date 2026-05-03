import streamlit as st
import torch

# Disable MPS on macOS to prevent convolution operation crashes
torch.backends.mps.enabled = False

import cv2
import numpy as np
from PIL import Image
import time
import os
import sys

# ---------------------------------------------------------
# SETUP & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Low-Light Image Enhancement",
    layout="wide"
)

# Custom CSS for Professional Minimalist Styling
st.markdown("""
    <style>
    /* Global Background */
    .stApp {
        background: linear-gradient(180deg, #0f1115 0%, #17191e 100%);
        color: #e2e8f0;
        font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    /* Center alignments */
    .center-text {
        text-align: center;
    }
    
    /* Header Divider */
    .header-divider {
        height: 1px;
        background: #2d3748;
        margin: 30px 0;
        border: none;
    }
    
    /* Image Containers */
    .stImage > img {
        border-radius: 6px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
    }
    
    /* Metrics Box Styling */
    .metric-container {
        background-color: #1a202c;
        padding: 24px;
        border-radius: 6px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        border: 1px solid #2d3748;
    }
    .metric-title {
        color: #a0aec0;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        color: #e2e8f0;
        font-size: 1.25rem;
        font-weight: 300;
        margin-top: 8px;
    }
    
    /* Clean Uploader Container */
    .css-1n76uvr, .css-1n76uvr:focus {
        border: 1px solid #4a5568;
        border-radius: 6px;
        background-color: #1a202c;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Define model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'working', 'model.pth')

# ---------------------------------------------------------
# IMPORT MODEL FROM SRC
# ---------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
try:
    from model import UNetEnhancer
except ImportError:
    st.error("Model structure not found. Ensure 'src/model.py' exists in your workspace.")
    st.stop()

# ---------------------------------------------------------
# MODEL CACHING & INITIALIZATION
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        # Limit CPU threads to reduce system load and heating issues
        torch.set_num_threads(4)
        
        # FORCE CPU INSTEAD OF MPS to prevent "convolution_overrideable not implemented" crashes on Mac
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            # Force CPU on Mac to avoid MPS issues with convolution operations
            device = torch.device("cpu")
        
        if not os.path.exists(MODEL_PATH):
            st.warning(f"⚠️ Model file not found at {MODEL_PATH}")
            return None, device
            
        model = UNetEnhancer().to(device)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        # Handle both direct state dict and checkpoint dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        return model, device
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, torch.device("cpu")

with st.spinner("Loading environment..."):
    model, device = load_model()

# ---------------------------------------------------------
# INFERENCE PIPELINE
# ---------------------------------------------------------
def preprocess_image(image: Image.Image):
    # Convert PIL Image to numpy array (RGB)
    img_rgb = np.array(image.convert('RGB'))
    
    # Normalize to [0,1]
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # Convert to tensor + permute (HWC -> CHW) + unsqueeze
    input_tensor = torch.tensor(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
    
    # Pad image to a multiple of 16 to support UNet architecture natively at FULL resolution
    import torch.nn.functional as F
    _, _, h, w = input_tensor.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h > 0 or pad_w > 0:
        input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        
    return input_tensor, img_rgb.shape[:2]

def postprocess_tensor(tensor: torch.Tensor, original_size):
    original_h, original_w = original_size
    
    # Crop back to exact original size to remove the padding
    tensor = tensor[:, :, :original_h, :original_w]
    
    # Remove batch dimension and move to CPU
    tensor = tensor.squeeze(0).cpu()
    
    # Convert tensor -> numpy (CHW -> HWC)
    output_np = tensor.permute(1, 2, 0).numpy()
    
    # Clip values explicitly
    output_clipped = np.clip(output_np, 0.0, 1.0)
    
    # Convert back to uint8 safely
    img_uint8 = (output_clipped * 255.0).astype(np.uint8)
    return Image.fromarray(img_uint8)

# ---------------------------------------------------------
# MAIN DASHBOARD UI
# ---------------------------------------------------------
st.markdown("<h1 class='center-text'>Low-Light Image Enhancement</h1>", unsafe_allow_html=True)
st.markdown("<p class='center-text' style='color: #a0aec0; font-size: 1.1rem; margin-bottom: 20px;'>Advanced neural enhancement pipeline for restoring visibility in underexposed photography.</p>", unsafe_allow_html=True)

st.markdown("<hr class='header-divider'>", unsafe_allow_html=True)

if model is None:
    st.error("Trained model weights not found. Please train the model first by running the training pipeline.")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Select an image for processing", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Side-by-Side Layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h4 class='center-text'>Original Image</h4>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            
        with col2:
            st.markdown("<h4 class='center-text'>Enhanced Image</h4>", unsafe_allow_html=True)
            
            start_time = time.time()
            input_tensor, orig_size = preprocess_image(image)
            input_tensor = input_tensor.to(device)
            
            with st.spinner("Enhancing image..."):
                with torch.no_grad():
                    output_tensor = model(input_tensor)
                    
                enhanced_image = postprocess_tensor(output_tensor, orig_size)
                
            process_time = time.time() - start_time
            
            st.image(enhanced_image, use_container_width=True)
            
        st.markdown("<br><hr class='header-divider'><br>", unsafe_allow_html=True)
        
        # Performance Metrics Footer
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"<div class='metric-container'><div class='metric-title'>Original Resolution</div><div class='metric-value'>{orig_size[1]} × {orig_size[0]}</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-container'><div class='metric-title'>Inference Time</div><div class='metric-value'>{process_time:.2f}s</div></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='metric-container'><div class='metric-title'>Compute Backend</div><div class='metric-value'>{str(device).upper()}</div></div>", unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
