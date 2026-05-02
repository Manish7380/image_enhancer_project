import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset

class LOLDataset(Dataset):
    def __init__(self, root_dir, mode='train', image_size=(256, 256)):
        """
        LOL Dataset Loader with Strict Checks and Augmentation.
        """
        super().__init__()
        self.image_size = image_size
        self.mode = mode
        
        # Determine paths based on mode
        if mode == 'train':
            self.low_dir = os.path.join(root_dir, 'our485', 'low')
            self.high_dir = os.path.join(root_dir, 'our485', 'high')
        elif mode == 'eval' or mode == 'test':
            self.low_dir = os.path.join(root_dir, 'eval15', 'low')
            self.high_dir = os.path.join(root_dir, 'eval15', 'high')
        else:
            raise ValueError("mode must be 'train', 'eval' or 'test'")
            
        if not os.path.exists(self.low_dir) or not os.path.exists(self.high_dir):
            raise FileNotFoundError(f"Dataset directories not found for mode '{mode}' at {root_dir}.")

        # Dataset Validation Checks
        low_images = sorted([f for f in os.listdir(self.low_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        high_images = sorted([f for f in os.listdir(self.high_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if len(low_images) != len(high_images):
            print(f"Warning: Number of low ({len(low_images)}) and high ({len(high_images)}) images differ in {mode} mode.")
        
        # Ensure exact matching of filenames (Strict check to prevent pseudo-targets mapping)
        self.valid_images = []
        for name in low_images:
            if name in high_images:
                self.valid_images.append(name)
            else:
                print(f"Warning: Filename mismatch. Ground truth missing for {name}. Skipping.")

        print(f"[{mode.upper()}] Loaded {len(self.valid_images)} valid image pairs.")
        
        # Print sample shapes for robustness verification
        if len(self.valid_images) > 0:
            sample_low = self.read_image(os.path.join(self.low_dir, self.valid_images[0]))
            print(f"[{mode.upper()}] Sample Low Image Before Augmentation - Shape: {sample_low.shape}, Max Val: {sample_low.max()}")

    def __len__(self):
        return len(self.valid_images)

    def read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            # Handle failure safely by returning black image and printing warning
            print(f"ERROR: Failed to load image at {path}. Returning zeros.")
            return np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def augment(self, low, high):
        """
        Applies strict, identical spatial augmentations to BOTH images.
        Applies color augmentations ONLY to the low-light image.
        """
        # 1. Random Crop
        h, w = low.shape[:2]
        ch, cw = self.image_size
        
        if h > ch and w > cw:
            y = random.randint(0, h - ch)
            x = random.randint(0, w - cw)
            low = low[y:y+ch, x:x+cw]
            high = high[y:y+ch, x:x+cw]
        else:
            low = cv2.resize(low, self.image_size)
            high = cv2.resize(high, self.image_size)

        # 2. Horizontal Flip
        if random.random() > 0.5:
            low = cv2.flip(low, 1)
            high = cv2.flip(high, 1)
            
        # 3. Brightness/Contrast Adjustment (only on low-light input)
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2) # Contrast control [0.8-1.2]
            beta = random.uniform(-15, 15)   # Brightness control
            low = cv2.convertScaleAbs(low, alpha=alpha, beta=beta)
            
        return low, high

    def __getitem__(self, idx):
        name = self.valid_images[idx]
        low_path = os.path.join(self.low_dir, name)
        high_path = os.path.join(self.high_dir, name)
        
        low_img = self.read_image(low_path)
        high_img = self.read_image(high_path)
        
        if self.mode == 'train':
            # Apply Augmentations during Training
            low_img, high_img = self.augment(low_img, high_img)
        else:
            # Simple resize for Evaluation
            low_img = cv2.resize(low_img, self.image_size)
            high_img = cv2.resize(high_img, self.image_size)
            
        # Normalize to [0, 1]
        low_img = low_img.astype(np.float32) / 255.0
        high_img = high_img.astype(np.float32) / 255.0
        
        # Convert to CHW tensor
        low_tensor = torch.from_numpy(low_img).permute(2, 0, 1).float()
        high_tensor = torch.from_numpy(high_img).permute(2, 0, 1).float()
        
        return {
            'input': low_tensor,
            'target': high_tensor,
            'name': name
        }
