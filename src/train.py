import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dataset import LOLDataset
from model import UNetEnhancer
from loss import CombinedEnhancementLoss

def plot_metrics(history, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss vs Epoch
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], marker='o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], marker='o', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'plot_loss.png'))
    plt.close()
    
    # 2. PSNR vs Epoch
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['val_psnr'], marker='o', label='Validation PSNR', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR vs Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'plot_psnr.png'))
    plt.close()

    # 3. SSIM vs Epoch
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['val_ssim'], marker='o', label='Validation SSIM', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM vs Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'plot_ssim.png'))
    plt.close()
    
    print(f"Metrics plots saved to {output_dir}/")

def check_overfitting(history):
    if len(history['val_loss']) < 5:
        return
    # Check if train loss decreases but val loss increases consistently over last 3 epochs
    val_trend = np.diff(history['val_loss'][-4:])
    train_trend = np.diff(history['train_loss'][-4:])
    if np.all(val_trend > 0) and np.all(train_trend < 0):
        print("\n!!! WARNING: Overfitting Detected !!!")
        print("Training loss is decreasing, but validation loss has increased for 3 consecutive epochs.")
        print("Suggested Actions: Increase regularization (dropout/weight decay), increase data augmentation, or stop training earlier.\n")

def train():
    DATA_ROOT = "../lol_dataset"
    if not os.path.exists(DATA_ROOT) and os.path.exists("lol_dataset"):
        DATA_ROOT = "lol_dataset"
        
    BATCH_SIZE = 8
    EPOCHS = 50 # Increased max epochs
    LEARNING_RATE = 1e-4 # Reduced learning rate for stability
    EARLY_STOPPING_PATIENCE = 10 # Increased patience
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    OUTPUT_DIR = "training_output"
    MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
    LOG_PATH = os.path.join(OUTPUT_DIR, "metrics_log.json")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")

    train_dataset = LOLDataset(DATA_ROOT, mode='train')
    val_dataset = LOLDataset(DATA_ROOT, mode='eval')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = UNetEnhancer().to(DEVICE)
    # Using upgraded Combined Loss: 0.4 MSE + 0.4 SSIM + 0.2 L1
    criterion = CombinedEnhancementLoss(mse_weight=0.4, ssim_weight=0.4, l1_weight=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': []}

    print("Starting optimized training loop...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for batch in train_pbar:
            inputs = batch['input'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_psnr_list = []
        val_ssim_list = []
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]")
        with torch.no_grad():
            for batch in val_pbar:
                inputs = batch['input'].to(DEVICE)
                targets = batch['target'].to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Compute Metrics (strictly using ground truth targets)
                output_rgb = outputs.clamp(0, 1).squeeze().permute(1, 2, 0).cpu().numpy()
                target_rgb = targets.squeeze().permute(1, 2, 0).cpu().numpy()
                
                psnr_val = peak_signal_noise_ratio(target_rgb, output_rgb, data_range=1.0)
                ssim_val = structural_similarity(target_rgb, output_rgb, channel_axis=-1, data_range=1.0)
                
                val_psnr_list.append(psnr_val)
                val_ssim_list.append(ssim_val)
                
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        val_loss /= len(val_loader)
        mean_psnr = np.mean(val_psnr_list)
        mean_ssim = np.mean(val_ssim_list)
        
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_psnr'].append(float(mean_psnr))
        history['val_ssim'].append(float(mean_ssim))

        print(f"Epoch {epoch} Summary | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PSNR: {mean_psnr:.2f} | Val SSIM: {mean_ssim:.4f}")
        
        # Save logs progressively
        with open(LOG_PATH, 'w') as f:
            json.dump(history, f, indent=4)

        check_overfitting(history)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"--> Saved new best model to {MODEL_PATH} with Val Loss: {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # Plot and save history
    plot_metrics(history, OUTPUT_DIR)

if __name__ == '__main__':
    train()
