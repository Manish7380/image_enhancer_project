import torch
import torch.nn as nn
import torch.nn.functional as F

def ssim_torch(prediction, target, window_size=11):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    padding = window_size // 2

    mu_x = F.avg_pool2d(prediction, kernel_size=window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, kernel_size=window_size, stride=1, padding=padding)

    sigma_x = F.avg_pool2d(prediction * prediction, kernel_size=window_size, stride=1, padding=padding) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(target * target, kernel_size=window_size, stride=1, padding=padding) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(prediction * target, kernel_size=window_size, stride=1, padding=padding) - (mu_x * mu_y)

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / (denominator + 1e-8)
    return ssim_map.mean()

class CombinedEnhancementLoss(nn.Module):
    def __init__(self, mse_weight=0.4, ssim_weight=0.4, l1_weight=0.2):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight

    def forward(self, prediction, target):
        # Mean Squared Error for pixel-level accuracy
        mse_loss = F.mse_loss(prediction, target)
        
        # L1 Loss for sharp edges and sparsity
        l1_loss = F.l1_loss(prediction, target)
        
        # Structural Similarity Index Measure (SSIM returns a similarity score, we minimize 1 - SSIM)
        ssim_loss = 1.0 - ssim_torch(prediction, target)
        
        # Combined Loss
        total_loss = (self.mse_weight * mse_loss + 
                      self.ssim_weight * ssim_loss + 
                      self.l1_weight * l1_loss)
        return total_loss
