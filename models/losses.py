
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import numpy as np


class DiceLoss(nn.Module):
    """Dice loss for semantic segmentation."""
    
    def __init__(self, smooth: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            predictions: Model predictions [B, C, H, W]
            targets: Ground truth targets [B, H, W]
            
        Returns:
            Dice loss scalar
        """
        # Convert predictions to probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot encoding
        num_classes = predictions.shape[1]
        targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Mask out ignore_index
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).float().unsqueeze(1)
            targets_onehot = targets_onehot * mask
            predictions = predictions * mask
        
        # Compute Dice coefficient for each class
        intersection = torch.sum(predictions * targets_onehot, dim=(2, 3))
        union = torch.sum(predictions, dim=(2, 3)) + torch.sum(targets_onehot, dim=(2, 3))
        
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - torch.mean(dice_coeff)
        
        return dice_loss


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal loss.
        
        Args:
            predictions: Model predictions [B, C, H, W]
            targets: Ground truth targets [B, H, W]
            
        Returns:
            Focal loss scalar
        """
        # Compute cross-entropy
        ce_loss = F.cross_entropy(predictions, targets, ignore_index=self.ignore_index, reduction='none')
        
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return torch.mean(focal_loss)


class TverskyLoss(nn.Module):
    """Tversky loss for handling imbalanced segmentation."""
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.smooth = smooth
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Tversky loss.
        
        Args:
            predictions: Model predictions [B, C, H, W]
            targets: Ground truth targets [B, H, W]
            
        Returns:
            Tversky loss scalar
        """
        # Convert predictions to probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot
        num_classes = predictions.shape[1]
        targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Compute Tversky components
        true_pos = torch.sum(predictions * targets_onehot, dim=(2, 3))
        false_neg = torch.sum(targets_onehot * (1 - predictions), dim=(2, 3))
        false_pos = torch.sum((1 - targets_onehot) * predictions, dim=(2, 3))
        
        # Compute Tversky index
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )
        
        return 1.0 - torch.mean(tversky)


class BackgroundSuppressionLoss(nn.Module):
    """Background suppression loss to reduce background over-confidence."""
    
    def __init__(self, background_class: int = 0, suppression_weight: float = 0.3):
        super().__init__()
        self.background_class = background_class
        self.suppression_weight = suppression_weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute background suppression loss.
        
        Args:
            predictions: Model predictions [B, C, H, W]
            targets: Ground truth targets [B, H, W]
            
        Returns:
            Background suppression loss scalar
        """
        # Get background probabilities
        bg_probs = F.softmax(predictions, dim=1)[:, self.background_class]  # [B, H, W]
        
        # Create background mask
        bg_mask = (targets == self.background_class).float()
        
        # Penalize high confidence on background regions
        bg_suppression = torch.mean(bg_probs * bg_mask)
        
        return self.suppression_weight * bg_suppression


class BoundaryLoss(nn.Module):
    """Boundary-aware loss for better edge segmentation."""
    
    def __init__(self, boundary_weight: float = 1.0):
        super().__init__()
        self.boundary_weight = boundary_weight
        
        # Sobel filters for edge detection
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2], 
            [-1, 0, 1]
        ]).float().view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]).float().view(1, 1, 3, 3))
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary loss.
        
        Args:
            predictions: Model predictions [B, C, H, W]
            targets: Ground truth targets [B, H, W]
            
        Returns:
            Boundary loss scalar
        """
        # Convert predictions to class predictions
        pred_classes = torch.argmax(predictions, dim=1).float().unsqueeze(1)  # [B, 1, H, W]
        target_classes = targets.float().unsqueeze(1)  # [B, 1, H, W]
        
        # Compute gradients for predicted and target boundaries
        pred_grad_x = F.conv2d(pred_classes, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_classes, self.sobel_y, padding=1)
        pred_boundary = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2)
        
        target_grad_x = F.conv2d(target_classes, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target_classes, self.sobel_y, padding=1)
        target_boundary = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2)
        
        # Compute boundary loss
        boundary_loss = F.mse_loss(pred_boundary, target_boundary)
        
        return self.boundary_weight * boundary_loss


class EnhancedMultiScaleLoss(nn.Module):
    """Enhanced multi-scale loss for Enhanced UNetMamba."""
    
    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        focal_weight: float = 0.5,
        tversky_weight: float = 0.3,
        bg_suppression_weight: float = 0.2,
        boundary_weight: float = 0.1,
        supervised_weight: float = 0.4,
        class_weights: Optional[List[float]] = None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.supervised_weight = supervised_weight
        
        # Initialize individual losses
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.tversky_loss = TverskyLoss(alpha=0.7, beta=0.3)
        self.bg_suppression_loss = BackgroundSuppressionLoss()
        self.boundary_loss = BoundaryLoss()
        
        # Cross-entropy with class weights
        if class_weights is not None:
            class_weights = torch.tensor(class_weights).float()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        # Loss weights
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.bg_suppression_weight = bg_suppression_weight
        self.boundary_weight = boundary_weight
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute enhanced multi-scale loss.
        
        Args:
            outputs: Model outputs dictionary
            targets: Ground truth targets [B, H, W]
            
        Returns:
            Dictionary of losses
        """
        main_pred = outputs['main_pred']
        
        # Main prediction losses
        dice_loss = self.dice_loss(main_pred, targets)
        ce_loss = self.ce_loss(main_pred, targets)
        focal_loss = self.focal_loss(main_pred, targets)
        tversky_loss = self.tversky_loss(main_pred, targets)
        bg_suppression = self.bg_suppression_loss(main_pred, targets)
        boundary_loss = self.boundary_loss(main_pred, targets)
        
        # Combine main losses
        main_loss = (
            self.dice_weight * dice_loss +
            self.ce_weight * ce_loss +
            self.focal_weight * focal_loss +
            self.tversky_weight * tversky_loss +
            self.bg_suppression_weight * bg_suppression +
            self.boundary_weight * boundary_loss
        )
        
        # Supervised losses (if available)
        supervised_loss = torch.tensor(0.0, device=main_pred.device)
        if 'supervised_preds' in outputs and outputs['supervised_preds'] is not None:
            for sup_pred in outputs['supervised_preds']:
                sup_ce = self.ce_loss(sup_pred, targets)
                supervised_loss += sup_ce
            supervised_loss /= len(outputs['supervised_preds'])
        
        # Total loss
        total_loss = main_loss + self.supervised_weight * supervised_loss
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'dice_loss': dice_loss,
            'ce_loss': ce_loss,
            'focal_loss': focal_loss,
            'tversky_loss': tversky_loss,
            'bg_suppression_loss': bg_suppression,
            'boundary_loss': boundary_loss,
            'supervised_loss': supervised_loss
        }


class AdaptiveLossScheduler:
    """Adaptive loss weight scheduler."""
    
    def __init__(self, initial_weights: Dict[str, float], schedule_type: str = 'cosine'):
        self.initial_weights = initial_weights
        self.schedule_type = schedule_type
        self.current_weights = initial_weights.copy()
        
    def step(self, epoch: int, max_epochs: int, metrics: Optional[Dict[str, float]] = None):
        """Update loss weights based on training progress."""
        
        progress = epoch / max_epochs
        
        if self.schedule_type == 'cosine':
            # Cosine annealing
            for key in self.current_weights:
                self.current_weights[key] = self.initial_weights[key] * (
                    0.5 * (1 + np.cos(np.pi * progress))
                )
        elif self.schedule_type == 'adaptive' and metrics is not None:
            # Adapt based on performance metrics
            if 'background_iou' in metrics and 'foreground_iou' in metrics:
                bg_iou = metrics['background_iou']
                fg_iou = metrics['foreground_iou']
                
                # Increase background suppression if background IoU is too high
                if bg_iou > 0.9:
                    self.current_weights['bg_suppression_weight'] *= 1.1
                
                # Increase focal weight if foreground IoU is low
                if fg_iou < 0.7:
                    self.current_weights['focal_weight'] *= 1.05
    
    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return self.current_weights


if __name__ == "__main__":
    # Test loss functions
    B, C, H, W = 2, 7, 64, 64
    predictions = torch.randn(B, C, H, W)
    targets = torch.randint(0, C, (B, H, W))
    
    # Test individual losses
    dice = DiceLoss()
    focal = FocalLoss()
    tversky = TverskyLoss()
    bg_suppress = BackgroundSuppressionLoss()
    boundary = BoundaryLoss()
    
    print("Dice loss:", dice(predictions, targets).item())
    print("Focal loss:", focal(predictions, targets).item())
    print("Tversky loss:", tversky(predictions, targets).item())
    print("Background suppression loss:", bg_suppress(predictions, targets).item())
    print("Boundary loss:", boundary(predictions, targets).item())
    
    # Test enhanced multi-scale loss
    enhanced_loss = EnhancedMultiScaleLoss(num_classes=C)
    outputs = {'main_pred': predictions}
    loss_dict = enhanced_loss(outputs, targets)
    print("\nEnhanced loss components:")
    for key, value in loss_dict.items():
        print(f"{key}: {value.item():.4f}")
