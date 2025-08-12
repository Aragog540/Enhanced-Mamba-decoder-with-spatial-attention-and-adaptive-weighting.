

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class AdaptiveBackgroundWeighting(nn.Module):
    """Adaptive background-foreground weighting module."""
    
    def __init__(self, channels: int, bg_threshold: float = 0.5):
        super().__init__()
        self.channels = channels
        self.bg_threshold = bg_threshold
        
        # Background probability estimation
        self.bg_estimator = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 8, 3, padding=1),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )
        
        # Foreground enhancement network
        self.fg_enhancer = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1)
        )
        
        # Dynamic weight computation
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
        # Learnable background suppression factor
        self.bg_suppression = nn.Parameter(torch.tensor(0.3))
        
        # Spatial consistency module
        self.consistency_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features: torch.Tensor, skip_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of adaptive background weighting.
        
        Args:
            features: Input feature tensor [B, C, H, W]
            skip_features: Optional skip connection features [B, C, H, W]
            
        Returns:
            Adaptively weighted features [B, C, H, W]
        """
        B, C, H, W = features.shape
        
        # Estimate background probability
        bg_prob = self.bg_estimator(features)  # [B, 1, H, W]
        fg_prob = 1.0 - bg_prob
        
        # Enhance foreground features
        fg_enhanced = self.fg_enhancer(features)
        
        # Compute adaptive weights based on global context
        adaptive_weights = self.weight_net(features)  # [B, C, 1, 1]
        
        # Apply background suppression with learnable factor
        suppression_factor = torch.clamp(self.bg_suppression, 0.1, 0.8)
        bg_suppressed = features * (1.0 - bg_prob * suppression_factor)
        
        # Apply foreground enhancement
        fg_boosted = fg_enhanced * fg_prob * adaptive_weights
        
        # Combine background suppression and foreground enhancement
        weighted_features = bg_suppressed + fg_boosted
        
        # Apply spatial consistency
        consistent_features = self.consistency_conv(weighted_features)
        
        # Add skip connection if provided
        if skip_features is not None:
            # Adaptive skip connection weighting
            skip_weight = torch.mean(fg_prob, dim=[2, 3], keepdim=True)  # [B, 1, 1, 1]
            consistent_features = consistent_features + skip_features * skip_weight * 0.1
        
        return consistent_features
    
    def get_attention_maps(self, features: torch.Tensor) -> dict:
        """Get attention maps for visualization."""
        with torch.no_grad():
            bg_prob = self.bg_estimator(features)
            fg_prob = 1.0 - bg_prob
            adaptive_weights = self.weight_net(features)
            
            return {
                'background_prob': bg_prob,
                'foreground_prob': fg_prob, 
                'adaptive_weights': adaptive_weights,
                'suppression_factor': self.bg_suppression.item()
            }


class CrossScaleFusion(nn.Module):
    """Cross-scale feature fusion with attention."""
    
    def __init__(self, channels_list: List[int]):
        super().__init__()
        self.channels_list = channels_list
        
        # Scale-wise attention modules
        self.scale_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // 4, 1),
                nn.BatchNorm2d(channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, 1, 1),
                nn.Sigmoid()
            ) for channels in channels_list
        ])
        
        # Feature transformation layers
        self.transform_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for channels in channels_list
        ])
        
        # Cross-scale interaction
        total_channels = sum(channels_list)
        self.cross_scale_conv = nn.Sequential(
            nn.Conv2d(total_channels, total_channels // 4, 1),
            nn.BatchNorm2d(total_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels // 4, channels_list[-1], 3, padding=1),
            nn.BatchNorm2d(channels_list[-1]),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of cross-scale fusion.
        
        Args:
            features_list: List of feature tensors at different scales
            
        Returns:
            Fused feature tensor at the finest scale
        """
        if not features_list:
            raise ValueError("features_list cannot be empty")
        
        # Determine target size (finest scale)
        target_size = features_list[-1].shape[2:]
        
        # Process each scale
        processed_features = []
        for feat, attention, transform in zip(features_list, self.scale_attentions, self.transform_layers):
            # Apply scale-specific attention
            att_weights = attention(feat)
            attended_feat = feat * att_weights
            
            # Transform features
            transformed_feat = transform(attended_feat)
            
            # Resize to target size
            if transformed_feat.shape[2:] != target_size:
                transformed_feat = F.interpolate(
                    transformed_feat, size=target_size,
                    mode='bilinear', align_corners=False
                )
            
            processed_features.append(transformed_feat)
        
        # Concatenate all processed features
        concat_features = torch.cat(processed_features, dim=1)
        
        # Cross-scale interaction
        fused_features = self.cross_scale_conv(concat_features)
        
        return fused_features


class AdaptiveLossWeighting(nn.Module):
    """Adaptive loss weighting for different regions."""
    
    def __init__(self, num_classes: int, temperature: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        
        # Per-class weight prediction
        self.weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_classes, num_classes // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes // 2, num_classes, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive loss weights.
        
        Args:
            predictions: Model predictions [B, C, H, W]
            targets: Ground truth targets [B, H, W]
            
        Returns:
            Adaptive loss weights [B, C, H, W]
        """
        B, C, H, W = predictions.shape
        
        # Convert targets to one-hot
        targets_onehot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()
        
        # Compute prediction confidence
        pred_probs = F.softmax(predictions / self.temperature, dim=1)
        
        # Compute per-class weights
        class_weights = self.weight_predictor(pred_probs)  # [B, C, 1, 1]
        
        # Compute difficulty-based weights
        difficulty = 1.0 - torch.sum(pred_probs * targets_onehot, dim=1, keepdim=True)
        difficulty_weights = torch.sigmoid(difficulty * 2 - 1)  # Enhance difficult regions
        
        # Combine weights
        final_weights = class_weights * difficulty_weights
        
        return final_weights


class UncertaintyWeighting(nn.Module):
    """Uncertainty-based weighting for predictions."""
    
    def __init__(self, channels: int):
        super().__init__()
        
        # Uncertainty estimation network
        self.uncertainty_net = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels // 4, 3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of uncertainty weighting.
        
        Args:
            features: Input features [B, C, H, W]
            
        Returns:
            Tuple of (weighted_features, uncertainty_map)
        """
        # Estimate uncertainty
        uncertainty = self.uncertainty_net(features)  # [B, 1, H, W]
        
        # Confidence is inverse of uncertainty
        confidence = 1.0 - uncertainty
        
        # Apply confidence weighting
        weighted_features = features * confidence
        
        return weighted_features, uncertainty


if __name__ == "__main__":
    # Test adaptive weighting modules
    x = torch.randn(2, 256, 64, 64)
    
    # Test AdaptiveBackgroundWeighting
    abw = AdaptiveBackgroundWeighting(256)
    out_abw = abw(x)
    print("AdaptiveBackgroundWeighting output shape:", out_abw.shape)
    
    # Test attention maps
    att_maps = abw.get_attention_maps(x)
    print("Background prob shape:", att_maps['background_prob'].shape)
    
    # Test CrossScaleFusion
    features = [
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 256, 16, 16), 
        torch.randn(2, 512, 8, 8)
    ]
    csf = CrossScaleFusion([128, 256, 512])
    out_csf = csf(features)
    print("CrossScaleFusion output shape:", out_csf.shape)
    
    # Test UncertaintyWeighting
    uw = UncertaintyWeighting(256)
    weighted_feat, uncertainty = uw(x)
    print("UncertaintyWeighting output shapes:", weighted_feat.shape, uncertainty.shape)
