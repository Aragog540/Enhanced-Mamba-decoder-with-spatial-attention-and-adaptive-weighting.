

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SpatialAttentionModule(nn.Module):
    """Multi-scale spatial attention module."""
    
    def __init__(self, channels: int, reduction: int = 8, kernel_sizes: List[int] = [1, 3, 5, 7]):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.kernel_sizes = kernel_sizes
        
        # Multi-scale spatial feature extraction
        self.multi_scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // reduction, kernel_size=k, 
                         padding=k // 2, groups=channels // reduction),
                nn.InstanceNorm2d(channels // reduction),
                nn.ReLU(inplace=True)
            ) for k in kernel_sizes
        ])
        
        # Attention weight generation
        self.attention_conv = nn.Sequential(
            nn.Conv2d(len(kernel_sizes) * (channels // reduction), channels // reduction, 1),
            nn.InstanceNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 1),
            nn.Sigmoid()
        )
        
        # Channel attention complement
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of spatial attention.
        
        Args:
            x: Input feature tensor [B, C, H, W]
            
        Returns:
            Attention-weighted features [B, C, H, W]
        """
        # Multi-scale spatial processing
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            multi_scale_features.append(conv(x))
        
        # Concatenate multi-scale features
        combined_features = torch.cat(multi_scale_features, dim=1)
        
        # Generate spatial attention weights
        spatial_attention = self.attention_conv(combined_features)
        
        # Generate channel attention weights  
        channel_attention = self.channel_attention(x)
        
        # Apply both attention mechanisms
        attended_features = x * spatial_attention * channel_attention
        
        # Residual connection
        return attended_features + x


class MultiScaleSpatialAttention(nn.Module):
    """Advanced multi-scale spatial attention with pyramid processing."""
    
    def __init__(self, channels: int, scales: List[int] = [1, 2, 4, 8]):
        super().__init__()
        self.channels = channels
        self.scales = scales
        
        # Pyramid pooling branches
        self.pyramid_branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(channels, channels // len(scales), 1),
                nn.BatchNorm2d(channels // len(scales)),
                nn.ReLU(inplace=True)
            ) for scale in scales
        ])
        
        # Feature fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-scale spatial attention.
        
        Args:
            x: Input feature tensor [B, C, H, W]
            
        Returns:
            Multi-scale attended features [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Process through pyramid branches
        pyramid_features = []
        for branch in self.pyramid_branches:
            feat = branch(x)
            # Upsample to original size
            feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            pyramid_features.append(feat)
        
        # Concatenate pyramid features
        pyramid_concat = torch.cat(pyramid_features, dim=1)
        
        # Generate attention weights
        attention_weights = self.fusion_conv(pyramid_concat)
        
        # Apply attention
        return x * attention_weights + x


class ChannelSpatialAttention(nn.Module):
    """Combined channel and spatial attention module."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),  # 2 channels: avg + max
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of channel-spatial attention.
        
        Args:
            x: Input feature tensor [B, C, H, W]
            
        Returns:
            Channel-spatial attended features [B, C, H, W]
        """
        # Channel attention
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        
        return x * spatial_att


class PyramidAttentionModule(nn.Module):
    """Pyramid attention module for multi-scale feature integration."""
    
    def __init__(self, channels_list: List[int], out_channels: int):
        super().__init__()
        self.channels_list = channels_list
        
        # Individual attention for each scale
        self.scale_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // 4, 1),
                nn.BatchNorm2d(channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, 1, 1),
                nn.Sigmoid()
            ) for channels in channels_list
        ])
        
        # Feature projection
        self.projections = nn.ModuleList([
            nn.Conv2d(channels, out_channels, 1)
            for channels in channels_list
        ])
        
    def forward(self, features_list: List[torch.Tensor], target_size: tuple) -> torch.Tensor:
        """
        Forward pass of pyramid attention.
        
        Args:
            features_list: List of feature tensors at different scales
            target_size: Target spatial size (H, W)
            
        Returns:
            Fused feature tensor [B, out_channels, H, W]
        """
        attended_features = []
        
        for feat, attention, projection in zip(features_list, self.scale_attentions, self.projections):
            # Apply scale-specific attention
            att_weights = attention(feat)
            attended_feat = feat * att_weights
            
            # Project to common channel dimension
            projected = projection(attended_feat)
            
            # Resize to target size
            if projected.shape[2:] != target_size:
                projected = F.interpolate(
                    projected, size=target_size, 
                    mode='bilinear', align_corners=False
                )
            
            attended_features.append(projected)
        
        # Sum all attended features
        return sum(attended_features)


if __name__ == "__main__":
    # Test spatial attention modules
    x = torch.randn(2, 256, 64, 64)
    
    # Test SpatialAttentionModule
    sam = SpatialAttentionModule(256)
    out_sam = sam(x)
    print("SpatialAttentionModule output shape:", out_sam.shape)
    
    # Test MultiScaleSpatialAttention
    msam = MultiScaleSpatialAttention(256)
    out_msam = msam(x)
    print("MultiScaleSpatialAttention output shape:", out_msam.shape)
    
    # Test PyramidAttentionModule
    features = [
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 256, 16, 16),
        torch.randn(2, 512, 8, 8)
    ]
    pam = PyramidAttentionModule([128, 256, 512], 256)
    out_pam = pam(features, (64, 64))
    print("PyramidAttentionModule output shape:", out_pam.shape)
