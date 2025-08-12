import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
import warnings

from .spatial_attention import SpatialAttentionModule, MultiScaleSpatialAttention
from .adaptive_weighting import AdaptiveBackgroundWeighting, CrossScaleFusion


class VSS_Block_Enhanced(nn.Module):
    """Enhanced VSS Block with integrated spatial attention and adaptive weighting."""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channels = channels
        
        # Core VSS components (simplified for demonstration)
        self.norm1 = nn.LayerNorm(channels)
        self.linear1 = nn.Linear(channels, channels)
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.activation = nn.SiLU()
        
        # Enhanced components
        self.spatial_attention = SpatialAttentionModule(channels, reduction)
        self.adaptive_weighting = AdaptiveBackgroundWeighting(channels)
        
        self.norm2 = nn.LayerNorm(channels)
        self.linear2 = nn.Linear(channels, channels)
        
    def forward(self, x: torch.Tensor, skip_connection: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        
        # Flatten for layer norm and linear operations
        x_flat = x.view(B, C, -1).transpose(1, 2)  # B, HW, C
        
        # First normalization and linear transformation
        normed = self.norm1(x_flat)
        linear_out = self.linear1(normed)
        
        # Reshape back for conv operations
        conv_input = linear_out.transpose(1, 2).view(B, C, H, W)
        conv_out = self.activation(self.dwconv(conv_input))
        
        # Apply spatial attention
        attended = self.spatial_attention(conv_out)
        
        # Apply adaptive background weighting
        weighted = self.adaptive_weighting(attended, skip_connection)
        
        # Second normalization and linear transformation
        weighted_flat = weighted.view(B, C, -1).transpose(1, 2)
        normed2 = self.norm2(weighted_flat)
        output_flat = self.linear2(normed2)
        
        # Reshape back and add residual connection
        output = output_flat.transpose(1, 2).view(B, C, H, W)
        return output + residual


class EnhancedMambaDecoder(nn.Module):
    """Enhanced Mamba decoder with spatial attention and adaptive weighting."""
    
    def __init__(self, channels: List[int], num_classes: int):
        super().__init__()
        self.channels = channels
        
        # Enhanced VSS blocks for each decoder stage
        self.vss_blocks = nn.ModuleList([
            VSS_Block_Enhanced(ch) for ch in channels[::-1]
        ])
        
        # Cross-scale feature fusion
        self.cross_scale_fusion = CrossScaleFusion(channels[::-1])
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose2d(channels[::-1][i], channels[::-1][i+1], 2, stride=2)
            if i < len(channels) - 1 else nn.Identity()
            for i in range(len(channels))
        ])
        
        # Final classification head
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels[0], channels[0] // 2, 3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, num_classes, 1)
        )
        
    def forward(self, encoder_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through enhanced decoder.
        
        Args:
            encoder_features: List of encoder features from deepest to shallowest
            
        Returns:
            Dictionary containing main prediction and auxiliary outputs
        """
        # Start from deepest encoder feature
        x = encoder_features[0]
        decoder_features = [x]
        
        # Process through enhanced VSS blocks
        for i, (vss_block, upsample) in enumerate(zip(self.vss_blocks, self.upsample_layers)):
            # Skip connection from encoder
            skip_conn = encoder_features[i] if i < len(encoder_features) else None
            
            # Enhanced VSS processing
            x = vss_block(x, skip_conn)
            
            # Upsample for next stage
            if i < len(self.vss_blocks) - 1:
                x = upsample(x)
                # Add skip connection if available
                if i + 1 < len(encoder_features):
                    x = x + encoder_features[i + 1]
            
            decoder_features.append(x)
        
        # Cross-scale feature fusion
        fused_features = self.cross_scale_fusion(decoder_features)
        
        # Final prediction
        main_pred = self.final_conv(fused_features[-1])
        
        return {
            'main_pred': main_pred,
            'decoder_features': decoder_features,
            'fused_features': fused_features
        }


class EnhancedUNetMamba(nn.Module):
    """
    Enhanced UNetMamba with spatial attention and adaptive background weighting.
    
    This model extends the original UNetMamba with:
    1. Multi-scale spatial attention mechanisms
    2. Adaptive background-foreground weighting
    3. Cross-scale feature fusion
    4. Enhanced local supervision
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        channels: List[int] = [64, 128, 256, 512],
        encoder_name: str = "rest",
        pretrained: bool = True,
        use_local_supervision: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.channels = channels
        self.use_local_supervision = use_local_supervision
        
        # Initialize encoder (ResT backbone)
        self.encoder = self._build_encoder(encoder_name, pretrained, **kwargs)
        
        # Enhanced Mamba decoder
        self.decoder = EnhancedMambaDecoder(channels, num_classes)
        
        # Local supervision module (optional)
        if use_local_supervision:
            from .local_supervision import EnhancedLocalSupervision
            self.local_supervision = EnhancedLocalSupervision(channels, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _build_encoder(self, encoder_name: str, pretrained: bool, **kwargs):
        """Build encoder backbone."""
        if encoder_name.lower() == "rest":
            from .backbones.rest import RestEncoder
            return RestEncoder(pretrained=pretrained, **kwargs)
        else:
            raise NotImplementedError(f"Encoder {encoder_name} not implemented")
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Enhanced UNetMamba.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Dictionary containing predictions and auxiliary outputs
        """
        B, C, H, W = x.shape
        
        # Encoder forward pass
        encoder_features = self.encoder(x)
        
        # Enhanced decoder forward pass
        decoder_outputs = self.decoder(encoder_features[::-1])  # Reverse for deepest first
        
        # Get main prediction
        main_pred = decoder_outputs['main_pred']
        
        # Upsample to input size
        main_pred = F.interpolate(
            main_pred, size=(H, W), mode='bilinear', align_corners=False
        )
        
        outputs = {
            'main_pred': main_pred,
            'decoder_features': decoder_outputs['decoder_features']
        }
        
        # Local supervision (training only)
        if self.use_local_supervision and self.training:
            supervised_outputs = self.local_supervision(
                decoder_outputs['decoder_features'], (H, W)
            )
            outputs.update(supervised_outputs)
        
        return outputs
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'Enhanced UNetMamba',
            'num_classes': self.num_classes,
            'channels': self.channels,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
            'use_local_supervision': self.use_local_supervision
        }


def enhanced_unetmamba_small(num_classes: int = 7, **kwargs) -> EnhancedUNetMamba:
    """Small Enhanced UNetMamba model."""
    return EnhancedUNetMamba(
        num_classes=num_classes,
        channels=[32, 64, 128, 256],
        **kwargs
    )


def enhanced_unetmamba_base(num_classes: int = 7, **kwargs) -> EnhancedUNetMamba:
    """Base Enhanced UNetMamba model."""
    return EnhancedUNetMamba(
        num_classes=num_classes,
        channels=[64, 128, 256, 512],
        **kwargs
    )


def enhanced_unetmamba_large(num_classes: int = 7, **kwargs) -> EnhancedUNetMamba:
    """Large Enhanced UNetMamba model.""" 
    return EnhancedUNetMamba(
        num_classes=num_classes,
        channels=[96, 192, 384, 768],
        **kwargs
    )


if __name__ == "__main__":
    # Test model
    model = enhanced_unetmamba_base(num_classes=7)
    x = torch.randn(2, 3, 1024, 1024)
    
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        print("Output shape:", outputs['main_pred'].shape)
        print("Model info:", model.get_model_info())
