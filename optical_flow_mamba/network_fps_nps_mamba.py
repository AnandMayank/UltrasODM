"""
FPS/NPS + Mamba Network Architecture

Enhanced version of simplified FPS/NPS with Mamba layers for better temporal modeling.
Combines the proven EfficientNet backbone with FPS/NPS sampling and Mamba processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b1
from fps_nps_sampling import CombinedFPSNPSSampling


class SimplifiedSSMLayer(nn.Module):
    """
    Simplified State Space Model (SSM) Layer
    Much simpler implementation to avoid dimension issues
    """
    def __init__(self, dim, state_dim=16):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim

        # Simple linear transformation with temporal processing
        self.temporal_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_proj = nn.Linear(dim, dim)

        # State-like processing
        self.state_proj = nn.Linear(dim, state_dim)
        self.output_proj = nn.Linear(state_dim, dim)

    def forward(self, x):
        """Input shape: (batch, seq_len, dim)"""
        batch_size, seq_len, dim = x.shape

        # Temporal convolution processing
        x_conv = x.transpose(1, 2)  # (B, dim, seq_len)
        x_conv = self.temporal_conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # (B, seq_len, dim)

        # Normalization and projection
        x_norm = self.temporal_norm(x_conv)
        x_proj = self.temporal_proj(x_norm)

        # State-like processing
        x_state = self.state_proj(x_proj)  # (B, seq_len, state_dim)
        output = self.output_proj(x_state)  # (B, seq_len, dim)

        # Residual connection
        return output + x


class FPSNPSMambaBlock(nn.Module):
    """
    FPS/NPS + Mamba Block
    
    Processes FPS and NPS sampled features through separate Mamba layers
    then combines them for enhanced temporal modeling.
    """
    def __init__(self, dim, state_dim=16):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        
        # Normalization
        self.norm = nn.LayerNorm(dim)
        
        # Separate processing for FPS and NPS features
        self.fps_proj = nn.Linear(dim, dim)
        self.nps_proj = nn.Linear(dim, dim)
        
        # Mamba layers for temporal modeling
        self.fps_mamba = SimplifiedSSMLayer(dim, state_dim)
        self.nps_mamba = SimplifiedSSMLayer(dim, state_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, fps_features, nps_features):
        """
        Args:
            fps_features: (B, num_fps_points, dim)
            nps_features: (B, num_nps_points, dim)
        Returns:
            combined_features: (B, total_points, dim)
        """
        # Normalize inputs
        fps_norm = self.norm(fps_features)
        nps_norm = self.norm(nps_features)
        
        # Project features
        fps_proj = self.fps_proj(fps_norm)
        nps_proj = self.nps_proj(nps_norm)
        
        # Apply Mamba for temporal modeling
        fps_mamba = self.fps_mamba(fps_proj)
        nps_mamba = self.nps_mamba(nps_proj)
        
        # Combine FPS and NPS features
        # Pad shorter sequence to match longer one
        if fps_mamba.shape[1] != nps_mamba.shape[1]:
            max_len = max(fps_mamba.shape[1], nps_mamba.shape[1])
            if fps_mamba.shape[1] < max_len:
                pad_size = max_len - fps_mamba.shape[1]
                fps_mamba = F.pad(fps_mamba, (0, 0, 0, pad_size))
            if nps_mamba.shape[1] < max_len:
                pad_size = max_len - nps_mamba.shape[1]
                nps_mamba = F.pad(nps_mamba, (0, 0, 0, pad_size))
        
        # Concatenate and fuse
        combined = torch.cat([fps_mamba, nps_mamba], dim=-1)  # (B, seq_len, 2*dim)
        fused = self.fusion(combined)  # (B, seq_len, dim)
        
        # Output projection with residual connection
        output = self.out_proj(fused)
        
        return output


class FPSNPSMambaNetwork(nn.Module):
    """
    FPS/NPS + Mamba Network
    
    Architecture:
    Input Frames → EfficientNet Backbone → FPS/NPS Sampling → Mamba Processing → 6-DOF Output
    
    Enhanced version that combines:
    - EfficientNet-B1 backbone (proven effective)
    - FPS/NPS sampling (our innovation)
    - Mamba layers (better temporal modeling)
    - Simple loss (PointDistance only)
    """
    
    def __init__(self, 
                 input_channels=1,
                 num_frames=4,
                 output_dim=6,
                 num_pairs=1,
                 num_fps_points=32,
                 num_nps_points=64,
                 mamba_state_dim=16,
                 backbone='efficientnet_b1'):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_frames = num_frames
        self.output_dim = output_dim
        self.num_pairs = num_pairs
        self.num_fps_points = num_fps_points
        self.num_nps_points = num_nps_points
        self.total_sampled_points = num_fps_points + num_nps_points
        self.mamba_state_dim = mamba_state_dim
        
        # Calculate actual output dimension (6 DOF parameters × num_pairs)
        self.actual_output_dim = output_dim * num_pairs
        
        # 1. Backbone Feature Extractor (EfficientNet-B1)
        if backbone == 'efficientnet_b1':
            self.backbone = efficientnet_b1(weights=None)
            
            # Modify input layer for multiple frames
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels=input_channels * num_frames,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias
            )
            
            # Remove classifier to get features
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Backbone {backbone} not supported")
        
        # 2. FPS/NPS Sampling Module
        self.fps_nps_sampler = CombinedFPSNPSSampling(
            num_fps_points=num_fps_points,
            num_nps_points=num_nps_points
        )
        
        # 3. Feature Projection for Mamba Processing
        self.feature_projection = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 4. Mamba Processing Block
        self.mamba_block = FPSNPSMambaBlock(dim=256, state_dim=mamba_state_dim)
        
        # 5. Temporal Fusion
        self.temporal_processor = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 6. Output Head
        self.output_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.actual_output_dim)  # 6-DOF × num_pairs
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Special initialization for output layer
        for m in self.output_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, frames):
        """
        Forward pass
        
        Args:
            frames: (B, T, H, W) input frame sequence
        Returns:
            output: (B, 6*num_pairs) 6-DOF pose parameters
        """
        B, T, H, W = frames.shape
        
        # 1. Stack frames for backbone processing
        stacked_frames = frames.view(B, T * self.input_channels, H, W)
        
        # 2. Extract features using backbone
        backbone_features = self.backbone(stacked_frames)  # (B, feature_dim)
        
        # 3. Create temporal dimension for processing
        frame_features = backbone_features.unsqueeze(1).repeat(1, T, 1)  # (B, T, feature_dim)
        
        # 4. Project features for Mamba processing
        projected_features = self.feature_projection(frame_features)  # (B, T, 256)
        
        # 5. Apply FPS/NPS sampling
        sampling_input = projected_features.unsqueeze(2)  # (B, T, 1, 256)
        combined_features, fps_indices, nps_indices = self.fps_nps_sampler(sampling_input)
        # combined_features: (B, total_sampled_points, 256)
        
        # 6. Split into FPS and NPS features for Mamba processing
        fps_features = combined_features[:, :self.num_fps_points, :]  # (B, num_fps_points, 256)
        nps_features = combined_features[:, self.num_fps_points:, :]   # (B, num_nps_points, 256)
        
        # 7. Apply Mamba processing
        mamba_output = self.mamba_block(fps_features, nps_features)  # (B, seq_len, 256)
        
        # 8. Global temporal pooling
        pooled_features = torch.mean(mamba_output, dim=1)  # (B, 256)
        
        # 9. Final temporal processing
        temporal_features = self.temporal_processor(pooled_features)  # (B, 256)
        
        # 10. Generate 6-DOF output for each pair
        output = self.output_head(temporal_features)  # (B, 6*num_pairs)
        
        return output
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone': 'EfficientNet-B1',
            'fps_points': self.num_fps_points,
            'nps_points': self.num_nps_points,
            'total_sampled_points': self.total_sampled_points,
            'input_frames': self.num_frames,
            'output_dim': self.output_dim,
            'num_pairs': self.num_pairs,
            'actual_output_dim': self.actual_output_dim,
            'mamba_state_dim': self.mamba_state_dim,
            'architecture': 'FPS/NPS + Mamba'
        }


def create_fps_nps_mamba_model(config):
    """
    Factory function to create FPS/NPS + Mamba model
    
    Args:
        config: Configuration dictionary
    Returns:
        model: FPSNPSMambaNetwork instance
    """
    model = FPSNPSMambaNetwork(
        input_channels=config.get('input_channels', 1),
        num_frames=config.get('num_frames', 4),
        output_dim=config.get('output_dim', 6),
        num_pairs=config.get('num_pairs', 1),
        num_fps_points=config.get('num_fps_points', 32),
        num_nps_points=config.get('num_nps_points', 64),
        mamba_state_dim=config.get('mamba_state_dim', 16),
        backbone=config.get('backbone', 'efficientnet_b1')
    )
    
    return model


def test_fps_nps_mamba_network():
    """Test function for the FPS/NPS + Mamba network"""
    print("Testing FPS/NPS + Mamba Network...")
    
    # Test configuration
    config = {
        'input_channels': 1,
        'num_frames': 4,
        'output_dim': 6,
        'num_pairs': 3,
        'num_fps_points': 32,
        'num_nps_points': 64,
        'mamba_state_dim': 16,
        'backbone': 'efficientnet_b1'
    }
    
    # Create model
    model = create_fps_nps_mamba_model(config)
    
    # Test input
    B, T, H, W = 2, 4, 224, 224
    frames = torch.randn(B, T, H, W)
    
    # Forward pass
    output = model(frames)
    print(f"Input shape: {frames.shape}")
    print(f"Output shape: {output.shape}")
    
    # Model info
    info = model.get_model_info()
    print(f"Model info: {info}")
    
    print("[OK] FPS/NPS + Mamba Network test passed!")


if __name__ == "__main__":
    test_fps_nps_mamba_network()
