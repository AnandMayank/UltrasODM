"""
Simplified FPS/NPS Network Architecture

Based on TUS-REC baseline success but enhanced with FPS/NPS sampling.
Target: ~10-15M parameters vs 45M in complex approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b1
from fps_nps_sampling import CombinedFPSNPSSampling


class SimpleFPSNPSNetwork(nn.Module):
    """
    Simplified FPS/NPS Network
    
    Architecture:
    Input Frames → EfficientNet Backbone → FPS/NPS Sampling → Temporal Fusion → 6-DOF Output
    
    Key Design:
    - EfficientNet-B1 backbone (proven effective in baseline)
    - FPS/NPS sampling for temporal modeling
    - Simple temporal fusion
    - Direct 6-DOF parameter output
    - ~10-15M parameters total
    """
    
    def __init__(self,
                 input_channels=1,
                 num_frames=4,
                 output_dim=6,
                 num_pairs=1,  # Number of frame pairs (calculated from data_pairs)
                 num_fps_points=32,
                 num_nps_points=64,
                 backbone='efficientnet_b1'):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_frames = num_frames
        self.output_dim = output_dim
        self.num_pairs = num_pairs
        self.num_fps_points = num_fps_points
        self.num_nps_points = num_nps_points
        self.total_sampled_points = num_fps_points + num_nps_points

        # Calculate actual output dimension (6 DOF parameters × num_pairs)
        self.actual_output_dim = output_dim * num_pairs
        
        # 1. Backbone Feature Extractor (based on TUS-REC baseline)
        if backbone == 'efficientnet_b1':
            self.backbone = efficientnet_b1(weights=None)
            
            # Modify input layer for ultrasound (grayscale) and multiple frames
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels=input_channels * num_frames,  # Stack frames as channels
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias
            )
            
            # Remove the classifier to get features
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Backbone {backbone} not supported")
        
        # 2. FPS/NPS Sampling Module
        self.fps_nps_sampler = CombinedFPSNPSSampling(
            num_fps_points=num_fps_points,
            num_nps_points=num_nps_points
        )
        
        # 3. Temporal Fusion Module
        # Convert backbone features to point-like features for sampling
        self.feature_projection = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 4. Temporal Processing
        self.temporal_processor = nn.Sequential(
            nn.Linear(256 * self.total_sampled_points, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 5. Output Head
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

        # Special initialization for output layer to prevent extreme values
        for m in self.output_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # Small weights
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, frames):
        """
        Forward pass

        Args:
            frames: (B, T, H, W) input frame sequence
        Returns:
            output: (B, 6) 6-DOF pose parameters
        """
        B, T, H, W = frames.shape

        # 1. Stack frames as channels for backbone processing
        # This follows the TUS-REC baseline approach
        stacked_frames = frames.view(B, T * self.input_channels, H, W)  # (B, T*1, H, W)

        # 2. Extract features using backbone
        # Process stacked frames (following baseline approach)
        backbone_features = self.backbone(stacked_frames)  # (B, feature_dim)

        # 3. Reshape for temporal processing
        # Since we processed stacked frames, we need to create temporal dimension
        frame_features = backbone_features.unsqueeze(1).repeat(1, T, 1)  # (B, T, feature_dim)
        
        # 4. Project features for FPS/NPS sampling
        projected_features = self.feature_projection(frame_features)  # (B, T, 256)

        # 5. Apply FPS/NPS sampling
        # Reshape for sampling: (B, T, 256) -> (B, T, 1, 256) for spatial sampling
        sampling_input = projected_features.unsqueeze(2)  # (B, T, 1, 256)
        sampled_features, fps_indices, nps_indices = self.fps_nps_sampler(sampling_input)
        # sampled_features: (B, total_sampled_points, 256)

        # 6. Temporal fusion
        # Flatten sampled features for temporal processing
        flattened_features = sampled_features.view(B, -1)  # (B, total_sampled_points * 256)
        temporal_features = self.temporal_processor(flattened_features)  # (B, 256)

        # 7. Generate 6-DOF output for each pair
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
            'actual_output_dim': self.actual_output_dim
        }


class SimpleCNNBackbone(nn.Module):
    """
    Alternative simple CNN backbone for comparison
    Even lighter than EfficientNet for ablation studies
    """
    
    def __init__(self, input_channels=1, num_frames=4):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels * num_frames, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.feature_dim = 512
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


def create_simple_fps_nps_model(config):
    """
    Factory function to create simplified FPS/NPS model

    Args:
        config: Configuration dictionary
    Returns:
        model: SimpleFPSNPSNetwork instance
    """
    model = SimpleFPSNPSNetwork(
        input_channels=config.get('input_channels', 1),
        num_frames=config.get('num_frames', 4),
        output_dim=config.get('output_dim', 6),
        num_pairs=config.get('num_pairs', 1),
        num_fps_points=config.get('num_fps_points', 32),
        num_nps_points=config.get('num_nps_points', 64),
        backbone=config.get('backbone', 'efficientnet_b1')
    )

    return model


def test_simple_fps_nps_network():
    """Test function for the simplified network"""
    print("Testing Simplified FPS/NPS Network...")
    
    # Test configuration
    config = {
        'input_channels': 1,
        'num_frames': 4,
        'output_dim': 6,
        'num_fps_points': 32,
        'num_nps_points': 64,
        'backbone': 'efficientnet_b1'
    }
    
    # Create model
    model = create_simple_fps_nps_model(config)
    
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
    
    print("[OK] Simplified FPS/NPS Network test passed!")


if __name__ == "__main__":
    test_simple_fps_nps_network()
