"""
Comparison Models for Contrastive Frame Grouping Study

Model A (Baseline): Optical Flow + EfficientNet
Model B (Proposed): Contrastive Frame Grouping → Optical Flow + EfficientNet

This module implements both models for fair comparison to demonstrate
the importance of contrastive frame grouping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b1

# Handle both relative and absolute imports
try:
    from .contrastive_grouping import ContrastiveFrameGrouping
except ImportError:
    from contrastive_grouping import ContrastiveFrameGrouping


class OpticalFlowEstimator(nn.Module):
    """Lightweight optical flow estimation between consecutive frames"""
    
    def __init__(self, in_channels=1):
        super().__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(in_channels * 2, 64, 7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        
        # Flow estimation
        self.flow_conv = nn.Conv2d(256, 2, 3, padding=1)
        
    def forward(self, frame1, frame2):
        """
        Estimate optical flow between two frames
        Args:
            frame1, frame2: (B, C, H, W) consecutive frames
        Returns:
            flow: (B, 2, H, W) optical flow field
        """
        # Concatenate frames
        x = torch.cat([frame1, frame2], dim=1)
        
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flow estimation
        flow = self.flow_conv(x)
        return flow


class BaselineModel(nn.Module):
    """
    Model A: Baseline - Optical Flow + EfficientNet
    
    Architecture:
    Input Frames → Optical Flow Estimation → EfficientNet → 6DoF Pose
                                          ↗
                       Original Frames ↗
    
    Losses: MSE + Correlation
    """
    
    def __init__(self, in_frames=8, pred_dim=6, input_channels=1):
        super().__init__()
        
        self.in_frames = in_frames
        self.pred_dim = pred_dim
        self.input_channels = input_channels
        
        # Optical flow estimator
        self.flow_estimator = OpticalFlowEstimator(input_channels)
        
        # EfficientNet backbone
        self.efficientnet = efficientnet_b1(weights=None)
        
        # Modify first layer to accept optical flow (2 channels) + original frames
        total_channels = 2 + in_frames  # 2 for flow + original frames
        self.efficientnet.features[0][0] = nn.Conv2d(
            in_channels=total_channels,
            out_channels=self.efficientnet.features[0][0].out_channels,
            kernel_size=self.efficientnet.features[0][0].kernel_size,
            stride=self.efficientnet.features[0][0].stride,
            padding=self.efficientnet.features[0][0].padding,
            bias=self.efficientnet.features[0][0].bias
        )
        
        # Modify classifier for pose prediction
        self.efficientnet.classifier[1] = nn.Linear(
            in_features=self.efficientnet.classifier[1].in_features,
            out_features=pred_dim
        )
        
        # Initialize modified layers
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize modified layers"""
        nn.init.kaiming_normal_(self.efficientnet.features[0][0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.efficientnet.classifier[1].weight, 0, 0.01)
        nn.init.constant_(self.efficientnet.classifier[1].bias, 0)
        
    def forward(self, frames):
        """
        Args:
            frames: (B, T, H, W) sequence of frames
        Returns:
            pose: (B, pred_dim) predicted pose parameters
        """
        B, T, H, W = frames.shape
        
        if T < 2:
            raise ValueError("Need at least 2 frames for optical flow")
        
        # Use consecutive frames for optical flow (middle frames for stability)
        mid_idx = T // 2
        frame1 = frames[:, mid_idx-1].unsqueeze(1) if mid_idx > 0 else frames[:, 0].unsqueeze(1)
        frame2 = frames[:, mid_idx].unsqueeze(1) if mid_idx < T else frames[:, -1].unsqueeze(1)
        
        # Estimate optical flow
        flow = self.flow_estimator(frame1, frame2)  # (B, 2, H', W')
        
        # Resize flow to match frame size if needed
        if flow.shape[-2:] != (H, W):
            flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
        
        # Flatten frames for concatenation: (B, T, H, W) → (B, T*1, H, W)
        frames_flat = frames.view(B, T, H, W)
        
        # Concatenate flow with original frames
        combined_input = torch.cat([flow, frames_flat], dim=1)  # (B, 2+T, H, W)
        
        # Pass through EfficientNet
        pose = self.efficientnet(combined_input)
        
        return pose


class EnhancedModel(nn.Module):
    """
    Model B: Enhanced - Contrastive Frame Grouping → Optical Flow + EfficientNet
    
    Architecture:
    Input Frames → Contrastive Frame Grouping → Optical Flow → EfficientNet → 6DoF Pose
                         ↓                           ↗
                  Triplet Loss              Original Frames ↗
    
    Losses: MSE + Correlation + Contrastive (Triplet)
    """
    
    def __init__(self, in_frames=8, pred_dim=6, input_channels=1,
                 margin_alpha=0.2, delta=2, Delta=4, tau_sim=0.5, embed_dim=256):
        super().__init__()
        
        self.in_frames = in_frames
        self.pred_dim = pred_dim
        self.input_channels = input_channels
        
        # Contrastive frame grouping module (Algorithm 1)
        self.contrastive_grouping = ContrastiveFrameGrouping(
            margin_alpha=margin_alpha,
            delta=delta,
            Delta=Delta,
            tau_sim=tau_sim,
            embed_dim=embed_dim,
            input_channels=input_channels
        )
        
        # Optical flow estimator
        self.flow_estimator = OpticalFlowEstimator(input_channels)
        
        # EfficientNet backbone (same as baseline for fair comparison)
        self.efficientnet = efficientnet_b1(weights=None)
        
        # Modify first layer to accept optical flow (2 channels) + original frames
        total_channels = 2 + in_frames  # 2 for flow + original frames
        self.efficientnet.features[0][0] = nn.Conv2d(
            in_channels=total_channels,
            out_channels=self.efficientnet.features[0][0].out_channels,
            kernel_size=self.efficientnet.features[0][0].kernel_size,
            stride=self.efficientnet.features[0][0].stride,
            padding=self.efficientnet.features[0][0].padding,
            bias=self.efficientnet.features[0][0].bias
        )
        
        # Modify classifier for pose prediction
        self.efficientnet.classifier[1] = nn.Linear(
            in_features=self.efficientnet.classifier[1].in_features,
            out_features=pred_dim
        )
        
        # Initialize modified layers
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize modified layers"""
        nn.init.kaiming_normal_(self.efficientnet.features[0][0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.efficientnet.classifier[1].weight, 0, 0.01)
        nn.init.constant_(self.efficientnet.classifier[1].bias, 0)
        
    def forward(self, frames, return_contrastive_loss=False):
        """
        Args:
            frames: (B, T, H, W) sequence of frames
            return_contrastive_loss: Whether to return contrastive loss (training mode)
        Returns:
            If return_contrastive_loss=False: pose (B, pred_dim)
            If return_contrastive_loss=True: (pose, contrastive_loss)
        """
        B, T, H, W = frames.shape
        
        # Step 1: Apply contrastive frame grouping
        if return_contrastive_loss and self.training:
            # Training mode: get embeddings and contrastive loss
            embeddings, contrastive_loss = self.contrastive_grouping(frames, training=True)
        else:
            # Inference mode: just get embeddings (no loss computation)
            with torch.no_grad():
                embeddings, _ = self.contrastive_grouping(frames, training=False)
            contrastive_loss = None
        
        # Step 2: Optical flow estimation (same as baseline)
        if T < 2:
            raise ValueError("Need at least 2 frames for optical flow")
        
        # Use consecutive frames for optical flow
        mid_idx = T // 2
        frame1 = frames[:, mid_idx-1].unsqueeze(1) if mid_idx > 0 else frames[:, 0].unsqueeze(1)
        frame2 = frames[:, mid_idx].unsqueeze(1) if mid_idx < T else frames[:, -1].unsqueeze(1)
        
        # Estimate optical flow
        flow = self.flow_estimator(frame1, frame2)  # (B, 2, H', W')
        
        # Resize flow to match frame size if needed
        if flow.shape[-2:] != (H, W):
            flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
        
        # Step 3: Combine flow with original frames
        frames_flat = frames.view(B, T, H, W)
        combined_input = torch.cat([flow, frames_flat], dim=1)  # (B, 2+T, H, W)
        
        # Step 4: Pass through EfficientNet
        pose = self.efficientnet(combined_input)
        
        if return_contrastive_loss and contrastive_loss is not None:
            return pose, contrastive_loss
        else:
            return pose


def build_comparison_model(model_type, **kwargs):
    """
    Factory function to build comparison models
    
    Args:
        model_type: 'baseline' or 'enhanced'
        **kwargs: Model parameters
        
    Returns:
        model: Initialized model
    """
    if model_type == 'baseline':
        return BaselineModel(**kwargs)
    elif model_type == 'enhanced':
        return EnhancedModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def test_comparison_models():
    """Test both comparison models"""
    print("Testing Comparison Models")
    print("=" * 40)
    
    # Test parameters
    B, T, H, W = 2, 8, 64, 64
    frames = torch.randn(B, T, H, W)
    
    # Test baseline model
    print("Testing Baseline Model (Optical Flow + EfficientNet)...")
    baseline = build_comparison_model('baseline', in_frames=T, pred_dim=6)
    baseline.eval()
    
    with torch.no_grad():
        pose_baseline = baseline(frames)
    print(f"[OK] Baseline output shape: {pose_baseline.shape}")
    
    # Test enhanced model
    print("\nTesting Enhanced Model (Contrastive + Optical Flow + EfficientNet)...")
    enhanced = build_comparison_model('enhanced', in_frames=T, pred_dim=6)
    enhanced.eval()
    
    with torch.no_grad():
        pose_enhanced = enhanced(frames, return_contrastive_loss=False)
    print(f"[OK] Enhanced output shape: {pose_enhanced.shape}")
    
    # Test enhanced model with contrastive loss (training mode)
    enhanced.train()
    pose_enhanced, contrastive_loss = enhanced(frames, return_contrastive_loss=True)
    print(f"[OK] Enhanced training output: pose {pose_enhanced.shape}, loss {contrastive_loss.item():.4f}")
    
    print("\n[SUCCESS] Comparison models test completed!")


if __name__ == "__main__":
    test_comparison_models()
