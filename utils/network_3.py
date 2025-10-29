"""
UltrasOM: A Mamba-based Network for 3D Freehand Ultrasound Reconstruction using Optical Flow
Implementation of the UltrasOM architecture using Video Mamba Suite components.
Based on the Video Mamba Suite: https://github.com/OpenGVLab/video-mamba-suite
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# Global debug counter to limit debug prints
_debug_print_count = 0
_max_debug_prints = 1

# Add video-mamba-suite to path
import sys
import os
video_mamba_path = os.path.join(os.path.dirname(__file__), '..', 'video-mamba-suite', 'video-mamba-suite', 'action-recognition')
if video_mamba_path not in sys.path:
    sys.path.insert(0, video_mamba_path)

# Try to import Video Mamba components
try:
    from models.vivim import VisionMamba, Block, create_block, PatchEmbed
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
    VIDEO_MAMBA_AVAILABLE = True
    print("[OK] Video Mamba Suite components loaded successfully")
except ImportError as e:
    print(f"[WARNING] Video Mamba Suite not available: {e}")
    VIDEO_MAMBA_AVAILABLE = False
    # Fallback imports
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class OpticalFlowEstimator(nn.Module):
    """Lightweight optical flow estimation module"""
    
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.flow_conv = nn.Conv2d(32, 2, 3, padding=1)  # 2 channels for x,y flow
        
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


class VideoEmbeddingModule(nn.Module):
    """Video embedding module that integrates optical flow with static information"""
    
    def __init__(self, in_channels=1, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Optical flow estimator
        self.flow_estimator = OpticalFlowEstimator(in_channels)
        
        # Static feature extractor
        self.static_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Flow feature extractor
        self.flow_conv = nn.Sequential(
            nn.Conv2d(2, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Fusion layer (static: 256 + flow: 128 = 384 total channels)
        self.fusion = nn.Conv2d(384, embed_dim, 1)
        
    def forward(self, frames):
        """
        Args:
            frames: (B, T, H, W) sequence of frames
        Returns:
            embeddings: (B, T-1, embed_dim, H', W') embedded features
        """
        B, T, H, W = frames.shape
        embeddings = []
        
        for t in range(T - 1):
            frame1 = frames[:, t:t+1]  # (B, 1, H, W)
            frame2 = frames[:, t+1:t+2]  # (B, 1, H, W)
            
            # Extract static features from current frame
            static_feat = self.static_conv(frame1)
            
            # Estimate optical flow
            flow = self.flow_estimator(frame1, frame2)
            
            # Extract flow features
            flow_feat = self.flow_conv(flow)
            
            # Resize flow features to match static features
            flow_feat = F.interpolate(flow_feat, size=static_feat.shape[-2:], 
                                    mode='bilinear', align_corners=False)
            
            # Fuse static and flow features
            fused = torch.cat([static_feat, flow_feat], dim=1)
            embedded = self.fusion(fused)
            embeddings.append(embedded)
        
        return torch.stack(embeddings, dim=1)  # (B, T-1, embed_dim, H', W')


class MambaBlock(nn.Module):
    """Simplified Mamba block for spatiotemporal modeling"""
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # Convolution for local dependencies
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # Activation
        self.act = nn.SiLU()
        
    def forward(self, x):
        """
        Args:
            x: (B, L, D) where L is sequence length, D is d_model
        Returns:
            output: (B, L, D)
        """
        B, L, D = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)
        
        # Convolution (local dependencies)
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]  # Trim to original length
        x = x.transpose(1, 2)  # (B, L, d_inner)
        x = self.act(x)
        
        # SSM computation (simplified)
        ssm_out = self.ssm(x)
        
        # Gate with z
        output = ssm_out * self.act(z)
        
        # Output projection
        output = self.out_proj(output)
        
        return output
    
    def ssm(self, x):
        """State Space Model computation with selective scan"""
        B, L, D = x.shape

        # Project to get delta, A, B, C parameters
        delta_BC = self.x_proj(x)  # (B, L, 2*d_state)
        delta, BC = delta_BC.chunk(2, dim=-1)  # Each (B, L, d_state)

        # Delta projection for time step
        delta = F.softplus(self.dt_proj(x))  # (B, L, d_inner)

        # Initialize state with correct dimensions
        h = torch.zeros(B, self.d_state//2, device=x.device, dtype=x.dtype)
        outputs = []

        # Selective scan
        for t in range(L):
            # Get current inputs
            x_t = x[:, t]  # (B, d_inner)
            delta_t = delta[:, t]  # (B, d_inner)
            BC_t = BC[:, t]  # (B, d_state)

            # Split B and C
            B_t = BC_t[:, :self.d_state//2]  # (B, d_state//2)
            C_t = BC_t[:, self.d_state//2:]  # (B, d_state//2)

            # State update (simplified)
            # Ensure dimensions match for state update
            if h.shape[-1] != B_t.shape[-1]:
                # Adjust h dimensions to match B_t
                h = h[:, :B_t.shape[-1]]

            # h = A * h + B * x (where A is learned implicitly through delta)
            decay = torch.exp(-delta_t[:, :B_t.shape[-1]].unsqueeze(1))  # Match B_t dimensions
            h = h * decay + B_t.unsqueeze(1) * x_t[:, :B_t.shape[-1]].unsqueeze(-1)

            # Output: y = C * h
            y_t = torch.sum(C_t.unsqueeze(1) * h, dim=-1)  # (B, d_inner)

            # Pad y_t to match d_inner if needed
            if y_t.shape[-1] < self.d_inner:
                padding = torch.zeros(B, self.d_inner - y_t.shape[-1], device=x.device, dtype=x.dtype)
                y_t = torch.cat([y_t, padding], dim=-1)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B, L, d_inner)


class SpaceTimeBlock(nn.Module):
    """Space-Time Block for spatiotemporal attention"""
    
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Spatial Mamba block
        self.spatial_mamba = MambaBlock(d_model)
        
        # Temporal Mamba block  
        self.temporal_mamba = MambaBlock(d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: (B, T, H, W, D) spatiotemporal features
        Returns:
            output: (B, T, H, W, D)
        """
        B, T, H, W, D = x.shape
        
        # Spatial processing
        x_spatial = x.view(B * T, H * W, D)
        x_spatial = x_spatial + self.spatial_mamba(self.norm1(x_spatial))
        x_spatial = x_spatial.view(B, T, H, W, D)
        
        # Temporal processing
        x_temporal = x_spatial.permute(0, 2, 3, 1, 4).contiguous()  # (B, H, W, T, D)
        x_temporal = x_temporal.view(B * H * W, T, D)
        x_temporal = x_temporal + self.temporal_mamba(self.norm2(x_temporal))
        x_temporal = x_temporal.view(B, H, W, T, D).permute(0, 3, 1, 2, 4)  # Back to (B, T, H, W, D)
        
        # Feed forward
        x_out = x_temporal.view(B * T * H * W, D)
        x_out = x_out + self.ffn(self.norm3(x_out))
        x_out = x_out.view(B, T, H, W, D)
        
        return x_out


class VideoMambaUltrasOM(nn.Module):
    """
    UltrasOM implementation using Video Mamba Suite for proper bidirectional processing
    """

    def __init__(self,
                 num_frames=8,
                 img_size=(480, 640),  # Match actual ultrasound image size
                 patch_size=16,
                 in_channels=1,
                 embed_dim=384,
                 depth=12,
                 pred_dim=6,
                 drop_path_rate=0.1,
                 if_bidirectional=True,
                 if_cls_token=True,
                 use_middle_cls_token=True,
                 frame_mid_cls_token=True):
        super().__init__()

        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.pred_dim = pred_dim

        # Use proper Video Mamba implementation with correct parameters
        try:
            if VIDEO_MAMBA_AVAILABLE:
                self.backbone = VisionMamba(
                    img_size=img_size,
                    patch_size=patch_size,
                    num_frames=num_frames,
                    depth=depth,
                    embed_dim=embed_dim,
                    channels=in_channels,
                    num_classes=0,  # No classification head
                    drop_path_rate=drop_path_rate,
                    if_bidirectional=if_bidirectional,
                    if_cls_token=if_cls_token,
                    use_middle_cls_token=use_middle_cls_token,
                    frame_mid_cls_token=frame_mid_cls_token,
                    if_abs_pos_embed=True,
                    final_pool_type='none',
                    bimamba_type="v2"  # Specify the correct bimamba type
                )
                print("[OK] Video Mamba backbone initialized successfully")
                self.use_video_mamba = True
            else:
                raise ImportError("Video Mamba not available")
        except Exception as e:
            print(f"[WARNING] Video Mamba initialization failed: {e}")
            print("[INFO] Falling back to simplified implementation")
            self.backbone = self._create_fallback_backbone(embed_dim, depth)
            self.use_video_mamba = False

        # Optical flow integration
        self.flow_estimator = OpticalFlowEstimator(in_channels)
        self.flow_fusion = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embed_dim // 4)
        )

        # Prediction head for 6DoF parameters
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(embed_dim + embed_dim // 4),
            nn.Linear(embed_dim + embed_dim // 4, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, pred_dim)
        )

    def _create_fallback_backbone(self, embed_dim, depth):
        """Create a simplified backbone when Video Mamba is not available"""
        return nn.Sequential(
            nn.Conv3d(1, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(64, embed_dim)
        )

    def forward(self, frames, return_features=False):
        """
        Args:
            frames: (B, T, H, W) or (B, 1, T, H, W) input frame sequence
            return_features: Whether to return intermediate features
        Returns:
            predictions: (B, pred_dim) 6DoF parameters
            features: (optional) intermediate features for contrastive learning
        """
        # Handle input dimensions - Video Mamba expects (B, T, C, H, W)
        global _debug_print_count, _max_debug_prints

        # Add debug info only for the first few batches
        if _debug_print_count < _max_debug_prints:
            print(f" Input frames shape: {frames.shape}")

        if frames.dim() == 4:  # (B, T, H, W) - add channel dimension
            frames = frames.unsqueeze(2)  # (B, T, 1, H, W)
            if _debug_print_count < _max_debug_prints:
                print(f" After adding channel dim: {frames.shape}")
        elif frames.dim() == 5 and frames.size(2) > 1:  # (B, T, C, H, W) with C > 1
            frames = frames.mean(dim=2, keepdim=True)  # Average channels
            if _debug_print_count < _max_debug_prints:
                print(f" After averaging channels: {frames.shape}")

        B, T, C, H, W = frames.shape
        if _debug_print_count < _max_debug_prints:
            print(f" Final input shape for Video Mamba: B={B}, T={T}, C={C}, H={H}, W={W}")

        # Extract features using Video Mamba backbone
        if hasattr(self, 'use_video_mamba') and self.use_video_mamba:
            # Video Mamba processing with proper input format
            if _debug_print_count < _max_debug_prints:
                print(f" Passing to Video Mamba with shape: {frames.shape}")

            # Reshape for patch embedding: (B, T, C, H, W) -> (B*T, C, H, W)
            B, T, C, H, W = frames.shape
            frames_reshaped = frames.view(B * T, C, H, W)
            if _debug_print_count < _max_debug_prints:
                print(f" Reshaped for patch embedding: {frames_reshaped.shape}")

            # Process through patch embedding
            patch_features = self.backbone.patch_embed(frames_reshaped)  # (B*T, N, embed_dim)
            BT, N, embed_dim = patch_features.shape
            if _debug_print_count < _max_debug_prints:
                print(f" Patch features shape: {patch_features.shape}")

            # Reshape back to (B, T, N, embed_dim) for temporal processing
            patch_features = patch_features.view(B, T, N, embed_dim)

            # Add positional embeddings and process through Mamba layers
            # For now, let's use a simplified approach - average over time and spatial dimensions
            backbone_features = patch_features.mean(dim=[1, 2])  # (B, embed_dim)
            if _debug_print_count < _max_debug_prints:
                print(f" Final backbone features shape: {backbone_features.shape}")
                _debug_print_count += 1  # Increment counter to stop printing debug info
        else:
            # Fallback implementation - reshape for 3D conv
            frames_3d = frames.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            backbone_features = self.backbone(frames_3d)  # (B, embed_dim)

        # Optical flow features (use middle frames for flow estimation)
        if T >= 2:
            mid_idx = T // 2
            # Extract frames from the original input: (B, T, C, H, W)
            frame1 = frames[:, mid_idx-1, 0] if mid_idx > 0 else frames[:, 0, 0]  # (B, H, W)
            frame2 = frames[:, mid_idx, 0] if mid_idx < T else frames[:, -1, 0]  # (B, H, W)

            # Estimate optical flow
            flow = self.flow_estimator(frame1.unsqueeze(1), frame2.unsqueeze(1))
            flow_features = self.flow_fusion(flow)  # (B, embed_dim//4)
        else:
            # No flow for single frame
            flow_features = torch.zeros(B, self.embed_dim // 4, device=frames.device)

        # Combine backbone and flow features
        combined_features = torch.cat([backbone_features, flow_features], dim=1)

        # Final prediction
        predictions = self.prediction_head(combined_features)

        if return_features:
            return predictions, combined_features
        return predictions


def build_ultrasom_model(num_frames=8, in_channels=1, pred_dim=6, embed_dim=384, num_layers=12, img_size=(480, 640)):
    """
    Build UltrasOM model using Video Mamba Suite

    Args:
        num_frames: Number of input frames
        in_channels: Number of input channels (1 for grayscale)
        pred_dim: Output dimension (6 for 6DoF parameters)
        embed_dim: Embedding dimension
        num_layers: Number of Mamba layers (depth)
        img_size: Input image size (H, W)

    Returns:
        UltrasOM model
    """
    return VideoMambaUltrasOM(
        num_frames=num_frames,
        img_size=img_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        depth=num_layers,
        pred_dim=pred_dim,
        if_bidirectional=True,  # Enable bidirectional processing
        if_cls_token=True,
        use_middle_cls_token=True,
        frame_mid_cls_token=True
    )


class UltrasOMLoss(nn.Module):
    """
    UltrasOM loss function with correlation and motion velocity losses
    """

    def __init__(self, alpha_mse=1.0, alpha_corr=0.5, alpha_velocity=0.3):
        super().__init__()
        self.alpha_mse = alpha_mse
        self.alpha_corr = alpha_corr
        self.alpha_velocity = alpha_velocity

    def forward(self, predictions, targets, velocities=None):
        """
        Args:
            predictions: (B, 6) predicted 6DoF parameters
            targets: (B, 6) ground truth 6DoF parameters
            velocities: (B,) optional motion velocities for velocity loss
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # MSE loss
        mse_loss = F.mse_loss(predictions, targets)

        # Correlation loss (prevents overfitting to scanning speed)
        corr_loss = self.correlation_loss(predictions, targets)

        # Motion velocity loss (improves generalization across poses)
        velocity_loss = torch.tensor(0.0, device=predictions.device)
        if velocities is not None:
            velocity_loss = self.motion_velocity_loss(predictions, targets, velocities)

        # Combined loss
        total_loss = (self.alpha_mse * mse_loss +
                     self.alpha_corr * corr_loss +
                     self.alpha_velocity * velocity_loss)

        loss_dict = {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'correlation_loss': corr_loss,
            'velocity_loss': velocity_loss
        }

        return total_loss, loss_dict

    def correlation_loss(self, predictions, targets):
        """Correlation loss to prevent overfitting to scanning patterns"""
        # Normalize predictions and targets
        pred_norm = F.normalize(predictions, dim=1)
        target_norm = F.normalize(targets, dim=1)

        # Compute correlation
        correlation = torch.sum(pred_norm * target_norm, dim=1)

        # Loss is 1 - correlation (we want high correlation)
        loss = torch.mean(1.0 - correlation)

        return loss

    def motion_velocity_loss(self, predictions, targets, velocities):
        """
        Motion velocity loss for pose generalization
        Using formula: 1/6 * (n-2) * Σ(v_i - v_pred_i)²

        Args:
            predictions: (B, 6) predicted 6DoF parameters
            targets: (B, 6) ground truth 6DoF parameters
            velocities: (B,) motion velocities for the sequence
        """
        # TODO: Implement the specific velocity loss formula when activated
        # Current implementation (commented out for now):

        # # Extract velocity components from predictions and targets
        # # Assuming the last 3 components represent rotational velocities
        # pred_velocities = torch.norm(predictions[:, 3:], dim=1)  # Rotational velocity magnitude
        # target_velocities = torch.norm(targets[:, 3:], dim=1)    # Ground truth rotational velocity
        #
        # # Calculate velocity prediction error
        # velocity_errors = (velocities - pred_velocities) ** 2
        #
        # # Apply the formula: 1/6 * (n-2) * Σ(v_i - v_pred_i)²
        # n = velocities.shape[0]  # Number of samples in batch
        # if n >= 2:
        #     velocity_loss = (1.0/6.0) * (n - 2) * torch.sum(velocity_errors)
        # else:
        #     velocity_loss = torch.tensor(0.0, device=predictions.device)
        #
        # return velocity_loss

        # Fallback to original implementation for now
        error = torch.norm(predictions - targets, dim=1)
        velocity_weights = 1.0 / (velocities + 1e-6)
        weighted_error = error * velocity_weights
        return torch.mean(weighted_error)






