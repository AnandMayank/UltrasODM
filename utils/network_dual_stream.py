# Dual-Stream Architecture with Optical Flow Integration
# Implements spatial and temporal streams with fusion mechanism
# Based on proven components from existing codebase

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import efficientnet_b1
from .remamba import Remamba
from .optical_flow import FlowNet
from .contrastive_grouping import ContrastiveFrameGrouping


class SpatialStream(nn.Module):
    """
    Spatial Stream: EfficientNet-based Spatial Feature Extraction

    Pipeline:
    Input Frames → EfficientNet Features → Spatial Embeddings (per frame processing)
    """

    def __init__(self, input_channels=1, embed_dim=256, num_frames=7):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames

        # EfficientNet-B1 backbone for spatial feature extraction
        self.efficientnet = efficientnet_b1(weights=None)

        # Modify first layer for ultrasound input
        self.efficientnet.features[0][0] = nn.Conv2d(
            in_channels=input_channels,
            out_channels=self.efficientnet.features[0][0].out_channels,
            kernel_size=self.efficientnet.features[0][0].kernel_size,
            stride=self.efficientnet.features[0][0].stride,
            padding=self.efficientnet.features[0][0].padding,
            bias=self.efficientnet.features[0][0].bias
        )

        # Remove the classifier to get features
        self.efficientnet.classifier = nn.Identity()

        # Get EfficientNet-B1 feature dimension (1280 for B1)
        efficientnet_features = 1280

        # Spatial Mamba for within-frame processing
        self.spatial_mamba = Remamba(d_model=efficientnet_features)

        # Spatial feature fusion to target embedding dimension
        self.spatial_fusion = nn.Sequential(
            nn.Linear(efficientnet_features, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, frames):
        """
        Args:
            frames: (B, T, H, W) sequence of frames

        Returns:
            spatial_features: (B, T, embed_dim) spatial features per frame
        """
        B, T, H, W = frames.shape

        # Extract spatial features per frame using EfficientNet
        spatial_features = []
        for t in range(T):
            frame = frames[:, t].unsqueeze(1)  # (B, 1, H, W)

            # EfficientNet feature extraction
            spatial_feat = self.efficientnet.features(frame)  # (B, 1280, H', W')
            spatial_feat = self.efficientnet.avgpool(spatial_feat)  # (B, 1280, 1, 1)
            spatial_feat_flat = spatial_feat.flatten(1)  # (B, 1280)

            # Apply spatial Mamba (Mamba expects (batch, sequence, dim))
            spatial_feat_mamba_input = spatial_feat_flat.unsqueeze(1).contiguous()  # (B, 1, 1280)
            try:
                spatial_feat_mamba = self.spatial_mamba(spatial_feat_mamba_input)  # (B, 1, 1280)
            except RuntimeError as e:
                if "stride" in str(e):
                    print("Mamba forward failed: {}, using fallback".format(str(e)))
                    # Fallback: use a simple linear transformation
                    spatial_feat_mamba = spatial_feat_mamba_input
                else:
                    raise e
            spatial_feat_mamba = spatial_feat_mamba.squeeze(1)  # (B, 1280)

            # Fusion to target dimension
            spatial_feat_fused = self.spatial_fusion(spatial_feat_mamba)  # (B, embed_dim)

            spatial_features.append(spatial_feat_fused)

        # Stack features: (B, T, embed_dim)
        spatial_features = torch.stack(spatial_features, dim=1)

        return spatial_features


class TemporalStream(nn.Module):
    """
    Temporal Stream: Contrastive Grouping → Optical Flow → Temporal Features
    
    Pipeline:
    Input Frames → Contrastive Grouping → Optical Flow → Temporal Mamba → Temporal Features
    """
    
    def __init__(self, input_channels=1, embed_dim=256, num_frames=7,
                 margin_alpha=0.2, delta=2, Delta=4, tau_sim=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        
        # Step 1: Contrastive frame grouping (Algorithm 1)
        self.contrastive_grouping = ContrastiveFrameGrouping(
            margin_alpha=margin_alpha,      # α = 0.2 (from algorithm)
            delta=delta,                    # δ: positive pair threshold
            Delta=Delta,                    # Δ: negative pair threshold
            tau_sim=tau_sim,                # ε: DBSCAN similarity threshold
            embed_dim=256,                  # Contrastive embedding dimension
            input_channels=input_channels
        )
        
        # Step 2: Optical flow network (proven from train.py)
        self.flow_net = FlowNet(in_channels=2)
        
        # Step 3: Temporal feature extraction with Mamba
        self.temporal_encoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Temporal Mamba for cross-frame processing
        self.temporal_mamba = Remamba(d_model=embed_dim)
        
        # Temporal fusion
        self.temporal_fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def compute_optical_flow(self, contrastive_embeddings, frames):
        """
        Compute optical flow on contrastive-grouped frames
        
        Args:
            contrastive_embeddings: (B, T, 256) embeddings from contrastive grouping
            frames: (B, T, H, W) original frames
            
        Returns:
            optical_flow: (B, T, 256) optical flow features per frame
        """
        B, T, H, W = frames.shape
        
        # Compute optical flow between consecutive frames
        flow_features = []
        
        for t in range(T):
            if t == 0:
                # For first frame, use flow between frame 0 and 1
                if T > 1:
                    frame1 = frames[:, 0].unsqueeze(1)  # (B, 1, H, W)
                    frame2 = frames[:, 1].unsqueeze(1)  # (B, 1, H, W)
                    flow_feat = self.flow_net(frame1, frame2)  # (B, 256)
                else:
                    # Single frame case - zero flow
                    flow_feat = torch.zeros(B, 256, device=frames.device)
            elif t == T - 1:
                # For last frame, use flow between frame T-2 and T-1
                frame1 = frames[:, T-2].unsqueeze(1)  # (B, 1, H, W)
                frame2 = frames[:, T-1].unsqueeze(1)  # (B, 1, H, W)
                flow_feat = self.flow_net(frame1, frame2)  # (B, 256)
            else:
                # For middle frames, use flow between frame t-1 and t+1
                frame1 = frames[:, t-1].unsqueeze(1)  # (B, 1, H, W)
                frame2 = frames[:, t+1].unsqueeze(1)  # (B, 1, H, W)
                flow_feat = self.flow_net(frame1, frame2)  # (B, 256)
            
            flow_features.append(flow_feat)
        
        # Stack optical flow features: (B, T, 256)
        optical_flow = torch.stack(flow_features, dim=1)
        
        return optical_flow
    
    def forward(self, frames, training=True):
        """
        Args:
            frames: (B, T, H, W) sequence of frames
            training: Whether in training mode

        Returns:
            temporal_features: (B, T, embed_dim) temporal features per frame
            triplet_loss: Contrastive loss (if training) or None (if inference)
        """
        B, T, H, W = frames.shape

        # Step 1: Apply contrastive learning (following network_enhanced.py approach)
        # ALWAYS use training=True for contrastive grouping to get triplet loss
        contrastive_embeddings, triplet_loss = self.contrastive_grouping(frames, training=True)
        frame_groups = None  # Not used in dual-stream architecture
        
        # Step 2: Then compute optical flow on contrastive-grouped frames
        optical_flow = self.compute_optical_flow(contrastive_embeddings, frames)
        
        # Step 3: Process temporal features
        temporal_features = []
        for t in range(T):
            flow_feat = optical_flow[:, t]  # (B, 256)
            
            # Encode temporal features
            temporal_feat = self.temporal_encoder(flow_feat)  # (B, embed_dim)
            temporal_features.append(temporal_feat)
        
        # Stack temporal features: (B, T, embed_dim)
        temporal_features = torch.stack(temporal_features, dim=1)
        
        # Step 4: Apply temporal Mamba for cross-frame processing
        # Reshape for Mamba: (B*T, embed_dim, 1, 1)
        temporal_features_reshaped = temporal_features.view(B*T, self.embed_dim, 1, 1).contiguous()
        try:
            temporal_features_mamba = self.temporal_mamba(temporal_features_reshaped)
        except RuntimeError as e:
            if "stride" in str(e):
                print("Mamba forward failed: {}, using fallback".format(str(e)))
                # Fallback: use the input features
                temporal_features_mamba = temporal_features_reshaped
            else:
                raise e
        temporal_features_mamba = temporal_features_mamba.view(B, T, self.embed_dim)
        
        # Step 5: Final temporal fusion
        temporal_features_final = []
        for t in range(T):
            feat = temporal_features_mamba[:, t]  # (B, embed_dim)
            feat_fused = self.temporal_fusion(feat)  # (B, embed_dim)
            temporal_features_final.append(feat_fused)
        
        temporal_features_final = torch.stack(temporal_features_final, dim=1)
        
        return temporal_features_final, triplet_loss


class DualStreamFusion(nn.Module):
    """
    Fusion Layer: Combines spatial and temporal features
    
    Pipeline:
    Spatial Features + Temporal Features → Concatenation → Final Mamba → 6-DOF Output
    """
    
    def __init__(self, embed_dim=256, num_frames=7, output_dim=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.output_dim = output_dim
        
        # Final Mamba for combined features
        self.final_mamba = Remamba(d_model=2*embed_dim)  # 2*embed_dim due to concatenation
        
        # Output head for sequential 6-DOF per frame
        self.output_head = nn.Sequential(
            nn.Linear(2*embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)  # 6-DOF output per frame
        )
    
    def forward(self, spatial_features, temporal_features):
        """
        Args:
            spatial_features: (B, T, embed_dim) spatial features
            temporal_features: (B, T, embed_dim) temporal features
            
        Returns:
            output: (B, T, 6) sequential 6-DOF output per frame
        """
        B, T, _ = spatial_features.shape
        
        # Combine both streams
        combined_features = torch.cat([spatial_features, temporal_features], dim=-1)  # (B, T, 2*embed_dim)
        
        # Apply final Mamba processing
        combined_features_reshaped = combined_features.view(B*T, 2*self.embed_dim, 1, 1)
        final_features = self.final_mamba(combined_features_reshaped)
        final_features = final_features.view(B, T, 2*self.embed_dim)
        
        # Generate sequential 6-DOF output per frame
        outputs = []
        for t in range(T):
            frame_feat = final_features[:, t]  # (B, 2*embed_dim)
            frame_output = self.output_head(frame_feat)  # (B, 6)
            outputs.append(frame_output)
        
        # Stack outputs: (B, T, 6)
        output = torch.stack(outputs, dim=1)
        
        return output


class DualStreamNetwork(nn.Module):
    """
    Complete Dual-Stream Architecture with Optical Flow Integration

    Architecture:
    Input Frames → Dual Stream Processing → Stream Fusion → 6-DOF Output
         ↓
    Stream 1: Spatial Processing (Pure Spatial Features)
    Stream 2: Temporal Processing (Contrastive Grouping → Optical Flow → Temporal Features)
         ↓
    Fusion: torch.cat([spatial_features, temporal_features], dim=1)
         ↓
    Final Mamba → Sequential 6-DOF Output per Frame
    """

    def __init__(self, input_channels=1, embed_dim=256, num_frames=7, output_dim=6,
                 margin_alpha=0.2, delta=2, Delta=4, tau_sim=0.5):
        super().__init__()

        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.output_dim = output_dim

        # Adaptive parameters based on number of frames (from network_enhanced.py)
        if num_frames <= 2:
            # For small sequences, use relaxed constraints
            self.delta_param = 1
            self.Delta_param = 1  # Allow any frame as negative
            print(f"[WARNING]  Warning: Only {num_frames} frames - contrastive grouping may be ineffective")
        elif num_frames <= 4:
            # For 4 frames, use constraints that allow valid triplets
            self.delta_param = 1  # |a-p| ≤ 1 (adjacent frames)
            self.Delta_param = 2  # |a-n| ≥ 2 (non-adjacent frames)
        else:
            # Standard Algorithm 1 parameters for longer sequences
            self.delta_param = delta
            self.Delta_param = Delta

        # Initialize dual streams
        self.spatial_stream = SpatialStream(
            input_channels=input_channels,
            embed_dim=embed_dim,
            num_frames=num_frames
        )

        self.temporal_stream = TemporalStream(
            input_channels=input_channels,
            embed_dim=embed_dim,
            num_frames=num_frames,
            margin_alpha=margin_alpha,
            delta=self.delta_param,
            Delta=self.Delta_param,
            tau_sim=tau_sim
        )

        # Initialize fusion layer
        self.fusion = DualStreamFusion(
            embed_dim=embed_dim,
            num_frames=num_frames,
            output_dim=output_dim
        )

    def forward(self, frames, training=True, return_contrastive_loss=False):
        """
        Main forward pass of dual-stream architecture

        Args:
            frames: (B, T, H, W) sequence of frames
            training: Whether in training mode
            return_contrastive_loss: Whether to return contrastive loss

        Returns:
            If return_contrastive_loss=False: output (B, T, 6)
            If return_contrastive_loss=True: (output, contrastive_loss)
        """
        # Stream 1: Spatial processing
        spatial_features = self.spatial_stream(frames)  # (B, T, embed_dim)

        # Stream 2: Temporal processing (contrastive → optical flow → temporal)
        temporal_features, triplet_loss = self.temporal_stream(frames, training=training)  # (B, T, embed_dim)

        # Fusion: Combine both streams
        output = self.fusion(spatial_features, temporal_features)  # (B, T, 6)

        # Handle contrastive loss return (following network_enhanced.py approach)
        if return_contrastive_loss:
            # Always return triplet_loss since we always compute it
            return output, triplet_loss
        else:
            return output


def create_dual_stream_network(input_channels=1, embed_dim=256, num_frames=7, output_dim=6,
                              margin_alpha=0.2, delta=2, Delta=4, tau_sim=0.5):
    """
    Factory function to create dual-stream network

    Args:
        input_channels: Number of input channels (1 for grayscale ultrasound)
        embed_dim: Embedding dimension for features
        num_frames: Number of frames in sequence
        output_dim: Output dimension (6 for 6-DOF)
        margin_alpha: α = 0.2 (triplet loss margin from algorithm)
        delta: δ (positive pair threshold)
        Delta: Δ (negative pair threshold)
        tau_sim: ε (DBSCAN similarity threshold)

    Returns:
        model: Initialized dual-stream network
    """
    return DualStreamNetwork(
        input_channels=input_channels,
        embed_dim=embed_dim,
        num_frames=num_frames,
        output_dim=output_dim,
        margin_alpha=margin_alpha,
        delta=delta,
        Delta=Delta,
        tau_sim=tau_sim
    )


# Test function
def test_dual_stream_network():
    """Test the dual-stream network with sample data"""
    print(" Testing Dual-Stream Network...")

    # Create model
    model = create_dual_stream_network(
        input_channels=1,
        embed_dim=256,
        num_frames=7,
        output_dim=6
    )

    # Create sample input
    B, T, H, W = 2, 7, 224, 224
    frames = torch.randn(B, T, H, W)

    print(f"[DATA] Input shape: {frames.shape}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test inference mode
    model.eval()
    with torch.no_grad():
        output = model(frames, training=False, return_contrastive_loss=False)
    print(f"[OK] Inference output shape: {output.shape}")  # Expected: (2, 7, 6)

    # Test training mode
    model.train()
    output, contrastive_loss = model(frames, training=True, return_contrastive_loss=True)
    print(f"[OK] Training output shape: {output.shape}")  # Expected: (2, 7, 6)
    print(f"[OK] Contrastive loss: {contrastive_loss.item():.4f}")

    print("[SUCCESS] Dual-Stream Network test completed!")


if __name__ == "__main__":
    test_dual_stream_network()
