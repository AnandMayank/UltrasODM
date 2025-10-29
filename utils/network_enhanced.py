# Enhanced Network based on network_1.py structure
# Implements Algorithm 1: Contrastive Frame Grouping BEFORE optical flow
# Follows the exact structure of network_1.py with contrastive enhancement

import torch
import torch.nn as nn
import torch.nn.functional as F
from .remamba import Remamba
from .optical_flow import FlowNet
from .contrastive_grouping import ContrastiveFrameGrouping


# Using ContrastiveFrameGrouping from contrastive_grouping.py




def build_model(opt, in_frames, pred_dim):
    """
    Enhanced build_model function following network_1.py structure
    Integrates Algorithm 1: Contrastive Frame Grouping BEFORE optical flow
    """

    class EnhancedCustomModel(nn.Module):
        def __init__(self):
            super().__init__()

            # Step 1: Contrastive Frame Grouping (Algorithm 1) - BEFORE optical flow
            # Adjust parameters based on number of frames
            if in_frames <= 2:
                # For small sequences, use relaxed constraints
                delta_param = 1
                Delta_param = 1  # Allow any frame as negative
            elif in_frames <= 4:
                # For 4 frames, use constraints that allow valid triplets
                delta_param = 1  # |a-p| ≤ 1 (adjacent frames)
                Delta_param = 2  # |a-n| ≥ 2 (non-adjacent frames)
            else:
                # Standard Algorithm 1 parameters for longer sequences
                delta_param = 2
                Delta_param = 4

            self.contrastive_grouping = ContrastiveFrameGrouping(
                margin_alpha=0.2,      # α = 0.2 (from algorithm)
                delta=delta_param,      # δ: positive pair threshold (adaptive)
                Delta=Delta_param,      # Δ: negative pair threshold (adaptive)
                tau_sim=0.5,           # ε: DBSCAN similarity threshold
                embed_dim=256,         # Embedding dimension
                input_channels=1       # Ultrasound grayscale
            )

            # Step 2: Optical Flow Network (real FlowNet from network_1.py)
            self.flow_net = FlowNet()

            # Step 3: Encoder (from network_1.py structure)
            # Modified to handle single channel input (ultrasound)
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Changed from 3 to 1 channel
                Remamba(64),  # Real Mamba implementation
                nn.MaxPool2d(2),
            )

            # Step 4: Channel adaptation for concatenated features
            self.channel_adapter = nn.Conv2d(128, 64, kernel_size=1)  # 128 -> 64 channels

            # Step 5: Decoder (from network_1.py structure)
            # Modified to output sequential 6-DOF per frame
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                Remamba(32),  # Real Mamba implementation
                nn.Conv2d(32, pred_dim, kernel_size=1),
            )

            # Additional components for enhanced functionality
            self.num_frames = in_frames
            self.pred_dim = pred_dim

        def forward(self, frames):
            """
            Enhanced forward pass with contrastive grouping BEFORE optical flow

            Args:
                frames: (B, T, H, W) input frame sequence

            Returns:
                If training:
                    output: Model predictions
                    triplet_loss: Contrastive triplet loss
                If inference:
                    output: Model predictions
                    frame_groups: Motion-coherent frame groups
            """
            B, T, H, W = frames.shape
            training = self.training

            # Step 1: Contrastive Frame Grouping (Algorithm 1) - BEFORE optical flow
            # Always compute triplet loss for both training and validation monitoring
            embeddings, triplet_loss = self.contrastive_grouping(frames, training=True)
            frame_groups = None

            # Step 2: Compute optical flow between consecutive frames (from network_1.py)
            flow_features = []
            for i in range(frames.shape[1]-1):
                # Extract individual frames for optical flow
                frame1 = frames[:,i]    # (B, H, W)
                frame2 = frames[:,i+1]  # (B, H, W)

                # Add channel dimension for FlowNet
                frame1 = frame1.unsqueeze(1)  # (B, 1, H, W)
                frame2 = frame2.unsqueeze(1)  # (B, 1, H, W)

                # Compute optical flow
                flow = self.flow_net(frame1, frame2)
                flow_features.append(flow)

            if flow_features:
                flow_features = torch.stack(flow_features, dim=1)  # (B, T-1, flow_dim)
            else:
                # Handle case with single frame
                flow_features = torch.zeros(B, 1, 256, device=frames.device)  # Adjust size as needed

            # Step 3: Combine with frame features (from network_1.py)
            # Process frames through encoder
            # Reshape frames for encoder: (B, T, H, W) -> (B*T, 1, H, W)
            frames_reshaped = frames.view(B*T, 1, H, W)
            encoded_frames = self.encoder(frames_reshaped)  # (B*T, 64, H', W')

            # Reshape back: (B*T, 64, H', W') -> (B, T, 64, H', W')
            _, C, H_enc, W_enc = encoded_frames.shape
            encoded_frames = encoded_frames.view(B, T, C, H_enc, W_enc)

            # Average over time dimension to match flow features
            encoded_frames = encoded_frames.mean(dim=1)  # (B, 64, H', W')

            # Resize flow features to match encoded frames if needed
            if flow_features.dim() == 3:  # (B, T-1, flow_dim)
                # Reshape flow features to spatial format for concatenation
                flow_dim = flow_features.shape[-1]
                # Create spatial representation
                flow_spatial = flow_features.mean(dim=1).unsqueeze(-1).unsqueeze(-1)  # (B, flow_dim, 1, 1)
                flow_spatial = flow_spatial.expand(-1, -1, H_enc, W_enc)  # (B, flow_dim, H', W')

                # Adjust channels to match encoder output
                if flow_dim != C:
                    flow_adapter = nn.Conv2d(flow_dim, C, kernel_size=1).to(frames.device)
                    flow_spatial = flow_adapter(flow_spatial)
            else:
                flow_spatial = torch.zeros_like(encoded_frames)

            # Concatenate encoded frames with flow features
            x = torch.cat([encoded_frames, flow_spatial], dim=1)  # (B, 2*C, H', W')

            # Step 4: Channel adaptation (128 -> 64 channels)
            x = self.channel_adapter(x)  # (B, 64, H', W')

            # Step 5: Decoder (from network_1.py)
            output = self.decoder(x)  # (B, pred_dim, H'', W'')

            # Global average pooling to get final prediction
            output = F.adaptive_avg_pool2d(output, (1, 1))  # (B, pred_dim, 1, 1)
            output = output.view(B, pred_dim)  # (B, pred_dim)

            # Return based on training mode
            if training:
                return output, triplet_loss
            else:
                return output, frame_groups

    return EnhancedCustomModel()
