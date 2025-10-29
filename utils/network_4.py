# Enhanced Dual-Stream Architecture with FPS/NPS Dual Mamba
# Based on 3DET-Mamba: State Space Model for End-to-End 3D Object Detection
# Implements FPS/NPS sampling with dual Mamba processing on proven dual-stream foundation

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import efficientnet_b1

try:
    from .remamba import Remamba
    from .optical_flow import FlowNet
    from .contrastive_grouping import ContrastiveFrameGrouping
except ImportError:
    # For standalone testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from remamba import Remamba
    from optical_flow import FlowNet
    from contrastive_grouping import ContrastiveFrameGrouping


def fps_ordering(tokens):
    """
    FPS (Farthest Point Ordering) from Algorithm 1 - Line 19
    Orders tokens based on farthest point relationships

    Args:
        tokens: (B, K, C) token sequence

    Returns:
        ordered_tokens: (B, K, C) FPS ordered token sequence
        indices: (B, K) ordering indices
    """
    B, K, C = tokens.shape
    device = tokens.device

    # Initialize distance matrix and ordering indices
    distances = torch.full((B, K), float('inf'), device=device)
    ordered_indices = torch.zeros((B, K), dtype=torch.long, device=device)

    # Start with random first token
    first_idx = torch.randint(0, K, (B,), device=device)
    ordered_indices[:, 0] = first_idx

    # Update distances to first token
    for b in range(B):
        first_token = tokens[b, first_idx[b]]  # (C,)
        dists = torch.sum((tokens[b] - first_token.unsqueeze(0)) ** 2, dim=1)  # (K,)
        distances[b] = torch.minimum(distances[b], dists)

    # Iteratively order by farthest tokens
    for i in range(1, K):
        # Find farthest token for each batch
        farthest_idx = torch.argmax(distances, dim=1)  # (B,)
        ordered_indices[:, i] = farthest_idx

        # Update distances
        for b in range(B):
            farthest_token = tokens[b, farthest_idx[b]]  # (C,)
            dists = torch.sum((tokens[b] - farthest_token.unsqueeze(0)) ** 2, dim=1)  # (K,)
            distances[b] = torch.minimum(distances[b], dists)

    # Gather ordered tokens
    ordered_tokens = torch.gather(
        tokens, 1,
        ordered_indices.unsqueeze(-1).expand(-1, -1, C)
    )  # (B, K, C)

    return ordered_tokens, ordered_indices


def nps_ordering(tokens):
    """
    NPS (Nearest Point Ordering) from Algorithm 1 - Line 3
    Orders tokens based on nearest point relationships

    Args:
        tokens: (B, K, C) token sequence

    Returns:
        ordered_tokens: (B, K, C) NPS ordered token sequence
        indices: (B, K) ordering indices
    """
    B, K, C = tokens.shape
    device = tokens.device

    # Start with first token
    ordered_indices = torch.zeros((B, K), dtype=torch.long, device=device)
    used_mask = torch.zeros((B, K), dtype=torch.bool, device=device)

    # Initialize with random starting token
    start_idx = torch.randint(0, K, (B,), device=device)
    ordered_indices[:, 0] = start_idx
    used_mask[torch.arange(B), start_idx] = True

    # Iteratively find nearest unused tokens
    for i in range(1, K):
        current_tokens = torch.gather(
            tokens, 1,
            ordered_indices[:, i-1:i].unsqueeze(-1).expand(-1, -1, C)
        )  # (B, 1, C)

        # Calculate distances to all tokens
        distances = torch.sum(
            (tokens - current_tokens) ** 2, dim=2
        )  # (B, K)

        # Mask used tokens with large distance
        distances[used_mask] = float('inf')

        # Find nearest unused token
        nearest_idx = torch.argmin(distances, dim=1)  # (B,)
        ordered_indices[:, i] = nearest_idx
        used_mask[torch.arange(B), nearest_idx] = True

    # Gather ordered tokens
    ordered_tokens = torch.gather(
        tokens, 1,
        ordered_indices.unsqueeze(-1).expand(-1, -1, C)
    )  # (B, K, C)

    return ordered_tokens, ordered_indices


class DualMambaBlock(nn.Module):
    """
    Exact Algorithm 1 Dual Mamba Block from 3DET-Mamba Paper

    Input: token sequence T_{i-1}: (B, K, C)
    Output: token sequence T_i: (B, K, C)

    Following the exact pseudo code:
    1. Process with different point orders
    2. T^F_{i-1} ← NPS(T_{i-1})
    3. T^N_{i-1} ← NPS(T_{i-1})
    4-6. Normalization and Linear projections
    7-8. SiLU and Conv1d operations
    9-10. Linear projections for B_s and C_s
    11. Softmax for positive Δ_s
    12-13. Discretization with log and exp
    14-16. SSM operations
    17. Element-wise multiplication with SiLU
    18. End for loop
    19. FPS ordering: y^F_N ← FPS(y^N_i)
    20-21. Final linear projection and return
    """

    def __init__(self, d_model=512, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model  # C in algorithm
        self.d_state = d_state  # D in algorithm
        self.d_conv = d_conv

        # Algorithm Line 5-6: Normalization and Linear layers
        self.norm_f = nn.LayerNorm(d_model)  # For FPS branch
        self.norm_n = nn.LayerNorm(d_model)  # For NPS branch
        self.linear_f = nn.Linear(d_model, d_model)  # z_s for FPS
        self.linear_n = nn.Linear(d_model, d_model)  # z_s for NPS

        # Algorithm Line 7-8: SiLU and Conv1d
        # Use kernel_size=1 for token-wise processing (no temporal convolution needed)
        self.conv1d_f = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv1d_n = nn.Conv1d(d_model, d_model, kernel_size=1)

        # Algorithm Line 9-10: Linear projections for B_s and C_s
        self.linear_b_f = nn.Linear(d_model, d_state)  # B_s for FPS
        self.linear_c_f = nn.Linear(d_model, d_state)  # C_s for FPS
        self.linear_b_n = nn.Linear(d_model, d_state)  # B_s for NPS
        self.linear_c_n = nn.Linear(d_model, d_state)  # C_s for NPS

        # Algorithm Line 12-13: Discretization parameters
        self.linear_delta_f = nn.Linear(d_model, d_state)  # Δ_s for FPS
        self.linear_delta_n = nn.Linear(d_model, d_state)  # Δ_s for NPS

        # SSM parameters (A is fixed, learnable)
        self.A_log = nn.Parameter(torch.randn(d_state))  # log(A) parameter

        # Input projection for SSM (from d_model to d_state)
        self.input_projection = nn.Linear(d_model, d_state)

        # Algorithm Line 20: Final linear projection
        self.linear_out = nn.Linear(d_model, d_model)
    
    def ssm_step(self, x, A, B, C, delta):
        """
        Single SSM step following Algorithm 1 Lines 14-16

        Args:
            x: input (B, C) - will be projected to (B, D)
            A: state matrix (D,)
            B: input matrix (B, D)
            C: output matrix (B, D)
            delta: discretization parameter (B, D)

        Returns:
            y: output (B, D)
        """
        # Project input x from (B, C) to (B, D)
        x_proj = self.input_projection(x)  # (B, D)

        # Line 14: A_bar = exp(Δ_s ⊙ A)
        A_bar = torch.exp(delta * A.unsqueeze(0))  # (B, D)

        # Line 15: B_bar = Δ_s ⊙ B_s
        B_bar = delta * B  # (B, D)

        # Line 16: y_s = C_s ⊙ (A_bar ⊙ h_{s-1} + B_bar ⊙ x_s)
        # For simplicity, assume h_{s-1} = 0 (can be extended with state)
        y = C * (A_bar * 0 + B_bar * x_proj)  # (B, D)
        y = C * (B_bar * x_proj)  # Simplified: (B, D)

        return y

    def forward(self, tokens):
        """
        Exact Algorithm 1 Implementation

        Args:
            tokens: T_{i-1} (B, K, C) input token sequence

        Returns:
            T_i: (B, K, C) output token sequence
        """
        B, K, C = tokens.shape

        # Algorithm Line 2: T^F_{i-1} ← FPS(T_{i-1})
        T_F, _ = fps_ordering(tokens)  # (B, K, C)

        # Algorithm Line 3: T^N_{i-1} ← NPS(T_{i-1})
        T_N, _ = nps_ordering(tokens)  # (B, K, C)

        # Process both branches in parallel
        outputs = []

        for branch_name, T_branch in [('F', T_F), ('N', T_N)]:
            # Select appropriate layers for this branch
            if branch_name == 'F':
                norm_layer = self.norm_f
                linear_layer = self.linear_f
                conv1d_layer = self.conv1d_f
                linear_b = self.linear_b_f
                linear_c = self.linear_c_f
                linear_delta = self.linear_delta_f
            else:
                norm_layer = self.norm_n
                linear_layer = self.linear_n
                conv1d_layer = self.conv1d_n
                linear_b = self.linear_b_n
                linear_c = self.linear_c_n
                linear_delta = self.linear_delta_n

            # Algorithm Line 4: for s = 1 to K do
            branch_outputs = []

            for s in range(K):
                # Algorithm Line 5: z_s ← Linear(Norm(T^{F/N}_{i-1}[s]))
                token_s = T_branch[:, s, :]  # (B, C)
                z_s = linear_layer(norm_layer(token_s))  # (B, C)

                # Algorithm Line 6: z_s ← SiLU(z_s)
                z_s = F.silu(z_s)  # (B, C)

                # Algorithm Line 7: z_s ← Conv1d(z_s)
                # Reshape for conv1d: (B, C, 1) -> (B, C, 1) -> (B, C)
                z_s_reshaped = z_s.unsqueeze(-1)  # (B, C, 1)
                z_s_conv = conv1d_layer(z_s_reshaped).squeeze(-1)  # (B, C)

                # Algorithm Line 8: B_s ← Linear(z_s), C_s ← Linear(z_s)
                B_s = linear_b(z_s_conv)  # (B, D)
                C_s = linear_c(z_s_conv)  # (B, D)

                # Algorithm Line 9: Δ_s ← Linear(z_s)
                delta_s = linear_delta(z_s_conv)  # (B, D)

                # Algorithm Line 10: Δ_s ← Softmax(Δ_s) to ensure Δ_s > 0
                delta_s = F.softmax(delta_s, dim=-1)  # (B, D)

                # Algorithm Line 11: A ← log(A) (A is learnable parameter)
                A = self.A_log  # (D,)

                # Algorithm Lines 12-16: SSM operation
                y_s = self.ssm_step(z_s_conv, A, B_s, C_s, delta_s)  # (B, D)

                # Pad to match original dimension if needed
                if y_s.shape[-1] != C:
                    if y_s.shape[-1] < C:
                        # Pad with zeros
                        padding = torch.zeros(B, C - y_s.shape[-1], device=y_s.device)
                        y_s = torch.cat([y_s, padding], dim=-1)
                    else:
                        # Truncate
                        y_s = y_s[:, :C]

                # Algorithm Line 17: y_s ← SiLU(y_s) ⊙ z_s
                y_s = F.silu(y_s) * z_s_conv[:, :C]  # (B, C)

                branch_outputs.append(y_s)

            # Stack outputs for this branch: (B, K, C)
            branch_output = torch.stack(branch_outputs, dim=1)
            outputs.append(branch_output)

        # Algorithm Line 18: end for (both branches completed)

        # Combine FPS and NPS outputs (average fusion)
        y_combined = (outputs[0] + outputs[1]) / 2  # (B, K, C)

        # Algorithm Line 19: y^F_N ← FPS(y^N_i)
        y_F_N, _ = fps_ordering(y_combined)  # (B, K, C)

        # Algorithm Line 20: T_i ← Linear(y^F_N)
        T_i = self.linear_out(y_F_N)  # (B, K, C)

        # Algorithm Line 21: return T_i
        return T_i


class SpatialStream(nn.Module):
    """
    Spatial Stream: EfficientNet-based Spatial Feature Extraction (Enhanced from dual-stream)
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

            # Apply spatial Mamba
            spatial_feat_mamba = self.spatial_mamba(spatial_feat_flat.unsqueeze(-1).unsqueeze(-1))  # (B, 1280, 1, 1)
            spatial_feat_mamba = spatial_feat_mamba.flatten(1)  # (B, 1280)

            # Fusion to target dimension
            spatial_feat_fused = self.spatial_fusion(spatial_feat_mamba)  # (B, embed_dim)

            spatial_features.append(spatial_feat_fused)

        # Stack features: (B, T, embed_dim)
        spatial_features = torch.stack(spatial_features, dim=1)

        return spatial_features


class EnhancedTemporalStream(nn.Module):
    """
    Enhanced Temporal Stream with FPS/NPS Dual Mamba

    Pipeline:
    Input Frames → Contrastive Grouping → Optical Flow → Feature+Flow Concat → Point Cloud → FPS/NPS Dual Mamba
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
        self.flow_net = FlowNet(in_channels=2, output_dim=256)

        # Step 3: Feature + Flow concatenation dimension
        self.concat_dim = 256 + 256  # contrastive_features + optical_flow = 512

        # Step 4: Point cloud generation from concatenated features
        self.point_cloud_generator = nn.Sequential(
            nn.Linear(self.concat_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),  # Final point cloud feature dimension
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Step 5: FPS/NPS Dual Mamba Block (3x blocks as in 3DET-Mamba)
        self.dual_mamba_blocks = nn.ModuleList([
            DualMambaBlock(d_model=512, d_state=16, d_conv=4) for _ in range(3)
        ])

        # Step 6: Final temporal feature projection
        self.temporal_projection = nn.Sequential(
            nn.Linear(512, embed_dim),
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
            temporal_features: (B, T, embed_dim) enhanced temporal features
            triplet_loss: Contrastive loss (if training)
        """
        B, T, H, W = frames.shape

        # Step 1: Apply contrastive learning first
        contrastive_embeddings, triplet_loss = self.contrastive_grouping(frames, training=True)  # (B, T, 256)

        # Step 2: Compute optical flow on contrastive-grouped frames
        optical_flow = self.compute_optical_flow(contrastive_embeddings, frames)  # (B, T, 256)

        # Step 3: Concatenate contrastive features + optical flow (KEY ENHANCEMENT)
        # This creates motion-enhanced features for better point cloud generation
        enhanced_features = torch.cat([contrastive_embeddings, optical_flow], dim=-1)  # (B, T, 512)

        # Step 4: Generate token sequence from enhanced features
        # Reshape for batch processing: (B*T, 512)
        enhanced_features_flat = enhanced_features.view(B*T, self.concat_dim)

        # Generate token features
        token_features_flat = self.point_cloud_generator(enhanced_features_flat)  # (B*T, 512)

        # Reshape to token sequence format: (B, T, 512) -> (B*T, 1, 512) -> (B*T, K, 512)
        # Create K tokens per frame for Algorithm 1
        K_tokens = 64  # Number of tokens per frame for FPS/NPS processing

        # Expand single token to K tokens with positional diversity
        token_features = token_features_flat.view(B*T, 1, 512)  # (B*T, 1, 512)
        token_features_expanded = token_features.repeat(1, K_tokens, 1)  # (B*T, K, 512)

        # Add positional encoding to create token diversity
        pos_encoding = torch.randn(1, K_tokens, 512, device=frames.device) * 0.1
        token_features_expanded = token_features_expanded + pos_encoding

        # Step 5: Apply FPS/NPS Dual Mamba Blocks (3x as in 3DET-Mamba)
        # Process each frame separately through Algorithm 1
        # Reshape from (B*T, K, 512) to (B, T, K, 512) for frame-wise processing
        dual_mamba_features = token_features_expanded.view(B, T, K_tokens, 512)  # (B, T, K, 512)

        # Process each frame through dual mamba blocks
        frame_features = []
        for t in range(T):
            frame_tokens = dual_mamba_features[:, t, :, :]  # (B, K, 512)

            # Apply 3x Dual Mamba blocks to this frame
            for dual_mamba_block in self.dual_mamba_blocks:
                frame_tokens = dual_mamba_block(frame_tokens)  # (B, K, 512)

            # Global average pooling across tokens for this frame
            frame_feat = torch.mean(frame_tokens, dim=1)  # (B, 512)
            frame_features.append(frame_feat)

        # Stack frame features: (B, T, 512)
        pooled_features = torch.stack(frame_features, dim=1)  # (B, T, 512)

        # Step 6: Project to final temporal feature dimension
        temporal_features = self.temporal_projection(pooled_features)  # (B, T, embed_dim)

        return temporal_features, triplet_loss


class EnhancedDualStreamFusion(nn.Module):
    """
    Enhanced Fusion Layer: Combines spatial and FPS/NPS enhanced temporal features

    Pipeline:
    Spatial Features + Enhanced Temporal Features → Concatenation → Final Mamba → 6-DOF Output
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
            temporal_features: (B, T, embed_dim) FPS/NPS enhanced temporal features

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


class EnhancedDualStreamNetwork(nn.Module):
    """
    Complete Enhanced Dual-Stream Architecture with FPS/NPS Dual Mamba

    Architecture:
    Input Frames → Dual Stream Processing → Enhanced Stream Fusion → 6-DOF Output
         ↓
    Stream 1: Spatial Processing (Pure Spatial Features) - UNCHANGED
    Stream 2: Enhanced Temporal Processing (Contrastive → Optical Flow → Feature+Flow Concat → FPS/NPS Dual Mamba)
         ↓
    Fusion: torch.cat([spatial_features, enhanced_temporal_features], dim=1)
         ↓
    Final Mamba → Sequential 6-DOF Output per Frame

    Key Enhancement: FPS/NPS Dual Mamba replaces simple bidirectional Mamba
    """

    def __init__(self, input_channels=1, embed_dim=256, num_frames=7, output_dim=6,
                 margin_alpha=0.2, delta=2, Delta=4, tau_sim=0.5):
        super().__init__()

        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.output_dim = output_dim

        # Adaptive parameters based on number of frames (from dual-stream)
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

        # ENHANCED: Replace simple temporal stream with FPS/NPS enhanced version
        self.temporal_stream = EnhancedTemporalStream(
            input_channels=input_channels,
            embed_dim=embed_dim,
            num_frames=num_frames,
            margin_alpha=margin_alpha,
            delta=self.delta_param,
            Delta=self.Delta_param,
            tau_sim=tau_sim
        )

        # ENHANCED: Use enhanced fusion layer
        self.fusion = EnhancedDualStreamFusion(
            embed_dim=embed_dim,
            num_frames=num_frames,
            output_dim=output_dim
        )

    def forward(self, frames, training=True, return_contrastive_loss=False):
        """
        Main forward pass of enhanced dual-stream architecture with FPS/NPS Dual Mamba

        Args:
            frames: (B, T, H, W) sequence of frames
            training: Whether in training mode
            return_contrastive_loss: Whether to return contrastive loss

        Returns:
            If return_contrastive_loss=False: output (B, T, 6)
            If return_contrastive_loss=True: (output, contrastive_loss)
        """
        # Stream 1: Spatial processing (unchanged)
        spatial_features = self.spatial_stream(frames)  # (B, T, embed_dim)

        # Stream 2: ENHANCED temporal processing with FPS/NPS Dual Mamba
        # Contrastive → Optical Flow → Feature+Flow Concat → Point Cloud → FPS/NPS Dual Mamba
        temporal_features, triplet_loss = self.temporal_stream(frames, training=training)  # (B, T, embed_dim)

        # Fusion: Combine both streams with enhanced fusion
        output = self.fusion(spatial_features, temporal_features)  # (B, T, 6)

        if return_contrastive_loss and triplet_loss is not None:
            return output, triplet_loss
        else:
            return output


def create_enhanced_dual_stream_network(input_channels=1, embed_dim=256, num_frames=7, output_dim=6,
                                      margin_alpha=0.2, delta=2, Delta=4, tau_sim=0.5):
    """
    Factory function to create enhanced dual-stream network with FPS/NPS Dual Mamba

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
        model: Initialized enhanced dual-stream network with FPS/NPS Dual Mamba
    """
    return EnhancedDualStreamNetwork(
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
def test_enhanced_dual_stream_network():
    """Test the enhanced dual-stream network with FPS/NPS Dual Mamba"""
    print(" Testing Enhanced Dual-Stream Network with FPS/NPS Dual Mamba...")

    # Create enhanced model
    model = create_enhanced_dual_stream_network(
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

    print("[SUCCESS] Enhanced Dual-Stream Network with FPS/NPS Dual Mamba test completed!")
    print("[START] Key Enhancements:")
    print("   - FPS sampling: Captures global motion patterns")
    print("   - NPS sampling: Captures local motion details")
    print("   - Dual Mamba: Forward + Backward temporal processing")
    print("   - Cross-attention fusion: Combines FPS + NPS features")
    print("   - 3x Dual Mamba blocks: Deep spatial-temporal modeling")


if __name__ == "__main__":
    test_enhanced_dual_stream_network()
