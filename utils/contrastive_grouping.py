"""
Algorithm 1: Contrastive Frame Grouping
Exact implementation following the provided pseudocode

Input: Frame sequence {It}T_t=1, margin α = 0.2
Initialize embedding network fϕ (EfficientNet backbone)
Training: Sample triplets and compute triplet loss
Inference: Cluster frames using DBSCAN
Output: Frame groups {Gk} with motion-coherent frames
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import DBSCAN
from torchvision.models import efficientnet_b1


class ContrastiveFrameGrouping(nn.Module):
    """
    Algorithm 1: Contrastive Frame Grouping

    Implements the exact algorithm from the pseudocode:
    1. EfficientNet backbone for embedding network fϕ (consistent with project)
    2. Triplet sampling with constraints |a-p| ≤ δ and |a-n| ≥ Δ
    3. Triplet loss: Ltri = max(||ea - ep||²₂ - ||ea - en||²₂ + α, 0)
    4. DBSCAN clustering for motion-coherent frame groups
    """

    def __init__(self,
                 margin_alpha=0.2,      # α = 0.2 (from algorithm)
                 delta=2,               # δ: positive pair threshold
                 Delta=4,               # Δ: negative pair threshold
                 tau_sim=0.5,           # ε: DBSCAN similarity threshold
                 embed_dim=256,         # Embedding dimension
                 input_channels=1):     # Ultrasound grayscale
        """
        Initialize Algorithm 1 components

        Args:
            margin_alpha: α = 0.2 (triplet loss margin from algorithm)
            delta: δ (positive pair constraint |a-p| ≤ δ)
            Delta: Δ (negative pair constraint |a-n| ≥ Δ)
            tau_sim: ε (DBSCAN similarity threshold)
            embed_dim: Embedding network output dimension
            input_channels: Input channels (1 for grayscale ultrasound)
        """
        super().__init__()

        # Algorithm 1 parameters
        self.alpha = margin_alpha  # α = 0.2
        self.delta = delta         # δ
        self.Delta = Delta         # Δ
        self.tau_sim = tau_sim     # ε for DBSCAN
        self.embed_dim = embed_dim

        # Step 2: Initialize embedding network fϕ (EfficientNet backbone)
        self.f_phi = self._initialize_efficientnet_backbone(input_channels, embed_dim)

    def _initialize_efficientnet_backbone(self, input_channels, embed_dim):
        """
        Step 2: Initialize embedding network fϕ (EfficientNet-B1 backbone)
        Using EfficientNet-B1 for consistent feature extraction across the architecture
        """
        # Create EfficientNet-B1 backbone for embeddings
        efficientnet = efficientnet_b1(weights=None)

        # Modify first layer for ultrasound input
        efficientnet.features[0][0] = nn.Conv2d(
            in_channels=input_channels,
            out_channels=efficientnet.features[0][0].out_channels,
            kernel_size=efficientnet.features[0][0].kernel_size,
            stride=efficientnet.features[0][0].stride,
            padding=efficientnet.features[0][0].padding,
            bias=efficientnet.features[0][0].bias
        )

        # Replace classifier with embedding layer
        efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(efficientnet.classifier[1].in_features, embed_dim),
            nn.ReLU(inplace=True)
        )

        return efficientnet

    def sample_anchor_positive_negative(self, T):
        """
        Step 4: Sample anchor Ia, positive Ip (|a − p| ≤ δ), negative In (|a − n| ≥ Δ)

        Args:
            T: Total number of frames in sequence

        Returns:
            a, p, n: Indices for anchor, positive, negative frames
        """
        # Sample anchor frame index
        a = np.random.randint(0, T)

        # Find valid positive indices: |a − p| ≤ δ
        valid_positives = []
        for p_candidate in range(T):
            if p_candidate != a and abs(a - p_candidate) <= self.delta:
                valid_positives.append(p_candidate)

        # Find valid negative indices: |a − n| ≥ Δ
        valid_negatives = []
        for n_candidate in range(T):
            if abs(a - n_candidate) >= self.Delta:
                valid_negatives.append(n_candidate)

        # Handle edge cases
        if len(valid_positives) == 0:
            # If no valid positives, use closest non-anchor frame
            valid_positives = [i for i in range(T) if i != a]

        if len(valid_negatives) == 0:
            # If no valid negatives, use farthest available frame
            distances = [(abs(a - i), i) for i in range(T) if i != a]
            distances.sort(reverse=True)
            valid_negatives = [distances[0][1]]

        # Sample positive and negative
        p = np.random.choice(valid_positives)
        n = np.random.choice(valid_negatives)

        return a, p, n

    def compute_embeddings(self, frames):
        """
        Step 5: Compute embeddings: ei = fϕ(Ii)

        Args:
            frames: Input frame sequence {It}T_t=1

        Returns:
            embeddings: {et}T_t=1 computed by fϕ
        """
        if frames.dim() == 4:  # (B, T, H, W)
            B, T, H, W = frames.shape
            # Reshape to process all frames: (B*T, 1, H, W)
            frames_flat = frames.view(B * T, 1, H, W)
            embeddings = self.f_phi(frames_flat)  # (B*T, embed_dim)
            # Reshape back: (B, T, embed_dim)
            embeddings = embeddings.view(B, T, self.embed_dim)
        elif frames.dim() == 3:  # (T, H, W) - single sequence
            T, H, W = frames.shape
            frames_with_channel = frames.unsqueeze(1)  # (T, 1, H, W)
            embeddings = self.f_phi(frames_with_channel)  # (T, embed_dim)
        else:
            raise ValueError(f"Unexpected frame dimensions: {frames.shape}")

        return embeddings
    
    def triplet_loss(self, embeddings, a, p, n):
        """
        Step 6: Triplet loss: Ltri = max(∥ea − ep∥²₂ − ∥ea − en∥²₂ + α, 0)

        Args:
            embeddings: Computed embeddings {et}
            a, p, n: Anchor, positive, negative indices

        Returns:
            Ltri: Triplet loss value
        """
        # Extract embeddings for anchor, positive, negative
        ea = embeddings[a]  # Anchor embedding
        ep = embeddings[p]  # Positive embedding
        en = embeddings[n]  # Negative embedding

        # Compute L2 squared distances
        dist_ap = torch.norm(ea - ep, p=2) ** 2  # ∥ea − ep∥²₂
        dist_an = torch.norm(ea - en, p=2) ** 2  # ∥ea − en∥²₂

        # Triplet loss: max(∥ea − ep∥²₂ − ∥ea − en∥²₂ + α, 0)
        Ltri = torch.clamp(dist_ap - dist_an + self.alpha, min=0.0)

        return Ltri

    def training_iteration(self, frame_sequence):
        """
        Steps 3-7: Training iteration loop

        Args:
            frame_sequence: {It}T_t=1 input frame sequence

        Returns:
            embeddings, triplet_loss: Computed embeddings and loss
        """
        # Step 5: Compute embeddings: ei = fϕ(Ii)
        embeddings = self.compute_embeddings(frame_sequence)

        if embeddings.dim() == 3:  # (B, T, embed_dim)
            B, T, _ = embeddings.shape
            # Process each sequence in batch
            total_loss = 0
            for b in range(B):
                # Step 4: Sample anchor, positive, negative for this sequence
                a, p, n = self.sample_anchor_positive_negative(T)

                # Step 6: Compute triplet loss
                loss = self.triplet_loss(embeddings[b], a, p, n)
                total_loss += loss

            # Average loss across batch
            Ltri = total_loss / B

        else:  # (T, embed_dim) - single sequence
            T, _ = embeddings.shape
            # Step 4: Sample anchor, positive, negative
            a, p, n = self.sample_anchor_positive_negative(T)

            # Step 6: Compute triplet loss
            Ltri = self.triplet_loss(embeddings, a, p, n)

        return embeddings, Ltri

    def inference_clustering(self, frame_sequence):
        """
        Steps 8-11: Inference phase

        Args:
            frame_sequence: {It}T_t=1 input frame sequence

        Returns:
            frame_groups: {Gk}K_k=1 motion-coherent frame groups
        """
        # Step 9: Compute all embeddings {et}T_t=1
        embeddings = self.compute_embeddings(frame_sequence)

        # Flatten embeddings for clustering
        if embeddings.dim() == 3:  # (B, T, embed_dim)
            B, T, embed_dim = embeddings.shape
            embeddings_flat = embeddings.view(B * T, embed_dim)
        else:  # (T, embed_dim)
            embeddings_flat = embeddings

        # Convert to numpy for DBSCAN
        embeddings_np = embeddings_flat.detach().cpu().numpy()

        # Step 10: Cluster frames using DBSCAN: {Gk}K_k=1 ← DBSCAN({et}, ε = τsim)
        clustering = DBSCAN(eps=self.tau_sim, min_samples=2)
        cluster_labels = clustering.fit_predict(embeddings_np)

        # Step 11: Output: Frame groups {Gk} with motion-coherent frames
        frame_groups = {}
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Ignore noise points (-1)
                if label not in frame_groups:
                    frame_groups[label] = []
                frame_groups[label].append(i)

        return frame_groups

    def forward(self, frame_sequence, training=True):
        """
        Main forward pass implementing Algorithm 1

        Args:
            frame_sequence: {It}T_t=1 input frame sequence
            training: Whether in training mode

        Returns:
            If training: (embeddings, triplet_loss)
            If inference: (embeddings, frame_groups)
        """
        if training:
            # Steps 3-7: Training iteration
            return self.training_iteration(frame_sequence)
        else:
            # Steps 8-11: Inference clustering
            embeddings = self.compute_embeddings(frame_sequence)
            frame_groups = self.inference_clustering(frame_sequence)
            return embeddings, frame_groups
    
    def cluster_frames(self, embeddings):
        """
        Cluster frames using DBSCAN for motion-coherent grouping
        
        Args:
            embeddings: (N, embed_dim) frame embeddings
            
        Returns:
            groups: Dictionary mapping cluster_id -> [frame_indices]
        """
        # Convert to numpy for DBSCAN
        embeddings_np = embeddings.detach().cpu().numpy()
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=self.tau_sim, min_samples=2)
        labels = clustering.fit_predict(embeddings_np)
        
        # Group frames by cluster
        groups = {}
        for i, label in enumerate(labels):
            if label != -1:  # Ignore noise points
                if label not in groups:
                    groups[label] = []
                groups[label].append(i)
        
        return groups
    
    def inference(self, frames):
        """
        Inference method for frame grouping
        
        Args:
            frames: (B, T, H, W) sequence of frames
            
        Returns:
            groups: Motion-coherent frame groups
        """
        self.eval()
        with torch.no_grad():
            _, groups = self.forward(frames, training=False)
        return groups


def test_algorithm_1():
    """Test Algorithm 1 implementation"""
    print("Testing Algorithm 1: Contrastive Frame Grouping")
    print("=" * 50)

    # Test parameters
    T = 8  # Frame sequence length
    H, W = 64, 64  # Frame dimensions

    # Create test frame sequence {It}T_t=1
    frame_sequence = torch.randn(1, T, H, W)  # (B=1, T=8, H=64, W=64)

    # Initialize Algorithm 1 with α = 0.2
    cfg = ContrastiveFrameGrouping(
        margin_alpha=0.2,  # α = 0.2 from algorithm
        delta=2,           # δ
        Delta=4,           # Δ
        tau_sim=0.5        # ε for DBSCAN
    )

    # Test training iteration (Steps 3-7)
    print("Testing Training Phase...")
    cfg.train()
    embeddings, Ltri = cfg(frame_sequence, training=True)
    print(f"[OK] Embeddings shape: {embeddings.shape}")
    print(f"[OK] Triplet loss Ltri: {Ltri.item():.4f}")

    # Test inference clustering (Steps 8-11)
    print("\nTesting Inference Phase...")
    cfg.eval()
    with torch.no_grad():
        embeddings, frame_groups = cfg(frame_sequence, training=False)

    print(f"[OK] Found {len(frame_groups)} motion-coherent groups {{Gk}}")
    for k, group in frame_groups.items():
        print(f"   Group G{k}: frames {group}")

    print("\n[SUCCESS] Algorithm 1 implementation test completed!")


if __name__ == "__main__":
    test_algorithm_1()
