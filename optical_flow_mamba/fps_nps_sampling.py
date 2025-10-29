"""
Simplified FPS/NPS Sampling for Ultrasound Pose Regression

Core sampling logic without complex dependencies.
Focus on effective point sampling for temporal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleFPSSampling(nn.Module):
    """
    Simplified Farthest Point Sampling (FPS)
    
    Captures global motion patterns by selecting points that are maximally distant
    from each other in feature space.
    """
    
    def __init__(self, num_points=64):
        super().__init__()
        self.num_points = num_points
    
    def forward(self, features):
        """
        Args:
            features: (B, T, H, W, C) or (B, T, N, C) feature tensor
        Returns:
            sampled_features: (B, num_points, C) sampled features
            indices: (B, num_points) sampling indices
        """
        B, T = features.shape[:2]
        
        # Flatten spatial dimensions if needed
        if features.dim() == 5:  # (B, T, H, W, C)
            B, T, H, W, C = features.shape
            features = features.view(B, T * H * W, C)
        else:  # (B, T, N, C)
            features = features.view(B, -1, features.shape[-1])
        
        # Apply FPS
        sampled_features, indices = self.farthest_point_sample(features, self.num_points)
        
        return sampled_features, indices
    
    def farthest_point_sample(self, points, num_samples):
        """
        Farthest Point Sampling algorithm
        
        Args:
            points: (B, N, C) point features
            num_samples: number of points to sample
        Returns:
            sampled_points: (B, num_samples, C)
            indices: (B, num_samples)
        """
        B, N, C = points.shape
        device = points.device
        
        # Initialize
        centroids = torch.zeros(B, num_samples, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        
        # Start with random point
        farthest = torch.randint(0, N, (B,), device=device)
        
        for i in range(num_samples):
            centroids[:, i] = farthest
            centroid = points[torch.arange(B), farthest].view(B, 1, C)
            
            # Calculate distances to current centroid
            dist = torch.sum((points - centroid) ** 2, dim=2)
            
            # Update minimum distances
            mask = dist < distance
            distance[mask] = dist[mask]
            
            # Find farthest point
            farthest = torch.argmax(distance, dim=1)
        
        # Gather sampled points
        batch_indices = torch.arange(B, device=device).view(B, 1).expand(B, num_samples)
        sampled_points = points[batch_indices, centroids]
        
        return sampled_points, centroids


class SimpleNPSSampling(nn.Module):
    """
    Simplified Nearest Point Sampling (NPS)
    
    Captures local motion details by selecting points based on density
    and local neighborhood relationships.
    """
    
    def __init__(self, num_points=128):
        super().__init__()
        self.num_points = num_points
    
    def forward(self, features):
        """
        Args:
            features: (B, T, H, W, C) or (B, T, N, C) feature tensor
        Returns:
            sampled_features: (B, num_points, C) sampled features
            indices: (B, num_points) sampling indices
        """
        B, T = features.shape[:2]
        
        # Flatten spatial dimensions if needed
        if features.dim() == 5:  # (B, T, H, W, C)
            B, T, H, W, C = features.shape
            features = features.view(B, T * H * W, C)
        else:  # (B, T, N, C)
            features = features.view(B, -1, features.shape[-1])
        
        # Apply NPS
        sampled_features, indices = self.nearest_point_sample(features, self.num_points)
        
        return sampled_features, indices
    
    def nearest_point_sample(self, points, num_samples):
        """
        Nearest Point Sampling based on local density
        
        Args:
            points: (B, N, C) point features
            num_samples: number of points to sample
        Returns:
            sampled_points: (B, num_samples, C)
            indices: (B, num_samples)
        """
        B, N, C = points.shape
        device = points.device
        
        # Calculate centroid
        centroid = torch.mean(points, dim=1, keepdim=True)  # (B, 1, C)
        
        # Calculate distances to centroid
        distances = torch.norm(points - centroid, dim=2)  # (B, N)
        
        # Sort by distance (nearest first)
        sorted_indices = torch.argsort(distances, dim=1)
        
        # Sample uniformly from sorted points to maintain local structure
        step = N // num_samples
        if step == 0:
            step = 1
        
        sampled_indices = sorted_indices[:, ::step][:, :num_samples]
        
        # If we don't have enough points, pad with random selection
        if sampled_indices.shape[1] < num_samples:
            remaining = num_samples - sampled_indices.shape[1]
            random_indices = torch.randint(0, N, (B, remaining), device=device)
            sampled_indices = torch.cat([sampled_indices, random_indices], dim=1)
        
        # Gather sampled points
        batch_indices = torch.arange(B, device=device).view(B, 1).expand(B, num_samples)
        sampled_points = points[batch_indices, sampled_indices]
        
        return sampled_points, sampled_indices


class CombinedFPSNPSSampling(nn.Module):
    """
    Combined FPS + NPS Sampling
    
    Uses both FPS (global patterns) and NPS (local details) for comprehensive
    temporal feature sampling.
    """
    
    def __init__(self, num_fps_points=32, num_nps_points=64):
        super().__init__()
        self.fps_sampler = SimpleFPSSampling(num_fps_points)
        self.nps_sampler = SimpleNPSSampling(num_nps_points)
        self.num_total_points = num_fps_points + num_nps_points
        
        # Feature fusion
        self.fusion_layer = nn.Linear(2, 1)  # Combine FPS and NPS weights
    
    def forward(self, features):
        """
        Args:
            features: (B, T, H, W, C) or (B, T, N, C) feature tensor
        Returns:
            combined_features: (B, num_total_points, C) combined sampled features
            fps_indices: (B, num_fps_points) FPS indices
            nps_indices: (B, num_nps_points) NPS indices
        """
        # Sample with both methods
        fps_features, fps_indices = self.fps_sampler(features)
        nps_features, nps_indices = self.nps_sampler(features)
        
        # Simple concatenation (can be enhanced with attention)
        combined_features = torch.cat([fps_features, nps_features], dim=1)
        
        return combined_features, fps_indices, nps_indices


def test_fps_nps_sampling():
    """Test function for FPS/NPS sampling"""
    print("Testing FPS/NPS Sampling...")
    
    # Create test data
    B, T, H, W, C = 2, 4, 32, 32, 256
    features = torch.randn(B, T, H, W, C)
    
    # Test FPS
    fps_sampler = SimpleFPSSampling(num_points=64)
    fps_features, fps_indices = fps_sampler(features)
    print(f"FPS output shape: {fps_features.shape}")
    
    # Test NPS
    nps_sampler = SimpleNPSSampling(num_points=128)
    nps_features, nps_indices = nps_sampler(features)
    print(f"NPS output shape: {nps_features.shape}")
    
    # Test Combined
    combined_sampler = CombinedFPSNPSSampling(num_fps_points=32, num_nps_points=64)
    combined_features, fps_idx, nps_idx = combined_sampler(features)
    print(f"Combined output shape: {combined_features.shape}")
    
    print("[OK] FPS/NPS Sampling tests passed!")


if __name__ == "__main__":
    test_fps_nps_sampling()
