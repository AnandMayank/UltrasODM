# Enhanced Loss Functions for UltrasOM with Point-Based Regression
# Inspired by TUS-REC baseline and optimized for direct point transformation

import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import PointDistance


class PointBasedLossFunction(nn.Module):
    """
    Point-based loss function following TUS-REC baseline approach:
    1. PointDistance loss (PRIMARY - direct point-to-point comparison)
    2. Manhattan distance loss (alternative geometric loss)
    3. Correlation loss (prevents overfitting to scanning speed)
    4. Optional contrastive learning (can be disabled for ablation)
    5. Minimal MSE loss (only for parameter space regularization)

    Key insight: Direct parameter-to-point transformation with PointDistance
    often outperforms complex weighted combinations.
    """

    def __init__(self,
                 alpha_point=1.0,       # PointDistance loss weight (PRIMARY)
                 alpha_manhattan=0.0,   # Manhattan distance weight (alternative)
                 alpha_corr=0.3,        # Correlation loss weight (reduced)
                 alpha_contrastive=0.0, # Contrastive loss weight (can be disabled)
                 alpha_mse=0.1,         # MSE loss weight (minimal)
                 margin=0.2,            # Margin for ranking loss
                 use_contrastive=True,  # Enable/disable contrastive learning
                 point_loss_mode='euclidean'): # 'euclidean', 'manhattan', 'combined'
        """
        Initialize enhanced loss function with focused losses

        Args:
            alpha_point: Weight for PointDistance loss (baseline metric)
            alpha_corr: Weight for correlation loss
            alpha_contrastive: Weight for contrastive loss
            alpha_mse: Weight for MSE loss
            margin: Margin for ranking loss
        """
        super().__init__()

        self.alpha_point = alpha_point
        self.alpha_corr = alpha_corr
        self.alpha_contrastive = alpha_contrastive
        self.alpha_mse = alpha_mse
        self.margin = margin

        # Loss functions
        self.point_distance = PointDistance(paired=True)  # From baseline
        self.mse_criterion = nn.MSELoss()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def correlation_loss_6dof(self, predictions, targets):
        """
        Correlation loss for 6 DOF parameters (from DCL-Net reference)
        Prevents overfitting to scanning speed variations

        Args:
            predictions: (B, 6) or (B, T, 6) predicted parameters
            targets: (B, 6) or (B, T, 6) ground truth parameters

        Returns:
            loss: Correlation loss (1 - correlation_coefficient)
        """
        try:
            # Flatten if multi-dimensional
            if predictions.dim() > 2:
                predictions = predictions.reshape(-1, predictions.shape[-1])
                targets = targets.reshape(-1, targets.shape[-1])

            # Calculate overall correlation across all DOFs
            pred_flat = predictions.flatten()
            target_flat = targets.flatten()

            # Calculate means
            pred_mean = torch.mean(pred_flat)
            target_mean = torch.mean(target_flat)

            # Calculate centered values
            pred_centered = pred_flat - pred_mean
            target_centered = target_flat - target_mean

            # Calculate correlation coefficient
            numerator = torch.sum(pred_centered * target_centered)
            pred_std = torch.sqrt(torch.sum(pred_centered ** 2) + 1e-8)
            target_std = torch.sqrt(torch.sum(target_centered ** 2) + 1e-8)

            correlation = numerator / (pred_std * target_std + 1e-8)
            correlation = torch.clamp(correlation, -1.0, 1.0)

            # Correlation loss: 1 - correlation (loss decreases as correlation increases)
            corr_loss = 1.0 - correlation

            return corr_loss

        except Exception as e:
            print(f"Warning: Correlation loss calculation failed: {e}")
            return torch.tensor(0.1, device=predictions.device)

    def margin_ranking_contrastive_loss(self, triplet_loss=None,
                                      anchor_embeddings=None, positive_embeddings=None, negative_embeddings=None):
        """
        Margin ranking loss for contrastive learning (preferred over contrastive loss)
        Uses triplet_loss if available, otherwise computes ranking loss

        Args:
            triplet_loss: Pre-computed triplet loss from Algorithm 1
            anchor_embeddings: Anchor embeddings for ranking loss
            positive_embeddings: Positive embeddings for ranking loss
            negative_embeddings: Negative embeddings for ranking loss

        Returns:
            loss: Contrastive loss
        """
        # Use triplet loss from Algorithm 1 if available (preferred)
        if triplet_loss is not None:
            return triplet_loss

        # Fallback to margin ranking loss
        if (anchor_embeddings is not None and
            positive_embeddings is not None and
            negative_embeddings is not None):

            # Calculate distances
            pos_distances = F.pairwise_distance(anchor_embeddings, positive_embeddings, p=2)
            neg_distances = F.pairwise_distance(anchor_embeddings, negative_embeddings, p=2)

            # Target: negative distance should be larger than positive distance
            target = torch.ones_like(pos_distances)

            # Margin ranking loss: max(0, margin + pos_dist - neg_dist)
            loss = self.ranking_loss(neg_distances, pos_distances, target)

            return loss

        # No contrastive information available
        return torch.tensor(0.0, device=anchor_embeddings.device if anchor_embeddings is not None else torch.device('cpu'))

    def forward(self, predictions, targets,
                pred_points=None, target_points=None,
                triplet_loss=None,
                anchor_embeddings=None, positive_embeddings=None, negative_embeddings=None):
        """
        Compute combined enhanced loss with focused losses

        Args:
            predictions: (B, 6) or (B, T, 6) predicted pose parameters
            targets: (B, 6) or (B, T, 6) ground truth pose parameters
            pred_points: Transformed prediction points for PointDistance
            target_points: Transformed target points for PointDistance
            triplet_loss: Pre-computed triplet loss from Algorithm 1
            anchor_embeddings: Anchor embeddings for ranking loss
            positive_embeddings: Positive embeddings for ranking loss
            negative_embeddings: Negative embeddings for ranking loss

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # 1. PointDistance loss (baseline - most important)
        if pred_points is not None and target_points is not None:
            point_loss = self.point_distance(pred_points, target_points)
        else:
            # Fallback to MSE if points not available
            point_loss = self.mse_criterion(predictions, targets)

        # 2. Correlation loss (prevents overfitting to scanning speed)
        corr_loss = self.correlation_loss_6dof(predictions, targets)

        # 3. Contrastive loss (motion-coherent frame grouping)
        contrastive_loss = self.margin_ranking_contrastive_loss(
            triplet_loss, anchor_embeddings, positive_embeddings, negative_embeddings
        )

        # Safety check: ensure contrastive_loss is a tensor
        if not isinstance(contrastive_loss, torch.Tensor):
            contrastive_loss = torch.tensor(0.0, device=predictions.device)

        # 4. MSE loss (standard reconstruction loss)
        mse_loss = self.mse_criterion(predictions, targets)

        # Combined loss with focused weights (ensure scalar output)
        total_loss = (self.alpha_point * point_loss +
                     self.alpha_corr * corr_loss +
                     self.alpha_contrastive * contrastive_loss +
                     self.alpha_mse * mse_loss)

        # Ensure total_loss is always a scalar for backward()
        if total_loss.numel() > 1:
            total_loss = total_loss.mean()

        # Loss dictionary for monitoring (handle multi-element tensors)
        loss_dict = {
            'total_loss': total_loss.mean().item() if total_loss.numel() > 1 else total_loss.item(),
            'point_loss': point_loss.mean().item() if isinstance(point_loss, torch.Tensor) and point_loss.numel() > 1 else (point_loss.item() if isinstance(point_loss, torch.Tensor) else 0.0),
            'corr_loss': corr_loss.mean().item() if isinstance(corr_loss, torch.Tensor) and corr_loss.numel() > 1 else (corr_loss.item() if isinstance(corr_loss, torch.Tensor) else 0.0),
            'contrastive_loss': contrastive_loss.mean().item() if isinstance(contrastive_loss, torch.Tensor) and contrastive_loss.numel() > 1 else (contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0.0),
            'mse_loss': mse_loss.mean().item() if isinstance(mse_loss, torch.Tensor) and mse_loss.numel() > 1 else (mse_loss.item() if isinstance(mse_loss, torch.Tensor) else 0.0)
        }

        return total_loss, loss_dict


def create_enhanced_loss(alpha_point=1.0, alpha_corr=0.5, alpha_contrastive=0.3, alpha_mse=0.2, margin=0.2):
    """
    Factory function to create enhanced loss function with focused losses

    Args:
        alpha_point: Weight for PointDistance loss (baseline)
        alpha_corr: Weight for correlation loss
        alpha_contrastive: Weight for contrastive loss
        alpha_mse: Weight for MSE loss
        margin: Margin for ranking loss

    Returns:
        Enhanced loss function
    """
    return EnhancedLossFunction(
        alpha_point=alpha_point,
        alpha_corr=alpha_corr,
        alpha_contrastive=alpha_contrastive,
        alpha_mse=alpha_mse,
        margin=margin
    )


# Convenience function for easy integration with existing code
def get_enhanced_criterion():
    """
    Get enhanced criterion with default weights optimized for UltrasOM

    Returns:
        Enhanced loss function with default weights
    """
    return create_enhanced_loss(
        alpha_point=1.0,      # PointDistance (baseline) - highest weight
        alpha_corr=0.5,       # Correlation loss - medium weight
        alpha_contrastive=0.3, # Contrastive loss - medium weight
        alpha_mse=0.2,        # MSE loss - lower weight (PointDistance is primary)
        margin=0.2            # Margin for contrastive ranking
    )
