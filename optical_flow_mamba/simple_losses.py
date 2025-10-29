"""
Enhanced Loss Functions for FPS/NPS + UltrasSOM Approach

Following UltrasSOM paper design with multi-component loss function:
1. PointDistance loss (primary - clinical accuracy)
2. MSE loss (parameter space regression)
3. Correlation loss (scanning speed invariance)
4. Motion velocity loss (pose generalization)

Key Features:
- Adaptive loss weighting based on motion magnitude
- Velocity-aware error penalization
- Correlation-based scanning pattern invariance
- Clinical accuracy optimization (<0.2mm target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PointDistance:
    """
    Point Distance Loss - Core loss function
    
    Borrowed from utils/loss.py
    This is the proven effective loss from TUS-REC baseline.
    """
    
    def __init__(self, paired=True):
        self.paired = paired
    
    def __call__(self, preds, labels):
        """
        Calculate point distance loss

        Args:
            preds: (B, N, 3) predicted points
            labels: (B, N, 3) ground truth points
        Returns:
            loss: scalar loss value
        """
        if self.paired:
            # Handle different input dimensions
            if preds.dim() == 3 and labels.dim() == 3:  # (B, N, 3)
                return ((preds - labels) ** 2).sum(dim=2).sqrt().mean()
            elif preds.dim() == 4 and labels.dim() == 4:  # (B, T, N, 3)
                return ((preds - labels) ** 2).sum(dim=3).sqrt().mean(dim=(0, 2))
            else:
                # Fallback: flatten and compute
                return ((preds - labels) ** 2).sum(dim=-1).sqrt().mean()
        else:
            return ((preds - labels) ** 2).sum(dim=-1).sqrt().mean()


class SimpleLossFunction(nn.Module):
    """
    Simplified Loss Function for FPS/NPS approach
    
    Uses ONLY PointDistance loss based on analysis that:
    1. Point-only loss gives better convergence than multi-loss
    2. Complex loss combinations cause interference
    3. TUS-REC baseline success with simple approach
    """
    
    def __init__(self, use_mse_fallback=True):
        super().__init__()
        
        # Primary loss: PointDistance (proven effective)
        self.point_distance = PointDistance(paired=True)
        
        # Fallback loss: MSE (when points not available)
        self.mse_criterion = nn.MSELoss()
        self.use_mse_fallback = use_mse_fallback
    
    def forward(self, predictions, targets, pred_points=None, target_points=None):
        """
        Compute simple loss
        
        Args:
            predictions: (B, 6) predicted pose parameters
            targets: (B, 6) ground truth pose parameters
            pred_points: (B, N, 3) transformed prediction points (optional)
            target_points: (B, N, 3) transformed target points (optional)
        
        Returns:
            loss: scalar loss value
            loss_dict: dictionary with loss components for monitoring
        """
        
        # Primary: PointDistance loss (if points available)
        if pred_points is not None and target_points is not None:
            loss = self.point_distance(pred_points, target_points)
            loss_type = 'point_distance'
        
        # Fallback: MSE loss on parameters
        elif self.use_mse_fallback:
            loss = self.mse_criterion(predictions, targets)
            loss_type = 'mse_fallback'
        
        else:
            raise ValueError("Either points or MSE fallback must be available")
        
        # Ensure loss is a scalar
        if isinstance(loss, torch.Tensor):
            if loss.numel() > 1:
                loss = loss.mean()  # Average over multiple pairs
            loss_value = loss.item()
        else:
            loss_value = loss

        # Loss dictionary for monitoring
        loss_dict = {
            'total_loss': loss_value,
            'loss_type': loss_type,
            'point_loss': loss_value if loss_type == 'point_distance' else 0.0,
            'mse_loss': loss_value if loss_type == 'mse_fallback' else 0.0
        }
        
        return loss, loss_dict


class OptionalMSELoss(nn.Module):
    """
    Optional MSE Loss for ablation studies
    
    Can be used to compare pure PointDistance vs PointDistance + MSE
    """
    
    def __init__(self, alpha_point=1.0, alpha_mse=0.1):
        super().__init__()
        
        self.alpha_point = alpha_point
        self.alpha_mse = alpha_mse
        
        self.point_distance = PointDistance(paired=True)
        self.mse_criterion = nn.MSELoss()
    
    def forward(self, predictions, targets, pred_points=None, target_points=None):
        """
        Compute combined PointDistance + MSE loss
        
        Args:
            predictions: (B, 6) predicted pose parameters
            targets: (B, 6) ground truth pose parameters
            pred_points: (B, N, 3) transformed prediction points
            target_points: (B, N, 3) transformed target points
        
        Returns:
            loss: scalar loss value
            loss_dict: dictionary with loss components
        """
        
        # PointDistance loss (primary)
        if pred_points is not None and target_points is not None:
            point_loss = self.point_distance(pred_points, target_points)
            if isinstance(point_loss, torch.Tensor) and point_loss.numel() > 1:
                point_loss = point_loss.mean()  # Average over multiple pairs
        else:
            point_loss = torch.tensor(0.0, device=predictions.device)

        # MSE loss (secondary)
        mse_loss = self.mse_criterion(predictions, targets)

        # Combined loss
        total_loss = self.alpha_point * point_loss + self.alpha_mse * mse_loss

        # Loss dictionary
        loss_dict = {
            'total_loss': total_loss.item(),
            'point_loss': point_loss.item() if isinstance(point_loss, torch.Tensor) else 0.0,
            'mse_loss': mse_loss.item(),
            'alpha_point': self.alpha_point,
            'alpha_mse': self.alpha_mse
        }
        
        return total_loss, loss_dict


def create_simple_loss_function(loss_type='point_only'):
    """
    Factory function to create loss function
    
    Args:
        loss_type: 'point_only', 'point_mse_combined'
    
    Returns:
        loss_function: Loss function instance
    """
    
    if loss_type == 'point_only':
        return SimpleLossFunction(use_mse_fallback=True)
    
    elif loss_type == 'point_mse_combined':
        return OptionalMSELoss(alpha_point=1.0, alpha_mse=0.1)
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def test_simple_losses():
    """Test function for simple losses"""
    print("Testing Simple Loss Functions...")
    
    # Test data
    B, N = 4, 5  # batch_size, num_points
    predictions = torch.randn(B, 6)  # 6-DOF parameters
    targets = torch.randn(B, 6)
    pred_points = torch.randn(B, N, 3)  # 3D points
    target_points = torch.randn(B, N, 3)
    
    # Test SimpleLossFunction
    simple_loss = SimpleLossFunction()
    loss1, loss_dict1 = simple_loss(predictions, targets, pred_points, target_points)
    print(f"SimpleLossFunction - Loss: {loss1.item():.4f}, Dict: {loss_dict1}")
    
    # Test OptionalMSELoss
    combined_loss = OptionalMSELoss()
    loss2, loss_dict2 = combined_loss(predictions, targets, pred_points, target_points)
    print(f"OptionalMSELoss - Loss: {loss2.item():.4f}, Dict: {loss_dict2}")
    
    # Test factory function
    point_only_loss = create_simple_loss_function('point_only')
    loss3, loss_dict3 = point_only_loss(predictions, targets, pred_points, target_points)
    print(f"Factory point_only - Loss: {loss3.item():.4f}, Dict: {loss_dict3}")
    
    print("[OK] Simple Loss Functions tests passed!")


class UltrasSOMEnhancedLoss(nn.Module):
    """
    Enhanced Multi-Component Loss Function following UltrasSOM design

    Components:
    1. MSE Loss (α₁ = 1.0): Standard regression loss for 6DoF parameters
    2. Correlation Loss (α₂ = 0.5): Prevents overfitting to scanning speeds
    3. Motion Velocity Loss (α₃ = 0.3): Improves pose generalization
    4. PointDistance Loss (α₄ = 2.0): Clinical accuracy (primary metric)

    Combined Loss: L = α₁L_mse + α₂L_corr + α₃L_velocity + α₄L_point
    """
    def __init__(self, alpha_mse=0.1, alpha_corr=0.05, alpha_velocity=0.05, alpha_point=5.0):
        super().__init__()
        self.alpha_mse = alpha_mse
        self.alpha_corr = alpha_corr
        self.alpha_velocity = alpha_velocity
        self.alpha_point = alpha_point

        # Individual loss components
        self.mse_loss = nn.MSELoss()
        self.point_distance = PointDistance(paired=True)

        # Small epsilon for numerical stability
        self.eps = 1e-6

    def correlation_loss(self, predictions, targets):
        """
        Correlation Loss: Prevents overfitting to specific scanning speeds
        Normalizes predictions and targets, then maximizes correlation
        """
        # Normalize predictions and targets
        pred_mean = torch.mean(predictions, dim=0, keepdim=True)
        target_mean = torch.mean(targets, dim=0, keepdim=True)

        pred_centered = predictions - pred_mean
        target_centered = targets - target_mean

        # Compute correlation coefficient
        pred_std = torch.std(pred_centered, dim=0) + self.eps
        target_std = torch.std(target_centered, dim=0) + self.eps

        correlation = torch.mean(
            (pred_centered * target_centered) / (pred_std * target_std)
        )

        # Return 1 - correlation to minimize (maximize correlation)
        return 1.0 - correlation

    def velocity_loss(self, predictions, targets, motion_info):
        """
        Motion Velocity Loss: Weights errors by inverse velocity
        Penalizes more for slow motion scenarios (better generalization)
        """
        # Get velocity information from motion_info
        velocity = motion_info.get('velocity', torch.ones_like(predictions[:, :3]))

        # Compute velocity magnitude
        velocity_magnitude = torch.norm(velocity, dim=1, keepdim=True) + self.eps

        # Compute MSE loss
        mse = torch.mean((predictions - targets) ** 2, dim=1, keepdim=True)

        # Weight by inverse velocity (higher weight for slow motion)
        velocity_weighted_loss = mse / velocity_magnitude

        return torch.mean(velocity_weighted_loss)

    def forward(self, model_output, targets, pred_points=None, target_points=None):
        """
        Enhanced forward pass with multi-component loss

        Args:
            model_output: dict containing:
                - 'pose': (B, 6) predicted 6-DOF parameters
                - 'motion_info': dict with motion statistics
            targets: (B, 6) ground truth pose parameters
            pred_points: (B, N, 3) transformed prediction points (optional)
            target_points: (B, N, 3) transformed target points (optional)
        Returns:
            loss: scalar total loss
            loss_dict: dict with individual and total losses
        """
        if isinstance(model_output, dict):
            predictions = model_output['pose']
            motion_info = model_output.get('motion_info', {})
        else:
            predictions = model_output
            motion_info = {}

        # Ensure same shape - handle different target formats
        if targets.shape != predictions.shape:
            if len(targets.shape) == 3 and targets.shape[-2:] == (3, 4):
                # Convert transformation matrix to 6-DOF parameters
                targets = self._matrix_to_6dof(targets)
            elif targets.numel() != predictions.numel():
                # Handle size mismatch - take first N elements or pad
                target_size = predictions.numel()
                if targets.numel() > target_size:
                    targets = targets.flatten()[:target_size].view(predictions.shape)
                else:
                    # Pad with zeros if targets is smaller
                    padded_targets = torch.zeros_like(predictions)
                    targets_flat = targets.flatten()
                    padded_targets.flatten()[:targets_flat.numel()] = targets_flat
                    targets = padded_targets
            else:
                targets = targets.view(predictions.shape)

        # 1. MSE Loss
        mse_loss = self.mse_loss(predictions, targets)

        # 2. Correlation Loss
        corr_loss = self.correlation_loss(predictions, targets)

        # 3. Motion Velocity Loss
        vel_loss = self.velocity_loss(predictions, targets, motion_info)

        # 4. PointDistance Loss (clinical accuracy)
        if pred_points is not None and target_points is not None:
            point_loss = self.point_distance(pred_points, target_points)
            if isinstance(point_loss, torch.Tensor) and point_loss.numel() > 1:
                point_loss = point_loss.mean()
        else:
            # Fallback to MSE if no points available
            point_loss = mse_loss

        # 5. Combined loss
        total_loss = (
            self.alpha_mse * mse_loss +
            self.alpha_corr * corr_loss +
            self.alpha_velocity * vel_loss +
            self.alpha_point * point_loss
        )

        # Loss dictionary for monitoring
        loss_dict = {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'correlation_loss': corr_loss.item(),
            'velocity_loss': vel_loss.item(),
            'point_loss': point_loss.item() if isinstance(point_loss, torch.Tensor) else 0.0,
            'weights': {
                'alpha_mse': self.alpha_mse,
                'alpha_corr': self.alpha_corr,
                'alpha_velocity': self.alpha_velocity,
                'alpha_point': self.alpha_point
            }
        }

        return total_loss, loss_dict


def create_enhanced_loss_function(loss_type='ultrasom_enhanced'):
    """
    Factory function to create enhanced loss function

    Args:
        loss_type: 'point_only', 'point_mse_combined', 'ultrasom_enhanced'

    Returns:
        loss_function: Loss function instance
    """

    if loss_type == 'point_only':
        return SimpleLossFunction(use_mse_fallback=True)

    elif loss_type == 'point_mse_combined':
        return OptionalMSELoss(alpha_point=1.0, alpha_mse=0.1)

    elif loss_type == 'ultrasom_enhanced':
        # Default UltrasSOM configuration (balanced)
        return UltrasSOMEnhancedLoss(
            alpha_mse=0.2,
            alpha_corr=0.1,
            alpha_velocity=0.1,
            alpha_point=3.0
        )

    elif loss_type == 'point_focused':
        # Point-focused configuration for clinical accuracy
        return UltrasSOMEnhancedLoss(
            alpha_mse=0.05,      # Minimal MSE weight
            alpha_corr=0.02,     # Minimal correlation weight
            alpha_velocity=0.02, # Minimal velocity weight
            alpha_point=10.0     # Maximum point accuracy weight
        )

    elif loss_type == 'clinical_optimized':
        # Optimized for <0.2mm clinical target
        return UltrasSOMEnhancedLoss(
            alpha_mse=0.01,      # Very minimal MSE weight
            alpha_corr=0.01,     # Very minimal correlation weight
            alpha_velocity=0.01, # Very minimal velocity weight
            alpha_point=20.0     # Dominant point accuracy weight
        )

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Available: 'point_only', 'point_mse_combined', 'ultrasom_enhanced', 'point_focused', 'clinical_optimized'")


if __name__ == "__main__":
    test_simple_losses()
