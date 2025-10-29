"""
Loss Functions for Contrastive Frame Grouping Comparison Study

This module implements loss functions for both baseline and enhanced models:
- Baseline: MSE + Correlation losses
- Enhanced: MSE + Correlation + Contrastive (Triplet) losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineLoss(nn.Module):
    """
    Loss function for Baseline Model (Model A)

    Combined Loss: L = α₁ * L_mse + α₂ * L_corr + α₃ * L_speed

    Where:
    - L_mse: Mean Squared Error on 6DoF parameters
    - L_corr: Correlation loss for feature consistency
    - L_speed: Motion speed loss for temporal consistency
    """

    def __init__(self, alpha_mse=1.0, alpha_corr=0.5, alpha_speed=0.3):
        super().__init__()
        self.alpha_mse = alpha_mse
        self.alpha_corr = alpha_corr
        self.alpha_speed = alpha_speed
        self.mse_criterion = nn.MSELoss()
        
    def mse_loss(self, predictions, targets):
        """Mean Squared Error loss on 6DoF parameters"""
        return self.mse_criterion(predictions, targets)
    
    def correlation_loss(self, predictions, targets):
        """
        Correlation loss for 6DoF parameters

        Args:
            predictions: (B, 6) predicted pose parameters
            targets: (B, 6) ground truth pose parameters

        Returns:
            loss: Correlation loss (1 - correlation)
        """
        try:
            # Flatten tensors using reshape instead of view
            pred_flat = predictions.reshape(-1)
            target_flat = targets.reshape(-1)

            # Ensure minimum size
            min_size = min(pred_flat.shape[0], target_flat.shape[0])
            pred_flat = pred_flat[:min_size]
            target_flat = target_flat[:min_size]

            # Calculate correlation
            pred_mean = torch.mean(pred_flat)
            target_mean = torch.mean(target_flat)

            # Covariance
            cov = torch.mean((pred_flat - pred_mean) * (target_flat - target_mean))

            # Standard deviations
            pred_std = torch.sqrt(torch.mean((pred_flat - pred_mean) ** 2) + 1e-8)
            target_std = torch.sqrt(torch.mean((target_flat - target_mean) ** 2) + 1e-8)

            # Correlation coefficient
            correlation = cov / (pred_std * target_std + 1e-8)

            # Clamp correlation to valid range [-1, 1]
            correlation = torch.clamp(correlation, -1.0, 1.0)

            # Return 1 - correlation (loss decreases as correlation increases)
            loss = 1.0 - correlation

            return loss

        except Exception as e:
            print(f"Warning: Correlation loss calculation failed: {e}")
            return torch.tensor(0.1, device=predictions.device)

    def motion_speed_loss(self, predictions, targets):
        """
        Motion speed loss for temporal consistency
        Formula: 1/6(n-2) * Σ(v_i - v_pred_i)²

        Args:
            predictions: (B, 6) predicted pose parameters for current frame
            targets: (B, 6) ground truth pose parameters for current frame

        Note: For single frame predictions, we simulate temporal sequences by
              treating batch dimension as temporal dimension for velocity calculation

        Returns:
            loss: Motion speed loss
        """
        try:
            # For single frame predictions (B, 6), we need to simulate temporal behavior
            # We'll treat the batch as a sequence to calculate velocities
            if predictions.dim() == 2 and predictions.shape[0] > 1:
                B, DOF = predictions.shape  # B=batch_size, DOF=6

                if B < 2:
                    # Need at least 2 samples to calculate velocity
                    return torch.tensor(0.0, device=predictions.device)

                # Calculate velocities (differences between consecutive samples in batch)
                # v_pred_i = pred[i] - pred[i-1] for i = 1, 2, ..., B-1
                pred_velocities = predictions[1:] - predictions[:-1]  # (B-1, 6)
                target_velocities = targets[1:] - targets[:-1]  # (B-1, 6)

                # Calculate squared differences: (v_i - v_pred_i)²
                velocity_diff_squared = (target_velocities - pred_velocities) ** 2  # (B-1, 6)

                # Sum across all velocity differences: Σ(v_i - v_pred_i)²
                total_velocity_diff = velocity_diff_squared.sum()  # Sum over all elements

                # Apply formula: 1/6(n-2) * Σ(v_i - v_pred_i)²
                # where n = B (number of frames), so n-2 = B-2
                n = B
                if n <= 2:
                    # For n <= 2, we have n-2 <= 0, so return small loss
                    return torch.tensor(1e-4, device=predictions.device)

                loss = (1.0 / (6.0 * (n - 2))) * total_velocity_diff

                # Add small epsilon for numerical stability
                loss = loss + 1e-6

                # Clamp to reasonable range
                loss = torch.clamp(loss, min=1e-4, max=10.0)

                return loss

            # For multi-frame sequences: (B, T, 6)
            elif predictions.dim() == 3:
                B, T, DOF = predictions.shape

                if T < 3:
                    # Need at least 3 frames for meaningful velocity calculation
                    return torch.tensor(1e-4, device=predictions.device)

                total_loss = 0
                for b in range(B):
                    # Calculate velocities for this sequence
                    pred_velocities = predictions[b, 1:] - predictions[b, :-1]  # (T-1, 6)
                    target_velocities = targets[b, 1:] - targets[b, :-1]  # (T-1, 6)

                    # Calculate squared differences
                    velocity_diff_squared = (target_velocities - pred_velocities) ** 2  # (T-1, 6)

                    # Sum across all velocity differences
                    total_velocity_diff = velocity_diff_squared.sum()

                    # Apply formula: 1/6(n-2) * Σ(v_i - v_pred_i)²
                    n = T
                    sequence_loss = (1.0 / (6.0 * (n - 2))) * total_velocity_diff
                    total_loss += sequence_loss

                # Average across batch
                loss = total_loss / B

                # Add small epsilon for numerical stability
                loss = loss + 1e-6

                # Clamp to reasonable range
                loss = torch.clamp(loss, min=1e-4, max=10.0)

                return loss
            else:
                # Fallback for unexpected shapes
                return torch.tensor(1e-4, device=predictions.device)

        except Exception as e:
            print(f"Warning: Motion speed loss calculation failed: {e}")
            return torch.tensor(1e-4, device=predictions.device)
    
    def forward(self, predictions, targets):
        """
        Compute combined baseline loss

        Args:
            predictions: (B, 6) or (B, T, 6) predicted pose parameters
            targets: (B, 6) or (B, T, 6) ground truth pose parameters

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Individual losses
        mse_loss = self.mse_loss(predictions, targets)
        corr_loss = self.correlation_loss(predictions, targets)
        speed_loss = self.motion_speed_loss(predictions, targets)

        # Combined loss
        total_loss = (self.alpha_mse * mse_loss +
                     self.alpha_corr * corr_loss +
                     self.alpha_speed * speed_loss)

        # Loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'correlation_loss': corr_loss.item(),
            'speed_loss': speed_loss.item(),
            'alpha_mse': self.alpha_mse,
            'alpha_corr': self.alpha_corr,
            'alpha_speed': self.alpha_speed
        }

        return total_loss, loss_dict


class EnhancedLoss(nn.Module):
    """
    Loss function for Enhanced Model (Model B)

    Combined Loss: L = α₁ * L_mse + α₂ * L_corr + α₃ * L_speed + α₄ * L_contrastive

    Where:
    - L_mse: Mean Squared Error on 6DoF parameters
    - L_corr: Correlation loss for feature consistency
    - L_speed: Motion speed loss for temporal consistency
    - L_contrastive: Triplet loss from contrastive frame grouping
    """

    def __init__(self, alpha_mse=1.0, alpha_corr=0.5, alpha_speed=0.3, alpha_contrastive=0.3):
        super().__init__()
        self.alpha_mse = alpha_mse
        self.alpha_corr = alpha_corr
        self.alpha_speed = alpha_speed
        self.alpha_contrastive = alpha_contrastive
        self.mse_criterion = nn.MSELoss()
        
    def mse_loss(self, predictions, targets):
        """Mean Squared Error loss on 6DoF parameters"""
        return self.mse_criterion(predictions, targets)
    
    def correlation_loss(self, predictions, targets):
        """
        Correlation loss for 6DoF parameters (same as baseline)
        """
        try:
            # Flatten tensors using reshape instead of view
            pred_flat = predictions.reshape(-1)
            target_flat = targets.reshape(-1)
            
            # Ensure minimum size
            min_size = min(pred_flat.shape[0], target_flat.shape[0])
            pred_flat = pred_flat[:min_size]
            target_flat = target_flat[:min_size]
            
            # Calculate correlation
            pred_mean = torch.mean(pred_flat)
            target_mean = torch.mean(target_flat)
            
            # Covariance
            cov = torch.mean((pred_flat - pred_mean) * (target_flat - target_mean))
            
            # Standard deviations
            pred_std = torch.sqrt(torch.mean((pred_flat - pred_mean) ** 2) + 1e-8)
            target_std = torch.sqrt(torch.mean((target_flat - target_mean) ** 2) + 1e-8)
            
            # Correlation coefficient
            correlation = cov / (pred_std * target_std + 1e-8)
            
            # Clamp correlation to valid range [-1, 1]
            correlation = torch.clamp(correlation, -1.0, 1.0)
            
            # Return 1 - correlation
            loss = 1.0 - correlation
            
            return loss
            
        except Exception as e:
            print(f"Warning: Correlation loss calculation failed: {e}")
            return torch.tensor(0.1, device=predictions.device)

    def motion_speed_loss(self, predictions, targets):
        """
        Motion speed loss for temporal consistency (same as baseline)
        Formula: 1/6(n-2) * Σ(v_i - v_pred_i)²

        Args:
            predictions: (B, 6) predicted pose parameters for current frame
            targets: (B, 6) ground truth pose parameters for current frame

        Returns:
            loss: Motion speed loss
        """
        try:
            # For single frame predictions (B, 6), we need to simulate temporal behavior
            # We'll treat the batch as a sequence to calculate velocities
            if predictions.dim() == 2 and predictions.shape[0] > 1:
                B, _ = predictions.shape  # B=batch_size, DOF=6

                if B < 2:
                    # Need at least 2 samples to calculate velocity
                    return torch.tensor(0.0, device=predictions.device)

                # Calculate velocities (differences between consecutive samples in batch)
                pred_velocities = predictions[1:] - predictions[:-1]  # (B-1, 6)
                target_velocities = targets[1:] - targets[:-1]  # (B-1, 6)

                # Calculate squared differences: (v_i - v_pred_i)²
                velocity_diff_squared = (target_velocities - pred_velocities) ** 2  # (B-1, 6)

                # Sum across all velocity differences: Σ(v_i - v_pred_i)²
                total_velocity_diff = velocity_diff_squared.sum()  # Sum over all elements

                # Apply formula: 1/6(n-2) * Σ(v_i - v_pred_i)²
                n = B
                if n <= 2:
                    return torch.tensor(1e-4, device=predictions.device)

                loss = (1.0 / (6.0 * (n - 2))) * total_velocity_diff

                # Add small epsilon for numerical stability
                loss = loss + 1e-6

                # Clamp to reasonable range
                loss = torch.clamp(loss, min=1e-4, max=10.0)

                return loss

            # For multi-frame sequences: (B, T, 6)
            elif predictions.dim() == 3:
                B, T, _ = predictions.shape

                if T < 3:
                    return torch.tensor(1e-4, device=predictions.device)

                total_loss = 0
                for b in range(B):
                    # Calculate velocities for this sequence
                    pred_velocities = predictions[b, 1:] - predictions[b, :-1]  # (T-1, 6)
                    target_velocities = targets[b, 1:] - targets[b, :-1]  # (T-1, 6)

                    # Calculate squared differences
                    velocity_diff_squared = (target_velocities - pred_velocities) ** 2  # (T-1, 6)

                    # Sum across all velocity differences
                    total_velocity_diff = velocity_diff_squared.sum()

                    # Apply formula: 1/6(n-2) * Σ(v_i - v_pred_i)²
                    n = T
                    sequence_loss = (1.0 / (6.0 * (n - 2))) * total_velocity_diff
                    total_loss += sequence_loss

                # Average across batch
                loss = total_loss / B

                # Add small epsilon for numerical stability
                loss = loss + 1e-6

                # Clamp to reasonable range
                loss = torch.clamp(loss, min=1e-4, max=10.0)

                return loss
            else:
                # Fallback for unexpected shapes
                return torch.tensor(1e-4, device=predictions.device)

        except Exception as e:
            print(f"Warning: Motion speed loss calculation failed: {e}")
            return torch.tensor(1e-4, device=predictions.device)

    def forward(self, predictions, targets, contrastive_loss):
        """
        Compute combined enhanced loss

        Args:
            predictions: (B, 6) or (B, T, 6) predicted pose parameters
            targets: (B, 6) or (B, T, 6) ground truth pose parameters
            contrastive_loss: Triplet loss from contrastive frame grouping

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Individual losses
        mse_loss = self.mse_loss(predictions, targets)
        corr_loss = self.correlation_loss(predictions, targets)
        speed_loss = self.motion_speed_loss(predictions, targets)

        # Combined loss
        total_loss = (self.alpha_mse * mse_loss +
                     self.alpha_corr * corr_loss +
                     self.alpha_speed * speed_loss +
                     self.alpha_contrastive * contrastive_loss)

        # Loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'correlation_loss': corr_loss.item(),
            'speed_loss': speed_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'alpha_mse': self.alpha_mse,
            'alpha_corr': self.alpha_corr,
            'alpha_speed': self.alpha_speed,
            'alpha_contrastive': self.alpha_contrastive
        }

        return total_loss, loss_dict


def build_loss_function(model_type, **kwargs):
    """
    Factory function to build loss functions
    
    Args:
        model_type: 'baseline' or 'enhanced'
        **kwargs: Loss function parameters
        
    Returns:
        loss_fn: Initialized loss function
    """
    if model_type == 'baseline':
        return BaselineLoss(**kwargs)
    elif model_type == 'enhanced':
        return EnhancedLoss(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def test_loss_functions():
    """Test both loss functions"""
    print("Testing Loss Functions")
    print("=" * 30)

    # Test data
    B, pred_dim = 4, 6
    predictions = torch.randn(B, pred_dim)
    targets = torch.randn(B, pred_dim)
    contrastive_loss = torch.tensor(0.15)  # Example contrastive loss

    # Test baseline loss
    print("Testing Baseline Loss (MSE + Correlation + Speed)...")
    baseline_loss_fn = build_loss_function('baseline', alpha_mse=1.0, alpha_corr=0.5, alpha_speed=0.3)

    total_loss, loss_dict = baseline_loss_fn(predictions, targets)
    print(f"[OK] Baseline Loss: {total_loss.item():.4f}")
    print(f"   MSE: {loss_dict['mse_loss']:.4f}, Correlation: {loss_dict['correlation_loss']:.4f}, Speed: {loss_dict['speed_loss']:.4f}")

    # Test enhanced loss
    print("\nTesting Enhanced Loss (MSE + Correlation + Speed + Contrastive)...")
    enhanced_loss_fn = build_loss_function('enhanced', alpha_mse=1.0, alpha_corr=0.5, alpha_speed=0.3, alpha_contrastive=0.3)

    total_loss, loss_dict = enhanced_loss_fn(predictions, targets, contrastive_loss)
    print(f"[OK] Enhanced Loss: {total_loss.item():.4f}")
    print(f"   MSE: {loss_dict['mse_loss']:.4f}, Correlation: {loss_dict['correlation_loss']:.4f}")
    print(f"   Speed: {loss_dict['speed_loss']:.4f}, Contrastive: {loss_dict['contrastive_loss']:.4f}")

    print("\n[SUCCESS] Loss functions test completed!")


if __name__ == "__main__":
    test_loss_functions()
