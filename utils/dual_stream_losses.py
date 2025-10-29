# Dual-Stream Loss Functions
# Based on enhanced_losses.py structure but modified for dual-stream architecture
# Handles sequential 6-DOF output per frame: (B, T, 6)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import PointDistance

# Import the proven motion speed loss implementation
try:
    from .loss_functions import motion_speed_loss as reference_motion_speed_loss
except ImportError:
    reference_motion_speed_loss = None


class DualStreamLossFunction(nn.Module):
    """
    Dual-stream loss function adapted from enhanced_losses.py
    Modified to handle sequential output: (B, T, 6) instead of (B, 6)
    
    Loss components:
    1. PointDistance loss (baseline - proven effective)
    2. Correlation loss (prevents overfitting to scanning speed)
    3. Contrastive loss (motion-coherent frame grouping)
    4. Motion speed loss (temporal consistency - from utils/loss_functions.py)
    5. MSE loss (standard reconstruction loss)
    """

    def __init__(self,
                 alpha_point=1.0,       # PointDistance loss weight (baseline)
                 alpha_corr=0.5,        # Correlation loss weight
                 alpha_contrastive=0.3, # Contrastive loss weight
                 alpha_mse=0.2,         # MSE loss weight (lower since PointDistance is primary)
                 alpha_speed=0.3,       # Motion speed loss weight (temporal consistency)
                 margin=0.2):           # Margin for ranking loss
        """
        Initialize dual-stream loss function with focused losses
        
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
        self.alpha_speed = alpha_speed
        self.margin = margin

        # Loss functions
        self.point_distance = PointDistance(paired=True)  # From baseline
        self.mse_criterion = nn.MSELoss()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def correlation_loss_6dof(self, predictions, targets):
        """
        Correlation loss for 6 DOF parameters (adapted from enhanced_losses.py)
        Modified to handle sequential output: (B, T, 6)
        
        Args:
            predictions: (B, 6) or (B, T, 6) predicted parameters
            targets: (B, 6) or (B, T, 6) ground truth parameters
            
        Returns:
            loss: Correlation loss (1 - correlation_coefficient)
        """
        try:
            # Handle sequential predictions: (B, T, 6) -> (B*T, 6) for correlation calculation
            if predictions.dim() == 3:
                B, T, DOF = predictions.shape
                if DOF == 6:
                    predictions = predictions.reshape(-1, 6)  # (B*T, 6)
                else:
                    print(f" Correlation loss: Incompatible prediction shape {predictions.shape}, expected (B, T, 6)")
                    return torch.tensor(0.1, device=predictions.device)

            # Handle targets - they might be points format or sequential 6-DOF
            if targets.dim() == 4:  # (B, 6, 3, 4) - points format
                # Convert points to 6-DOF parameters using proper transformation matrix decomposition
                B, num_points, _, _ = targets.shape
                targets_6dof = torch.zeros(B, 6, device=targets.device)

                # Extract translation (last column, first 3 elements) from first transformation matrix
                targets_6dof[:, :3] = targets[:, 0, :3, 3]  # Translation: tx, ty, tz

                # Extract rotation using proper rotation matrix to Euler angles conversion
                # Use the rotation matrix from the first transformation
                rot_matrix = targets[:, 0, :3, :3]  # (B, 3, 3)

                # Convert rotation matrix to Euler angles (ZYX convention)
                # This is a simplified version - for better results, use proper rotation matrix decomposition
                # Extract Euler angles: roll (x), pitch (y), yaw (z)
                targets_6dof[:, 3] = torch.atan2(rot_matrix[:, 2, 1], rot_matrix[:, 2, 2])  # Roll (rotation around x-axis)
                targets_6dof[:, 4] = torch.asin(-torch.clamp(rot_matrix[:, 2, 0], -1.0, 1.0))  # Pitch (rotation around y-axis)
                targets_6dof[:, 5] = torch.atan2(rot_matrix[:, 1, 0], rot_matrix[:, 0, 0])  # Yaw (rotation around z-axis)

                targets = targets_6dof  # (B, 6)

                # Debug: Print some values to see if they're reasonable
                if not hasattr(self, '_debug_corr_values'):
                    print(f" Correlation loss debug:")
                    print(f"   Targets 6-DOF sample: {targets[0].detach().cpu().numpy()}")
                    print(f"   Translation range: {targets[:, :3].min().item():.4f} to {targets[:, :3].max().item():.4f}")
                    print(f"   Rotation range: {targets[:, 3:].min().item():.4f} to {targets[:, 3:].max().item():.4f}")
                    self._debug_corr_values = True
            elif targets.dim() == 3:
                if targets.shape[-1] == 6:  # (B, T, 6) - sequential 6-DOF
                    targets = targets.reshape(-1, 6)  # (B*T, 6)
                else:
                    print(f" Correlation loss: Incompatible target shape {targets.shape}, expected (B, T, 6)")
                    return torch.tensor(0.1, device=predictions.device)
            elif targets.dim() == 2 and targets.shape[-1] == 6:  # (B, 6) - single 6-DOF
                # Repeat targets to match predictions
                B_pred = predictions.shape[0] // targets.shape[0] if predictions.shape[0] % targets.shape[0] == 0 else 1
                targets = targets.repeat(B_pred, 1)  # Repeat to match (B*T, 6)
            else:
                print(f" Correlation loss: Incompatible target shape {targets.shape}, expected 6-DOF format")
                return torch.tensor(0.1, device=predictions.device)

            # Ensure same batch size
            if predictions.shape[0] != targets.shape[0]:
                min_size = min(predictions.shape[0], targets.shape[0])
                predictions = predictions[:min_size]
                targets = targets[:min_size]

            # Debug: Print prediction values too
            if not hasattr(self, '_debug_pred_values'):
                print(f" Correlation loss predictions debug:")
                print(f"   Predictions 6-DOF sample: {predictions[0].detach().cpu().numpy()}")
                print(f"   Pred translation range: {predictions[:, :3].min().item():.4f} to {predictions[:, :3].max().item():.4f}")
                print(f"   Pred rotation range: {predictions[:, 3:].min().item():.4f} to {predictions[:, 3:].max().item():.4f}")
                self._debug_pred_values = True

            # Calculate correlation for each DOF dimension
            total_corr_loss = 0
            num_dofs = min(predictions.shape[-1], targets.shape[-1])  # Should be 6

            # Debug: Print some correlation computation details
            debug_correlations = []

            for dof in range(num_dofs):
                pred_dof = predictions[:, dof]  # (B*T,) or (B,)
                target_dof = targets[:, dof]    # (B*T,) or (B,)

                # Calculate correlation
                pred_mean = torch.mean(pred_dof)
                target_mean = torch.mean(target_dof)

                # Covariance
                cov = torch.mean((pred_dof - pred_mean) * (target_dof - target_mean))

                # Standard deviations
                pred_std = torch.sqrt(torch.mean((pred_dof - pred_mean) ** 2) + 1e-8)
                target_std = torch.sqrt(torch.mean((target_dof - target_mean) ** 2) + 1e-8)

                # Correlation coefficient
                correlation = cov / (pred_std * target_std + 1e-8)

                # Clamp correlation to valid range [-1, 1]
                correlation = torch.clamp(correlation, -1.0, 1.0)

                # Store for debugging
                debug_correlations.append(correlation.item())

                # Accumulate loss (1 - correlation)
                total_corr_loss += (1.0 - correlation)

            # Average across DOF dimensions
            loss = total_corr_loss / num_dofs

            # Debug: Print correlation details occasionally
            if hasattr(self, '_debug_corr_count'):
                self._debug_corr_count += 1
            else:
                self._debug_corr_count = 1

            if self._debug_corr_count <= 3:  # Print first 3 times
                print(f" Correlation loss debug #{self._debug_corr_count}:")
                print(f"   Individual correlations: {[f'{c:.4f}' for c in debug_correlations]}")
                print(f"   Final correlation loss: {loss.item():.4f}")
                print(f"   Pred shape: {predictions.shape}, Target shape: {targets.shape}")

            return loss

        except Exception as e:
            print(f"Warning: Dual-stream correlation loss calculation failed: {e}")
            print(f"  Predictions shape: {predictions.shape}")
            print(f"  Targets shape: {targets.shape}")
            return torch.tensor(0.1, device=predictions.device)

    def margin_ranking_contrastive_loss(self, triplet_loss, anchor_embeddings=None, 
                                      positive_embeddings=None, negative_embeddings=None):
        """
        Margin ranking contrastive loss (adapted from enhanced_losses.py)
        
        Args:
            triplet_loss: Pre-computed triplet loss from contrastive grouping
            anchor_embeddings: Anchor embeddings (optional)
            positive_embeddings: Positive embeddings (optional)
            negative_embeddings: Negative embeddings (optional)
            
        Returns:
            contrastive_loss: Contrastive loss value
        """
        # If triplet loss is already computed, use it directly (including small values)
        if triplet_loss is not None and isinstance(triplet_loss, torch.Tensor):
            # Accept any triplet loss value, including small ones from training
            return triplet_loss

        # Fallback: compute margin ranking loss if embeddings available
        if (anchor_embeddings is not None and
            positive_embeddings is not None and
            negative_embeddings is not None):

            # Compute distances
            pos_dist = F.pairwise_distance(anchor_embeddings, positive_embeddings)
            neg_dist = F.pairwise_distance(anchor_embeddings, negative_embeddings)

            # Margin ranking loss: pos_dist should be smaller than neg_dist
            target = torch.ones_like(pos_dist)  # pos_dist < neg_dist
            ranking_loss = self.ranking_loss(neg_dist, pos_dist, target)

            return ranking_loss

        # No contrastive information available (validation mode) - return zero loss
        if triplet_loss is not None and isinstance(triplet_loss, torch.Tensor):
            # Return zero loss with same device as triplet_loss
            return torch.tensor(0.0, device=triplet_loss.device)
        else:
            # Fallback device (should not happen)
            return torch.tensor(0.0)

    def dof_MSE(self, labels, outputs, criterion=None, dof_based=False):
        """
        MSE loss function following the correct structure from train_dual_stream.py

        Args:
            labels: Ground truth labels
            outputs: Model predictions
            criterion: MSE criterion (uses self.mse_criterion if None)
            dof_based: Whether to compute DOF-based loss with debugging

        Returns:
            loss: MSE loss value
        """
        import time

        if criterion is None:
            criterion = self.mse_criterion

        if dof_based:
            dof_losses = []
            for dof_id in range(labels.shape[1]):
                # print(labels[:, dof_id].shape)
                x = outputs[:, dof_id]
                y = labels[:, dof_id]

                dof_loss = criterion(x, y)
                dof_losses.append(dof_loss)
            print(dof_losses)
            loss = sum(dof_losses) / 6
            print(loss)
            print(criterion(labels, outputs))
            time.sleep(30)
        else:
            loss = criterion(labels, outputs)

        return loss

    def motion_speed_loss(self, predictions, targets):
        """
        Motion speed loss for temporal consistency (following utils/loss_functions.py implementation)
        Formula: L_speed = (1/6) * (1/(n-2)) * sum[(vi - v̂i)^2] from i=1 to n-2

        Args:
            predictions: (B, T, 6) predicted pose parameters (sequential)
            targets: (B, T, 6) ground truth pose parameters (sequential)

        Returns:
            loss: Motion speed loss
        """
        # This loss is specifically designed for sequential output (B, T, 6)
        if predictions.dim() == 3 and targets.dim() == 3:
            B, T, _ = predictions.shape

            # Return small constant if we don't have enough frames (need at least 3 for n-2 velocities)
            if T < 3:
                return torch.tensor(1e-4, device=predictions.device)

            # Calculate velocities (differences between consecutive frames)
            pred_velocities = predictions[:, 1:] - predictions[:, :-1]  # (B, T-1, 6)
            target_velocities = targets[:, 1:] - targets[:, :-1]  # (B, T-1, 6)

            # Calculate squared differences between predicted and ground truth velocities
            velocity_diff_squared = (pred_velocities - target_velocities) ** 2

            # Sum across DOF dimensions (6)
            velocity_diff_squared_sum = velocity_diff_squared.sum(dim=-1)  # (B, T-1)

            # Calculate mean across temporal dimension (T-2) and batch
            n = T  # number of original frames
            loss = (1.0/6.0) * (1.0/(n-2)) * velocity_diff_squared_sum.mean()

            # Add small epsilon for numerical stability
            eps = 1e-6
            loss = loss + eps

            # Clamp to reasonable range (reduced min clamp to allow smaller natural values)
            loss = torch.clamp(loss, min=1e-6, max=100.0)

            return loss

        # Fallback for non-sequential data
        elif predictions.dim() == 2 and targets.dim() == 2:
            # Treat batch dimension as temporal for velocity calculation
            B, _ = predictions.shape

            if B < 2:
                return torch.tensor(1e-4, device=predictions.device)

            # Calculate velocities (differences between consecutive samples in batch)
            pred_velocities = predictions[1:] - predictions[:-1]  # (B-1, 6)
            target_velocities = targets[1:] - targets[:-1]  # (B-1, 6)

            # Calculate squared differences
            velocity_diff_squared = (target_velocities - pred_velocities) ** 2  # (B-1, 6)

            # Sum across all velocity differences
            total_velocity_diff = velocity_diff_squared.sum()

            # Apply formula: 1/6(B-1) * Σ(v_i - v_pred_i)²
            loss = (1.0 / (6.0 * (B - 1))) * total_velocity_diff

            # Add small epsilon for numerical stability
            loss = loss + 1e-6

            # Clamp to reasonable range (reduced min clamp to allow smaller natural values)
            loss = torch.clamp(loss, min=1e-6, max=10.0)

            return loss

        else:
            # Fallback for unexpected shapes
            raise ValueError(f"Motion speed loss: Incompatible shapes - pred: {predictions.shape}, target: {targets.shape}")

    def forward(self, predictions, targets,
                pred_points=None, target_points=None,
                triplet_loss=None,
                anchor_embeddings=None, positive_embeddings=None, negative_embeddings=None,
                sequential_predictions=None, sequential_targets=None):
        """
        Compute combined dual-stream loss with focused losses

        Args:
            predictions: (B, T, 6) predicted pose parameters (sequential output)
            targets: (B, 6) or (B, T, 6) or other shapes from transform_label
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
        # Handle sequential predictions: (B, T, 6) -> (B, 6) for compatibility
        if predictions.dim() == 3:
            # Average across temporal dimension for loss calculation
            predictions_avg = predictions.mean(dim=1)  # (B, 6)
        else:
            predictions_avg = predictions  # (B, 6)

        # Handle targets - they might come from transform_label with different shapes
        if targets.dim() == 4:  # (B, 1, 3, 4) - transformation matrices
            # Extract translation and rotation from transformation matrices
            # This is a fallback - we'll use PointDistance loss instead
            targets_avg = torch.zeros_like(predictions_avg)  # Dummy targets
        elif targets.dim() == 3:
            if targets.shape[-1] == 6:  # (B, T, 6)
                targets_avg = targets.mean(dim=1)  # (B, 6)
            else:  # (B, 1, N) or other 3D shapes
                targets_avg = torch.zeros_like(predictions_avg)  # Dummy targets
        elif targets.dim() == 2:
            if targets.shape[-1] == 6:  # (B, 6)
                targets_avg = targets
            else:  # (B, N) where N != 6
                targets_avg = torch.zeros_like(predictions_avg)  # Dummy targets
        else:
            targets_avg = torch.zeros_like(predictions_avg)  # Dummy targets

        # 1. PointDistance loss (baseline - most important)
        if pred_points is not None and target_points is not None:
            point_loss = self.point_distance(pred_points, target_points)
        else:
            # Fallback to MSE if points not available - using correct dof_MSE function
            point_loss = self.dof_MSE(targets_avg, predictions_avg, criterion=self.mse_criterion, dof_based=False)

        # 2. Correlation loss (prevents overfitting to scanning speed)
        corr_loss = self.correlation_loss_6dof(predictions_avg, targets_avg)

        # 3. Contrastive loss (motion-coherent frame grouping)
        contrastive_loss = self.margin_ranking_contrastive_loss(
            triplet_loss, anchor_embeddings, positive_embeddings, negative_embeddings
        )

        # Safety check: ensure contrastive_loss is a tensor
        if not isinstance(contrastive_loss, torch.Tensor):
            contrastive_loss = torch.tensor(0.0, device=predictions.device)

        # 4. Motion speed loss (temporal consistency - use sequential data if available)
        # Use sequential predictions/targets if provided, otherwise fall back to regular predictions
        if sequential_predictions is not None and sequential_targets is not None:
            if reference_motion_speed_loss is not None:
                speed_loss = reference_motion_speed_loss(sequential_predictions, sequential_targets)
            else:
                speed_loss = self.motion_speed_loss(sequential_predictions, sequential_targets)
        elif predictions.dim() == 3 and targets.dim() == 3:
            # Fallback: use regular predictions if they're sequential
            speed_loss = self.motion_speed_loss(predictions, targets)
        else:
            # Not sequential data - this should not happen in dual-stream training
            raise ValueError(f"Motion speed loss: Expected sequential data but got pred: {predictions.shape}, target: {targets.shape}")

        # 5. MSE loss (standard reconstruction loss) - using correct dof_MSE implementation
        # Convert targets to 6-DOF format if needed
        if targets_avg.dim() == 4:  # Points format (B, 6, 3, 4)
            # Convert points to 6-DOF parameters using proper transformation matrix decomposition
            B, _, _, _ = targets_avg.shape
            targets_6dof = torch.zeros(B, 6, device=targets_avg.device)

            # Extract translation (last column, first 3 elements) from first transformation matrix
            targets_6dof[:, :3] = targets_avg[:, 0, :3, 3]  # Translation: tx, ty, tz

            # Extract rotation using proper rotation matrix to Euler angles conversion
            rot_matrix = targets_avg[:, 0, :3, :3]  # (B, 3, 3)

            # Convert rotation matrix to Euler angles (ZYX convention)
            targets_6dof[:, 3] = torch.atan2(rot_matrix[:, 2, 1], rot_matrix[:, 2, 2])  # Roll
            targets_6dof[:, 4] = torch.asin(-torch.clamp(rot_matrix[:, 2, 0], -1.0, 1.0))  # Pitch
            targets_6dof[:, 5] = torch.atan2(rot_matrix[:, 1, 0], rot_matrix[:, 0, 0])  # Yaw

            targets_avg = targets_6dof  # (B, 6)

        # Now compute MSE loss with proper 6-DOF targets using the correct dof_MSE function
        if predictions_avg.shape != targets_avg.shape:
            raise ValueError(f"MSE loss: Shape mismatch - pred: {predictions_avg.shape}, target: {targets_avg.shape}")

        # Use the correct dof_MSE function following the structure from train_dual_stream.py
        mse_loss = self.dof_MSE(targets_avg, predictions_avg, criterion=self.mse_criterion, dof_based=False)

        # Combined loss with focused weights (ensure scalar output)
        total_loss = (self.alpha_point * point_loss +
                     self.alpha_corr * corr_loss +
                     self.alpha_contrastive * contrastive_loss +
                     self.alpha_speed * speed_loss +
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
            'speed_loss': speed_loss.mean().item() if isinstance(speed_loss, torch.Tensor) and speed_loss.numel() > 1 else (speed_loss.item() if isinstance(speed_loss, torch.Tensor) else 0.0),
            'mse_loss': mse_loss.mean().item() if isinstance(mse_loss, torch.Tensor) and mse_loss.numel() > 1 else (mse_loss.item() if isinstance(mse_loss, torch.Tensor) else 0.0)
        }

        return total_loss, loss_dict


def create_dual_stream_loss(alpha_point=1.0, alpha_corr=0.5, alpha_contrastive=0.3, alpha_speed=0.3, alpha_mse=0.2, margin=0.2):
    """
    Factory function to create dual-stream loss function

    Args:
        alpha_point: Weight for PointDistance loss (baseline)
        alpha_corr: Weight for correlation loss
        alpha_contrastive: Weight for contrastive loss
        alpha_speed: Weight for motion speed loss (temporal consistency)
        alpha_mse: Weight for MSE loss
        margin: Margin for ranking loss

    Returns:
        Dual-stream loss function
    """
    return DualStreamLossFunction(
        alpha_point=alpha_point,
        alpha_corr=alpha_corr,
        alpha_contrastive=alpha_contrastive,
        alpha_speed=alpha_speed,
        alpha_mse=alpha_mse,
        margin=margin
    )


def get_dual_stream_criterion():
    """
    Get dual-stream criterion with default weights optimized for dual-stream architecture

    Returns:
        Dual-stream loss function with default weights
    """
    return create_dual_stream_loss(
        alpha_point=1.0,      # PointDistance (baseline) - highest weight
        alpha_corr=0.5,       # Correlation loss - medium weight
        alpha_contrastive=0.3, # Contrastive loss - medium weight
        alpha_speed=0.3,      # Motion speed loss - medium weight (temporal consistency)
        alpha_mse=0.2,        # MSE loss - lower weight (PointDistance is primary)
        margin=0.2            # Margin for contrastive ranking
    )

def get_optimized_dual_stream_criterion():
    """
    Get OPTIMIZED dual-stream criterion with enhanced weights for better convergence

    Returns:
        Optimized dual-stream loss function with enhanced weights
    """
    return create_dual_stream_loss(
        alpha_point=1.5,      # PointDistance (INCREASED) - even higher emphasis on tracking accuracy
        alpha_corr=0.7,       # Correlation loss (INCREASED) - stronger prevention of speed overfitting
        alpha_contrastive=0.5, # Contrastive loss (INCREASED) - stronger motion-coherent grouping
        alpha_speed=0.8,      # Motion speed loss (INCREASED) - stronger temporal consistency
        alpha_mse=0.1,        # MSE loss (DECREASED) - reduce conflict with PointDistance
        margin=0.3            # Margin (INCREASED) - stronger contrastive separation
    )

def get_aggressive_dual_stream_criterion():
    """
    Get AGGRESSIVE dual-stream criterion for fastest convergence to sub-millimeter accuracy

    Returns:
        Aggressive dual-stream loss function optimized for rapid convergence
    """
    return create_dual_stream_loss(
        alpha_point=2.0,      # PointDistance (AGGRESSIVE) - maximum emphasis on tracking
        alpha_corr=1.0,       # Correlation loss (HIGH) - strong speed-independent learning
        alpha_contrastive=0.8, # Contrastive loss (HIGH) - strong motion coherence
        alpha_speed=1.2,      # Motion speed loss (HIGH) - maximum temporal consistency
        alpha_mse=0.05,       # MSE loss (MINIMAL) - minimal conflict with primary losses
        margin=0.4            # Margin (HIGH) - maximum contrastive separation
    )

def get_fps_nps_balanced_criterion():
    """
    Get BALANCED FPS/NPS criterion - stable training for complex architecture

    Returns:
        Balanced FPS/NPS loss function for stable training
    """
    return create_dual_stream_loss(
        alpha_point=1.5,      # PointDistance (MODERATE) - strong but stable emphasis
        alpha_corr=0.6,       # Correlation loss (MODERATE) - good speed independence
        alpha_contrastive=0.4, # Contrastive loss (MODERATE) - good motion coherence
        alpha_speed=0.8,      # Motion speed loss (MODERATE) - good temporal consistency
        alpha_mse=0.1,        # MSE loss (LOW) - minimal conflict
        margin=0.25           # Margin (MODERATE) - good contrastive separation
    )

def get_fps_nps_optimized_criterion():
    """
    Get OPTIMIZED FPS/NPS criterion - enhanced for better convergence with FPS/NPS architecture

    Returns:
        Optimized FPS/NPS loss function for enhanced convergence
    """
    return create_dual_stream_loss(
        alpha_point=2.5,      # PointDistance (HIGH) - strong emphasis for sub-mm accuracy
        alpha_corr=0.8,       # Correlation loss (HIGH) - strong speed independence
        alpha_contrastive=0.6, # Contrastive loss (HIGH) - strong motion coherence
        alpha_speed=1.5,      # Motion speed loss (HIGH) - strong temporal consistency
        alpha_mse=0.05,       # MSE loss (MINIMAL) - minimal conflict
        margin=0.35           # Margin (HIGH) - strong contrastive separation
    )
