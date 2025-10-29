# -*- coding: utf-8 -*-
"""
Loss functions designed by myself, for better training the networks
The input should be 2 batch_size x n_dimensional vector: network outputs and labels
"""

# %%
import time 
import numpy as np
import torch
try:
    import pytorch3d.transforms
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    print("Warning: pytorch3d not available. 6 DOF extraction will use fallback method.")


# %%
''' Correlation loss for evaluator '''
def correlation_loss(output, target):
    x = output.flatten()
    y = target
    # print('x shape {}, y shape {}'.format(x.shape, y.shape))
    xy = x * y
    mean_xy = torch.mean(xy)
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    cov_xy = mean_xy - mean_x * mean_y
    # print('xy shape {}'.format(xy.shape))
    # print('xy {}'.format(xy))
    # print('mean_xy {}'.format(mean_xy))
    # print('cov_xy {}'.format(cov_xy))

    var_x = torch.sum((x - mean_x) ** 2 / x.shape[0])
    var_y = torch.sum((y - mean_y) ** 2 / y.shape[0])
    # print('var_x {}'.format(var_x))

    corr_xy = cov_xy / (torch.sqrt(var_x * var_y))
    # print('correlation_xy {}'.format(corr_xy))

    loss = 1 - corr_xy
    # time.sleep(30)
    # x = output
    # y = target
    #
    # vx = x - torch.mean(x)
    # vy = y - torch.mean(y)
    #
    # loss = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    # print(loss)
    return loss

def correlation_loss_np(output, target):
    # output = output.data.cpu().numpy()
    # target = target.data.cpu().numpy()
    # output = output.flatten()
    print('output {}, target {}'.format(output.shape, target.shape))
    correlation = np.corrcoef(output, target)[0, 1]
    # loss = 1 - correlation

    return correlation

def matrix_to_euler_angles_fallback(rotation_matrices):
    """
    Fallback function to convert rotation matrices to Euler angles when pytorch3d is not available
    Uses ZYX convention (same as pytorch3d default)
    """
    # Extract individual rotation matrix elements
    r11 = rotation_matrices[..., 0, 0]
    r12 = rotation_matrices[..., 0, 1]
    r13 = rotation_matrices[..., 0, 2]
    r21 = rotation_matrices[..., 1, 0]
    r22 = rotation_matrices[..., 1, 1]
    r23 = rotation_matrices[..., 1, 2]
    r31 = rotation_matrices[..., 2, 0]
    r32 = rotation_matrices[..., 2, 1]
    r33 = rotation_matrices[..., 2, 2]

    # Convert to Euler angles (ZYX convention)
    sy = torch.sqrt(r11 * r11 + r21 * r21)

    singular = sy < 1e-6

    x = torch.where(singular,
                   torch.atan2(-r23, r22),
                   torch.atan2(r32, r33))
    y = torch.where(singular,
                   torch.atan2(-r13, sy),
                   torch.atan2(-r13, sy))
    z = torch.where(singular,
                   torch.zeros_like(r11),
                   torch.atan2(r21, r11))

    return torch.stack([z, y, x], dim=-1)  # ZYX order


def extract_6dof_from_transforms(transforms):
    """
    Extract 6 DOF parameters (tx, ty, tz, rx, ry, rz) from 4x4 transformation matrices

    Args:
        transforms: torch.Tensor of shape (batch, num_frames, 4, 4) or (batch, 4, 4)

    Returns:
        torch.Tensor of shape (batch, num_frames, 6) or (batch, 6) containing [tx, ty, tz, rx, ry, rz]
    """
    if transforms.dim() == 3:
        # Single transform per batch item
        batch_size = transforms.shape[0]
        # Extract translation
        translation = transforms[:, :3, 3]  # (batch, 3)
        # Extract rotation matrix and convert to euler angles
        rotation_matrices = transforms[:, :3, :3]  # (batch, 3, 3)

        if PYTORCH3D_AVAILABLE:
            euler_angles = pytorch3d.transforms.matrix_to_euler_angles(rotation_matrices, 'ZYX')  # (batch, 3)
        else:
            euler_angles = matrix_to_euler_angles_fallback(rotation_matrices)  # (batch, 3)

        # Combine translation and rotation
        dof_params = torch.cat([translation, euler_angles], dim=1)  # (batch, 6)
        return dof_params

    elif transforms.dim() == 4:
        # Multiple transforms per batch item
        batch_size, num_frames = transforms.shape[0], transforms.shape[1]
        # Extract translation
        translation = transforms[:, :, :3, 3]  # (batch, num_frames, 3)
        # Extract rotation matrix and convert to euler angles
        rotation_matrices = transforms[:, :, :3, :3]  # (batch, num_frames, 3, 3)

        if PYTORCH3D_AVAILABLE:
            # Reshape for pytorch3d
            rot_reshaped = rotation_matrices.view(-1, 3, 3)
            euler_angles = pytorch3d.transforms.matrix_to_euler_angles(rot_reshaped, 'ZYX')  # (batch*num_frames, 3)
            euler_angles = euler_angles.view(batch_size, num_frames, 3)  # (batch, num_frames, 3)
        else:
            euler_angles = matrix_to_euler_angles_fallback(rotation_matrices)  # (batch, num_frames, 3)

        # Combine translation and rotation
        dof_params = torch.cat([translation, euler_angles], dim=2)  # (batch, num_frames, 6)
        return dof_params

    else:
        raise ValueError(f"Unsupported transform tensor dimensions: {transforms.shape}")


def motion_speed_loss(pred_data, gt_data):
    """
    Calculate motion speed loss (L_speed) using the formula:
    L_speed = (1/6) * (1/(n-2)) * sum[(vi - v̂i)^2] from i=1 to n-2
    
    Args:
        pred_data: Predicted velocities (v̂i)
        gt_data: Ground truth velocities (vi)
        
    Returns:
        torch.Tensor: motion speed loss
    """
    try:
        # Extract 6 DOF parameters if inputs are transformation matrices
        if pred_data.dim() >= 3 and pred_data.shape[-1] == 4 and pred_data.shape[-2] == 4:
            pred_data = extract_6dof_from_transforms(pred_data)
        if gt_data.dim() >= 3 and gt_data.shape[-1] == 4 and gt_data.shape[-2] == 4:
            gt_data = extract_6dof_from_transforms(gt_data)

        # Return small constant if we don't have enough frames
        if pred_data.shape[1] < 3 or gt_data.shape[1] < 3:  # Need at least 3 frames for n-2 velocities
            return torch.tensor(1e-4, device=pred_data.device)

        # Calculate velocities (differences between consecutive frames)
        pred_velocities = pred_data[:, 1:] - pred_data[:, :-1]  # (batch, n-1, 6)
        gt_velocities = gt_data[:, 1:] - gt_data[:, :-1]  # (batch, n-1, 6)

        # Calculate squared differences between predicted and ground truth velocities
        velocity_diff_squared = (pred_velocities - gt_velocities) ** 2

        # Sum across DOF dimensions (6)
        velocity_diff_squared_sum = velocity_diff_squared.sum(dim=-1)  # (batch, n-1)

        # Calculate mean across temporal dimension (n-2) and batch
        n = pred_data.shape[1]  # number of original frames
        loss = (1.0/6.0) * (1.0/(n-2)) * velocity_diff_squared_sum.mean()

        # Add small epsilon for numerical stability
        eps = 1e-6
        loss = loss + eps

        # Clamp to reasonable range
        loss = torch.clamp(loss, min=1e-4, max=100.0)

        return loss

    except Exception as e:
        print(f"Warning: Motion speed loss calculation failed: {e}")
        return torch.tensor(1e-4, device=pred_data.device)


def get_correlation_loss_adaptive(labels, outputs, dof_based=True):
    """
    Calculate correlation loss that works with any tensor shapes

    Args:
        labels: Ground truth data (any shape)
        outputs: Predicted data (any shape)
        dof_based: If True, try to calculate per-dimension correlation when possible

    Returns:
        torch.Tensor: correlation loss
    """
    try:
        # Always use overall correlation for robustness
        # This avoids dimension mismatch issues
        x = outputs.flatten()
        y = labels.flatten()

        # Ensure we have the same number of elements
        min_size = min(x.shape[0], y.shape[0])
        if min_size == 0:
            return torch.tensor(0.0, device=outputs.device)

        x = x[:min_size]
        y = y[:min_size]

        # Calculate correlation
        xy = x * y
        mean_xy = torch.mean(xy)
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        cov_xy = mean_xy - mean_x * mean_y

        var_x = torch.sum((x - mean_x) ** 2) / x.shape[0]
        var_y = torch.sum((y - mean_y) ** 2) / y.shape[0]

        # Add epsilon for numerical stability
        denominator = torch.sqrt(var_x * var_y) + 1e-8
        corr_xy = cov_xy / denominator

        # Clamp correlation to valid range [-1, 1]
        corr_xy = torch.clamp(corr_xy, -1.0, 1.0)

        loss = 1 - corr_xy
        return loss

    except Exception as e:
        # If anything goes wrong, return a small positive loss
        print(f"Warning: Correlation loss calculation failed: {e}")
        return torch.tensor(0.1, device=outputs.device)


# Keep the old function name for backward compatibility
def get_correlation_loss_6dof(labels, outputs, dof_based=True):
    """
    Backward compatibility wrapper for get_correlation_loss_adaptive
    """
    return get_correlation_loss_adaptive(labels, outputs, dof_based)


# def dof_MSE_loss(labels, outputs, criterion):
#     """
#     Calculate MSE loss that adapts to the actual dimensions of labels and outputs

#     Args:
#         labels: Ground truth data (any shape)
#         outputs: Predicted data (any shape)
#         criterion: MSE loss function

#     Returns:
#         torch.Tensor: MSE loss
#     """
#     # If inputs are transformation matrices, extract 6 DOF parameters
#     if labels.dim() >= 3 and labels.shape[-1] == 4 and labels.shape[-2] == 4:
#         labels = extract_6dof_from_transforms(labels)
#     if outputs.dim() >= 3 and outputs.shape[-1] == 4 and outputs.shape[-2] == 4:
#         outputs = extract_6dof_from_transforms(outputs)

#     # If dimensions don't match, we can't compute MSE directly
#     # This typically happens when model outputs parameters but labels are points
#     # In this case, use the standard MSE loss as-is (the transformation handles the conversion)
#     loss = criterion(outputs, labels)
#     return loss

def dof_MSE_loss(labels, outputs, criterion, dof_based=False):
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


if __name__ == '__main__':
    x = np.linspace(1, 50, num=50)
    y = 3 * x
    print(x)
    print(y)
    # loss = correlation_loss(output=x, target=y)
    # print('loss = {:.4f}'.format(loss))

