"""
Improved DCL-Net with Anti-Spiral Mechanisms
Addresses spiral artifacts in trajectory prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class TemporalSmoothingLayer(nn.Module):
    """Applies temporal smoothing to prevent abrupt changes"""
    def __init__(self, input_dim, smoothing_factor=0.1):
        super(TemporalSmoothingLayer, self).__init__()
        self.smoothing_factor = smoothing_factor
        self.register_buffer('prev_output', torch.zeros(1, input_dim))
        
    def forward(self, x):
        if self.training:
            # During training, apply smoothing constraint
            smoothed = (1 - self.smoothing_factor) * x + self.smoothing_factor * self.prev_output
            self.prev_output = x.detach()
            return smoothed
        else:
            # During inference, apply stronger smoothing
            smoothed = 0.7 * x + 0.3 * self.prev_output
            self.prev_output = x.detach()
            return smoothed

class TrajectoryConstraintLayer(nn.Module):
    """Constrains trajectory parameters to realistic ranges"""
    def __init__(self, output_dim=6):
        super(TrajectoryConstraintLayer, self).__init__()
        self.output_dim = output_dim
        
        # Define realistic parameter ranges (in radians and mm)
        self.register_buffer('max_rotation', torch.tensor([0.2, 0.2, 0.2]))  # ~11 degrees max per step
        self.register_buffer('max_translation', torch.tensor([10.0, 10.0, 10.0]))  # 10mm max per step
        
    def forward(self, x):
        # Split into rotation and translation components
        if self.output_dim == 6:
            rotation = x[:, :3]
            translation = x[:, 3:]
            
            # Apply tanh activation with scaling to constrain ranges
            rotation = torch.tanh(rotation) * self.max_rotation
            translation = torch.tanh(translation) * self.max_translation
            
            return torch.cat([rotation, translation], dim=1)
        else:
            # For other output dimensions, apply general constraint
            return torch.tanh(x) * 0.1

class ImprovedDC2Net(nn.Module):
    def __init__(self, block, layers, num_frames=7, in_channels=1, output_dim=6):
        super(ImprovedDC2Net, self).__init__()
        self.inplanes = 64
        self.num_frames = num_frames
        
        # Initial convolution
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv3d(512 * block.expansion, 512, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Feature extraction for contrastive learning
        self.feature_dim = 512 * block.expansion * num_frames
        
        # Improved trajectory head with anti-spiral mechanisms
        self.trajectory_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
        
        # Anti-spiral components
        self.temporal_smoother = TemporalSmoothingLayer(output_dim)
        self.trajectory_constraint = TrajectoryConstraintLayer(output_dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights to prevent extreme outputs"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input shape: (B, C, D, H, W) = (B, 1, T, 224, 224)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Attention mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Feature vector for contrastive learning
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        
        # Trajectory prediction with anti-spiral mechanisms
        traj_params = self.trajectory_head(features)
        
        # Apply temporal smoothing
        traj_params = self.temporal_smoother(traj_params)
        
        # Apply trajectory constraints
        traj_params = self.trajectory_constraint(traj_params)
        
        return traj_params, features

def create_improved_dc2net(num_frames=7, in_channels=1, output_dim=6):
    """Create improved DCL-Net with anti-spiral mechanisms"""
    return ImprovedDC2Net(block=Bottleneck3D, layers=[3, 4, 6, 3], 
                         num_frames=num_frames, in_channels=in_channels, 
                         output_dim=output_dim)

# Additional loss functions to prevent spiraling
class AntiSpiralLoss(nn.Module):
    """Loss function to penalize spiral-like trajectories"""
    def __init__(self, spiral_penalty=1.0, smoothness_penalty=0.5):
        super(AntiSpiralLoss, self).__init__()
        self.spiral_penalty = spiral_penalty
        self.smoothness_penalty = smoothness_penalty
        
    def forward(self, predictions, targets):
        # Standard MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # Spiral penalty: penalize large rotations
        rotation_penalty = torch.mean(torch.sum(predictions[:, :3] ** 2, dim=1))
        
        # Smoothness penalty: penalize abrupt changes
        if predictions.size(0) > 1:
            diff = predictions[1:] - predictions[:-1]
            smoothness_penalty = torch.mean(torch.sum(diff ** 2, dim=1))
        else:
            smoothness_penalty = 0
        
        total_loss = mse_loss + self.spiral_penalty * rotation_penalty + self.smoothness_penalty * smoothness_penalty
        
        return total_loss

# Example usage
if __name__ == "__main__":
    model = create_improved_dc2net(num_frames=7, in_channels=1)
    input_tensor = torch.randn(2, 1, 7, 224, 224)
    traj_params, features = model(input_tensor)
    
    print("Improved trajectory parameters shape:", traj_params.shape)
    print("Feature vector shape:", features.shape)
    print("Trajectory parameter ranges:")
    print("  Rotation (rad):", traj_params[:, :3].min().item(), "to", traj_params[:, :3].max().item())
    print("  Translation (mm):", traj_params[:, 3:].min().item(), "to", traj_params[:, 3:].max().item())
